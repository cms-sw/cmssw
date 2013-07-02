// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDConsumerBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue, 02 Apr 2013 21:36:06 GMT
// $Id: EDConsumerBase.cc,v 1.6 2013/06/07 17:58:32 chrjones Exp $
//

// system include files
#include <cassert>
#include <utility>

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/ProductHolderIndexHelper.h"

using namespace edm;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//EDConsumerBase::EDConsumerBase()
//{
//}

// EDConsumerBase::EDConsumerBase(const EDConsumerBase& rhs)
// {
//    // do actual copying here;
// }

EDConsumerBase::~EDConsumerBase()
{
}

//
// assignment operators
//
// const EDConsumerBase& EDConsumerBase::operator=(const EDConsumerBase& rhs)
// {
//   //An exception safe implementation is
//   EDConsumerBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
ConsumesCollector
EDConsumerBase::consumesCollector() {
  ConsumesCollector c{this};
  return std::move(c);
}


unsigned int
EDConsumerBase::recordConsumes(BranchType iBranch, TypeToGet const& iType, edm::InputTag const& iTag, bool iAlwaysGets) {
  unsigned int index =m_tokenInfo.size();

  const size_t labelSize = iTag.label().size();
  const size_t productInstanceSize = iTag.instance().size();
  unsigned int labelStart = m_tokenLabels.size();
  unsigned short delta1 = labelSize+1;
  unsigned short delta2 = labelSize+2+productInstanceSize;
  m_tokenInfo.emplace_back(TokenLookupInfo{iType.type(),ProductHolderIndexInvalid,iBranch},
                           iAlwaysGets,
                           LabelPlacement{labelStart,delta1,delta2},
                           iType.kind());

  bool skipCurrentProcess = iTag.willSkipCurrentProcess();

  const size_t additionalSize =
      skipCurrentProcess ?
      labelSize+productInstanceSize+3 :
      labelSize+productInstanceSize+iTag.process().size()+3;

  m_tokenLabels.reserve(m_tokenLabels.size()+additionalSize);
  {
    const std::string& m =iTag.label();
    m_tokenLabels.insert(m_tokenLabels.end(),m.begin(),m.end());
    m_tokenLabels.push_back('\0');
  }
  {
    const std::string& m =iTag.instance();
    m_tokenLabels.insert(m_tokenLabels.end(),m.begin(),m.end());
    m_tokenLabels.push_back('\0');
  }
  {
    if (!skipCurrentProcess) {
      const std::string& m =iTag.process();
      m_tokenLabels.insert(m_tokenLabels.end(),m.begin(),m.end());
      m_tokenLabels.push_back('\0');
    } else {
      m_tokenLabels.push_back('\0');
    }
  }
  return index;
}

void
EDConsumerBase::updateLookup(BranchType iBranchType,
                             ProductHolderIndexHelper const& iHelper)
{
  {
    auto itKind = m_tokenInfo.begin<kKind>();
    auto itLabels = m_tokenInfo.begin<kLabels>();
    for(auto itInfo = m_tokenInfo.begin<kLookupInfo>(),itEnd = m_tokenInfo.end<kLookupInfo>();
        itInfo != itEnd; ++itInfo,++itKind,++itLabels) {
      const unsigned int labelStart = itLabels->m_startOfModuleLabel;
      const char* moduleLabel = &(m_tokenLabels[labelStart]);
      itInfo->m_index = iHelper.index(*itKind,
                                      itInfo->m_type,
                                      moduleLabel,
                                      moduleLabel+itLabels->m_deltaToProductInstance,
                                      moduleLabel+itLabels->m_deltaToProcessName);
    }
  }
  
  //now add resolved requests to get many to the end of our list
  // a get many will have an empty module label
  for(size_t i=0, iEnd = m_tokenInfo.size(); i!=iEnd;++i) {
    //need to copy since pointer could be invalidated by emplace_back
    auto const info = m_tokenInfo.get<kLookupInfo>(i);
    if(info.m_branchType == iBranchType &&
       info.m_index == ProductHolderIndexInvalid &&
       m_tokenLabels[m_tokenInfo.get<kLabels>(i).m_startOfModuleLabel]=='\0') {
      //find all matching types
      const auto kind=m_tokenInfo.get<kKind>(i);
      auto matches = iHelper.relatedIndexes(kind,info.m_type);

      //NOTE: This could be changed to contain the true labels for what is being
      // requested but for now I want to remember these are part of a get many
      const LabelPlacement labels= m_tokenInfo.get<kLabels>(i);
      bool alwaysGet = m_tokenInfo.get<kAlwaysGets>(i);
      for(unsigned int j=0;j!=matches.numberOfMatches();++j) {
        //only keep the ones that are for a specific data item and not a collection
        if(matches.isFullyResolved(j)) {
          auto index =matches.index(j);
          m_tokenInfo.emplace_back(TokenLookupInfo{info.m_type,index,info.m_branchType},
                                   alwaysGet,
                                   labels,
                                   kind);
        }
      }
    }
  }
  m_tokenInfo.shrink_to_fit();
}


//
// const member functions
//
ProductHolderIndex
EDConsumerBase::indexFrom(EDGetToken iToken, BranchType iBranch, TypeID const& iType) const
{
  if(unlikely(iToken.index()>=m_tokenInfo.size())) {
    throwBadToken(iType,iToken);
  }
  const auto& info = m_tokenInfo.get<kLookupInfo>(iToken.index());
  if (likely(iBranch == info.m_branchType)) {
    if (likely(iType == info.m_type)) {
      return info.m_index;
    } else {
      throwTypeMismatch(iType, iToken);
    }
  } else {
    throwBranchMismatch(iBranch,iToken);
  }
  return edm::ProductHolderIndexInvalid;
}

void
EDConsumerBase::itemsToGet(BranchType iBranch, std::vector<ProductHolderIndex>& oIndices) const
{
  //how many are we adding?
  unsigned int count=0;
  {
    auto itAlwaysGet = m_tokenInfo.begin<kAlwaysGets>();
    for(auto it = m_tokenInfo.begin<kLookupInfo>(),
        itEnd = m_tokenInfo.end<kLookupInfo>();
        it != itEnd; ++it,++itAlwaysGet) {
      if(iBranch==it->m_branchType) {
        if (it->m_index != ProductHolderIndexInvalid) {
          if(*itAlwaysGet) {
            ++count;
          }
        }
      }
    }
  }
  oIndices.reserve(oIndices.size()+count);
  {
    auto itAlwaysGet = m_tokenInfo.begin<kAlwaysGets>();
    for(auto it = m_tokenInfo.begin<kLookupInfo>(),
        itEnd = m_tokenInfo.end<kLookupInfo>();
        it != itEnd; ++it,++itAlwaysGet) {
      if(iBranch==it->m_branchType) {
        if (it->m_index != ProductHolderIndexInvalid) {
          if(*itAlwaysGet) {
            oIndices.push_back(it->m_index);
          }
        }
      }
    }
  }
}

void
EDConsumerBase::itemsMayGet(BranchType iBranch, std::vector<ProductHolderIndex>& oIndices) const
{
  //how many are we adding?
  unsigned int count=0;
  {
    auto itAlwaysGet = m_tokenInfo.begin<kAlwaysGets>();
    for(auto it = m_tokenInfo.begin<kLookupInfo>(),
        itEnd = m_tokenInfo.end<kLookupInfo>();
        it != itEnd; ++it,++itAlwaysGet) {
      if(iBranch==it->m_branchType) {
        if (it->m_index != ProductHolderIndexInvalid) {
          if(not *itAlwaysGet) {
            ++count;
          }
        }
      }
    }
  }
  oIndices.reserve(oIndices.size()+count);
  {
    auto itAlwaysGet = m_tokenInfo.begin<kAlwaysGets>();
    for(auto it = m_tokenInfo.begin<kLookupInfo>(),
        itEnd = m_tokenInfo.end<kLookupInfo>();
        it != itEnd; ++it,++itAlwaysGet) {
      if(iBranch==it->m_branchType) {
        if (it->m_index != ProductHolderIndexInvalid) {
          if(not *itAlwaysGet) {
            oIndices.push_back(it->m_index);
          }
        }
      }
    }
  }
}

void
EDConsumerBase::labelsForToken(EDGetToken iToken, Labels& oLabels) const
{
  unsigned int index = iToken.index();
  auto labels = m_tokenInfo.get<kLabels>(index);
  unsigned int start = labels.m_startOfModuleLabel;
  oLabels.module = &(m_tokenLabels[start]);
  oLabels.productInstance = oLabels.module+labels.m_deltaToProductInstance;
  oLabels.process = oLabels.module+labels.m_deltaToProcessName;
}


void
EDConsumerBase::throwTypeMismatch(edm::TypeID const& iType, EDGetToken iToken) const
{
  throw cms::Exception("TypeMismatch")<<"A get using a EDGetToken used the C++ type '"<<iType.className()<<"' but the consumes call was for type '"<<m_tokenInfo.get<kLookupInfo>(iToken.index()).m_type.className()<<"'.\n Please modify either the consumes or get call so the types match.";
}
void
EDConsumerBase::throwBranchMismatch(BranchType iBranch, EDGetToken iToken) const {
  throw cms::Exception("BranchTypeMismatch")<<"A get using a EDGetToken was done in "<<BranchTypeToString(iBranch)<<" but the consumes call was for "<<BranchTypeToString(m_tokenInfo.get<kLookupInfo>(iToken.index()).m_branchType)<<".\n Please modify the consumes call to use the correct branch type.";
}

void
EDConsumerBase::throwBadToken(edm::TypeID const& iType, EDGetToken iToken) const
{
  if(iToken.isUnitialized()) {
    throw cms::Exception("BadToken")<<"A get using a EDGetToken with the C++ type '"<<iType.className()<<"' was made using an uninitialized token.\n Please check that the variable is being initialized from a 'consumes' call.";
  }
  throw cms::Exception("BadToken")<<"A get using a EDGetToken with the C++ type '"<<iType.className()<<"' was made using a token with a value "<<iToken.index()<<" which is beyond the range used by this module.\n Please check that the variable is being initialized from a 'consumes' call from this module.\n You can not share EDGetToken values between modules.";
}


//
// static member functions
//
