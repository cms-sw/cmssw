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
// $Id$
//

// system include files
#include <cassert>

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"
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

unsigned int
EDConsumerBase::recordConsumes(BranchType iBranch, TypeToGet const& iType, edm::InputTag const& iTag, bool iAlwaysGets) {
  unsigned int index =m_tokenToLookup.size();
  m_tokenToLookup.emplace_back(iType.type(),ProductHolderIndexInvalid,iBranch);
  m_tokenAlwaysGets.push_back(iAlwaysGets);
  m_tokenKind.push_back(iType.kind());
  const size_t additionalSize = iTag.label().size()+iTag.instance().size()+iTag.process().size()+3;
  m_tokenLabels.reserve(m_tokenLabels.size()+additionalSize);
  {
    m_tokenStartOfLabels.push_back(m_tokenLabels.size());
    const std::string& m =iTag.label();
    m_tokenLabels.insert(m_tokenLabels.end(),m.begin(),m.end());
    m_tokenLabels.push_back('\0');
  }
  {
    m_tokenStartOfLabels.push_back(m_tokenLabels.size());
    const std::string& m =iTag.instance();
    m_tokenLabels.insert(m_tokenLabels.end(),m.begin(),m.end());
    m_tokenLabels.push_back('\0');
  }
  {
    m_tokenStartOfLabels.push_back(m_tokenLabels.size());
    const std::string& m =iTag.process();
    m_tokenLabels.insert(m_tokenLabels.end(),m.begin(),m.end());
    m_tokenLabels.push_back('\0');
  }
  return index;
}

void
EDConsumerBase::updateLookup(BranchType iBranchType,
                             ProductHolderIndexHelper const& iHelper)
{
  
  size_t i = 0;
  for(auto & info: m_tokenToLookup){
    if(info.m_branchType == iBranchType) {
      unsigned int l = 3*i;
      info.m_index = iHelper.index(m_tokenKind[i],
                                   info.m_type,
                                   &(m_tokenLabels[m_tokenStartOfLabels[l]]),
                                   &(m_tokenLabels[m_tokenStartOfLabels[l+1]]),
                                   &(m_tokenLabels[m_tokenStartOfLabels[l+2]]));
    }
    ++i;
  }
  //now add resolved requests to get many to the end of our list
  // a get many will have an empty module label
  for(size_t i=0, iEnd = m_tokenToLookup.size(); i!=iEnd;++i) {
    //need to copy since pointer could be invalidated by emplace_back
    auto const info = m_tokenToLookup[i];
    if(info.m_branchType == iBranchType &&
       info.m_index == ProductHolderIndexInvalid &&
       m_tokenLabels[m_tokenStartOfLabels[i*3]]=='\0') {
      //find all matching types
      auto matches = iHelper.relatedIndexes(m_tokenKind[i],info.m_type);
      for(unsigned int j=0;j!=matches.numberOfMatches();++j) {
        //only keep the ones that are for a specific data item and not a collection
        if(matches.isFullyResolved(j)) {
          auto index =matches.index(j);
          m_tokenToLookup.emplace_back(info.m_type,index,info.m_branchType);

          //NOTE: must keep all of these data structures in synch
          m_tokenAlwaysGets.push_back(m_tokenAlwaysGets[i]);
          m_tokenKind.push_back(m_tokenKind[i]);
          //NOTE: This could be changed to contain the true labels for what is being
          // requested but for now I want to remember these are part of a get many
          m_tokenStartOfLabels.push_back(m_tokenStartOfLabels[i*3]);
          m_tokenStartOfLabels.push_back(m_tokenStartOfLabels[i*3+1]);
          m_tokenStartOfLabels.push_back(m_tokenStartOfLabels[i*3+2]);
        }
      }
    }
  }
}


//
// const member functions
//
ProductHolderIndex
EDConsumerBase::indexFrom(EDGetToken iToken, BranchType iBranch, TypeID const& iType) const
{
  assert(iToken.value()<m_tokenToLookup.size());
  const auto& info = m_tokenToLookup[iToken.value()];
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
  unsigned int i=0;
  for(auto const& info: m_tokenToLookup) {
    if(iBranch==info.m_branchType) {
      if (info.m_index != ProductHolderIndexInvalid) {
        if(m_tokenAlwaysGets[i]) {
          ++count;
        }
      }
    }
  }
  oIndices.reserve(oIndices.size()+count);

  for(auto const& info: m_tokenToLookup) {
    if(iBranch==info.m_branchType) {
      if (info.m_index != ProductHolderIndexInvalid) {
        if(m_tokenAlwaysGets[i]) {
          oIndices.push_back(info.m_index);
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
  unsigned int i=0;
  for(auto const& info: m_tokenToLookup) {
    if(iBranch==info.m_branchType) {
      if (info.m_index != ProductHolderIndexInvalid) {
        if(not m_tokenAlwaysGets[i]) {
          ++count;
        }
      }
    }
  }
  oIndices.reserve(oIndices.size()+count);
  
  for(auto const& info: m_tokenToLookup) {
    if(iBranch==info.m_branchType) {
      if (info.m_index != ProductHolderIndexInvalid) {
        if(not m_tokenAlwaysGets[i]) {
          oIndices.push_back(info.m_index);
        }
      }
    }
  }  
}

void
EDConsumerBase::labelsForToken(EDGetToken iToken, Labels& oLabels) const
{
  unsigned int index = 3*iToken.value();
  oLabels.module = &(m_tokenLabels[m_tokenStartOfLabels[index]]);
  oLabels.productInstance = &(m_tokenLabels[m_tokenStartOfLabels[index+1]]);
  oLabels.process = &(m_tokenLabels[m_tokenStartOfLabels[index+2]]);
}


void
EDConsumerBase::throwTypeMismatch(edm::TypeID const& iType, EDGetToken iToken) const
{
  throw cms::Exception("TypeMismatch")<<"A get using a EDGetToken used the C++ type '"<<iType.className()<<"' but the consumes call was for type '"<<m_tokenToLookup[iToken.value()].m_type.className()<<"'.\n Please modify either the consumes or get call so the types match.";
}
void
EDConsumerBase::throwBranchMismatch(BranchType iBranch, EDGetToken iToken) const {
  throw cms::Exception("BranchTypeMismatch")<<"A get using a EDGetToken was done in "<<BranchTypeToString(iBranch)<<" but the consumes call was for "<<BranchTypeToString(m_tokenToLookup[iToken.value()].m_branchType)<<".\n Please modify the consumes call to use the correct branch type.";
}


//
// static member functions
//
