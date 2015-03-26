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
//

// system include files
#include <algorithm>
#include <cassert>
#include <cstring>
#include <set>
#include <utility>

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/ProductHolderIndexHelper.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

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
  return c;
}

static const edm::InputTag kWasEmpty("@EmptyLabel@");

edm::InputTag const&
EDConsumerBase::checkIfEmpty(edm::InputTag const& iTag) {
  if (iTag.label().empty()) {
    return kWasEmpty;
  }
  return iTag;
}

unsigned int
EDConsumerBase::recordConsumes(BranchType iBranch, TypeToGet const& iType, edm::InputTag const& iTag, bool iAlwaysGets) {

  if(frozen_) {
    throwConsumesCallAfterFrozen(iType, iTag);
  }

  unsigned int index =m_tokenInfo.size();

  bool skipCurrentProcess = iTag.willSkipCurrentProcess();

  const size_t labelSize = iTag.label().size();
  const size_t productInstanceSize = iTag.instance().size();
  unsigned int labelStart = m_tokenLabels.size();
  unsigned short delta1 = labelSize+1;
  unsigned short delta2 = labelSize+2+productInstanceSize;
  m_tokenInfo.emplace_back(TokenLookupInfo{iType.type(), ProductHolderIndexInvalid, skipCurrentProcess, iBranch},
                           iAlwaysGets,
                           LabelPlacement{labelStart,delta1,delta2},
                           iType.kind());

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
  frozen_ = true;
  {
    auto itKind = m_tokenInfo.begin<kKind>();
    auto itLabels = m_tokenInfo.begin<kLabels>();
    for(auto itInfo = m_tokenInfo.begin<kLookupInfo>(),itEnd = m_tokenInfo.end<kLookupInfo>();
        itInfo != itEnd; ++itInfo,++itKind,++itLabels) {
      if(itInfo->m_branchType == iBranchType) {
        const unsigned int labelStart = itLabels->m_startOfModuleLabel;
        const char* moduleLabel = &(m_tokenLabels[labelStart]);
        itInfo->m_index = ProductHolderIndexAndSkipBit(iHelper.index(*itKind,
                                                                     itInfo->m_type,
                                                                     moduleLabel,
                                                                     moduleLabel+itLabels->m_deltaToProductInstance,
                                                                     moduleLabel+itLabels->m_deltaToProcessName),
                                                       itInfo->m_index.skipCurrentProcess());
      }
    }
  }

  //now add resolved requests to get many to the end of our list
  // a get many will have an empty module label
  for(size_t i=0, iEnd = m_tokenInfo.size(); i!=iEnd;++i) {
    //need to copy since pointer could be invalidated by emplace_back
    auto const info = m_tokenInfo.get<kLookupInfo>(i);
    if(info.m_branchType == iBranchType &&
       info.m_index.productHolderIndex() == ProductHolderIndexInvalid &&
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
          m_tokenInfo.emplace_back(TokenLookupInfo{info.m_type, index, info.m_index.skipCurrentProcess(), info.m_branchType},
                                   alwaysGet,
                                   labels,
                                   kind);
        }
      }
    }
  }
  m_tokenInfo.shrink_to_fit();

  if(iBranchType == InEvent) {
    itemsToGet(iBranchType, itemsToGetFromEvent_);
  }
}

//
// const member functions
//
ProductHolderIndexAndSkipBit
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
  return ProductHolderIndexAndSkipBit(edm::ProductHolderIndexInvalid, false);
}

void
EDConsumerBase::itemsToGet(BranchType iBranch, std::vector<ProductHolderIndexAndSkipBit>& oIndices) const
{
  //how many are we adding?
  unsigned int count=0;
  {
    auto itAlwaysGet = m_tokenInfo.begin<kAlwaysGets>();
    for(auto it = m_tokenInfo.begin<kLookupInfo>(),
        itEnd = m_tokenInfo.end<kLookupInfo>();
        it != itEnd; ++it,++itAlwaysGet) {
      if(iBranch==it->m_branchType) {
        if (it->m_index.productHolderIndex() != ProductHolderIndexInvalid) {
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
        if (it->m_index.productHolderIndex() != ProductHolderIndexInvalid) {
          if(*itAlwaysGet) {
            oIndices.push_back(it->m_index);
          }
        }
      }
    }
  }
}

void
EDConsumerBase::itemsMayGet(BranchType iBranch, std::vector<ProductHolderIndexAndSkipBit>& oIndices) const
{
  //how many are we adding?
  unsigned int count=0;
  {
    auto itAlwaysGet = m_tokenInfo.begin<kAlwaysGets>();
    for(auto it = m_tokenInfo.begin<kLookupInfo>(),
        itEnd = m_tokenInfo.end<kLookupInfo>();
        it != itEnd; ++it,++itAlwaysGet) {
      if(iBranch==it->m_branchType) {
        if (it->m_index.productHolderIndex() != ProductHolderIndexInvalid) {
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
        if (it->m_index.productHolderIndex() != ProductHolderIndexInvalid) {
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

bool
EDConsumerBase::registeredToConsume(ProductHolderIndex iIndex, bool skipCurrentProcess, BranchType iBranch) const
{
  for(auto it = m_tokenInfo.begin<kLookupInfo>(),
      itEnd = m_tokenInfo.end<kLookupInfo>();
      it != itEnd; ++it) {
    if(it->m_index.productHolderIndex() == iIndex and
       it->m_index.skipCurrentProcess() == skipCurrentProcess and
       it->m_branchType == iBranch) {
      return true;
    }
  }
  //TEMPORARY: Remember so we do not have to do this again
  //non thread-safe
  EDConsumerBase* nonConstThis = const_cast<EDConsumerBase*>(this);
  nonConstThis->m_tokenInfo.emplace_back(TokenLookupInfo{TypeID{}, iIndex, skipCurrentProcess, iBranch},
                                         true,
                                         LabelPlacement{0,0,0},
                                         PRODUCT_TYPE);

  return false;
}

bool
EDConsumerBase::registeredToConsumeMany(TypeID const& iType, BranchType iBranch) const
{
  for(auto it = m_tokenInfo.begin<kLookupInfo>(),
      itEnd = m_tokenInfo.end<kLookupInfo>();
      it != itEnd; ++it) {
    //consumesMany entries do not have their index resolved
    if(it->m_index.productHolderIndex() == ProductHolderIndexInvalid and
       it->m_type == iType and
       it->m_branchType == iBranch) {
      return true;
    }
  }
  //TEMPORARY: Remember so we do not have to do this again
  //non thread-safe
  EDConsumerBase* nonConstThis = const_cast<EDConsumerBase*>(this);
  nonConstThis->m_tokenInfo.emplace_back(TokenLookupInfo{iType,ProductHolderIndexInvalid, false, iBranch},
                           true,
                           LabelPlacement{0,0,0},
                           PRODUCT_TYPE);
  return false;
  
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
  if(iToken.isUninitialized()) {
    throw cms::Exception("BadToken")<<"A get using a EDGetToken with the C++ type '"<<iType.className()<<"' was made using an uninitialized token.\n Please check that the variable is being initialized from a 'consumes' call.";
  }
  throw cms::Exception("BadToken")<<"A get using a EDGetToken with the C++ type '"<<iType.className()<<"' was made using a token with a value "<<iToken.index()<<" which is beyond the range used by this module.\n Please check that the variable is being initialized from a 'consumes' call from this module.\n You can not share EDGetToken values between modules.";
}

void
EDConsumerBase::throwConsumesCallAfterFrozen(TypeToGet const& typeToGet, InputTag const& inputTag) const {
  throw cms::Exception("LogicError") << "A module declared it consumes a product after its constructor.\n"
                                     << "This must be done in the contructor\n"
                                     << "The product type was: " << typeToGet.type() << "\n"
                                     << "and " << inputTag << "\n";
}

namespace {
  struct CharStarComp {
    bool operator()(const char* iLHS, const char* iRHS) const {
      return strcmp(iLHS,iRHS) < 0;
    }
  };
}

void
EDConsumerBase::modulesDependentUpon(const std::string& iProcessName,
                                     std::vector<const char*>& oModuleLabels) const {
  std::set<const char*, CharStarComp> uniqueModules;
  for(unsigned int index=0, iEnd=m_tokenInfo.size();index <iEnd; ++index) {
    auto const& info = m_tokenInfo.get<kLookupInfo>(index);
    if( not info.m_index.skipCurrentProcess() ) {
      auto labels = m_tokenInfo.get<kLabels>(index);
      unsigned int start = labels.m_startOfModuleLabel;
      const char* processName = &(m_tokenLabels[start+labels.m_deltaToProcessName]);
      if( (not processName) or processName[0]==0 or
         iProcessName == processName) {
        uniqueModules.insert(&(m_tokenLabels[start]));
      }
    }
  }
  
  oModuleLabels = std::vector<const char*>(uniqueModules.begin(),uniqueModules.end());
}


namespace {
  void
  insertFoundModuleLabel(const char* consumedModuleLabel,
                         std::vector<ModuleDescription const*>& modules,
                         std::set<std::string>& alreadyFound,
                         std::map<std::string, ModuleDescription const*> const& labelsToDesc,
                         ProductRegistry const& preg) {
    // Convert from label string to module description, eliminate duplicates,
    // then insert into the vector of modules
    auto it = labelsToDesc.find(consumedModuleLabel);
    if(it != labelsToDesc.end()) {
      if(alreadyFound.insert(consumedModuleLabel).second) {
        modules.push_back(it->second);
      }
      return;
    }
    // Deal with EDAlias's by converting to the original module label first
    std::vector<std::pair<std::string, std::string> > const& aliasToOriginal = preg.aliasToOriginal();
    std::pair<std::string, std::string> target(consumedModuleLabel, std::string());
    auto iter = std::lower_bound(aliasToOriginal.begin(), aliasToOriginal.end(), target);
    if(iter != aliasToOriginal.end() && iter->first == consumedModuleLabel) {

      std::string const& originalModuleLabel = iter->second;
      auto iter2 = labelsToDesc.find(originalModuleLabel);
      if(iter2 != labelsToDesc.end()) {
        if(alreadyFound.insert(originalModuleLabel).second) {
          modules.push_back(iter2->second);
        }
        return;
      }
    }
    // Ignore the source products, we are only interested in module products.
    // As far as I know, it should never be anything else so throw if something
    // unknown gets passed in.
    if(std::string(consumedModuleLabel) != "source") {
      throw cms::Exception("EDConsumerBase", "insertFoundModuleLabel")
        << "Couldn't find ModuleDescription for the consumed module label: "
        << std::string(consumedModuleLabel) << "\n";
    }
  }
}

void
EDConsumerBase::modulesWhoseProductsAreConsumed(std::vector<ModuleDescription const*>& modules,
                                                ProductRegistry const& preg,
                                                std::map<std::string, ModuleDescription const*> const& labelsToDesc,
                                                std::string const& processName) const {

  ProductHolderIndexHelper const& iHelper = *preg.productLookup(InEvent);

  std::set<std::string> alreadyFound;

  auto itKind = m_tokenInfo.begin<kKind>();
  auto itLabels = m_tokenInfo.begin<kLabels>();
  for(auto itInfo = m_tokenInfo.begin<kLookupInfo>(),itEnd = m_tokenInfo.end<kLookupInfo>();
      itInfo != itEnd; ++itInfo,++itKind,++itLabels) {

    if(itInfo->m_branchType == InEvent) {

      const unsigned int labelStart = itLabels->m_startOfModuleLabel;
      const char* consumedModuleLabel = &(m_tokenLabels[labelStart]);
      const char* consumedProcessName = consumedModuleLabel+itLabels->m_deltaToProcessName;

      if(*consumedModuleLabel != '\0') { // not a consumesMany
        if(*consumedProcessName != '\0') { // process name is specified in consumes call
          if (processName == consumedProcessName &&
              iHelper.index(*itKind,
                            itInfo->m_type,
                            consumedModuleLabel,
                            consumedModuleLabel+itLabels->m_deltaToProductInstance,
                            consumedModuleLabel+itLabels->m_deltaToProcessName) != ProductHolderIndexInvalid) {
            insertFoundModuleLabel(consumedModuleLabel, modules, alreadyFound, labelsToDesc, preg);
          }
        } else { // process name was empty
          auto matches = iHelper.relatedIndexes(*itKind,
                                                itInfo->m_type,
                                                consumedModuleLabel,
                                                consumedModuleLabel+itLabels->m_deltaToProductInstance);
          for(unsigned int j = 0; j < matches.numberOfMatches(); ++j) {
            if(processName == matches.processName(j)) {
              insertFoundModuleLabel(consumedModuleLabel, modules, alreadyFound, labelsToDesc, preg);
            }
          }
        }
      // consumesMany case
      } else if(itInfo->m_index.productHolderIndex() == ProductHolderIndexInvalid) {
        auto matches = iHelper.relatedIndexes(*itKind,
                                              itInfo->m_type);
        for(unsigned int j = 0; j < matches.numberOfMatches(); ++j) {
          if(processName == matches.processName(j)) {
            insertFoundModuleLabel(matches.moduleLabel(j), modules, alreadyFound, labelsToDesc, preg);
          }
        }
      }
    }
  }
}

std::vector<ConsumesInfo>
EDConsumerBase::consumesInfo() const {

  // Use this to eliminate duplicate entries related
  // to consumesMany items where only the type was specified
  // and the there are multiple matches. In these cases the
  // label, instance, and process will be empty.
  std::set<edm::TypeID> alreadySeenTypes;

  std::vector<ConsumesInfo> result;
  auto itAlways = m_tokenInfo.begin<kAlwaysGets>();
  auto itKind = m_tokenInfo.begin<kKind>();
  auto itLabels = m_tokenInfo.begin<kLabels>();
  for(auto itInfo = m_tokenInfo.begin<kLookupInfo>(),itEnd = m_tokenInfo.end<kLookupInfo>();
      itInfo != itEnd; ++itInfo,++itKind,++itLabels, ++itAlways) {

    const unsigned int labelStart = itLabels->m_startOfModuleLabel;
    const char* consumedModuleLabel = &(m_tokenLabels[labelStart]);
    const char* consumedInstance = consumedModuleLabel+itLabels->m_deltaToProductInstance;
    const char* consumedProcessName = consumedModuleLabel+itLabels->m_deltaToProcessName;

    // consumesMany case
    if(*consumedModuleLabel == '\0') {
      if(!alreadySeenTypes.insert(itInfo->m_type).second) {
        continue;
      }
    }

    // Just copy the information into the ConsumesInfo data structure
    result.emplace_back(itInfo->m_type,
                        consumedModuleLabel,
                        consumedInstance,
                        consumedProcessName,
                        itInfo->m_branchType,
                        *itKind,
                        *itAlways,
                        itInfo->m_index.skipCurrentProcess());
  }
  return result;
}
