#ifndef FWCore_Framework_EDConsumerBase_h
#define FWCore_Framework_EDConsumerBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDConsumerBase
// 
/**\class EDConsumerBase EDConsumerBase.h "EDConsumerBase.h"

 Description: Allows declaration of what data is being consumed

 Usage:
    The EDM modules all inherit from this base class

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 02 Apr 2013 21:35:53 GMT
// $Id: EDConsumerBase.h,v 1.1 2013/04/14 19:06:48 chrjones Exp $
//

// system include files
#include <vector>

// user include files
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeToGet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductHolderIndex.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"


// forward declarations

namespace edm {
  class ProductHolderIndexHelper;
  
  class EDConsumerBase
  {
    
  public:
    EDConsumerBase() {}
    virtual ~EDConsumerBase();
    
    // ---------- const member functions ---------------------
    ProductHolderIndex indexFrom(EDGetToken, BranchType, TypeID const&) const;
    
    void itemsToGet(BranchType, std::vector<ProductHolderIndex>&) const;
    void itemsMayGet(BranchType, std::vector<ProductHolderIndex>&) const;
    
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    void updateLookup(BranchType iBranchType,
                      ProductHolderIndexHelper const&);
    
    struct Labels {
      const char*  module;
      const char*  productInstance;
      const char*  process;
    };
    void labelsForToken(EDGetToken iToken, Labels& oLabels) const;
    
  protected:
    
    template <typename ProductType, BranchType B=InEvent>
    EDGetTokenT<ProductType> consumes(edm::InputTag const& tag) {
      TypeToGet tid=TypeToGet::make<ProductType>();
      return EDGetTokenT<ProductType>{recordConsumes(B,tid, tag,true)};
    }

    EDGetToken consumes(const TypeToGet& id, edm::InputTag const& tag) {
      return EDGetToken{recordConsumes(InEvent,id,tag,true)};
    }
    
    template <BranchType B>
    EDGetToken consumes(TypeToGet const& id, edm::InputTag const& tag) {
      return EDGetToken{recordConsumes(B,id,tag,true)};
    }

    template <typename ProductType, BranchType B=InEvent>
    EDGetTokenT<ProductType> mayConsume(edm::InputTag const& tag) {
      TypeToGet tid=TypeToGet::make<ProductType>();
      return mayConsume<B>(tid,tag);
    }
    
    
    EDGetToken mayConsume(const TypeToGet& id, edm::InputTag const& tag) {
      return mayConsume<InEvent>(id,tag);
    }
    
    template <BranchType B>
    EDGetToken mayConsume(const TypeToGet& id, edm::InputTag const& tag) {
      return EDGetToken{recordConsumes(B,id,tag,false)};
    }

    template <typename ProductType, BranchType B=InEvent>
    void consumesMany() {
      TypeToGet tid=TypeToGet::make<ProductType>();
      consumesMany<B>(tid);
    }
    
    
    void consumesMany(const TypeToGet& id) {
      consumesMany<InEvent>(id);
    }
    
    template <BranchType B>
    void consumesMany(const TypeToGet& id) {
      recordConsumes(B,id,edm::InputTag{},true);
    }

  private:
    EDConsumerBase(const EDConsumerBase&) = delete;
    
    const EDConsumerBase& operator=(const EDConsumerBase&) = delete;
    
    unsigned int recordConsumes(BranchType iBranch, TypeToGet const& iType, edm::InputTag const& iTag, bool iAlwaysGets);
    
    void throwTypeMismatch(edm::TypeID const&, EDGetToken) const;
    void throwBranchMismatch(BranchType, EDGetToken) const;
    void throwBadToken(edm::TypeID const& iType, EDGetToken iToken) const;
    
    // ---------- member data --------------------------------
    struct TokenLookupInfo {
      TokenLookupInfo(edm::TypeID const& iID,
                      ProductHolderIndex iIndex,
                      BranchType iBranch):
      m_type(iID),m_index(iIndex),m_branchType(iBranch){}
      edm::TypeID m_type;
      ProductHolderIndex m_index;
      BranchType m_branchType;
    };
    std::vector<TokenLookupInfo> m_tokenToLookup;

    //Each of these has one entry per token
    std::vector<bool> m_tokenAlwaysGets;
    std::vector<unsigned int> m_tokenStartOfLabels;
    std::vector<edm::KindOfType> m_tokenKind;

    //m_tokenStartOfLabels holds the entries into this container
    // for each of the 3 labels needed to id the data
    std::vector<char> m_tokenLabels;
    
  };
}


#endif
