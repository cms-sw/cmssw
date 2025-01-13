#ifndef Subsystem_Package_GMTPutTokens_h
#define Subsystem_Package_GMTPutTokens_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     GMTPutTokens
//
/**\class GMTPutTokens GMTPutTokens.h "GMTPutTokens.h"

 Description: Holder for the EDPutTokens

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 11 Dec 2024 13:41:10 GMT
//
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"

#include "L1TObjectCollections.h"

// system include files
#include <vector>

// user include files
#include "FWCore/Utilities/interface/EDPutToken.h"

// forward declarations

namespace l1t {
  namespace stage2 {
    struct GMTPutTokens {
      edm::EDPutTokenT<RegionalMuonCandBxCollection> bmtf_;
      edm::EDPutTokenT<RegionalMuonCandBxCollection> omtf_;
      edm::EDPutTokenT<RegionalMuonCandBxCollection> emtf_;

      edm::EDPutTokenT<MuonBxCollection> muon_;
      std::vector<edm::EDPutTokenT<MuonBxCollection>> muonCopies_;

      edm::EDPutTokenT<MuonBxCollection> imdMuonsBMTF_;
      edm::EDPutTokenT<MuonBxCollection> imdMuonsEMTFNeg_;
      edm::EDPutTokenT<MuonBxCollection> imdMuonsEMTFPos_;
      edm::EDPutTokenT<MuonBxCollection> imdMuonsOMTFNeg_;
      edm::EDPutTokenT<MuonBxCollection> imdMuonsOMTFPos_;

      edm::EDPutTokenT<RegionalMuonShowerBxCollection> showerEMTF_;
      edm::EDPutTokenT<MuonShowerBxCollection> muonShower_;
      std::vector<edm::EDPutTokenT<MuonShowerBxCollection>> muonShowerCopy_;
    };
  }  // namespace stage2
}  // namespace l1t
#endif
