#ifndef _HLTHcalPhiSymFilter_H
#define _HLTHcalPhiSymFilter_H

// -*- C++ -*-
//
// Package:    HLTHcalPhiSymFilter
// Class:      HLTHcalPhiSymFilter
//
/**\class HLTHcalPhiSymFilter HLTHcalPhiSymFilter.cc

 Description: Producer for HcalRecHits to be used for phi-symmetry HCAL calibration . Discard events in which no suitable rechit is available

*/
//
// Original Author:  Grigory Safronov
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// class decleration
//

class HLTHcalPhiSymFilter : public HLTFilter {
   public:
      explicit HLTHcalPhiSymFilter(const edm::ParameterSet&);
      ~HLTHcalPhiSymFilter() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event &, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      // ----------member data ---------------------------

 edm::EDGetTokenT<HBHERecHitCollection> HBHEHitsToken_;
 edm::EDGetTokenT<HORecHitCollection> HOHitsToken_;
 edm::EDGetTokenT<HFRecHitCollection> HFHitsToken_;
 edm::InputTag HBHEHits_;
 edm::InputTag HOHits_;
 edm::InputTag HFHits_;
 std::string phiSymHBHEHits_;
 std::string phiSymHOHits_;
 std::string phiSymHFHits_;
 double eCut_HB_;
 double eCut_HE_;
 double eCut_HO_;
 double eCut_HF_;

};

#endif

