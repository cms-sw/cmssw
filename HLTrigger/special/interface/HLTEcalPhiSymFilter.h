#ifndef _HLTEcalPhiSymFilter_H
#define _HLTEcalPhiSymFilter_H

// -*- C++ -*-
//
// Package:    HLTEcalPhiSymFilter
// Class:      HLTEcalPhiSymFilter
// 
/**\class HLTEcalPhiSymFilter HLTEcalPhiSymFilter.cc Calibration/EcalAlCaRecoProducers/src/HLTEcalPhiSymFilter.cc

* Description: Producer for EcalRecHits to be used for phi-symmetry ECAL 
* calibration . Discard events in which no suitable rechit is available.
* Rechits are accepted if their energy is above a threshold, eCut_barl_ 
* for the barrel and  eCut_endc_. However, if the status of the channel is 
* marked bad at some level, given by statusThreshold_, then a higher 
* threshold (e.g. eCut_barl_high_) is applied. If  parameter useRecoFlag_  
* is true, statusThreshold_ acts on EcalRecHit::recoFlag(), while if it is 
* false, it acts on the ChannelStatus record from the database.

*/

//
// Original Author:  David Futyan
// HLT Port       :  Stefano Argiro
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTEcalPhiSymFilter : public edm::global::EDFilter<> {
public:
  HLTEcalPhiSymFilter(const edm::ParameterSet&);
  ~HLTEcalPhiSymFilter();

  virtual bool filter(edm::StreamID, edm::Event & event, const edm::EventSetup & setup) const override final;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  const edm::EDGetTokenT<EBDigiCollection> barrelDigisToken_;
  const edm::EDGetTokenT<EEDigiCollection> endcapDigisToken_;
  const edm::EDGetTokenT<EcalUncalibratedRecHitCollection> barrelUncalibHitsToken_;
  const edm::EDGetTokenT<EcalUncalibratedRecHitCollection> endcapUncalibHitsToken_;
  const edm::EDGetTokenT<EBRecHitCollection> barrelHitsToken_;
  const edm::EDGetTokenT<EERecHitCollection> endcapHitsToken_;
  const std::string phiSymBarrelDigis_;
  const std::string phiSymEndcapDigis_;
  const std::vector<double> ampCut_barlP_;
  const std::vector<double> ampCut_barlM_;
  const std::vector<double> ampCut_endcP_; 
  const std::vector<double> ampCut_endcM_; 
  const uint32_t statusThreshold_; ///< accept channels with up to this status
  const bool   useRecoFlag_;       ///< use recoflag instead of DB for bad channels
};

#endif
