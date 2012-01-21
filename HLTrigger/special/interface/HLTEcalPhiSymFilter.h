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
//         Created:  $Date: 2009/10/06 08:27:49 $
// $Id: HLTEcalPhiSymFilter.h,v 1.3 2009/10/06 08:27:49 argiro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class HLTEcalPhiSymFilter : public HLTFilter {
   public:
      explicit HLTEcalPhiSymFilter(const edm::ParameterSet&);
      ~HLTEcalPhiSymFilter();


      virtual bool hltFilter(edm::Event &, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
   private:
      // ----------member data ---------------------------

 
 edm::InputTag barrelHits_;
 edm::InputTag endcapHits_;
 std::string phiSymBarrelHits_;
 std::string phiSymEndcapHits_;
 double eCut_barl_;
 double eCut_endc_;  
 double eCut_barl_high_;
 double eCut_endc_high_; 
 uint32_t statusThreshold_; ///< accept channels with up to this status
 bool   useRecoFlag_;       ///< use recoflag instead of DB for bad channels
};

#endif
