#ifndef _HLTEcalPhiSymFilter_H
#define _HLTEcalPhiSymFilter_H

// -*- C++ -*-
//
// Package:    HLTEcalPhiSymFilter
// Class:      HLTEcalPhiSymFilter
// 
/**\class HLTEcalPhiSymFilter HLTEcalPhiSymFilter.cc Calibration/EcalAlCaRecoProducers/src/HLTEcalPhiSymFilter.cc

 Description: Producer for EcalRecHits to be used for phi-symmetry ECAL calibration . Discard events in which no suitable rechit is available

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  David Futyan
//         Created:  $Date: 2006/11/21 16:53:03 $
// $Id: HLTEcalPhiSymFilter.h,v 1.2 2006/11/21 16:53:03 malgeri Exp $
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


      virtual bool filter(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 std::string ecalHitsProducer_;
 std::string barrelHits_;
 std::string endcapHits_;
 std::string phiSymBarrelHits_;
 std::string phiSymEndcapHits_;
 double eCut_barl_;
 double eCut_endc_;  
};

#endif
