#ifndef L1TriggerConfig_L1ScalesProducers_L1CaloInputScalesProducer_h
#define L1TriggerConfig_L1ScalesProducers_L1CaloInputScalesProducer_h
// -*- C++ -*-
//
// Package:     L1ScalesProducers
// Class  :     L1CaloInputScalesProducer
// 
/**\class L1CaloInputScalesProducer L1CaloInputScalesProducer.h L1TriggerConfig/L1ScalesProducers/interface/L1CaloInputScalesProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Fri May 30 18:25:56 CEST 2008
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"


// forward declarations

class L1CaloInputScalesProducer : public edm::ESProducer {
   public:
      L1CaloInputScalesProducer(const edm::ParameterSet&);
      ~L1CaloInputScalesProducer() override;

      //typedef std::shared_ptr<L1CaloInputScale> ReturnType;

      std::unique_ptr<L1CaloEcalScale>
	produceEcalScale(const L1CaloEcalScaleRcd&);
      std::unique_ptr<L1CaloHcalScale>
	produceHcalScale(const L1CaloHcalScaleRcd&);
   private:
      // ----------member data ---------------------------
  std::vector<double> m_ecalEtThresholdsPosEta;
  std::vector<double> m_ecalEtThresholdsNegEta;
  std::vector<double> m_hcalEtThresholdsPosEta;
  std::vector<double> m_hcalEtThresholdsNegEta;

};

#endif
