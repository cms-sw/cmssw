#ifndef L1GCTEMULATOR_H
#define L1GCTEMULATOR_H

/**\class L1GctEmulator L1GctEmulator.h src/L1Trigger/GlobalCaloTrigger/src/L1GctEmulator.h

 Description:  Framework module that runs the GCT bit-level emulator

 Implementation:
       An EDProducer that contains an instance of L1GlobalCaloTrigger.
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 15:04:56 CEST 2006
// $Id: L1GctEmulator.h,v 1.2 2007/05/20 22:21:43 jbrooke Exp $
//
//

// system includes


// EDM includes
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Utilities/interface/Exception.h"


class L1GlobalCaloTrigger;
class L1GctJetEtCalibrationLut;


class L1GctEmulator : public edm::EDProducer {
 public:

  /// constructor
  explicit L1GctEmulator(const edm::ParameterSet& ps);

  /// destructor
  ~L1GctEmulator();

 private:
  void beginJob(const edm::EventSetup& c) ;
  void produce(edm::Event& e, const edm::EventSetup& c);
  void endJob() ;

  void configureGct(const edm::EventSetup& c) ;

  // input label
  std::string m_inputLabel;

  // pointer to the actual emulator
  L1GlobalCaloTrigger* m_gct;

  // pointer to the jet Et LUT
  L1GctJetEtCalibrationLut* m_jetEtCalibLut;

  // untracked parameters
  bool m_verbose;

  // tracked parameters

};

#endif
