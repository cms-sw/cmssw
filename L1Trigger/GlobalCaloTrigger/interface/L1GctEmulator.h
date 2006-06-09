#ifndef L1GCTEMULATOR_H
#define L1GCTEMULATOR_H

/*! 
 * \author Jim Brooke
 * \date May 2006
 */

/*! \class L1GctEmulator L1GctEmulator.h src/L1Trigger/GlobalCaloTrigger/src/L1GctEmulator.h
 *  \brief Framework module that runs the GCT Emulator
 *
 *  An EDProducer that contains an instance of L1GlobalCaloTrigger.
 *
 */


// EDM includes
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

// GCT includes
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

class L1GctEmulator : public edm::EDProducer {
 public:

  /// constructor
  explicit L1GctEmulator(const edm::ParameterSet& ps);

  /// destructor
  virtual ~L1GctEmulator();

  /// method inherited from EDProducer
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:

  // pointer to the actual emulator
  L1GlobalCaloTrigger* m_gct;

  // untracked parameters
  bool m_verbose;

  // tracked parameters
  double m_jetEtLut_A;
  double m_jetEtLut_B;
  double m_jetEtLut_C;


};

#endif
