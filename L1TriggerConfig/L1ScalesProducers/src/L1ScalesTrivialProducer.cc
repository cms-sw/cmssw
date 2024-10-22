#include "L1TriggerConfig/L1ScalesProducers/interface/L1ScalesTrivialProducer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1ScalesTrivialProducer::L1ScalesTrivialProducer(const edm::ParameterSet& ps) {
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &L1ScalesTrivialProducer::produceEmScale);
  setWhatProduced(this, &L1ScalesTrivialProducer::produceJetScale);
  setWhatProduced(this, &L1ScalesTrivialProducer::produceHtMissScale);
  setWhatProduced(this, &L1ScalesTrivialProducer::produceHfRingScale);

  //now do what ever other initialization is needed

  // get numbers from the config file -  all units are GeV
  m_emEtScaleInputLsb = ps.getParameter<double>("L1CaloEmEtScaleLSB");
  m_emEtThresholds = ps.getParameter<std::vector<double> >("L1CaloEmThresholds");

  m_jetEtScaleInputLsb = ps.getParameter<double>("L1CaloRegionEtScaleLSB");
  m_jetEtThresholds = ps.getParameter<std::vector<double> >("L1CaloJetThresholds");

  m_htMissThresholds = ps.getParameter<std::vector<double> >("L1HtMissThresholds");
  m_hfRingThresholds = ps.getParameter<std::vector<double> >("L1HfRingThresholds");
}

L1ScalesTrivialProducer::~L1ScalesTrivialProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
std::unique_ptr<L1CaloEtScale> L1ScalesTrivialProducer::produceEmScale(const L1EmEtScaleRcd& iRecord) {
  return std::make_unique<L1CaloEtScale>(m_emEtScaleInputLsb, m_emEtThresholds);
}

std::unique_ptr<L1CaloEtScale> L1ScalesTrivialProducer::produceJetScale(const L1JetEtScaleRcd& iRecord) {
  return std::make_unique<L1CaloEtScale>(m_jetEtScaleInputLsb, m_jetEtThresholds);
}

std::unique_ptr<L1CaloEtScale> L1ScalesTrivialProducer::produceHtMissScale(const L1HtMissScaleRcd& iRecord) {
  return std::make_unique<L1CaloEtScale>(0, 0x7f, m_jetEtScaleInputLsb, m_htMissThresholds);
}

std::unique_ptr<L1CaloEtScale> L1ScalesTrivialProducer::produceHfRingScale(const L1HfRingEtScaleRcd& iRecord) {
  return std::make_unique<L1CaloEtScale>(0xff, 0x7, m_jetEtScaleInputLsb, m_hfRingThresholds);
}
