#ifndef PhysicsTools_RecoAlgos_MassKinFitterCandProducer_h
#define PhysicsTools_RecoAlgos_MassKinFitterCandProducer_h
/* \class MassKinFitterCandProducer
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/RecoUtils/interface/CandMassKinFitter.h"

class MassKinFitterCandProducer : public edm::EDProducer {
public:
  explicit MassKinFitterCandProducer(const edm::ParameterSet&, CandMassKinFitter * = 0);
private:
  edm::InputTag src_;
  std::auto_ptr<CandMassKinFitter> fitter_;
  void produce( edm::Event &, const edm::EventSetup & );
};

#endif
