/* \class MassKinFitterCandProducer
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/RecoAlgos/interface/MassKinFitterCandProducer.h"

class CustomKinFitter : public CandMassKinFitter {
public:
  CustomKinFitter(double mass) : CandMassKinFitter(mass) { }
private:
  virtual double errEt(double et, double eta) const { return 0.2; }
  virtual double errEta(double et, double eta) const { return 0.2; }
  virtual double errPhi(double et, double eta) const { return 0.2; }
};

class MassKinFitterCandCustomProducer : public MassKinFitterCandProducer {
public:
  explicit MassKinFitterCandCustomProducer(const edm::ParameterSet& cfg) :
    MassKinFitterCandProducer(cfg, new CustomKinFitter(cfg.getParameter<double>("mass"))) { }
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MassKinFitterCandCustomProducer );

