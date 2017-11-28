#include "RecoEgamma//ElectronIdentification/interface/ElectronMVAEstimatorRun2.h"

class ElectronMVAEstimatorRun2Fall17iso : public ElectronMVAEstimatorRun2 {

 public:

  ElectronMVAEstimatorRun2Fall17iso(const edm::ParameterSet& conf) : ElectronMVAEstimatorRun2(conf) {}
  ~ElectronMVAEstimatorRun2Fall17iso() {}

  const std::string& getName() const final { return name_; }

  std::vector<float> fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event&) const;
  std::vector<float> fillMVAVariables( const reco::GsfElectron * particle, const edm::Handle<reco::ConversionCollection> conversions, const reco::BeamSpot *beamSpot, const edm::Handle<double> rho) const;

 private:

  const std::string name_ = "ElectronMVAEstimatorRun2Fall17iso";

};
