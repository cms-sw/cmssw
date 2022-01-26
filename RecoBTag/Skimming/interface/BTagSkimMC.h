#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class BTagSkimMC : public edm::one::EDFilter<> {
public:
  /// constructor
  BTagSkimMC(const edm::ParameterSet&);
  bool filter(edm::Event& evt, const edm::EventSetup& es) override;
  void endJob() override;

private:
  bool verbose;
  double overallLumi;

  unsigned int nEvents_;
  unsigned int nAccepted_;
  double pthatMin, pthatMax;
  std::string process_;
};
