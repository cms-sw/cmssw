#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class HiPFCandCleaner : public edm::EDProducer {
public:
  explicit HiPFCandCleaner(const edm::ParameterSet&);
  ~HiPFCandCleaner() override;

  // class methods

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::PFCandidateCollection> candidatesToken_;

  double ptMin_;
  double absEtaMax_;
};
