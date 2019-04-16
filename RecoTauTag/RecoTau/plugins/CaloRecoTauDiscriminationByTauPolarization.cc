#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

/* class CaloRecoTauDiscriminationByTauPolarization
 * created : September 22 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 */


namespace {
using namespace reco;
using namespace std;

class CaloRecoTauDiscriminationByTauPolarization : public CaloTauDiscriminationProducerBase  {
  public:
    explicit CaloRecoTauDiscriminationByTauPolarization(
        const edm::ParameterSet& iConfig)
        :CaloTauDiscriminationProducerBase(iConfig) {
          rTauMin = iConfig.getParameter<double>("rtau");
          booleanOutput = iConfig.getParameter<bool>("BooleanOutput");
        }

    ~CaloRecoTauDiscriminationByTauPolarization() override{}
    double discriminate(const CaloTauRef&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  private:
    bool booleanOutput;
    double rTauMin;
};

double
CaloRecoTauDiscriminationByTauPolarization::discriminate(const CaloTauRef& tau) const {
  double rTau = 0;
  if(tau.isNonnull() && tau->p() > 0 && tau->leadTrack().isNonnull())
    rTau = tau->leadTrack()->p()/tau->p();
  if(booleanOutput) return ( rTau > rTauMin ? 1. : 0. );
  return rTau;
}
}

void
CaloRecoTauDiscriminationByTauPolarization::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // caloRecoTauDiscriminationByTauPolarization
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("CaloTauProducer", edm::InputTag("caloRecoTauProducer"));
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut");
      psd1.add<edm::InputTag>("Producer");
      psd0.addOptional<edm::ParameterSetDescription>("leadTrack", psd1);
    }
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<double>("rtau", 0.8);
  desc.add<edm::InputTag>("PVProducer", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("BooleanOutput", true);
  descriptions.add("caloRecoTauDiscriminationByTauPolarization", desc);
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByTauPolarization);
