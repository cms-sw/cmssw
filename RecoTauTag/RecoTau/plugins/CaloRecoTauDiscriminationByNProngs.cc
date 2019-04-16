#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

/* class CaloRecoTauDiscriminationByNProngs
 * created : September 23 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

namespace {

using namespace reco;
using namespace std;

class CaloRecoTauDiscriminationByNProngs final 
  : public CaloTauDiscriminationProducerBase  {
  public:
    explicit CaloRecoTauDiscriminationByNProngs(const edm::ParameterSet& iConfig)
        :CaloTauDiscriminationProducerBase(iConfig) {
      nprongs			= iConfig.getParameter<uint32_t>("nProngs");
      booleanOutput = iConfig.getParameter<bool>("BooleanOutput");
    }
    ~CaloRecoTauDiscriminationByNProngs() override{}
    double discriminate(const reco::CaloTauRef&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  private:
    uint32_t nprongs;
    bool booleanOutput;
};


double CaloRecoTauDiscriminationByNProngs::discriminate(const CaloTauRef& tau) const {
  bool accepted = false;
  int np = tau->signalTracks().size();
  if((np == 1 && (nprongs == 1 || nprongs == 0)) ||
     (np == 3 && (nprongs == 3 || nprongs == 0)) ) accepted = true;
  if(!accepted) np = 0;
  if(booleanOutput) return accepted;
  return np;
}

}

void
CaloRecoTauDiscriminationByNProngs::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // caloRecoTauDiscriminationByNProngs
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("nProngs", 0);
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
  desc.add<edm::InputTag>("CaloTauProducer", edm::InputTag("caloRecoTauProducer"));
  desc.add<bool>("BooleanOutput", true);
  descriptions.add("caloRecoTauDiscriminationByNProngs", desc);
}
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByNProngs);
