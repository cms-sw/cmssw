#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByNProngs);
