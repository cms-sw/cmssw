#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

/* class CaloRecoTauDiscriminationByTauPolarization
 * created : September 22 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 */

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

    ~CaloRecoTauDiscriminationByTauPolarization(){}
    double discriminate(const CaloTauRef&);

  private:
    bool booleanOutput;
    double rTauMin;
};

double
CaloRecoTauDiscriminationByTauPolarization::discriminate(const CaloTauRef& tau){
  double rTau = 0;
  if(tau.isNonnull() && tau->p() > 0 && tau->leadTrack().isNonnull())
    rTau = tau->leadTrack()->p()/tau->p();
  if(booleanOutput) return ( rTau > rTauMin ? 1. : 0. );
  return rTau;
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByTauPolarization);
