#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

/* class CaloRecoTauDiscriminationByInvMass
 * created : September 23 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "TLorentzVector.h"

using namespace reco;
using namespace std;

class CaloRecoTauDiscriminationByInvMass : public CaloTauDiscriminationProducerBase  {
  public:
    explicit CaloRecoTauDiscriminationByInvMass(
        const edm::ParameterSet& iConfig)
        :CaloTauDiscriminationProducerBase(iConfig) {
      invMassMin		= iConfig.getParameter<double>("invMassMin");
      invMassMax		= iConfig.getParameter<double>("invMassMax");
      chargedPionMass 	= 0.139;
      booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
    }

    ~CaloRecoTauDiscriminationByInvMass(){}

    double discriminate(const reco::CaloTauRef&);

  private:
    double threeProngInvMass(const CaloTauRef&);
    double chargedPionMass;
    double invMassMin,invMassMax;
    bool booleanOutput;
};

double CaloRecoTauDiscriminationByInvMass::discriminate(const CaloTauRef& tau){

  double invMass = threeProngInvMass(tau);
  if(booleanOutput) return (
      invMass > invMassMin && invMass < invMassMax ? 1. : 0. );
  return invMass;
}

double CaloRecoTauDiscriminationByInvMass::threeProngInvMass(
    const CaloTauRef& tau){
  TLorentzVector sum;
  reco::TrackRefVector signalTracks = tau->signalTracks();
  for(size_t i = 0; i < signalTracks.size(); ++i){
    TLorentzVector p4;
    p4.SetXYZM(signalTracks[i]->px(),
               signalTracks[i]->py(),
               signalTracks[i]->pz(),
               chargedPionMass);
    sum += p4;
  }
  return sum.M();
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByInvMass);
