#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

/* class CaloRecoTauDiscriminationByInvMass
 * created : September 23 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "TLorentzVector.h"

namespace {

using namespace reco;
using namespace std;

class CaloRecoTauDiscriminationByInvMass final : public CaloTauDiscriminationProducerBase  {
  public:
    explicit CaloRecoTauDiscriminationByInvMass(
        const edm::ParameterSet& iConfig)
        :CaloTauDiscriminationProducerBase(iConfig) {
      invMassMin		= iConfig.getParameter<double>("invMassMin");
      invMassMax		= iConfig.getParameter<double>("invMassMax");
      chargedPionMass 	= 0.139;
      booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
    }

    ~CaloRecoTauDiscriminationByInvMass() override{}

    double discriminate(const reco::CaloTauRef&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  private:
    double threeProngInvMass(const CaloTauRef&) const ;
    double chargedPionMass;
    double invMassMin,invMassMax;
    bool booleanOutput;
};

double CaloRecoTauDiscriminationByInvMass::discriminate(const CaloTauRef& tau) const {

  double invMass = threeProngInvMass(tau);
  if(booleanOutput) return (
      invMass > invMassMin && invMass < invMassMax ? 1. : 0. );
  return invMass;
}

double CaloRecoTauDiscriminationByInvMass::threeProngInvMass(
    const CaloTauRef& tau) const {
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

}

void
CaloRecoTauDiscriminationByInvMass::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // caloRecoTauDiscriminationByInvMass
  edm::ParameterSetDescription desc;
  desc.add<double>("invMassMin", 0.0);
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
  desc.add<bool>("BooleanOutput", true);
  desc.add<double>("invMassMax", 1.4);
  descriptions.add("caloRecoTauDiscriminationByInvMass", desc);
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByInvMass);
