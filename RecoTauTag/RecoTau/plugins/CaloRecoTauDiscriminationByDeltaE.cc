#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

/* class CaloRecoTauDiscriminationByDeltaE
 * created : September 23 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "TLorentzVector.h"

using namespace reco;
using namespace std;
using namespace edm;

class CaloRecoTauDiscriminationByDeltaE final : public CaloTauDiscriminationProducerBase  {
    public:
	explicit CaloRecoTauDiscriminationByDeltaE(const ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){
		deltaEmin		= iConfig.getParameter<double>("deltaEmin");
		deltaEmax               = iConfig.getParameter<double>("deltaEmax");
		chargedPionMass         = 0.139;
		booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
	}

      	~CaloRecoTauDiscriminationByDeltaE() override{}

	void beginEvent(const edm::Event&, const edm::EventSetup&) override;
	double discriminate(const reco::CaloTauRef&) const override;

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    private:
	double DeltaE(const CaloTauRef&) const ;

	double chargedPionMass;

	double deltaEmin,deltaEmax;
	bool booleanOutput;
};

void CaloRecoTauDiscriminationByDeltaE::beginEvent(const Event& iEvent, const EventSetup& iSetup){
}

double CaloRecoTauDiscriminationByDeltaE::discriminate(const CaloTauRef& tau) const {

	double dE = DeltaE(tau);
	if(booleanOutput) return ( dE > deltaEmin && dE < deltaEmax ? 1. : 0. );
	return dE;
}

double CaloRecoTauDiscriminationByDeltaE::DeltaE(const CaloTauRef& tau) const {
	double tracksE = 0;
	reco::TrackRefVector signalTracks = tau->signalTracks();
	for(size_t i = 0; i < signalTracks.size(); ++i){
		TLorentzVector p4;
		p4.SetXYZM(signalTracks[i]->px(),
               signalTracks[i]->py(),
               signalTracks[i]->pz(),
               chargedPionMass);
		tracksE += p4.E();
	}
	if(tau->leadTrackHCAL3x3hitsEtSum() == 0) return -1; // electron
	return tracksE/tau->leadTrackHCAL3x3hitsEtSum() - 1.0;
}

void
CaloRecoTauDiscriminationByDeltaE::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // caloRecoTauDiscriminationByDeltaE
  edm::ParameterSetDescription desc;
  desc.add<double>("deltaEmin", -0.15);
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
  desc.add<double>("deltaEmax", 1.0);
  desc.add<bool>("BooleanOutput", true);
  desc.add<edm::InputTag>("TauProducer", edm::InputTag("caloRecoTauProducer"));
  descriptions.add("caloRecoTauDiscriminationByDeltaE", desc);
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByDeltaE);

