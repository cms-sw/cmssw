#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

/* class PFRecoTauDiscriminationByDeltaE
 * created : August 30 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

#include "TLorentzVector.h"

using namespace reco;
using namespace std;
using namespace edm;

class PFRecoTauDiscriminationByDeltaE : public PFTauDiscriminationProducerBase  {
    public:
	explicit PFRecoTauDiscriminationByDeltaE(const ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig){
		deltaEmin		= iConfig.getParameter<double>("deltaEmin");
		deltaEmax               = iConfig.getParameter<double>("deltaEmax");
		chargedPionMass         = 0.139;
		booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
	}

      	~PFRecoTauDiscriminationByDeltaE() override{}

	void beginEvent(const edm::Event&, const edm::EventSetup&) override;
	double discriminate(const reco::PFTauRef&) const override;

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    private:
	double DeltaE(const PFTauRef&) const ;

	double chargedPionMass;

	double deltaEmin,deltaEmax;
	bool booleanOutput;
};

void PFRecoTauDiscriminationByDeltaE::beginEvent(const Event& iEvent, const EventSetup& iSetup){
}

double PFRecoTauDiscriminationByDeltaE::discriminate(const PFTauRef& tau) const{

	double dE = DeltaE(tau);
	if(booleanOutput) return ( dE > deltaEmin && dE < deltaEmax ? 1. : 0. );
	return dE;
}

double PFRecoTauDiscriminationByDeltaE::DeltaE(const PFTauRef& tau) const {
	double tracksE = 0;
	const std::vector<PFCandidatePtr>& signalTracks = tau->signalPFChargedHadrCands();
	for(size_t i = 0; i < signalTracks.size(); ++i){
		TLorentzVector p4;
		p4.SetXYZM(signalTracks[i]->px(),
                           signalTracks[i]->py(),
                           signalTracks[i]->pz(),
                           chargedPionMass);
		tracksE += p4.E();
	}

	double hadrTauP = tau->momentum().r() * (1.0 - tau->emFraction());
 	if (tau->emFraction() >= 1.0) {
  		return -1.0; // electron
 	} else {
  		return tracksE / hadrTauP - 1.0;
 	}
}

void
PFRecoTauDiscriminationByDeltaE::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationByDeltaE
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
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));
  descriptions.add("pfRecoTauDiscriminationByDeltaE", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByDeltaE);
