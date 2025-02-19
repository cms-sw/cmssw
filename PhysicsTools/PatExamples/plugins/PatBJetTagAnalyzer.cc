#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#include <TH1.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

class PatBJetTagAnalyzer : public edm::EDAnalyzer  {
    public: 
	/// constructor and destructor
	PatBJetTagAnalyzer(const edm::ParameterSet &params);
	~PatBJetTagAnalyzer();

	// virtual methods called from base class EDAnalyzer
	virtual void beginJob();
	virtual void analyze(const edm::Event &event, const edm::EventSetup &es);

    private:
	// configuration parameters
	edm::InputTag jets_;

	double jetPtCut_;		// minimum (uncorrected) jet energy
	double jetEtaCut_;		// maximum |eta| for jet

	// jet flavour constants

	enum Flavour {
		ALL_JETS = 0,
		UDSG_JETS,
		C_JETS,
		B_JETS,
		NONID_JETS,
		N_JET_TYPES
	};

	TH1 * flavours_;

	// one group of plots per jet flavour;
	struct Plots {
		TH1 *discrTC, *discrSSV, *discrCSV;
	} plots_[N_JET_TYPES];
};

PatBJetTagAnalyzer::PatBJetTagAnalyzer(const edm::ParameterSet &params) :
	jets_(params.getParameter<edm::InputTag>("jets")),
	jetPtCut_(params.getParameter<double>("jetPtCut")),
	jetEtaCut_(params.getParameter<double>("jetEtaCut"))
{
}

PatBJetTagAnalyzer::~PatBJetTagAnalyzer()
{
}

void PatBJetTagAnalyzer::beginJob()
{
	// retrieve handle to auxiliary service
	//  used for storing histograms into ROOT file
	edm::Service<TFileService> fs;

	flavours_ = fs->make<TH1F>("flavours", "jet flavours", 5, 0, 5);

	// book histograms for all jet flavours
	for(unsigned int i = 0; i < N_JET_TYPES; i++) {
		Plots &plots = plots_[i];
		const char *flavour, *name;

		switch((Flavour)i) {
		    case ALL_JETS:
			flavour = "all jets";
			name = "all";
			break;
		    case UDSG_JETS:
			flavour = "light flavour jets";
			name = "udsg";
			break;
		    case C_JETS:
			flavour = "charm jets";
			name = "c";
			break;
		    case B_JETS:
			flavour = "bottom jets";
			name = "b";
			break;
		    default:
			flavour = "unidentified jets";
			name = "ni";
			break;
		}

		plots.discrTC = fs->make<TH1F>(Form("discrTC_%s", name),
		                               Form("track counting (\"high efficiency\") in %s", flavour),
		                               100, 0, 20);
		plots.discrSSV = fs->make<TH1F>(Form("discrSSV_%s", name),
		                                Form("simple secondary vertex in %s", flavour),
		                                100, 0, 10);
		plots.discrCSV = fs->make<TH1F>(Form("discrCSV_%s", name),
		                                Form("combined secondary vertex in %s", flavour),
		                                100, 0, 1);
	}
}

void PatBJetTagAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &es)
{  
	// handle to the jets collection
	edm::Handle<pat::JetCollection> jetsHandle;
	event.getByLabel(jets_, jetsHandle);

	// now go through all jets
	for(pat::JetCollection::const_iterator jet = jetsHandle->begin();
	    jet != jetsHandle->end(); ++jet) {

		// only look at jets that pass the pt and eta cut
		if (jet->pt() < jetPtCut_ ||
		    std::abs(jet->eta()) > jetEtaCut_)
			continue;

		Flavour flavour;
		// find out the jet flavour (differs between quark and anti-quark)
		switch(std::abs(jet->partonFlavour())) {
		    case 1:
		    case 2:
		    case 3:
		    case 21:
			flavour = UDSG_JETS;
			break;
		    case 4:
			flavour = C_JETS;
			break;
		    case 5:
			flavour = B_JETS;
			break;
		    default:
			flavour = NONID_JETS;
		}

		// simply count the number of accepted jets
		flavours_->Fill(ALL_JETS);
		flavours_->Fill(flavour);

		double discrTC = jet->bDiscriminator("trackCountingHighEffBJetTags");
		double discrSSV = jet->bDiscriminator("simpleSecondaryVertexBJetTags");
		double discrCSV = jet->bDiscriminator("combinedSecondaryVertexBJetTags");

		plots_[ALL_JETS].discrTC->Fill(discrTC);
		plots_[flavour].discrTC->Fill(discrTC);

		plots_[ALL_JETS].discrSSV->Fill(discrSSV);
		plots_[flavour].discrSSV->Fill(discrSSV);

		plots_[ALL_JETS].discrCSV->Fill(discrCSV);
		plots_[flavour].discrCSV->Fill(discrCSV);
	}
}
	
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PatBJetTagAnalyzer);
