
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoJets/JetProducers/interface/SubEventGenJetProducer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace std;
using namespace reco;
using namespace edm;
using namespace cms;

namespace {
	bool checkHydro(const reco::GenParticle * p){
		const Candidate* m1 = p->mother();
		while(m1){
			int pdg = abs(m1->pdgId());
			int st = m1->status();
			LogDebug("SubEventMothers")<<"Pdg ID : "<<pdg<<endl;
			if(st == 3 || pdg < 9 || pdg == 21){
				LogDebug("SubEventMothers")<<"Sub-Collision  Found! Pdg ID : "<<pdg<<endl;
				return false;
			}
			const Candidate* m = m1->mother();
			m1 = m;
			if(!m1) LogDebug("SubEventMothers")<<"No Mother, particle is : "<<pdg<<" with status "<<st<<endl;
		}
		//      return true;
		return true; // Debugging - to be changed
	}
}

SubEventGenJetProducer::SubEventGenJetProducer(edm::ParameterSet const& conf):
	VirtualJetProducer( conf )
{
	//   mapSrc_ = conf.getParameter<edm::InputTag>( "srcMap");
	ignoreHydro_ = conf.getParameter<bool>("ignoreHydro");
	produces<reco::BasicJetCollection>();
	// the subjet collections are set through the config file in the "jetCollInstanceName" field.

	input_cand_token_ = consumes<reco::CandidateView>(conf.getParameter<edm::InputTag>("src"));

}


void SubEventGenJetProducer::inputTowers( ) {

	std::vector<edm::Ptr<reco::Candidate> >::const_iterator inBegin = inputs_.begin(),
	inEnd = inputs_.end(), i = inBegin;
	for (; i != inEnd; ++i ) {
		reco::CandidatePtr input = inputs_[i - inBegin];
		if (edm::isNotFinite(input->pt()))           continue;
		if (input->et()    <inputEtMin_)  continue;
		if (input->energy()<inputEMin_)   continue;
		if (isAnomalousTower(input))      continue;

		edm::Ptr<reco::Candidate> p = inputs_[i - inBegin];
		const GenParticle * pref = dynamic_cast<const GenParticle *>(p.get());
		int subevent = pref->collisionId();
		LogDebug("SubEventContainers")<<"SubEvent is : "<<subevent<<endl;
		LogDebug("SubEventContainers")<<"SubSize is : "<<subInputs_.size()<<endl;

		if(subevent >= (int)subInputs_.size()){ 
			hydroTag_.resize(subevent+1, -1);
			subInputs_.resize(subevent+1);
			LogDebug("SubEventContainers")<<"SubSize is : "<<subInputs_.size()<<endl;
			LogDebug("SubEventContainers")<<"HydroTagSize is : "<<hydroTag_.size()<<endl;
		}

		LogDebug("SubEventContainers")<<"HydroTag is : "<<hydroTag_[subevent]<<endl;
		if(hydroTag_[subevent] != 0) hydroTag_[subevent] = (int)checkHydro(pref);

		subInputs_[subevent].push_back(fastjet::PseudoJet(input->px(),input->py(),input->pz(),
							input->energy()));

		subInputs_[subevent].back().set_user_index(i - inBegin);

	}

}

void SubEventGenJetProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){

	LogDebug("VirtualJetProducer") << "Entered produce\n";

	fjJets_.clear();
	subInputs_.clear();
	nSubParticles_.clear();
	hydroTag_.clear();
	inputs_.clear();

	// get inputs and convert them to the fastjet format (fastjet::PeudoJet)
	edm::Handle<reco::CandidateView> inputsHandle;
	iEvent.getByToken(input_cand_token_, inputsHandle);
	for (size_t i = 0; i < inputsHandle->size(); ++i) {
		inputs_.push_back(inputsHandle->ptrAt(i));
	}
	LogDebug("VirtualJetProducer") << "Got inputs\n";

	inputTowers();
	// Convert candidates to fastjet::PseudoJets.
	// Also correct to Primary Vertex. Will modify fjInputs_
	// and use inputs_

	////////////////

	auto jets = std::make_unique<std::vector<GenJet>>();
	subJets_ = jets.get();

	LogDebug("VirtualJetProducer") << "Inputted towers\n";

	size_t nsub = subInputs_.size();

	for(size_t isub = 0; isub < nsub; ++isub){
		if(ignoreHydro_ && hydroTag_[isub]) continue;
		fjJets_.clear();
		fjInputs_.clear();
		fjInputs_ = subInputs_[isub];
		runAlgorithm( iEvent, iSetup );
	}

	//Finalize
	LogDebug("SubEventJetProducer") << "Wrote jets\n";

	iEvent.put(std::move(jets));  
	return;
}

void SubEventGenJetProducer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup) {

	// run algorithm
	fjJets_.clear();

	fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, *fjJetDefinition_ ) );
	fjJets_ = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));

	using namespace reco;

	for (unsigned int ijet=0;ijet<fjJets_.size();++ijet) {

		GenJet jet;
		const fastjet::PseudoJet& fjJet = fjJets_[ijet];

		std::vector<fastjet::PseudoJet> fjConstituents = sorted_by_pt(fjClusterSeq_->constituents(fjJet));
		std::vector<CandidatePtr> constituents = getConstituents(fjConstituents);

		double px=fjJet.px();
		double py=fjJet.py();
		double pz=fjJet.pz();
		double E=fjJet.E();
		double jetArea=0.0;
		double pu=0.;

		writeSpecific( jet,
			   Particle::LorentzVector(px, py, pz, E),
			   vertex_,
			   constituents, iSetup);

		jet.setJetArea (jetArea);
		jet.setPileup (pu);

		subJets_->push_back(jet);
	}   
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SubEventGenJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

	edm::ParameterSetDescription desc;
	desc.add<bool> 	("ignoreHydro", 	true);
	//// From VirtualJetProducer
	desc.add<string> ("@module_label",	"" );
	desc.add<edm::InputTag>("src",		edm::InputTag("particleFlow") );
	desc.add<edm::InputTag>("srcPVs",	edm::InputTag("") );
	desc.add<string>("jetType",		"PFJet" );
	desc.add<string>("jetAlgorithm",	"AntiKt" );
	desc.add<double>("rParam",		0.4 );
	desc.add<double>("inputEtMin",		0.0 );
	desc.add<double>("inputEMin",		0.0 );
	desc.add<double>("jetPtMin",		5. );
	desc.add<bool> 	("doPVCorrection",	false );
	desc.add<bool> 	("doAreaFastjet",	false );
	desc.add<bool>  ("doRhoFastjet",	false );
	desc.add<string>("jetCollInstanceName", ""	);
	desc.add<bool> 	("doPUOffsetCorr", 	false	);
	desc.add<string>("subtractorName", 	""	);
	desc.add<bool> 	("useExplicitGhosts", 	false	);
	desc.add<bool> 	("doAreaDiskApprox", 	false 	);
	desc.add<double>("voronoiRfact", 	-0.9 	);
	desc.add<double>("Rho_EtaMax", 	 	4.4 	);
	desc.add<double>("Ghost_EtaMax",	5. 	);
	desc.add<int> 	("Active_Area_Repeats",	1 	);
	desc.add<double>("GhostArea",	 	0.01 	);
	desc.add<bool> 	("restrictInputs", 	false 	);
	desc.add<unsigned int> 	("maxInputs", 	1 	);
	desc.add<bool> 	("writeCompound", 	false 	);
	desc.add<bool> 	("doFastJetNonUniform", false 	);
	desc.add<bool> 	("useDeterministicSeed",false 	);
	desc.add<unsigned int> 	("minSeed", 	14327 	);
	desc.add<int> 	("verbosity", 		1 	);
	desc.add<double>("puWidth",	 	0. 	);
	desc.add<unsigned int>("nExclude", 	0 	);
	desc.add<unsigned int>("maxBadEcalCells", 	9999999	);
	desc.add<unsigned int>("maxBadHcalCells",	9999999 );
	desc.add<unsigned int>("maxProblematicEcalCells",	9999999 );
	desc.add<unsigned int>("maxProblematicHcalCells",	9999999 );
	desc.add<unsigned int>("maxRecoveredEcalCells",	9999999 );
	desc.add<unsigned int>("maxRecoveredHcalCells",	9999999 );
	///// From PileUpSubstractor
	desc.add<double> ("puPtMin", 	10.);
	desc.add<double> ("nSigmaPU", 	1.);
	desc.add<double> ("radiusPU", 	0.5);
	/////////////////////
	descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(SubEventGenJetProducer);


