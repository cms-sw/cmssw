#include "RecoParticleFlow/PFClusterTools/interface/CalibratableTest.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFNuclearInteraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace edm;
using namespace pftools;
using namespace reco;

CalibratableTest::CalibratableTest(const edm::ParameterSet& parameters) :
	debug_(0), nParticleWrites_(0), nParticleFails_(0), nEventWrites_(0), nEventFails_(0),
			 deltaRCandToSim_(0.4) {

	std::cout << __PRETTY_FUNCTION__ << std::endl;
	
	/* This procedure is GENERIC to storing any dictionary enable class in a ROOT tree. */
	tree_ = fileservice_->make<TTree>("CalibratableTest", "");
	calib_ = new Calibratable();
	tree_->Branch("Calibratable", "pftools::Calibratable", &calib_, 32000, 2);
	
	inputTagCandidates_= parameters.getParameter<InputTag>("PFCandidates");
	inputTagSimParticles_= parameters.getParameter<InputTag>("PFSimParticles");
	inputTagClustersEcal_= parameters.getParameter<InputTag>("PFClustersEcal");
	inputTagClustersHcal_= parameters.getParameter<InputTag>("PFClustersHcal");
	deltaRCandToSim_ = parameters.getParameter<double>("deltaRCandToSim");
	debug_= parameters.getParameter<int>("debug");
}

CalibratableTest::~CalibratableTest() {

}

void CalibratableTest::beginJob() {
	if (debug_ > 1)
		std::cout << __PRETTY_FUNCTION__ << "\n";

}

void CalibratableTest::analyze(const edm::Event& event,
		const edm::EventSetup& iSetup) {
	if (debug_ > 1)
		std::cout << __PRETTY_FUNCTION__ << "\n";

	//Extract new collection references
	pfCandidates_ = new Handle<PFCandidateCollection>;
	simParticles_ = new Handle<PFSimParticleCollection>;
	clustersEcal_ = new Handle<PFClusterCollection>;
	clustersHcal_ = new Handle<PFClusterCollection>;

	getCollection(*pfCandidates_, inputTagCandidates_, event);
	getCollection(*simParticles_, inputTagSimParticles_, event);
	getCollection(*clustersEcal_, inputTagClustersEcal_, event);
	getCollection(*clustersHcal_, inputTagClustersHcal_, event);

	//Reset calibratable branch
	thisEventPasses_ = true;
	thisParticlePasses_ = true;
	calib_->reset();

	if (debug_ > 1)
		std::cout << "\tStarting event..."<< std::endl;

	PFSimParticleCollection sims = **simParticles_;
	PFCandidateCollection candidates = **pfCandidates_;
	PFClusterCollection clustersEcal = **clustersEcal_;
	PFClusterCollection clustersHcal = **clustersHcal_;

	if (sims.size() == 0) {
		std::cout << "\tNo sim particles found!" << std::endl;
		thisEventPasses_ = false;
	}

	//Find primary pions in the event
	std::vector<unsigned> primarySims = findPrimarySimParticles(sims);
	if (debug_) {
		std::cout << "\tFound "<< primarySims.size()
				<< " primary sim particles, "<< (**pfCandidates_).size() << " pfCandidates\n";
	}
	for (std::vector<unsigned>::const_iterator cit = primarySims.begin(); cit
			!= primarySims.end(); ++cit) {
		//There will be one write to the tree for each pion found.
		if (debug_ > 1)
			std::cout << "\t**Starting particle...**\n";
		const PFSimParticle& sim = sims[*cit];
		//One sim per calib =>
		calib_->sim_numEvent_ = 1;
		calib_->sim_isMC_ = true;
		calib_->sim_energyEvent_ = sim.trajectoryPoint(PFTrajectoryPoint::ClosestApproach).momentum().E();
		calib_->sim_phi_ = sim.trajectoryPoint(PFTrajectoryPoint::ClosestApproach).momentum().Phi();
		calib_->sim_eta_ = sim.trajectoryPoint(PFTrajectoryPoint::ClosestApproach).momentum().Eta();

		if (sim.nTrajectoryPoints() > PFTrajectoryPoint::ECALEntrance) {
			calib_->sim_etaEcal_ = sim.trajectoryPoint(PFTrajectoryPoint::ECALEntrance).positionREP().Eta();
			calib_->sim_phiEcal_ = sim.trajectoryPoint(PFTrajectoryPoint::ECALEntrance).positionREP().Phi();
		}
		if (sim.nTrajectoryPoints() > PFTrajectoryPoint::HCALEntrance) {
			calib_->sim_etaHcal_ = sim.trajectoryPoint(PFTrajectoryPoint::HCALEntrance).positionREP().Eta();
			calib_->sim_phiHcal_ = sim.trajectoryPoint(PFTrajectoryPoint::HCALEntrance).positionREP().Phi();
		}

		// Find candidates near this sim particle
		std::vector<unsigned> matchingCands = findCandidatesInDeltaR(sim,
				candidates, deltaRCandToSim_);
		if (debug_ > 3)
			std::cout << "\t\tFound candidates near sim, found "
					<< matchingCands.size()<< " of them.\n";
		if (matchingCands.size() == 0)
			thisParticlePasses_ = false;
		for (std::vector<unsigned>::const_iterator mcIt = matchingCands.begin(); mcIt
				!= matchingCands.end(); ++mcIt) {
			const PFCandidate& theCand = candidates[*mcIt];
			extractCandidate(theCand);
		}
		//Finally,
		fillTreeAndReset();
		
	}

	delete pfCandidates_;
	delete simParticles_;
	delete clustersEcal_;
	delete clustersHcal_;

	if (thisEventPasses_)
		++nEventWrites_;
	else
		++nEventFails_;

}

std::vector<unsigned> CalibratableTest::findPrimarySimParticles(
		const std::vector<PFSimParticle>& sims) {
	std::vector<unsigned> answers;
	unsigned index(0);
	for (std::vector<PFSimParticle>::const_iterator cit = sims.begin(); cit
			!= sims.end(); ++cit) {
		PFSimParticle theSim = *cit;
		//TODO: what about rejected events?
		if (theSim.motherId() >= 0)
			continue;
		int particleId = abs(theSim.pdgCode());
		if (particleId != 211)
			continue;
		//TODO: ...particularly interacting pions?
		if (theSim.daughterIds().size() > 0)
			continue;
		answers.push_back(index);
		++index;
	}
	return answers;
}

void CalibratableTest::extractCandidate(const PFCandidate& cand) {
	if (debug_ > 3)
		std::cout << "\tCandidate: "<< cand << "\n";

	//There may be several PFCandiates per sim particle. So we create a mini-class
	//to represent each one. CandidateWrapper is defined in Calibratable.
	//It's very easy to use, as we shall see...
	CandidateWrapper cw;
	cw.energy_ = cand.energy();
	cw.eta_ = cand.eta();
	cw.phi_ = cand.phi();
	cw.type_ = cand.particleId();
	cw.energyEcal_ = cand.ecalEnergy();
	cw.energyHcal_ = cand.hcalEnergy();
	if (debug_ > 4)
		std::cout << "\t\tECAL energy = " << cand.ecalEnergy()
				<< ", HCAL energy = " << cand.hcalEnergy() << "\n";

	//Now, extract block elements from the pfCandidate:
	PFCandidate::ElementsInBlocks eleInBlocks = cand.elementsInBlocks();
	if (debug_ > 3)
		std::cout << "\tLooping over elements in blocks, "
				<< eleInBlocks.size() << " of them."<< std::endl;
	for (PFCandidate::ElementsInBlocks::iterator bit = eleInBlocks.begin(); bit
			!= eleInBlocks.end(); ++bit) {

		/* 
		 * Find PFClusters associated with this candidate.
		 */
		
		//Extract block reference
		PFBlockRef blockRef((*bit).first);
		//Extract index
		unsigned indexInBlock((*bit).second);
		//Dereference the block (what a palava!)
		const PFBlock& block = *blockRef;
		//And finally get a handle on the elements
		const edm::OwnVector<reco::PFBlockElement> & elements = block.elements();
		//get references to the candidate's track, ecal clusters and hcal clusters
		switch (elements[indexInBlock].type()) {
		case PFBlockElement::ECAL: {
			reco::PFClusterRef clusterRef = elements[indexInBlock].clusterRef();
			const PFCluster theRealCluster = *clusterRef;
			CalibratableElement d(theRealCluster.energy(),
					theRealCluster.positionREP().eta(), theRealCluster.positionREP().phi(), theRealCluster.layer() );
			calib_->cluster_ecal_.push_back(d);
			if (debug_ > 4)
				std::cout << "\t\tECAL cluster: "<< theRealCluster << "\n";

			break;
		}

		case PFBlockElement::HCAL: {
			reco::PFClusterRef clusterRef = elements[indexInBlock].clusterRef();
			const PFCluster theRealCluster = *clusterRef;
			CalibratableElement d(theRealCluster.energy(),
					theRealCluster.positionREP().eta(), theRealCluster.positionREP().phi(), theRealCluster.layer() );
			calib_->cluster_hcal_.push_back(d);
			if (debug_ > 4)
				std::cout << "\t\tHCAL cluster: "<< theRealCluster << "\n";

			break;
		}

		default:
			if (debug_ > 3)
				std::cout << "\t\tOther block type: "
						<< elements[indexInBlock].type() << "\n";
			break;
		}

	}
	//For each candidate found,
	calib_->cands_.push_back(cw);
}

std::vector<unsigned> CalibratableTest::findCandidatesInDeltaR(
		const PFSimParticle& pft, const std::vector<PFCandidate>& cands,
		const double& deltaRCut) {

	unsigned index(0);
	std::vector<unsigned> answers;

	double trEta = pft.trajectoryPoint(PFTrajectoryPoint::ECALEntrance).positionREP().Eta();
	double trPhi = pft.trajectoryPoint(PFTrajectoryPoint::ECALEntrance).positionREP().Phi();

	for (std::vector<PFCandidate>::const_iterator cit = cands.begin(); cit
			!= cands.end(); ++cit) {

		PFCandidate cand = *cit;
		double cEta = cand.eta();
		double cPhi = cand.phi();

		if (deltaR(cEta, trEta, cPhi, trPhi) < deltaRCut) {
			//accept
			answers.push_back(index);
		}

		++index;
	}
	return answers;
}

void CalibratableTest::fillTreeAndReset() {
	if (thisEventPasses_ && thisParticlePasses_) {
		++nParticleWrites_;
		calib_->recompute();
		if (debug_> 4) {
			//print a summary
			std::cout << *calib_;
		}
		tree_->Fill();
	} else {
		++nParticleFails_;
	}
	if (debug_ > 1)
		std::cout << "\t**Finished particle.**\n";
	calib_->reset();
}

void CalibratableTest::endJob() {

	if (debug_> 0) {
		std::cout << __PRETTY_FUNCTION__ << std::endl;

		std::cout << "\tnParticleWrites: "<< nParticleWrites_
				<< ", nParticleFails: "<< nParticleFails_ << "\n";
		std::cout << "\tnEventWrites: "<< nEventWrites_ << ", nEventFails: "
				<< nEventFails_ << "\n";
		std::cout << "Leaving "<< __PRETTY_FUNCTION__ << "\n";
	}

}

double CalibratableTest::deltaR(const double& eta1, const double& eta2,
		const double& phi1, const double& phi2) {
	double deltaEta = fabs(eta1 - eta2);
	double deltaPhi = fabs(phi1 - phi2);
	if (deltaPhi > M_PI) {
		deltaPhi = 2 * M_PI- deltaPhi;
	}
	return sqrt(pow(deltaEta, 2) + pow(deltaPhi, 2));
}


