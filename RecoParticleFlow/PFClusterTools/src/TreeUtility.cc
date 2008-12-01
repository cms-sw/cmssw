#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "TBranch.h"
#include "TTree.h"
#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include <cmath>

using namespace pftools;
TreeUtility::TreeUtility() {
}

TreeUtility::~TreeUtility() {
}

unsigned TreeUtility::getCalibratablesFromRootFile(TChain& tree,
		std::vector<Calibratable>& toBeFilled) {

//	f.cd("extraction");
//	TTree* tree = (TTree*) f.Get("extraction/Extraction");
//	if (tree == 0) {
//		PFToolsException me("Couldn't open tree!");
//		throw me;
//	}
//	std::cout << "Successfully opened file. Getting branches..."<< std::endl;
	CalibratablePtr calib_ptr(new Calibratable());
	//TBranch* calibBr = tree.GetBranch("Calibratable");
	//spwBr->SetAddress(&spw);
	tree.SetBranchAddress("Calibratable", &calib_ptr);
	
	std::cout << "Looping over tree's "<< tree.GetEntries() << " entries...\n";
	for (unsigned entries(0); entries < tree.GetEntries(); entries++) {
		tree.GetEntry(entries);
		Calibratable c(*calib_ptr);
		toBeFilled.push_back(c);
	}

	return tree.GetEntries();

}

unsigned TreeUtility::convertCalibratablesToParticleDeposits(
		const std::vector<Calibratable>& input,
		std::vector<ParticleDepositPtr>& toBeFilled, CalibrationTarget target,
		DetectorElementPtr offset, DetectorElementPtr ecal,
		DetectorElementPtr hcal, bool includeOffset) {

	std::cout << __PRETTY_FUNCTION__ << std::endl;
	std::cout << "WARNING: Using fabs() for eta value assignments!\n";
	std::cout << "Input Calibratable has size " << input.size() << "\n"; 
	std::cout << "Cutting on > 1 PFCandidate.\n";
	
	//neither of these two are supported yet
	if (target == UNDEFINED || target == PFELEMENT)
		return 0;
	unsigned count(0);
	for (std::vector<Calibratable>::const_iterator cit = input.begin(); cit
			!= input.end(); ++cit) {
		Calibratable c = *cit;
		ParticleDepositPtr pd(new ParticleDeposit());
		bool veto(false);
		if (c.sim_isMC_) {
			pd->setTruthEnergy(c.sim_energyEvent_);
			pd->setEta(fabs(c.sim_eta_));
			pd->setPhi(c.sim_phi_);
			//TODO:: sort this out
			if (c.sim_energyEvent_== 0)
				veto = true;
		}
		
		if(c.cands_num_ > 1)
			veto = true;

		if (target == CLUSTER) {
			if (c.cluster_ecal_.size() == 0 && c.cluster_hcal_.size() ==0)
				veto = true;
			//			if (c.cluster_numEcal_ > 1|| c.cluster_numHcal_ > 1)
			//				veto = true;
			//TODO: using fabs for eta! WARNING!!!
			Deposition decal(ecal, fabs(c.cluster_meanEcal_.eta_),
					c.cluster_meanEcal_.phi_, c.cluster_energyEcal_, 0);
			Deposition dhcal(hcal, fabs(c.cluster_meanHcal_.eta_),
					c.cluster_meanHcal_.phi_, c.cluster_energyHcal_, 0);
			Deposition doffset(offset, fabs(c.cluster_meanEcal_.eta_),
					c.cluster_meanEcal_.phi_, 0.001, 0);

				pd->addTruthDeposition(decal);
				pd->addRecDeposition(decal);

				pd->addTruthDeposition(dhcal);
				pd->addRecDeposition(dhcal);


			if (includeOffset) {
				pd->addTruthDeposition(doffset);
				pd->addRecDeposition(doffset);
			}

		}

		else if (target == PFCANDIDATE) {
			//			if(c.cands_num_ != 1)
			//				veto = true;
			Deposition decal(ecal, c.cand_eta_, c.cand_phi_,
					c.cand_energyEcal_, 0);
			Deposition dhcal(hcal, c.cand_eta_, c.cand_phi_,
					c.cand_energyHcal_, 0);
			Deposition doffset(offset, c.cand_eta_, c.cand_phi_, 1.0, 0);

			pd->addTruthDeposition(decal);
			pd->addTruthDeposition(dhcal);
			pd->addRecDeposition(decal);
			pd->addRecDeposition(dhcal);

			if (includeOffset) {
				pd->addTruthDeposition(doffset);
				pd->addRecDeposition(doffset);
			}
		}

		else if (target == RECHIT) {
			if (c.rechits_ecal_.size() == 0&& c.rechits_hcal_.size() == 0)
			veto = true;
			Deposition decal(ecal, c.rechits_meanEcal_.eta_,
					c.rechits_meanEcal_.phi_, c.rechits_meanEcal_.energy_
					* c.rechits_ecal_.size(), 0);
			Deposition dhcal(hcal, c.rechits_meanHcal_.eta_,
					c.rechits_meanHcal_.phi_, c.rechits_meanHcal_.energy_
					* c.rechits_hcal_.size(), 0);
			Deposition doffset(offset, c.rechits_meanEcal_.eta_,
					c.rechits_meanEcal_.phi_, 1.0, 0);

			pd->addTruthDeposition(decal);
			pd->addTruthDeposition(dhcal);
			pd->addRecDeposition(decal);
			pd->addRecDeposition(dhcal);

			if (includeOffset) {
				pd->addTruthDeposition(doffset);
				pd->addRecDeposition(doffset);
			}

		}
		if (!veto)
		toBeFilled.push_back(pd);
	
		++count;
	}
	
	return toBeFilled.size();

}


