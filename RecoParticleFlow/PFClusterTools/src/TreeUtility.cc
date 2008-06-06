#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "TBranch.h"
#include "TTree.h"
#include "RecoParticleFlow/PFClusterTools/interface/SingleParticleWrapper.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibratable.h"
#include <cmath>

using namespace pftools;
TreeUtility::TreeUtility() {
}

TreeUtility::~TreeUtility() {
}

unsigned TreeUtility::getCalibratablesFromRootFile(TFile& f,
		std::vector<Calibratable>& toBeFilled) {

	f.cd("extraction");
	TTree* tree = (TTree*) f.Get("extraction/Extraction");
	if (tree == 0) {
		PFToolsException me("Couldn't open tree!");
		throw me;
	}
	std::cout << "Successfully opened file. Getting branches..."<< std::endl;
	CalibratablePtr calib_ptr(new Calibratable());
	TBranch* calibBr = tree->GetBranch("Calibratable");
	//spwBr->SetAddress(&spw);
	calibBr->SetAddress(&calib_ptr);
	std::cout << "Looping over tree's "<< tree->GetEntries() << " entries...\n";
	for (unsigned entries(0); entries < tree->GetEntries(); entries++) {
		tree->GetEntry(entries);
		Calibratable c(*calib_ptr);
		toBeFilled.push_back(c);
	}

	return tree->GetEntries();

}

unsigned TreeUtility::convertCalibratablesToParticleDeposits(
		const std::vector<Calibratable>& input,
		std::vector<ParticleDepositPtr>& toBeFilled, CalibrationTarget target, DetectorElementPtr offset, 
		DetectorElementPtr ecal, DetectorElementPtr hcal) {

	std::cout << __PRETTY_FUNCTION__ << std::endl;
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
			pd->setEta(c.cand_eta_);
			pd->setPhi(c.cand_phi_);
			//TODO:: sort this out
			if (c.cand_eta_ == NAN|| c.cand_phi_ == NAN|| c.sim_energyEvent_ == 0)
				veto = true;
		}

		if (target == CLUSTER) {
			Deposition decal(ecal, c.cluster_etaEcal_, c.cluster_phiEcal_,
					c.cluster_energyEcal_, 0);
			Deposition dhcal(hcal, c.cluster_etaHcal_, c.cluster_phiHcal_,
					c.cluster_energyHcal_, 0);
			Deposition doffset(offset, c.cluster_etaHcal_, c.cluster_phiHcal_,
					1.0, 0);
			pd->addTruthDeposition(doffset);
			pd->addTruthDeposition(decal);
			pd->addTruthDeposition(dhcal);
			pd->addRecDeposition(decal);
			pd->addRecDeposition(dhcal);
			pd->addRecDeposition(doffset);

		}

		if (target == PFCANDIDATE) {
			Deposition decal(ecal, c.cand_eta_, c.cand_phi_,
					c.cluster_energyEcal_, 0);
			Deposition dhcal(hcal, c.cand_eta_, c.cand_phi_,
					c.cluster_energyHcal_, 0);
			Deposition doffset(offset, c.cand_eta_, c.cand_phi_,
								1.0, 0);
			pd->addTruthDeposition(doffset);
			pd->addTruthDeposition(decal);
			pd->addTruthDeposition(dhcal);
			pd->addRecDeposition(decal);
			pd->addRecDeposition(dhcal);
			pd->addRecDeposition(doffset);
		}

		if (target == RECHIT) {
			Deposition decal(ecal, c.rechits_etaEcal_, c.rechits_phiEcal_,
					c.rechits_energyEcal_, 0);
			Deposition dhcal(hcal, c.rechits_etaHcal_, c.rechits_phiHcal_,
					c.rechits_energyHcal_, 0);
			Deposition doffset(offset, c.rechits_etaEcal_, c.rechits_phiEcal_,
								1.0, 0);
			pd->addTruthDeposition(doffset);
			pd->addTruthDeposition(decal);
			pd->addTruthDeposition(dhcal);
			pd->addRecDeposition(decal);
			pd->addRecDeposition(dhcal);
			pd->addRecDeposition(doffset);

		}
		if (!veto)
			toBeFilled.push_back(pd);
		++count;
	}

	return toBeFilled.size();

}

void TreeUtility::recreateFromRootFile(TFile& file,
		std::vector<DetectorElementPtr>& elements,
		std::vector<ParticleDepositPtr>& toBeFilled) {

	std::cout << __PRETTY_FUNCTION__
			<< ": This method is highly specific to detector element types and may fail if their definitions change. Please be advised of this limitation!\n";
	typedef boost::shared_ptr<SingleParticleWrapper> SingleParticleWrapperPtr;
	DetectorElementPtr ecal;
	DetectorElementPtr hcal;
	DetectorElementPtr offset;
	SingleParticleWrapperPtr spw_ptr(new SingleParticleWrapper);
	//SingleParticleWrapper* spw = new SingleParticleWrapper;

	for (std::vector<DetectorElementPtr>::iterator it = elements.begin(); it
			!= elements.end(); ++it) {
		DetectorElementPtr de = *it;
		if (de->getType() == ECAL)
			ecal = de;
		if (de->getType() == HCAL)
			hcal = de;
		if (de->getType() == OFFSET)
			offset = de;

	}
	if (offset == 0) {
		std::cout
				<< "Offset element NOT found in input collection; no 'a' coefficient will be added to ParticleDeposits.\n";
	}

	std::cout << "Initialised detector elements."<< std::endl;

	if (offset != 0)
		std::cout << "\t"<< *offset << "\n";
	std::cout << "\t"<< *ecal << "\n";
	std::cout << "\t"<< *hcal << "\n";
	file.ls();
	TTree* tree = (TTree*) file.Get("CaloData");
	tree->ls();
	if (tree == 0) {
		PFToolsException me("Couldn't open tree!");
		throw me;
	}
	std::cout << "Opened Tree CaloData...\n";

	std::cout << "Assigning branch: \n";
	TBranch* spwBr = tree->GetBranch("SingleParticleWrapper");
	//spwBr->SetAddress(&spw);
	spwBr->SetAddress(&spw_ptr);
	std::cout << "Looping over entries...\n";
	for (unsigned entries(0); entries < tree->GetEntries(); entries++) {

		tree->GetEntry(entries);

		ParticleDepositPtr pd(new ParticleDeposit(spw_ptr->trueEnergy, spw_ptr->etaEcal, spw_ptr->phiEcal));

		if (offset != 0) {
			Deposition dOffset(offset, spw_ptr->etaEcal, spw_ptr->phiEcal, 1.0);
			pd->addRecDeposition(dOffset);
			pd->addTruthDeposition(dOffset);
		}
		Deposition dE(ecal, spw_ptr->etaEcal, spw_ptr->phiEcal, spw_ptr->eEcal);
		Deposition dH(hcal, spw_ptr->etaEcal, spw_ptr->phiEcal, spw_ptr->eHcal);

		//RecDepositions are what the detector element should detect
		//when well calibrated
		//yet includes detector effects

		pd->addRecDeposition(dE);
		pd->addRecDeposition(dH);

		//TruthDepositions are the true MC particle energy depositions
		//Here we set the same as rec depositions - they aren't used in this test case.

		pd->addTruthDeposition(dE);
		pd->addTruthDeposition(dH);

		//std::cout << *pd << std::endl;
		toBeFilled.push_back(pd);

	}

}
void TreeUtility::recreateFromRootFile(TFile& f) {

	DetectorElementPtr ecal(new DetectorElement(ECAL, 1.0));
	DetectorElementPtr hcal(new DetectorElement(HCAL, 1.0));
	std::vector<DetectorElementPtr> elements;
	elements.push_back(ecal);
	elements.push_back(hcal);
	std::cout << "Made detector elements...\n";
	std::cout << "Recreating from root file...\n";
	std::vector<ParticleDepositPtr> particles;
	recreateFromRootFile(f, elements, particles);
	std::cout << "Finished.\n";
}

std::vector<ParticleDepositPtr> TreeUtility::extractParticles(TFile& f) {
	DetectorElementPtr ecal(new DetectorElement(ECAL, 1.0));
	DetectorElementPtr hcal(new DetectorElement(HCAL, 1.0));
	std::vector<DetectorElementPtr> elements;
	elements.push_back(ecal);
	elements.push_back(hcal);
	std::cout << "Made detector elements...\n";
	std::cout << "Recreating from root file...\n";
	std::vector<ParticleDepositPtr> particles;
	recreateFromRootFile(f, elements, particles);
	return particles;
}

