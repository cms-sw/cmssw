#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "TBranch.h"
#include "TTree.h"
#include "RecoParticleFlow/PFClusterTools/interface/SingleParticleWrapper.h"

using namespace pftools;
TreeUtility::TreeUtility() {
}

TreeUtility::~TreeUtility() {
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

