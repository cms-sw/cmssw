#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "TBranch.h"
#include "TTree.h"
#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <TF1.h>

using namespace pftools;
TreeUtility::TreeUtility() {
}

TreeUtility::~TreeUtility() {
}

double deltaR(double eta1, double eta2, double phi1, double phi2) {
	return sqrt(pow(eta1 - eta2, 2) + pow(phi1 - phi2, 2));
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
		if (c.cands_num_ == 1 && (c.cluster_ecal_.size() +   c.cluster_hcal_.size()) > 0)
			toBeFilled.push_back(c);
	}
	std::cout << "Done." << std::endl;
	return tree.GetEntries();

}

void TreeUtility::dumpCaloDataToCSV(TChain& tree, std::string csvFilename, double range) {
	CalibratablePtr calib_ptr(new Calibratable());

	tree.SetBranchAddress("Calibratable", &calib_ptr);
	ofstream csvFile;
	csvFile.open(csvFilename.c_str());

	std::cout << "Looping over tree's "<< tree.GetEntries() << " entries...\n";
	unsigned writes(0);
	TFile freq("freq.root", "recreate");
	TH1F frequencies("f", "f", 50, 0, range);
	TF1 g("g", "gaus(0)");
	g.FixParameter(1, range/2.0);
	g.FixParameter(0, 1),
	g.FixParameter(2, range/4.0);

	for (unsigned entries(0); entries < tree.GetEntries(); entries++) {
		tree.GetEntry(entries);
		Calibratable c(*calib_ptr);
		bool veto(false);

		//Check vetos as usual
		if (c.cands_num_ > 1)
			veto = true;
		 if(c.cluster_ecal_.size() == 0  && c.cluster_hcal_.size() == 0)
		 	veto = true;
		 if(!veto) {
			if(frequencies.GetBinContent(static_cast<int>(floor(c.sim_energyEvent_)) + 1) < (3000) * g.Eval(c.sim_energyEvent_) ) {
			 	frequencies.Fill(static_cast<int>(floor(c.sim_energyEvent_)));
				c.recompute();
				csvFile << c.sim_energyEvent_ << "\t";
				csvFile << c.sim_energyEcal_ << "\t";
				csvFile << c.sim_energyHcal_ << "\t";
				
				
				//csvFile << c.cluster_energyEcal_/range << "\t";
				//csvFile << c.cluster_energyHcal_/range << "\t";
				csvFile << c.cluster_ecal_.size() << "\t";
				csvFile << c.cluster_hcal_.size() << "\t";
				csvFile << c.rechits_energyEcal_/range << "\t";
				csvFile << c.rechits_energyHcal_/range << "\t";
				csvFile << c.rechits_ecal_.size()/20 << "\t";
				csvFile << c.rechits_hcal_.size()/20 << "\t";
				
				/*
				unsigned count(0);
				unsigned leadingHits(5);
				std::vector<CalibratableElement>::const_iterator eIt = c.rechits_ecal_.begin();
				for(; eIt != c.rechits_ecal_.end(); ++eIt) {
					const CalibratableElement ce = *eIt;
					csvFile << ce.energy_/range << "\t";
					double dR =deltaR(ce.eta_, c.rechits_meanEcal_.eta_, ce.phi_, c.rechits_meanEcal_.phi_);
					//csvFile << dR << "\t";
					++count;
					if(count == 2)
						break;
				}
				unsigned numEcal = c.rechits_ecal_.size();
				for(unsigned q(numEcal); q < 2; ++q) {
					csvFile << "0\t";
					
				} 
				
				count = 0;
				std::vector<CalibratableElement>::const_iterator hIt = c.rechits_hcal_.begin();
				for(; hIt != c.rechits_hcal_.end(); ++hIt) {
					const CalibratableElement ce = *hIt;
					csvFile << ce.energy_/range << "\t";
					double dR =deltaR(ce.eta_, c.rechits_meanHcal_.eta_, ce.phi_, c.rechits_meanHcal_.phi_);
					//csvFile << dR << "\t";
					++count;
					if(count == leadingHits)
						break;
				}
				unsigned numHcal = c.rechits_hcal_.size();
				for(unsigned q(numHcal); q < leadingHits; ++q) {
					csvFile << "0\t";
				} */
				
				csvFile << fabs(c.sim_eta_/2) << "\n";
				
				++writes;
		 	}
		}


	}
	frequencies.Print("frequencies.eps");
	frequencies.Write();
	freq.Close();
	std::cout << "Closing file " << csvFilename << " with " << writes << " entries.\n" << std::endl;

	csvFile.close();

}

unsigned TreeUtility::getParticleDepositsDirectly(TChain& sourceChain,
		std::vector<ParticleDepositPtr>& toBeFilled, CalibrationTarget target,
		DetectorElementPtr offset, DetectorElementPtr ecal,
		DetectorElementPtr hcal, bool includeOffset) {

	CalibratablePtr calib_ptr(new Calibratable());
	sourceChain.SetBranchAddress("Calibratable", &calib_ptr);
	std::cout << __PRETTY_FUNCTION__ << std::endl;
	std::cout << "WARNING: Using fabs() for eta value assignments!\n";
	std::cout << "Cutting on > 1 PFCandidate.\n";
	std::cout << "Looping over tree's "<< sourceChain.GetEntries() << " entries...\n";
	//neither of these two are supported yet
	if (target == UNDEFINED || target == PFELEMENT)
		return 0;
	unsigned count(0);

	for (unsigned entries(0); entries < sourceChain.GetEntries(); entries++) {
		sourceChain.GetEntry(entries);
		Calibratable c(*calib_ptr);

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
		if (c.tb_isTB_) {
			pd->setTruthEnergy(c.sim_energyEvent_);
			pd->setEta(c.tb_eta_);
			pd->setPhi(c.tb_phi_);
			veto = false;
		}

		if (c.cands_num_ > 1)
			veto = true;

		std::cout << "WARNING: HARD CUT ON 100 GeV SIM PARTICLES!\n";
		if(c.sim_energyEvent_ > 100)
			veto = true;

		if (target == CLUSTER) {
			if (c.cluster_ecal_.size() == 0  && c.cluster_hcal_.size() ==0)
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

unsigned TreeUtility::convertCalibratablesToParticleDeposits(
		const std::vector<Calibratable>& input,
		std::vector<ParticleDepositPtr>& toBeFilled, CalibrationTarget target,
		DetectorElementPtr offset, DetectorElementPtr ecal,
		DetectorElementPtr hcal, bool includeOffset) {

	std::cout << __PRETTY_FUNCTION__ << std::endl;
	std::cout << "WARNING: Using fabs() for eta value assignments!\n";
	std::cout << "Input Calibratable has size "<< input.size() << "\n";
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
		if (c.tb_isTB_) {
			pd->setTruthEnergy(c.sim_energyEvent_);
			pd->setEta(c.tb_eta_);
			pd->setPhi(c.tb_phi_);
			veto = false;
		}

		if (c.cands_num_ > 1)
			veto = true;

		if (target == CLUSTER) {
			if (c.cluster_ecal_.size() == 0&& c.cluster_hcal_.size() ==0)
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

