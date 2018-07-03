#include "DataFormats/ParticleFlowReco/interface/Calibratable.h"
#include <algorithm>

using namespace pftools;

std::ostream& pftools::operator<<(std::ostream& s, const Calibratable& calib_) {
	s << "Calibratable summary:\n\tsim:\t\t(" << calib_.sim_energyEvent_
			<< "),\t[" << calib_.sim_etaEcal_ << ", " << calib_.sim_phiEcal_
			<< "]\n";
	s << "\ttestbeam:\t(" << calib_.tb_energyEvent_ << ", "
			<< calib_.tb_energyEcal_ << ", " << calib_.tb_energyHcal_ << "), ["
			<< calib_.tb_eta_ << ", " << calib_.tb_phi_ << "]\n";
	s << "\trechits:\t(" << calib_.rechits_energyEvent_ << ", "
			<< calib_.rechits_energyEcal_ << ", " << calib_.rechits_energyHcal_
			<< ")\n";
	s << "\tcluster:\t(" << calib_.cluster_energyEvent_ << ", "
			<< calib_.cluster_energyEcal_ << ", " << calib_.cluster_energyHcal_
			<< ")\n";
	s << "\tcands:\t\t(" << calib_.cand_energyEvent_ << ", "
			<< calib_.cand_energyEcal_ << ", " << calib_.cand_energyHcal_
			<< "), ";
	s << "\t[" << calib_.cand_eta_ << ", " << calib_.cand_phi_ << "], "
			<< calib_.cands_num_ << " of them\n";
	for (std::vector<CandidateWrapper>::const_iterator c =
			calib_.cands_.begin(); c != calib_.cands_.end(); ++c) {
		const CandidateWrapper& cw = *c;
		s << "\t\t\t\tType: " << cw.type_ << ", (" << cw.energy_ << ", "
				<< cw.energyEcal_ << ", " << cw.energyHcal_ << ") at [ "
				<< cw.eta_ << ", " << cw.phi_ << "]\n";
	}
	s << "\t\tNeutral EM energy: " << calib_.cand_energyNeutralEM_ << "\n";
	s << "\t\tNeutral Had energy: " << calib_.cand_energyNeutralHad_ << "\n";

	return s;
}

std::ostream& pftools::operator<<(std::ostream& s,
		const CalibratableElement& ce) {
	s << "CalibratableElement: (energy, eta, phi) = (" << ce.energy_ << ", "
			<< ce.eta_ << ", " << ce.phi_ << ")";

	return s;
}

void Calibratable::recompute() {

	cluster_meanEcal_ = computeMean(cluster_ecal_);
	cluster_meanHcal_ = computeMean(cluster_hcal_);
	rechits_meanEcal_ = computeMean(rechits_ecal_);
	rechits_meanHcal_ = computeMean(rechits_hcal_);
	tb_meanEcal_ = computeMean(tb_ecal_);
	tb_meanHcal_ = computeMean(tb_hcal_);

	cluster_numEcal_ = cluster_ecal_.size();
	cluster_numHcal_ = cluster_hcal_.size();
	rechits_numEcal_ = rechits_ecal_.size();
	rechits_numHcal_ = rechits_hcal_.size();
	tb_numEcal_ = tb_ecal_.size();
	tb_numHcal_ = tb_hcal_.size();

	cluster_energyEvent_ = cluster_meanEcal_.energy_ * cluster_ecal_.size()
			+ cluster_meanHcal_.energy_ * cluster_hcal_.size();
	cluster_energyEcal_ = cluster_meanEcal_.energy_ * cluster_ecal_.size();
	cluster_energyHcal_ = cluster_meanHcal_.energy_ * cluster_hcal_.size();

	rechits_energyEvent_ = rechits_meanEcal_.energy_ * rechits_ecal_.size()
			+ rechits_meanHcal_.energy_ * rechits_hcal_.size();
	rechits_energyEcal_ = rechits_meanEcal_.energy_ * rechits_ecal_.size();
	rechits_energyHcal_ = rechits_meanHcal_.energy_ * rechits_hcal_.size();

	tb_energyEvent_ = tb_meanEcal_.energy_ * tb_ecal_.size()
			+ tb_meanHcal_.energy_ * tb_hcal_.size();
	tb_energyEcal_ = tb_meanEcal_.energy_ * tb_ecal_.size();
	tb_energyHcal_ = tb_meanHcal_.energy_ * tb_hcal_.size();

	cands_num_ = cands_.size();
	cands_mean_ = computeMean(cands_);

	cand_energyEvent_ = 0;
	cand_energyEcal_ = 0;
	cand_energyHcal_ = 0;
	cand_energyNeutralEM_ = 0;
	cand_energyNeutralHad_ = 0;
	cand_type_ = 0;
	cand_eta_ = cands_mean_.eta_;
	cand_phi_ = cands_mean_.phi_;

	for (std::vector<CandidateWrapper>::iterator it = cands_.begin(); it
			!= cands_.end(); ++it) {
		CandidateWrapper& c = *it;
		if (c.type_ == 4)
			cand_energyNeutralEM_ += c.energy_;
		if (c.type_ == 5)
			cand_energyNeutralHad_ += c.energy_;
		cand_energyEvent_ += c.energy_;
		cand_energyEcal_ += c.energyEcal_;
		cand_energyHcal_ += c.energyHcal_;
		cand_type_ += c.type_;
		c.recompute();
	}

	std::sort(tb_ecal_.begin(), tb_ecal_.end());
	std::sort(tb_hcal_.begin(), tb_hcal_.end());
	std::sort(rechits_ecal_.begin(), rechits_ecal_.end());
	std::sort(rechits_hcal_.begin(), rechits_hcal_.end());
	std::sort(cluster_ecal_.begin(), cluster_ecal_.end());
	std::sort(cluster_hcal_.begin(), cluster_hcal_.end());

}

CandidateWrapper Calibratable::computeMean(
		const std::vector<CandidateWrapper>& wrappers) {
	CandidateWrapper cw;

	if (wrappers.empty())
		return cw;
	for (std::vector<CandidateWrapper>::const_iterator it = wrappers.begin(); it
			!= wrappers.end(); ++it) {
		const CandidateWrapper& c = *it;
		cw.energy_ += c.energy_;
		cw.phi_ += c.phi_;
		cw.eta_ += c.eta_;
		cw.energyEcal_ += c.energyEcal_;
		cw.energyHcal_ += c.energyHcal_;
		cw.type_ += c.type_;
	}

	cw.energy_ /= wrappers.size();
	cw.phi_ /= wrappers.size();
	cw.eta_ /= wrappers.size();
	cw.energyEcal_ /= wrappers.size();
	cw.energyHcal_ /= wrappers.size();
	cw.type_ /= wrappers.size();

	return cw;
}

CalibratableElement Calibratable::computeMean(const std::vector<
		CalibratableElement>& diets) {
	CalibratableElement dmean;
	if (diets.empty())
		return dmean;
	for (std::vector<CalibratableElement>::const_iterator cit = diets.begin(); cit
			!= diets.end(); ++cit) {
		CalibratableElement d = *cit;
		dmean.energy_ += d.energy_;
		dmean.eta_ += d.eta_;
		dmean.phi_ += d.phi_;
		dmean.extent_ += d.extent_;
		dmean.time_ += d.time_;
	}
	dmean.energy_ /= diets.size();
	dmean.eta_ /= diets.size();
	dmean.phi_ /= diets.size();
	dmean.extent_ /= diets.size();
	dmean.time_ /= diets.size();
	return dmean;
}

void Calibratable::fillCaloWindow(
		const std::vector<CalibratableElement>& source, CaloWindow& destination) const {
	std::vector<CalibratableElement>::const_iterator cit = source.begin();
	for (; cit != source.end(); ++cit) {
		const CalibratableElement& ce = *cit;
		bool ok = destination.addHit(ce.eta_, ce.phi_, ce.energy_);
		if (!ok)
			std::cout << __PRETTY_FUNCTION__
					<< ": couldn't fill CaloWindow with " << ce << "\n";
	}
}

void Calibratable::reset() {

	calibrations_.clear();

	sim_energyEvent_ = 0;
	sim_energyEcal_ = 0;
	sim_energyHcal_ = 0;
	sim_eta_ = 0;
	sim_phi_ = 0;
	sim_numEvent_ = 0;
	sim_isMC_ = false;

	tb_isTB_ = false;
	tb_eta_ = 0.0;
	tb_phi_ = 0.0;
	tb_run_ = 0;
	tb_pdg_ = 0;
	tb_tof_ = 0;
	tb_ck3_ = 0;
	tb_ck2_ = 0;
	tb_vetosPassed_ = 0;
	tb_energyEvent_ = 0;
	tb_energyEcal_ = 0;
	tb_energyHcal_ = 0;
	tb_ecal_.clear();
	tb_hcal_.clear();
	tb_numEcal_ = 0;
	tb_numHcal_ = 0;
	tb_meanEcal_.reset();
	tb_meanHcal_.reset();

	sim_etaEcal_ = 0;
	sim_etaHcal_ = 0;
	sim_phiEcal_ = 0;
	sim_phiHcal_ = 0;

	recotrk_numHits_ = 0;
	recotrk_quality_ = 0;
	recotrk_charge_ = 0;
	recotrk_etaEcal_ = 0;
	recotrk_phiEcal_ = 0;
	//TODO:: check this is sufficient
	recotrk_momentum_.SetPxPyPzE(0, 0, 0, 0);
	recotrk_deltaRWithSim_ = 0.0;

	cluster_energyEvent_ = 0;
	cluster_energyEcal_ = 0;
	cluster_energyHcal_ = 0;
	cluster_numEcal_ = 0;
	cluster_numHcal_ = 0;
	cluster_ecal_.clear();
	cluster_hcal_.clear();
	cluster_meanEcal_.reset();
	cluster_meanHcal_.reset();

	rechits_energyEvent_ = 0;
	rechits_ecal_.clear();
	rechits_hcal_.clear();
	rechits_energyEcal_ = 0;
	rechits_energyHcal_ = 0;
	rechits_numEcal_ = 0;
	rechits_numHcal_ = 0;
	rechits_meanEcal_.reset();
	rechits_meanHcal_.reset();

	cands_.clear();
	cands_num_ = 0;
	cands_mean_.reset();
	cand_energyEvent_ = 0;
	cand_energyEcal_ = 0;
	cand_energyHcal_ = 0;
	cand_eta_ = 0;
	cand_phi_ = 0;
	cand_type_ = -1;
	cand_energyNeutralEM_ = 0;
	cand_energyNeutralHad_ = 0;

	calowindow_ecal_.reset();
	calowindow_hcal_.reset();

}
