#ifndef CALIBRATABLE_H_
#define CALIBRATABLE_H_

//#include <boost/shared_ptr.hpp>

#include <vector>
#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/CalibrationResultWrapper.h"
#include "DataFormats/ParticleFlowReco/interface/CalibrationProvenance.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowReco/interface/CaloWindow.h"
#include "DataFormats/ParticleFlowReco/interface/CaloEllipse.h"

namespace pftools {

/**
 * \class CalibratableElement
 * \brief Small wrapper class for storing individual rechit and cluster information.
 *
 * \author Jamie Ballin
 * \date	June 2008
 */
class CalibratableElement {
public:
	CalibratableElement() {
		reset();
	}

	CalibratableElement(double energy, double eta, double phi, int layer,
			double extent = 0.0, double time = 0.0) :
		energy_(energy), eta_(eta), phi_(phi), time_(time), layer_(layer), extent_(extent) {
	}

	double energy_, eta_, phi_, time_;
	int layer_;

	//RMS of deltaR of hits in the cluster to its centre
	double extent_;

	void reset() {
		energy_ = 0.0;
		eta_ = 0.0;
		phi_ = 0.0;
		layer_ = 0;
		extent_ = 0.0;
		time_ = 0.0;
	}
	bool operator<(const CalibratableElement& em) const {
		if (em.energy_ < energy_)
			return true;
		return false;
	}


};

/**
 * \class CandidateWrapper
 * \brief Small wrapper class to store information associated with PFCandidates
 * \author Jamie Ballin
 * \date May 2008
 *
 * Documentation added Dec 08.
 */
class CandidateWrapper {
public:
	CandidateWrapper() {
		reset();
	}

	CandidateWrapper(double energy, double eta, double phi, double energyEcal,
			double energyHcal, int type) :
		energy_(energy), eta_(eta), phi_(phi), energyEcal_(energyEcal),
				energyHcal_(energyHcal), type_(type) {
		calowindow_ecal_.reset();
		calowindow_hcal_.reset();
		caloellipse_ecal_.reset();
		caloellipse_hcal_.reset();

	}

	double energy_, eta_, phi_, energyEcal_, energyHcal_;
	int cluster_numEcal_, cluster_numHcal_;
	int type_;

	CaloWindow calowindow_ecal_;
	CaloWindow calowindow_hcal_;

	CaloEllipse caloellipse_ecal_;
	CaloEllipse caloellipse_hcal_;

	void reset() {
		energy_ = 0;
		eta_ = 0;
		phi_ = 0;
		energyEcal_ = 0;
		energyHcal_ = 0;
		cluster_numEcal_ = 0;
		cluster_numHcal_ = 0;
		type_ = -1;
		calowindow_ecal_.reset();
		calowindow_hcal_.reset();
		caloellipse_ecal_.reset();
		caloellipse_hcal_.reset();
	}

	void recompute() {
		caloellipse_ecal_.makeCaches();
		caloellipse_hcal_.makeCaches();
	}
};
/**
 \class Calibratable
 \brief Wraps essential single particle calibration data ready for export to a Root file.

 Note that a Reflex dictionary must be produced for this class, for the Root branching mechanisms to work.

 \author Jamie Ballin
 \date   May 2008
 */
class Calibratable {
public:

	Calibratable() {
		reset();
	}

	virtual ~Calibratable() {
	}

	/**
	 * Call to reset() (useful for TTree users)
	 */
	virtual void reset();

	/**
	 * For each collection: candidates, clusters, rechits and truth
	 * overall energy, ecal, hcal, n of each, eta and phi
	 * naming scheme: collection_variable_
	 * All _energyEcal_, _energyHcal_, _eta_ and _phi_ are mean values
	 * (i.e. value = sum of elements/number of elements)
	 * BUT _energyEvent_ fields are sums of all elements.
	 */
	//truth first
	double sim_energyEvent_, sim_eta_, sim_phi_;
	double sim_energyEcal_, sim_energyHcal_;
	double sim_etaEcal_, sim_etaHcal_, sim_phiEcal_, sim_phiHcal_;
	int sim_numEvent_;
	//set to true if this event is not real data
	bool sim_isMC_;
	//test beam specific
	bool tb_isTB_;
	//where was the TB table?
	double tb_eta_, tb_phi_;
	//TB run number and PDG particle type
	int tb_run_, tb_pdg_;
	//Veto counter values
	double tb_tof_, tb_ck3_, tb_ck2_;
	//Erm, bit complicated this one...
	char tb_vetosPassed_;

	//DYNAMIC: Computed from tb_ecal_ and tb_hcal_
	double tb_energyEvent_, tb_energyEcal_, tb_energyHcal_;
	std::vector<CalibratableElement> tb_ecal_, tb_hcal_;
	int tb_numEcal_, tb_numHcal_; //DYNAMIC

	CalibratableElement tb_meanEcal_, tb_meanHcal_; //DYNAMIC

	//leading track
	double recotrk_numHits_, recotrk_quality_, recotrk_charge_;
	double recotrk_etaEcal_, recotrk_phiEcal_;
	//delta phi between sim particle and leading track
	double recotrk_deltaRWithSim_;
	math::XYZTLorentzVector recotrk_momentum_;

	//CaloWindow class (new for 3_1_X)
	CaloWindow calowindow_ecal_;
	CaloWindow calowindow_hcal_;

	//pf clusters
	//DYNAMIC: Computed from cluster_ecal_ and cluster_hcal_
	double cluster_energyEvent_, cluster_energyEcal_, cluster_energyHcal_; //DYNAMIC
	std::vector<CalibratableElement> cluster_ecal_, cluster_hcal_;
	int cluster_numEcal_, cluster_numHcal_; //DYNAMIC
	CalibratableElement cluster_meanEcal_, cluster_meanHcal_; //DYNAMIC

	//pf rechits
	//DYNAMIC: Computed from rechits_ecal_ and rechits_hcal_
	double rechits_energyEvent_, rechits_energyEcal_, rechits_energyHcal_;
	std::vector<CalibratableElement> rechits_ecal_, rechits_hcal_;
	int rechits_numEcal_, rechits_numHcal_; //DYNAMIC
	CalibratableElement rechits_meanEcal_, rechits_meanHcal_; //DYNAMIC

	//pf candidates
	std::vector<CandidateWrapper> cands_;
	CandidateWrapper cands_mean_; //DYNAMIC
	int cands_num_; //DYNAMIC

	//DYNAMIC: Computed from cands_
	double cand_energyEvent_, cand_energyEcal_, cand_energyHcal_, cand_eta_,
			cand_phi_;
	double cand_energyNeutralEM_, cand_energyNeutralHad_; //DYNAMIC
	int cand_type_; //DYNAMIC

	std::vector<CalibrationResultWrapper> calibrations_;

	//Recomputes cluster and rechit averages using the vectors of DepositDiets
	//Users should call this before filling the tree.
	virtual void recompute();

	/**
	 * Compute the mean of a vector of CalibratableElements
	 * @param diets
	 * @return
	 */
	virtual CalibratableElement computeMean(const std::vector<
			CalibratableElement>& diets);

	/**
	 * Compute the mean of a vector of CandidateWrappers
	 * @param wrappers
	 * @return
	 */
	virtual CandidateWrapper computeMean(
			const std::vector<CandidateWrapper>& wrappers);

	/**
	 * Helper method to fill a CaloWindow with CalibratableElement objects
	 * You should initialise the CaloWindow first!
	 * @param source
	 * @param destination
	 */
	virtual void fillCaloWindow(const std::vector<CalibratableElement>& source,
			CaloWindow& destination) const;

};

//typedef boost::shared_ptr<Calibratable> CalibratablePtr;

std::ostream& operator<<(std::ostream& s, const Calibratable& calib_);
std::ostream& operator<<(std::ostream& s, const CalibratableElement& ce_);

}
#endif /*CALIBRATABLE_H_*/

