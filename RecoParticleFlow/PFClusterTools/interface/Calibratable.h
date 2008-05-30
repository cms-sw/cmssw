#ifndef CALIBRATABLE_H_
#define CALIBRATABLE_H_

#include <boost/shared_ptr.hpp>

#include <vector>

#include "RecoParticleFlow/PFClusterTools/interface/CalibrationResultWrapper.h"

namespace pftools {
/**
 \class Calibratable 
 \brief Wraps essential single particle calibration data ready for export to a Root file.
 
 Note that a Reflex dictionary must be produced for this class, for the Root branching mechanisms to work.

 \author Jamie Ballin
 \date   May 2008
 */
class Calibratable {
public:

	typedef boost::shared_ptr<Calibratable> CalibratablePtr;

	Calibratable() {
		reset();
	}

	virtual ~Calibratable() {
	}
	;

	virtual void reset() {

		calibrations_.clear();

		sim_energyEvent_ = 0;
		sim_eta_ = 0;
		sim_phi_ = 0;
		sim_numEvent_ = 0;
		sim_isMC_ = false;

		cluster_energyEvent_ = 0;
		cluster_energyEcal_ = 0;
		cluster_energyHcal_ = 0;
		cluster_etaEcal_ = 0;
		cluster_phiEcal_ = 0;
		cluster_etaHcal_ = 0;
		cluster_phiHcal_ = 0;
		cluster_numEcal_ = 0;
		cluster_numHcal_= 0;

		rechits_energyEvent_ = 0;
		rechits_energyEcal_ = 0;
		rechits_energyHcal_ = 0;
		rechits_etaEcal_ = 0;
		rechits_phiEcal_ = 0;
		rechits_etaHcal_ = 0;
		rechits_phiHcal_ = 0;
		rechits_numEcal_ = 0;
		rechits_numHcal_ = 0;

		cand_energyEvent_ = 0;
		cand_energyEcal_ = 0;
		cand_energyHcal_ = 0;
		cand_eta_ = 0;
		cand_phi_ = 0;
		cand_numEvent_ = 0;

		pfele_energyEvent_ = 0;
		pfele_energyEcal_ = 0;
		pfele_energyHcal_ = 0;
		pfele_numEcal_ = 0;
		pfele_numHcal_ = 0;
	}
	;

	/*
	 * For each collection: candidates, clusters, rechits and truth
	 * overall energy, ecal, hcal, n of each, eta and phi
	 * naming scheme: collection_variable_
	 * eta and phi given at ECAL front face
	 * All _energyEcal_, _energyHcal_, _eta_ and _phi_ are mean values
	 * (i.e. value = sum of elements/number of elements)
	 * BUT _energyEvent_ fields are sums of all elements.
	 */
	//truth first
	double sim_energyEvent_, sim_eta_, sim_phi_;
	int sim_numEvent_;
	//set to true if this event is not real data
	bool sim_isMC_;
	//clusters
	double cluster_energyEvent_, cluster_energyEcal_, cluster_energyHcal_,
			cluster_etaEcal_, cluster_phiEcal_, cluster_etaHcal_,
			cluster_phiHcal_;
	int cluster_numEcal_, cluster_numHcal_;
	//rechits
	double rechits_energyEvent_, rechits_energyEcal_, rechits_energyHcal_,
			rechits_etaEcal_, rechits_phiEcal_, rechits_etaHcal_,
			rechits_phiHcal_;
	int rechits_numEcal_, rechits_numHcal_;
	//pf candidates
	double cand_energyEvent_, cand_energyEcal_, cand_energyHcal_, cand_eta_,
			cand_phi_;
	int cand_numEvent_;
	//pf elements
	double pfele_energyEvent_, pfele_energyEcal_, pfele_energyHcal_;
	int pfele_numEcal_, pfele_numHcal_;

	std::vector<CalibrationResultWrapper> calibrations_;

};
}
#endif /*CALIBRATABLE_H_*/
