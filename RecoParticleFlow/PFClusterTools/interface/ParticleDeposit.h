#ifndef PARTICLEDEPOSIT_HH_
#define PARTICLEDEPOSIT_HH_

#include <vector>

#include "RecoParticleFlow/PFClusterTools/interface/Deposition.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"


#include <iostream>

/**
 \class ParticleDeposit 
 \brief An object that encapsualtes energy depositions (real, MC or otherwise) made by  particle in N DetectorElement objects.

 \author Jamie Ballin
 \date   April 2008
 */
namespace pftools {

class ParticleDeposit {
public:

	ParticleDeposit(double truthEnergy = -1.0, double eta = 0, double phi = 0);
	virtual ~ParticleDeposit();

	virtual void addRecDeposition(const Deposition& rec);
	virtual void addTruthDeposition(const Deposition& truth);

	virtual const std::vector<Deposition>& getRecDepositions() const;
	virtual std::vector<Deposition> getTruthDepositions() const;

	/*
	 * Returns the overall MC particle energy.
	 */
	virtual double getTruthEnergy() const {
		return myTruthEnergy;
	}

	/* 
	 * Returns the detected energy from this detector element, including calibration.
	 */
	virtual double getRecEnergy(const DetectorElementPtr de) const;

	virtual double getRecEnergy() const;
	
	virtual void setRecEnergy(const DetectorElementPtr de, double energy);

	virtual double getEnergyResolution() const;

	/*
	 * Returns the raw MC energy input into this detector element.
	 */
	virtual double getTruthEnergy(const DetectorElementPtr de) const;

	virtual unsigned getId() const {
		return myId;
	}

	inline double getEta() const {
		return myEta;
	}

	virtual double getPhi() const {
		return myPhi;
	}
	
	void setTruthEnergy(const double truth) {
		myTruthEnergy = truth;
	}
	
	void setPhi(const double phi) {
		myPhi = phi;
	}
	
	void setEta(const double eta) {
		myEta = eta;
	}

	double getTargetFunctionContrib() const;

	friend std::ostream& operator<<(std::ostream& s, const ParticleDeposit& p);

private:
	static unsigned count;
	//ParticleDeposit(const ParticleDeposit& pd);
	std::vector<Deposition> myRecDepositions;
	std::vector<Deposition> myTruthDepositions;
	unsigned myId;

	double myTruthEnergy;
	double myEta;
	double myPhi;

	/*
	 * For general ROOT dictionary building happiness!
	 */
	std::vector<ParticleDeposit*> pdps_;
	//	std::vector<ParticleDeposit> pds_;
};


typedef boost::shared_ptr<ParticleDeposit> ParticleDepositPtr;

}

#endif /*PARTICLEDEPOSIT_HH_*/
