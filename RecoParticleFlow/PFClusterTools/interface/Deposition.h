#ifndef DEPOSITION_HH_
#define DEPOSITION_HH_
#include <iostream>

#include <boost/shared_ptr.hpp>

#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"

namespace pftools {

/**
 * \class Deposition
 * 
 * \brief This class holds an arbitrary energy deposition, specified in terms of angular 
 * position, energy, depth (optional) and detector element type.
 * 
 * \author Jamie Balin
 * \date April 2008
 * */
class Deposition {
public:
	
	inline void setEnergy(double energy) {
		myEnergy = energy;
	}

	
	inline double getEta() const {
		return myEta;
	}
	;

	inline double getPhi() const {
		return myPhi;
	}
	;

	inline double getEnergy() const {
		return myEnergy;
	}
	;

	inline DetectorElementPtr getDetectorElement() const {
		return myElementPtr;
	}
	;

	/* 
	 * Returns the user specified "depth" for this deposition. 
	 * Usually presumed to be zero. 
	 * */
	inline double getDepth() const {
		return myDepth;
	}
	;

	Deposition(DetectorElementPtr element, double eta = 0.0, double phi = 0.0,
			double energy = 0.0, double depth = 0.0);

	virtual ~Deposition();

private:
	//DetectorElement* myElement;
	DetectorElementPtr myElementPtr;
	
	double myEta;
	double myPhi;
	double myEnergy;
	double myDepth;
};

typedef boost::shared_ptr<Deposition> DepositionPtr;

/*
 * Streams a description of this deposition into the supplied stream.
 * */
std::ostream& operator<<(std::ostream& s, const Deposition& d);
 
}

#endif /*DEPOSITION_HH_*/
