#ifndef DEPOSITION_HH_
#define DEPOSITION_HH_
#include <iostream>
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.hh"

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

	inline DetectorElement* getDetectorElement() const {
		return myElement;
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

	Deposition(DetectorElement* element, double eta = 0.0, double phi = 0.0,
			double energy = 0.0, double depth = 0.0);

	virtual ~Deposition();

	/*
	 * Streams a description of this deposition into the supplied stream.
	 * */
	friend std::ostream& operator<<(std::ostream& s, const Deposition& d);

private:
	DetectorElement* myElement;
	double myEta;
	double myPhi;
	double myEnergy;
	double myDepth;
};
}

#endif /*DEPOSITION_HH_*/
