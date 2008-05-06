#ifndef SPACEVOXEL_HH_
#define SPACEVOXEL_HH_
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <string>

namespace pftools {

/**\class SpaceVoxel 
 \brief A multi-dimensional volume element to subdivide the detector into different calibration regions.

 \author Jamie Ballin
 \date   April 2008
 */
class SpaceVoxel {
public:
	/*
	 * Constructs a SpaceVoxel over the specified region.
	 * 
	 * Upper edges are exclusive; Lower edges are inclusive.
	 * 
	 * If any of XXXBegin == XXXEnd, then this criterion is ignored and the corresponding
	 * containsXXX(y) will return true.
	 */
	SpaceVoxel(double etaBegin = 0, double etaEnd = 0, double phiBegin = 0,
			double phiEnd = 0, double energyBegin = 0, double energyEnd = 0);

	virtual ~SpaceVoxel();

	virtual bool contains(const double& eta, const double& phi,
			const double& energy) const;

	virtual bool containsEta(const double& eta) const;

	virtual bool containsPhi(const double& phi) const;

	virtual bool containsEnergy(const double& energy) const;

	void print(std::ostream& s) const;
	
	void printMsg() {
		std::cout << "Hello!\n";
	}
	
	int getInt() {
		return 2;
	}
	
	//Prints this SpaceVoxel's name into the supplied string
	void getName(std::string& s) const;

	friend std::ostream& operator<<(std::ostream& s,
			const pftools::SpaceVoxel& sv);
	

private:
	double myEtaMin;
	double myEtaMax;
	double myPhiMin;
	double myPhiMax;
	double myEnergyMin;
	double myEnergyMax;

};

typedef boost::shared_ptr<SpaceVoxel> SpaceVoxelPtr;

}
#endif /*SPACEVOXEL_HH_*/
