#ifndef SPACEMANAGER_HH_
#define SPACEMANAGER_HH_
#include "RecoParticleFlow/PFClusterTools/interface/SpaceVoxel.hh"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.hh"
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace pftools {
/**
 \class SpaceManager 
 \brief A tool to associate SpaceVoxels with Calibrator objects

 \author Jamie Ballin
 \date   April 2008
 */
class SpaceManager {
public:
	SpaceManager();

	virtual ~SpaceManager();

	/*
	 * Initialises the internal map of calibrators and space voxels according to the
	 * type of calibrator supplied and the specified eta, phi and energy segmentation.
	 */
	void createCalibrators(const Calibrator& toClone, const double etaSeg,
			const double phiSeg, const double energySeg);

	/*
	 * As above but only for the specified ranges. 
	 * (Compare with ROOT TH3F histogram constructor!)
	 */
	void createCalibrators(const Calibrator& toClone, const unsigned nEta,
			const double etaMin, const double etaMax, const unsigned nPhi,
			const double phiMin, const double phiMax, const unsigned nEnergy,
			const double energyMin, const double energyMax) throw(PFToolsException&);

	std::map<SpaceVoxelPtr, CalibratorPtr>* getCalibrators() {
		std::map<SpaceVoxelPtr, CalibratorPtr>* ptr = &myAddressBook;
		return ptr;
	}

	/* 
	 * Adds a calibrator for the specified volume element.
	 * Returns a pointer to it once it's been created, and returns a pointer to
	 * any exisitng calibrator should that SpaceVoxel already exist.
	 */
	CalibratorPtr createCalibrator(const Calibrator& toClone, SpaceVoxelPtr s);

	/*
	 * Returns a pointer to the calibrator you need for the specified space point.
	 * Returns 0 if it's not found.
	 */
	CalibratorPtr findCalibrator(const double eta, const double phi,
			const double energy = 0) const;
	
	void clear();

private:
	std::map<SpaceVoxelPtr, CalibratorPtr> myAddressBook;
	std::vector<SpaceVoxelPtr> myKnownSpaceVoxels;
};



}
#endif /*SPACEMANAGER_HH_*/
