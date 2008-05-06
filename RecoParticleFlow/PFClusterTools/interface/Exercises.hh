#ifndef EXERCISES_HH_
#define EXERCISES_HH_

#include <TFile.h>
#include <TH1F.h>
#include <map>

#include "RecoParticleFlow/PFClusterTools/interface/SpaceVoxel.hh"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.hh"


namespace pftools {
/**
 * \class Exercises
 * \brief Simple test harness for the PFClusterTools package. 
 * 
 * Instantiate one of these classes, and call methods to exercise the objects in the PFClusterTools library.
 * \author Jamie Balin
 * \date April 2008
 */
class Exercises {
public:
	Exercises();
	virtual ~Exercises();

	/* 
	 * This tries to recreate SingleParticleWrappers from the supplied TFile
	 */
	void testTreeUtility(TFile& f) const;

	/*
	 * This does a test with random data that the calibration chain is working properly.
	 */
	void testCalibrators() const;

	/*
	 * This calibrates SingleParticleWrappers found in the supplied TFile.
	 */
	void testCalibrationFromTree(TFile& f) const;

	/*
	 * This (temporary) method evaluates the energy resolution before and after calibration.
	 */
	void evaluatePerformance(
			const std::map<SpaceVoxelPtr, CalibratorPtr>* const detectorMap,
			std::map<SpaceVoxelPtr, TH1F>& before,
			std::map<SpaceVoxelPtr, TH1F>& after) const;
	
	void writeOutHistos(std::map<SpaceVoxelPtr, TH1F>& input) const;
	
};
}

#endif /*EXERCISES_HH_*/
