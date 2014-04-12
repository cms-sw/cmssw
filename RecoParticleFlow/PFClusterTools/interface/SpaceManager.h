#ifndef SPACEMANAGER_HH_
#define SPACEMANAGER_HH_
#include "RecoParticleFlow/PFClusterTools/interface/SpaceVoxel.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"
#include "RecoParticleFlow/PFClusterTools/interface/Region.h"
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <TGraph.h>
#include <TF1.h>
#include <TF2.h>
#include <string>
namespace pftools {
/**
 \class SpaceManager 
 \brief A tool to associate SpaceVoxels with Calibrator objects

 \author Jamie Ballin
 \date   April 2008
 */
class SpaceManager {
public:
	SpaceManager(std::string name);

	virtual ~SpaceManager();
	
	std::string getName() {
		return name_;
	}

	
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
	
	void createCalibrators(const Calibrator& toClone);

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
	
	void assignCalibration(const CalibratorPtr& c, const std::map<DetectorElementPtr, double>& result);
	
	std::map<DetectorElementPtr, double> getCalibration(CalibratorPtr c);
	
	std::ostream& printCalibrations(std::ostream& stream);
	
	TH1* extractEvolution(DetectorElementPtr det, Region region, TF1& f1, bool useTruth = true);

	void addEvolution(const DetectorElementPtr& det, Region region, const TF1& f) {
		if(region == BARREL_POS)
			barrelPosEvolutions_[det] = f;
		if(region == ENDCAP_POS)
			endcapPosEvolutions_[det] = f;
	}
	
	double interpolateCoefficient(DetectorElementPtr det, double energy, double eta, double  phi);
	
	double evolveCoefficient(DetectorElementPtr det, double energy, double eta, double  phi);
	
	int getNCalibrations() {
		return calibrationCoeffs_.size();
	}
	void clear();
	
	void makeInverseAddressBook();
	
	void setBarrelLimit(double limit) {
		barrelLimit_ = limit;
	}

private:
	
	std::string name_;
	
	double barrelLimit_;
	double transitionLimit_;
	double endcapLimit_;
	
	std::map<SpaceVoxelPtr, CalibratorPtr> myAddressBook;
	std::map<CalibratorPtr, SpaceVoxelPtr> inverseAddressBook_;
	std::map<CalibratorPtr, std::map<DetectorElementPtr, double> > calibrationCoeffs_;
	std::vector<SpaceVoxelPtr> myKnownSpaceVoxels;
	
	std::vector<SpaceVoxelPtr> barrelPosRegion_;
	std::vector<SpaceVoxelPtr> transitionPosRegion_;
	std::vector<SpaceVoxelPtr> endcapPosRegion_;
	
	std::map<DetectorElementPtr, TF1> barrelPosEvolutions_;
	std::map<DetectorElementPtr, TF1> endcapPosEvolutions_;
	
	std::map<Region, std::vector<SpaceVoxelPtr> > regionsToSVs_;
	
	
};

typedef boost::shared_ptr<SpaceManager> SpaceManagerPtr;

}
#endif /*SPACEMANAGER_HH_*/
