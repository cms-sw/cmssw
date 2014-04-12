#ifndef GUARD_surveypxbimagelocalfit_h
#define GUARD_surveypxbimagelocalfit_h

#include <sstream>
#include <vector>
#include <utility>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include <iostream>

//! Class to hold one picture of the BPix survey and the local fit
class SurveyPxbImageLocalFit : public SurveyPxbImage
{
public:
	typedef std::vector<value_t> localpars_t;
	typedef std::vector<coord_t> fidpoint_t;
	typedef unsigned int count_t;
	static const count_t nGlD = 3; // no of global derivs
	static const count_t nLcD = 4; // no of local derivs
	static const count_t nMsrmts = 8; // no of measurements per image
	static const count_t nLcPars = 4; // no of local parameters
	static const count_t nFidpoints = 4; // no of fiducial points
	// Typedefs for pede
	typedef int pede_label_t;
	typedef float pede_deriv_t;

	// Constructors
	SurveyPxbImageLocalFit(): 
		SurveyPxbImage(), a_(nLcPars,0), fidpoints_(nFidpoints), fitValidFlag_(false), derivsValidFlag_(false)
	{
		initFidPoints();
	};

	//! Constructor from istringstream
	SurveyPxbImageLocalFit(std::istringstream &iss): 
	    SurveyPxbImage(iss), a_(nLcPars,0), fidpoints_(nFidpoints), fitValidFlag_(false), derivsValidFlag_(false)
	{
		initFidPoints();
	};

	//! Invoke the fit
	void doFit(const fidpoint_t &fidpointvec);
	void doFit(const fidpoint_t &fidpointvec, const pede_label_t &label1, const pede_label_t &label2);
	void doFit(value_t x1, value_t y1, value_t g1, value_t x2, value_t y2, value_t g2);

	//! returns validity flag
	bool isFitValid() { return fitValidFlag_; };

	//! returns local parameters after fit
	localpars_t getLocalParameters();

	//! returns the chi^2 of the fit
	value_t getChi2();

	pede_label_t getLocalDerivsSize()                      {return nLcD; };
	pede_label_t getGlobalDerivsSize()                     {return nGlD; };
	const pede_deriv_t* getLocalDerivsPtr(count_t i)       {return localDerivsMatrix_.Array()+i*nLcD; };
	const pede_deriv_t* getGlobalDerivsPtr(count_t i)      {return globalDerivsMatrix_.Array()+i*nGlD;};
	const pede_label_t* getGlobalDerivsLabelPtr(count_t i) {return i<4 ? &labelVec1_[0] : &labelVec2_[0];};
	pede_deriv_t getResiduum(count_t i) { return (pede_deriv_t) r(i); };
	pede_deriv_t getSigma(count_t i) { return i%2 ? sigma_x_ : sigma_y_ ; };

	void setLocalDerivsToZero(count_t i);
	void setGlobalDerivsToZero(count_t i);

private:
	//! Local parameters
	localpars_t a_;
	
	//! Vector of residuals
	ROOT::Math::SVector<value_t, nMsrmts> r;

	//! Matrix with global derivs
	//std::vector<localDerivs_t> globalDerivsVec_;
	ROOT::Math::SMatrix<pede_deriv_t,nMsrmts,nGlD> globalDerivsMatrix_;

	//! Matrix with local derivs
	//std::vector<globalDerivs_t> localDerivsVec_;
	ROOT::Math::SMatrix<pede_deriv_t,nMsrmts,nLcD> localDerivsMatrix_;

	//! Vector with labels to global derivs
	std::vector<pede_label_t> labelVec1_, labelVec2_;

	//! True position of the fiducial points on a sensor wrt. local frame (u,v)
	std::vector<coord_t> fidpoints_;

	//! chi2 of the local fit
	value_t chi2_;

	//! Validity Flag
	bool fitValidFlag_;

	//! 
	bool derivsValidFlag_;

	//! Initialise the fiducial points
	void initFidPoints()
	{
		fidpoints_[0] = coord_t(-0.91,-3.30);
		fidpoints_[1] = coord_t(+0.91,-3.30);
		fidpoints_[2] = coord_t(-0.91,+3.30);
		fidpoints_[3] = coord_t(+0.91,+3.30);
	}

	//! Distance
	value_t dist(const coord_t& p1, const coord_t& p2)
	{
	    value_t dx = p1.x()-p2.x();
	    value_t dy = p1.y()-p2.y();
	    return sqrt(dx*dx+dy*dy);
	}
	
};

#endif

