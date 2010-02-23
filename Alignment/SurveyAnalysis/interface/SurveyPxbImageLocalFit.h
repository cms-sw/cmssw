#ifndef GUARD_surveypxbimagelocalfit_h
#define GUARD_surveypxbimagelocalfit_h

#include <sstream>
#include <vector>
#include <utility>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"
//#include <iostream>

//! Class to hold one picture of the BPix survey and the local fit
class SurveyPxbImageLocalFit : public SurveyPxbImage
{
public:
	typedef std::vector<value_t> localpars_t;
	//typedef std::vector<LocalPoint> fidpoint_t;
	typedef std::vector<coord_t> fidpoint_t;

	// Constructors
	SurveyPxbImageLocalFit(): 
		SurveyPxbImage(), a_(4,0), fidpoints_(4), fitValidFlag_(false) 
	{
		initFidPoints();
	};

	//! Constructor from istringstream
	SurveyPxbImageLocalFit(std::istringstream &iss): 
		SurveyPxbImage(iss), a_(4,0), fidpoints_(4), fitValidFlag_(false) 
	{
		initFidPoints();
	};

	//! Invoke the fit
	void doFit(const fidpoint_t &fidpointvec);
	void doFit(value_t x1, value_t y1, value_t g1, value_t x2, value_t y2, value_t g2);

	//! returns validity flag
	bool isFitValid() { return fitValidFlag_; };

	//! returns local parameters after fit
	localpars_t getLocalParameters();

	//! returns the chi^2 of the fit
	value_t getChi2();

private:
	//! Local parameters
	localpars_t a_;

	//! True position of the fiducial points on a sensor wrt. local frame (u,v)
	std::vector<coord_t> fidpoints_;

	//! chi2 of the local fit
	value_t chi2_;

	//! Validity Flag
	bool fitValidFlag_;

	//! Initialise the fiducial points
	void initFidPoints()
	{
		fidpoints_[0] = coord_t(-0.91,+3.30);
		fidpoints_[1] = coord_t(+0.91,+3.30);
		fidpoints_[2] = coord_t(+0.91,-3.30);
		fidpoints_[3] = coord_t(-0.91,-3.30);
	}
	
};

#endif

