#ifndef GUARD_surveypxbimagelocalfit_h
#define GUARD_surveypxbimagelocalfit_h

#include <sstream>
#include <vector>
#include <utility>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"

//! Class to hold one picture of the BPix survey and the local fit
class SurveyPxbImageLocalFit : public SurveyPxbImage
{
public:
	typedef std::vector<value_t> localpars_t;

	// Constructors
	SurveyPxbImageLocalFit(): 
		SurveyPxbImage(), a(4,0), fitValidFlag(false) {};

	SurveyPxbImageLocalFit(std::istringstream &iss): 
		SurveyPxbImage(iss), a(4,0), fitValidFlag(false) {};

	//! returns validity flag
	bool isFitValid() { return fitValidFlag; };
	

private:
	//! Local parameters
	localpars_t a;

	//! Global parameters
	value_t x, y, gamma;

	//! Validity Flag
	bool fitValidFlag;
	
};

#endif

