#ifndef GUARD_surveypxbdicer_h
#define GUARD_surveypxbdicer_h

#include <vector>
#include <functional>
#include <utility>
#include <fstream>
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include <CLHEP/Random/RandGauss.h>
#include <CLHEP/Random/Random.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <stdexcept>

//! Class to dice a picture from a given set of fiducial points
//! This class has its use for toy MC simulations to validate the PXB survey
class SurveyPxbDicer
{
public:
	typedef SurveyPxbImage::coord_t coord_t;
	typedef double value_t;
	typedef std::vector<coord_t> fidpoint_t;
	typedef unsigned int count_t;
	typedef SurveyPxbImage::id_t id_t;
	typedef std::pair<id_t,id_t> idPair_t;

	// Constructors
	SurveyPxbDicer() {};
	// Constructor from VPSet
	SurveyPxbDicer(const std::vector<edm::ParameterSet>& pars, unsigned int seed);

	//! Invoke the dicer
	//! \param fidpointvec vector with fiducial points where values need to be diced for and transformed to the photo fram
	//! \param idPair pair of the id values
	std::string doDice(const fidpoint_t &fidpointvec, const idPair_t &id, const bool rotate=false);
	void doDice(const fidpoint_t &fidpointvec, const idPair_t &id, std::ofstream &outfile, const bool rotate=false);

private:
	//! invoke the RNG to geat a gaussian smeared value
	//! \param mean mean value
	//! \param sigma 
	value_t ranGauss(value_t mean, value_t sigma) {return CLHEP::RandGauss::shoot(mean,sigma);};

	value_t mean_a0, sigma_a0;
	value_t mean_a1, sigma_a1;
	value_t mean_scale, sigma_scale;
	value_t mean_phi, sigma_phi;
	value_t mean_x, sigma_x;
	value_t mean_y, sigma_y;

	//! Gets parameter by name from the VPSet
	//! \param name name of the parameter to be searched for in field 'name' of the VPSet
	//! \param par selects the value, i.e. mean or sigma
	//! \param pars reference to VPSet
	value_t getParByName(const std::string &name, const std::string &par, const std::vector<edm::ParameterSet>& pars);

	//! Transform the diced values to the frame of a toy photograph
	//! \param x coordinate to be transformed from local frame to photo frame
	//! \param a0 Transformation parameter, shift in x
	//! \param a1 Transformation parameter, shift in y
	//! \param a2 Transformation parameter, scale*cos(phi)
	//! \param a3 Transformation parameter, scale*sin(phi)
	coord_t transform(const coord_t &x, const value_t &a0, const value_t &a1, const value_t &a2, const value_t &a3);

	//! Function object for searching for a parameter in the VPSet
	struct findParByName: public std::binary_function< std::string, edm::ParameterSet, bool >
	{
		bool operator () ( const std::string &name, const edm::ParameterSet &parset )
			const {	return (parset.getParameter<std::string>("name") == name);}
	};

};

#endif

