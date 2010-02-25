#ifndef GUARD_surveypxbdicer_h
#define GUARD_surveypxbdicer_h

//#include <sstream>
#include <vector>
#include <functional>
#include <utility>
//#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
//#include "Math/TRandom3.h"
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
	SurveyPxbDicer(std::vector<edm::ParameterSet> pars, unsigned int seed);

	std::string doDice(const fidpoint_t &fidpointvec, const idPair_t &id);

private:
	value_t ranGauss(value_t mean, value_t sigma) {return CLHEP::RandGauss::shoot(mean,sigma);};

	value_t mean_a0, sigma_a0;
	value_t mean_a1, sigma_a1;
	value_t mean_scale, sigma_scale;
	value_t mean_phi, sigma_phi;
	value_t mean_u, sigma_u;
	value_t mean_v, sigma_v;

	value_t getParByName(const std::string &name, const std::string &par, const std::vector<edm::ParameterSet>& pars);

	coord_t transform(const coord_t &x, const value_t &a0, const value_t &a1, const value_t &a2, const value_t &a3);


	std::string toString(const int &i);
	std::string toString(const double &x);

	struct findParByName: public std::binary_function< std::string, edm::ParameterSet, bool >
	{
		bool operator () ( const std::string &name, const edm::ParameterSet &parset )
			const {	return (parset.getParameter<std::string>("name") == name);}
	};

};

#endif

