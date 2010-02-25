#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbDicer.h"

//#include <stdexcept>
//#include <utility>
//#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <functional>
//#include <map>
#include <algorithm>
//#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
//#include "Math/TRandom3.h"
#include <CLHEP/Random/RandGauss.h>
#include <CLHEP/Random/Random.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

SurveyPxbDicer::SurveyPxbDicer(std::vector<edm::ParameterSet> pars, unsigned int seed)
{
	CLHEP::HepRandom::setTheSeed( seed );
	mean_a0 = getParByName("a0", "mean", pars);
	sigma_a0 = getParByName("a0", "sigma", pars);
	mean_a1 = getParByName("a1", "mean", pars);
	sigma_a1 = getParByName("a1", "sigma", pars);
	mean_scale = getParByName("scale", "mean", pars);
	sigma_scale = getParByName("scale", "sigma", pars);
	mean_phi = getParByName("phi", "mean", pars);
	sigma_phi = getParByName("phi", "sigma", pars);
	mean_u = getParByName("u", "mean", pars);
	sigma_u = getParByName("u", "sigma", pars);
	mean_v = getParByName("v", "mean", pars);
	sigma_v = getParByName("v", "sigma", pars);
}

std::string SurveyPxbDicer::doDice(const fidpoint_t &fidpointvec, const idPair_t &id)
{
	// Dice the local parameters
	const value_t a0 = ranGauss(mean_a0, sigma_a0);
	const value_t a1 = ranGauss(mean_a1, sigma_a1);
	const value_t scale = ranGauss(mean_scale, sigma_scale);
	const value_t phi = ranGauss(mean_phi, sigma_phi);
	const value_t a2 = scale * cos(phi);
	const value_t a3 = scale * sin(phi);
	const coord_t p0 = transform(fidpointvec[0],a0,a1,a2,a3);
	const coord_t p1 = transform(fidpointvec[1],a0,a1,a2,a3);
	const coord_t p2 = transform(fidpointvec[2],a0,a1,a2,a3);
	const coord_t p3 = transform(fidpointvec[3],a0,a1,a2,a3);
	std::ostringstream oss;
	oss << id.first << " " 
		 << ranGauss(p2.y(),sigma_v) << " " << ranGauss(p2.x(),sigma_u) << " "
		 << ranGauss(p3.y(),sigma_v) << " " << ranGauss(p3.x(),sigma_u) << " "
		 << id.second << " "
		 << ranGauss(p1.y(),sigma_v) << " " << ranGauss(p1.x(),sigma_u) << " "
		 << ranGauss(p0.y(),sigma_v) << " " << ranGauss(p0.x(),sigma_u) << " "
		 << sigma_u << " " << sigma_v;
	return oss.str();
}

std::string SurveyPxbDicer::toString(const int &i) { std::ostringstream oss; oss << i; return oss.str(); }
std::string SurveyPxbDicer::toString(const double &x) { std::ostringstream oss; oss << x; return oss.str(); }

SurveyPxbDicer::coord_t SurveyPxbDicer::transform(const coord_t &x, const value_t &a0, const value_t &a1, const value_t &a2, const value_t &a3)
{
	return coord_t(a0+a2*x.x()+a3*x.y(),a1-a3*x.x()+a2*x.y());
}

SurveyPxbDicer::value_t SurveyPxbDicer::getParByName(const std::string &name, const std::string &par, const std::vector<edm::ParameterSet>& pars)
{
	std::vector<edm::ParameterSet>::const_iterator it;
	it = std::find_if(pars.begin(), pars.end(), std::bind1st(findParByName(),name));
	if (it == pars.end()) { throw std::runtime_error("Parameter not found in SurveyPxbDicer::getParByName"); }
	return (*it).getParameter<value_t>(par);
}
/*
bool SurveyPxbDicer::findParByName(std::string name, edm::ParameterSet parset)
{
	return (parset.getParameter<std::string>("name") == name);
}
*/

