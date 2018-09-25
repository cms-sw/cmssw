#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbDicer.h"

//#include <stdexcept>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <functional>
#include <algorithm>
#include <fstream>
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/Random.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

SurveyPxbDicer::SurveyPxbDicer(const std::vector<edm::ParameterSet>& pars, unsigned int seed)
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
	mean_x = getParByName("x", "mean", pars);
	sigma_x = getParByName("x", "sigma", pars);
	mean_y = getParByName("y", "mean", pars);
	sigma_y = getParByName("y", "sigma", pars);
}

std::string SurveyPxbDicer::doDice(const fidpoint_t &fidpointvec, const idPair_t &id, const bool rotate)
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
	const value_t sign = rotate ? -1 : 1;
	std::ostringstream oss;
	oss << id.first << " " 
		 << sign*ranGauss(p0.x(),sigma_x) << " " << -sign*ranGauss(p0.y(),sigma_y) << " "
		 << sign*ranGauss(p1.x(),sigma_x) << " " << -sign*ranGauss(p1.y(),sigma_y) << " "
		 << id.second << " "
		 << sign*ranGauss(p2.x(),sigma_x) << " " << -sign*ranGauss(p2.y(),sigma_y) << " "
		 << sign*ranGauss(p3.x(),sigma_x) << " " << -sign*ranGauss(p3.y(),sigma_y) << " "
		 << sigma_x << " " << sigma_y << " "
		 << rotate
		 << " # MC-truth:"
		 << " a0-a3: " << a0 << " " << a1 << " " << a2 << " " << a3
		 << " S: " << scale
		 << " phi: " << phi
		 << " x0: " << fidpointvec[0].x() << " " << fidpointvec[0].y()
		 << " x1: " << fidpointvec[1].x() << " " << fidpointvec[1].y()
		 << " x2: " << fidpointvec[2].x() << " " << fidpointvec[2].y()
		 << " x3: " << fidpointvec[3].x() << " " << fidpointvec[3].y()
		 << std::endl;
	return oss.str();
}

void SurveyPxbDicer::doDice(const fidpoint_t &fidpointvec, const idPair_t &id, std::ofstream &outfile, const bool rotate)
{
	outfile << doDice(fidpointvec, id, rotate);
}

SurveyPxbDicer::coord_t SurveyPxbDicer::transform(const coord_t &x, const value_t &a0, const value_t &a1, const value_t &a2, const value_t &a3)
{
	return coord_t(a0+a2*x.x()+a3*x.y(), a1-a3*x.x()+a2*x.y());
}

SurveyPxbDicer::value_t SurveyPxbDicer::getParByName(const std::string &name, const std::string &par, const std::vector<edm::ParameterSet>& pars)
{
	std::vector<edm::ParameterSet>::const_iterator it;
	it = std::find_if(pars.begin(), pars.end(), [&name] (auto const& c){return findParByName()(name,c);});
	if (it == pars.end()) { throw std::runtime_error("Parameter not found in SurveyPxbDicer::getParByName"); }
	return (*it).getParameter<value_t>(par);
}

