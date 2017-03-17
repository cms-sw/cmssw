#include "DataFormats/HcalCalibObjects/interface/HBHEDarkening.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <cassert>

HBHEDarkening::HBHEDarkening(const edm::ParameterSet & p) : 
	ieta_shift(p.getParameter<int>("ieta_shift")), drdA(p.getParameter<double>("drdA")), drdB(p.getParameter<double>("drdB"))
{
	//initialize dose maps
	std::vector<edm::ParameterSet> p_dosemaps = p.getParameter<std::vector<edm::ParameterSet>>("dosemaps");
	for(const auto& p_dosemap : p_dosemaps){
		edm::FileInPath fp = p_dosemap.getParameter<edm::FileInPath>("file");
		int file_energy = p_dosemap.getParameter<int>("energy");
		dosemaps.emplace(file_energy,readDoseMap(fp.fullPath()));
	}
	
	//initialize years
	std::vector<edm::ParameterSet> p_years = p.getParameter<std::vector<edm::ParameterSet>>("years");
	years.reserve(p_years.size());
	for(const auto& p_year : p_years){
		years.emplace_back(p_year);
	}
	std::sort(years.begin(),years.end());
	//sum up int lumi
	double sumlumi = 0.0;
	for(auto& year : years){
		sumlumi += year.intlumi;
		year.sumlumi = sumlumi;
	}
}

std::vector<std::vector<double>> HBHEDarkening::readDoseMap(const std::string& fullpath) {
	std::ifstream infile(fullpath.c_str());
	if(!infile.is_open()){
		throw cms::Exception("FileNotFound") << "Unable to open '" << fullpath << "'" << std::endl;
	}
	std::string line;
	std::vector<std::vector<double>> result;
	while(getline(infile,line)){
		//space-separated
		std::stringstream linestream(line);
		std::vector<double> lineresult;
		double doseval;
		while(linestream >> doseval) lineresult.push_back(doseval);
		result.push_back(lineresult);
	}
	return result;
}

double HBHEDarkening::dose(int ieta, int lay, int energy) const {
	//existence check
	const auto dosemapIt = dosemaps.find(energy);
	if(dosemapIt == dosemaps.end()) return 0.0;

	//bounds check
	const auto& dosemap = dosemapIt->second;
	if(ieta<0 or ieta>=int(dosemap.size())) return 0.0;
	
	//bounds check
	const auto& doserow = dosemap[ieta];
	if(lay<0 or lay>=int(doserow.size())) return 0.0;
	
	return doserow[lay];
}

std::string HBHEDarkening::getYearForLumi(double intlumi) const {
	//compare based on sum lumi value
	auto lb = std::lower_bound(years.begin(),years.end(),intlumi,LumiYearComp());
	if(lb == years.end() or lb->sumlumi < intlumi) {
		throw cms::Exception("ValueError") << "HBHEDarkening: insufficient LHC run information provided to simulate " << intlumi << "/fb - check the python config" << std::endl;
	}
	return lb->year;
}

double HBHEDarkening::degradationYear(const LumiYear& year, double intlumi, int ieta, int lay) const {
	double doseToUse = dose(ieta,lay,year.energy);
	if(doseToUse==0.0) return 1.0;
	
	//apply dose rate dependence model to the provided year
	//get krad/hr from Mrad/fb-1 and fb-1/hr
	double decayConst = drdA*std::pow(1000*doseToUse*year.lumirate,drdB);
	
	//determine if this is a partial year
	double intlumiToUse = year.intlumi;
	if(intlumi < year.sumlumi) intlumiToUse = year.sumlumi - intlumi;
	
	//calculate degradation
	return std::exp(-(intlumiToUse*doseToUse)/decayConst);
}

double HBHEDarkening::degradation(double intlumi, int ieta, int lay) const {
	ieta = abs(ieta);
	//shift ieta tower index to act as array index
	ieta -= ieta_shift;
	//shift layer index by 1 to act as array index
	lay -= 1;
	
	//accumulate degradation over years
	double response = 1.0;
	std::string yearForLumi = getYearForLumi(intlumi);
	assert(yearForLumi.size());
	
	for(const auto& year : years){
		response *= degradationYear(year,intlumi,ieta,lay);
		if(year.year==yearForLumi) break;
	}
	
	return response;
}
