#ifndef HcalCalibObjects_HBHEDarkening_h
#define HcalCalibObjects_HBHEDarkening_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>
#include <map>

// Scintillator darkening model for HB and HE
// ingredients:
// 1) dose map (Mrad/fb-1), from Fluka
// 2) decay constant D as function of dose rate d (Mrad vs krad/hr): D(d) = A*d^B
// 3) inst lumi per year (fb-1/hr)
// 4) int lumi per year (fb-1)
// layer number for HB: (0,16) = (1,17) in HcalTestNumbering
// layer number for HE: (-1,17) = (1,19) in HcalTestNumbering

class HBHEDarkening {
	public:
		HBHEDarkening(const edm::ParameterSet & p);
		~HBHEDarkening() {}
		
		//public accessor
		double degradation(double intlumi, int ieta, int lay) const;
		
		//helper function
		static std::vector<std::vector<double>> readDoseMap(const std::string& fullpath);

		//helper classes
		struct LumiYear {
			//constructors
			LumiYear() : 
				year(""), intlumi(0.), lumirate(0.), energy(0), sumlumi(0.) {}
			LumiYear(std::string year_, double intlumi_, double lumirate_, int energy_) : 
				year(year_), intlumi(intlumi_), lumirate(lumirate_), energy(energy_), sumlumi(0.) {}
			LumiYear(const edm::ParameterSet & p) : 
				year(p.getParameter<std::string>("year")), intlumi(p.getParameter<double>("intlumi")), lumirate(p.getParameter<double>("lumirate")), energy(p.getParameter<int>("energy")), sumlumi(0.) {}
			
			//sorting
			bool operator<(const LumiYear& yr) const {
				return year < yr.year;
			}
			
			//member variables
			std::string year;
			double intlumi;
			double lumirate;
			int energy;
			double sumlumi;
		};
		struct LumiYearComp {
			bool operator()(const LumiYear& yr, const double& lum) const {
				return yr.sumlumi < lum;
			}
		};

	private:
		//helper functions
		double dose(int ieta, int lay, int energy) const;
		std::string getYearForLumi(double intlumi) const;
		double degradationYear(const LumiYear& year, double intlumi, int ieta, int lay) const;
	
		//member variables
		int ieta_shift;
		double drdA, drdB;
		std::map<int,std::vector<std::vector<double>>> dosemaps; //one map for each center of mass energy
		std::vector<LumiYear> years;
};

#endif // HBHEDarkening_h
