#ifndef CondFormats_HcalObjects_HBHEDarkening_h
#define CondFormats_HcalObjects_HBHEDarkening_h

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
  //helper classes
  struct LumiYear {
    //constructors
    LumiYear() : year_(""), intlumi_(0.), lumirate_(0.), energy_(0), sumlumi_(0.) {}
    LumiYear(std::string year, float intlumi, float lumirate, int energy)
        : year_(year), intlumi_(intlumi), lumirate_(lumirate), energy_(energy), sumlumi_(0.) {}

    //sorting
    bool operator<(const LumiYear& yr) const { return year_ < yr.year_; }

    //member variables
    std::string year_;
    float intlumi_;
    float lumirate_;
    int energy_;
    float sumlumi_;
  };
  struct LumiYearComp {
    bool operator()(const LumiYear& yr, const float& lum) const { return yr.sumlumi_ < lum; }
  };

  HBHEDarkening(int ieta_shift,
                float drdA,
                float drdB,
                std::map<int, std::vector<std::vector<float>>> dosemaps,
                std::vector<LumiYear> years);
  ~HBHEDarkening() {}

  //public accessors
  float degradation(float intlumi, int ieta, int lay) const;
  int get_ieta_shift() const { return ieta_shift_; }

  //helper function
  static std::vector<std::vector<float>> readDoseMap(const std::string& fullpath);

private:
  //helper functions
  float dose(int ieta, int lay, int energy) const;
  std::string getYearForLumi(float intlumi) const;
  float degradationYear(const LumiYear& year, float intlumi, int ieta, int lay) const;

  //member variables
  int ieta_shift_;
  float drdA_, drdB_;
  std::map<int, std::vector<std::vector<float>>> dosemaps_;  //one map for each center of mass energy
  std::vector<LumiYear> years_;
};

#endif  // HBHEDarkening_h
