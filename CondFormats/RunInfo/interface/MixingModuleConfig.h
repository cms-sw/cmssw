#ifndef MixingModuleConfig_H
#define MixingModuleConfig_H

#include <vector>
#include <string>
#include <iostream>

namespace edm{
  class ParameterSet;
}

class MixingInputConfig {
 public:
  MixingInputConfig();
  virtual ~MixingInputConfig(){};

  const int itype() const {return t_;}
  std::string type() const { 
    switch(t_){
    case 0:      return "none";
    case 1:      return "fixed";
    case 2:      return "poisson";
    case 3:      return "histo";
    case 4:      return "prodFunction";
    }
    return "";
  }
  int itype(std::string s)const {
    if (s=="none")      return 0;
    if (s=="fixed")      return 1;
    if (s=="poisson")      return 2;
    if (s=="histo")      return 3;
    if (s=="probFunction")      return 4;
    return 0;
  }

  const double averageNumber() const { return an_;}
  //  const int intAverage() const { return ia_;}
  const std::vector<int> & probFunctionVariable() const { return dpfv_;}
  const std::vector<double> & probValue() const { return dp_;}
  const int outOfTime() const { return moot_;}
  const int fixedOutOfTime() const { return ioot_;}

  void read(edm::ParameterSet & pset);

 private:
  int t_;
  double an_;
  //  int ia_;
  std::vector<int> dpfv_;
  std::vector<double>dp_;
  int moot_;
  int ioot_;
  
};

class MixingModuleConfig {
 public:
  MixingModuleConfig();
  virtual ~MixingModuleConfig(){};

  const MixingInputConfig & config (unsigned int i=0) const { return configs_[i];}
  
  const int & bunchSpace() const { return bs_;}
  const int & minBunch() const { return minb_;}
  const int & maxBunch() const { return maxb_;}

  void read(edm::ParameterSet & pset);
  
 private:
  std::vector<MixingInputConfig> configs_;
  
  int minb_;
  int maxb_;
  int bs_;
};



std::ostream& operator<< ( std::ostream&, const MixingModuleConfig & beam );
std::ostream& operator<< ( std::ostream&, const MixingInputConfig & beam );

#endif
