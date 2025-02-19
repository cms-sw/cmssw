#include "CondFormats/RunInfo/interface/MixingModuleConfig.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <sstream>

MixingModuleConfig::MixingModuleConfig(){
  configs_.resize(4);
}
MixingInputConfig::MixingInputConfig(){}

std::ostream& operator<< ( std::ostream& os, const MixingModuleConfig & c) {
  std::stringstream ss;
  os <<c.bunchSpace()<<"\n"<<c.config();
  return os;
}
std::ostream& operator<< ( std::ostream& os, const MixingInputConfig & c) {
  std::stringstream ss;
  os <<c.type();
  return os;
}

void MixingModuleConfig::read(edm::ParameterSet & pset){
  
  bs_=pset.getParameter<int>("bunchspace");
  minb_=(pset.getParameter<int>("minBunch")*25)/pset.getParameter<int>("bunchspace");
  maxb_=(pset.getParameter<int>("maxBunch")*25)/pset.getParameter<int>("bunchspace");

  //FIXME. not covering all possible cases (not used anyways)
  edm::ParameterSet p0=pset.getParameter<edm::ParameterSet>("input");
  configs_[0].read(p0);
}


void MixingInputConfig::read(edm::ParameterSet & pset){
  t_=itype(pset.getParameter<std::string>("type"));
  an_=0;
  //  ia_=0;
  dpfv_.clear();
  dp_.clear();
  moot_=0;
  ioot_=0;

  switch(t_){
  case 1:
    an_=pset.getParameter<double>("averageNumber");
    break;
  case 2:
    an_=pset.getParameter<double>("averageNumber");
    break;
  case 3:
    //not supposed to be valid
  case 4:
    dpfv_=pset.getParameter<edm::ParameterSet>("nbPileupEvents").getParameter<std::vector<int> >("probFunctionVariable");
    dp_=pset.getParameter<edm::ParameterSet>("nbPileupEvents").getParameter<std::vector<double> >("probValue");
    break;
  }

  if (pset.getUntrackedParameter<bool>("manage_OOT"))
    {
      std::string OOT_type = pset.getUntrackedParameter<std::string>("OOT_type");

      if(OOT_type == "Poisson" || OOT_type == "poisson")
	moot_=2;
      else if (OOT_type == "Fixed" || OOT_type == "fixed") {
	moot_=1;
	ioot_=pset.getUntrackedParameter<int>("intFixed_OOT", -1);
      }
    }
}
