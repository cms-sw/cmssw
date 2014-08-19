#include "GeneratorInterface/GenFilters/interface/HeavyQuarkFromMPIFilter.h"

using namespace edm;
using namespace std;


HeavyQuarkFromMPIFilter::HeavyQuarkFromMPIFilter(const edm::ParameterSet& iConfig) { 
  
  ParameterSet filterPSet=iConfig.getParameter<edm::ParameterSet>("filterAlgoPSet");
  
  HeavyQuarkFromMPIFilterAlgo_=new HeavyQuarkFromMPIFilterAlgo(filterPSet);

}

HeavyQuarkFromMPIFilter::~HeavyQuarkFromMPIFilter() {
}


bool HeavyQuarkFromMPIFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

  
  bool result=HeavyQuarkFromMPIFilterAlgo_->filter(iEvent);

  return result;

}


