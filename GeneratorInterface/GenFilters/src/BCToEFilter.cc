#include "GeneratorInterface/GenFilters/interface/BCToEFilter.h"

using namespace edm;
using namespace std;


BCToEFilter::BCToEFilter(const edm::ParameterSet& iConfig) { 
  
  ParameterSet filterPSet=iConfig.getParameter<edm::ParameterSet>("filterAlgoPSet");
  
  BCToEAlgo_=new BCToEFilterAlgo(filterPSet);

}

BCToEFilter::~BCToEFilter() {
}


bool BCToEFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

  
  bool result=BCToEAlgo_->filter(iEvent);

  return result;

}


