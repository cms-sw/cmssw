#include "GeneratorInterface/GenFilters/interface/HighETPhotonsFilter.h"

using namespace edm;
using namespace std;


HighETPhotonsFilter::HighETPhotonsFilter(const edm::ParameterSet& iConfig) { 
  
  ParameterSet filterPSet=iConfig.getParameter<edm::ParameterSet>("filterAlgoPSet");
  
  HighETPhotonsAlgo_=new HighETPhotonsFilterAlgo(filterPSet);

}

HighETPhotonsFilter::~HighETPhotonsFilter() {
}


bool HighETPhotonsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

  
  bool result=HighETPhotonsAlgo_->filter(iEvent);

  return result;

}


