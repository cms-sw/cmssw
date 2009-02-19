#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/RunInfo/interface/HLTScaler.h"
#include "HLTScalerDummyReader.h"
#include "CondTools/RunInfo/interface/HLTScalerReaderFactory.h"
#include <iostream>
void
lumi::HLTScalerDummyReader::fill(int startRun, 
			    int numberOfRuns, 
       std::vector< std::pair<lumi::HLTScaler*,cond::Time_t> >& result){
  //fake 10 runs with 30 lumisection each with 3564 bunchcrossing,100 hlt trigger
  for(int i=startRun; i<=startRun+numberOfRuns; ++i){
    for(int j=1; j<30; ++j){
      edm::LuminosityBlockID lu(i,j);
      cond::Time_t current=(cond::Time_t)(lu.value());
      std::vector<lumi::HLTInfo> hltdata;
      for(int h=0; h<100; ++h){
	lumi::HLTInfo hltinfo(12+h,10+h,2+h);
	hltdata.push_back(hltinfo);
      }
      lumi::HLTScaler* l=new lumi::HLTScaler;
      l->setHLTData(hltdata);
      std::cout<<"current "<<current<<std::endl;
      result.push_back(std::make_pair<lumi::HLTScaler*,cond::Time_t>(l,current));
    }
  }
  std::cout<<"result size "<<result.size()<<std::endl;
}
DEFINE_EDM_PLUGIN(lumi::HLTScalerReaderFactory,lumi::HLTScalerDummyReader,"dummy");
