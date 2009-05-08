#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
#include "LumiDummyReader.h"
#include "CondTools/RunInfo/interface/LumiReaderFactory.h"
#include <iostream>
/**fill lumi info for n consecutive runs. The last one is valid forever. If a run is missing in the source, implicit NULL values -99 are filled everywhere. The version of the data is identified by a lumi version number in the source database. 
 **/
void
lumi::LumiDummyReader::fill(int startRun, 
			    int numberOfRuns, 
    std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >& result,
    short lumiVersionNumber){
  //
  //fake 10 runs with 30 lumisection each with 3564 bunchcrossing, 
  //run 5 is missing, writing one lumi block with null data as placeholder
  //
  short lumiversion=lumiVersionNumber;
  std::cout<<"lumiversion "<<lumiversion<<std::endl;
  for(int i=startRun; i<=startRun+numberOfRuns; ++i){
    for(int j=1; j<30; ++j){
      edm::LuminosityBlockID lu(i,j);
      if(i==5){
	edm::LuminosityBlockID lu5(5,1);
	cond::Time_t current=(cond::Time_t)(lu5.value());
	lumi::LuminosityInfo* l5=new lumi::LuminosityInfo;
	l5->setLumiNull();
	std::cout<<"current "<<current<<std::endl;
	result.push_back(std::make_pair<lumi::LuminosityInfo*,cond::Time_t>(l5,current));
      }else{
	cond::Time_t current=(cond::Time_t)(lu.value());
	lumi::LuminosityInfo* l=new lumi::LuminosityInfo;
	l->setLumiVersionNumber(1);
	l->setLumiSectionId(j);
	l->setDeadtimeNormalization(0.5);
	lumi::LumiAverage avg(1.1,0.2,1,5);
	l->setLumiAverage(avg,lumi::ET);
	l->setLumiAverage(avg,lumi::OCCD1);
	l->setLumiAverage(avg,lumi::OCCD2);
	std::vector<lumi::BunchCrossingInfo> bxinfo;
	bxinfo.reserve(3564);
	for(int bxidx=1;bxidx<=3564;++bxidx){
	bxinfo.push_back(lumi::BunchCrossingInfo(bxidx,2.1,0.6,3,4));
	}
	l->setBunchCrossingData(bxinfo,lumi::ET);
	l->setBunchCrossingData(bxinfo,lumi::OCCD1);
	l->setBunchCrossingData(bxinfo,lumi::OCCD2);
	std::cout<<"current "<<current<<std::endl;
	result.push_back(std::make_pair<lumi::LuminosityInfo*,cond::Time_t>(l,current));
      }
  }
  std::cout<<"result size "<<result.size()<<std::endl;
  }
}

DEFINE_EDM_PLUGIN(lumi::LumiReaderFactory,lumi::LumiDummyReader,"dummy");
 
