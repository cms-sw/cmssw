#ifndef RecoLuminosity_LumiProducer_LumiCalculator_h
#define RecoLuminosity_LumiProducer_LumiCalculator_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include <boost/regex.hpp>
//#include <cmath>
#include <iostream>

class LumiCalculator : public edm::EDAnalyzer{
public:
  
  explicit LumiCalculator(edm::ParameterSet const&);
  virtual ~LumiCalculator();

private:  
  virtual void beginJob(const edm::EventSetup& );
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, 
				  edm::EventSetup const& c);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void endJob();
  std::vector<std::string> splitpathstr(const std::string& strValue,const std::string separator);
  HLTConfigProvider hltConfig_;
  
};//end class

// -----------------------------------------------------------------

LumiCalculator::LumiCalculator(edm::ParameterSet const& iConfig){
}

// -----------------------------------------------------------------

LumiCalculator::~LumiCalculator(){
}

// -----------------------------------------------------------------

void LumiCalculator::analyze(edm::Event const& e,edm::EventSetup const&){
  
}

// -----------------------------------------------------------------

void LumiCalculator::beginJob(const edm::EventSetup& c){
  
}

// -----------------------------------------------------------------

void LumiCalculator::beginRun(const edm::Run& run, const edm::EventSetup& c){
  if(!hltConfig_.init("HLT")){
    throw cms::Exception("HLT process cannot be initialized");
  }
  hltConfig_.dump("processName");
  hltConfig_.dump("TableName");
  //hltConfig_.dump("Triggers");
  //hltConfig_.dump("Modules");
  
}

// -----------------------------------------------------------------
void LumiCalculator::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, 
					edm::EventSetup const& c){
}

// -----------------------------------------------------------------
void LumiCalculator::endRun(edm::Run const& run, edm::EventSetup const& c){
  unsigned int totaltrg=hltConfig_.size();
  for (unsigned int t=0;t<totaltrg;++t){
    std::string hltname(hltConfig_.triggerName(t));
    std::vector<std::string> numpathmodules=hltConfig_.moduleLabels(hltname);
    std::cout<<t<<" HLT path\t"<<hltname<<std::endl;
    std::vector<std::string>::iterator hltpathBeg=numpathmodules.begin();
    std::vector<std::string>::iterator hltpathEnd=numpathmodules.end();
    for(std::vector<std::string>::iterator numpathmodule = hltpathBeg;
	numpathmodule!=hltpathEnd; ++numpathmodule ) {
      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed"){
	edm::ParameterSet l1GTPSet=hltConfig_.modulePSet(*numpathmodule);
	std::string l1pathname=l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
	if(l1pathname.find("OR")!=std::string::npos){
	  std::cout<<"    L1Seeds(ORed)\t"<< l1pathname<< std::endl;
	  std::vector<std::string> seeds=splitpathstr(l1pathname," OR ");
	  for(std::vector<std::string>::iterator i=seeds.begin();i!=seeds.end();++i){
	    if(i->size()!=0)std::cout<<"\t\tseed: "<<*i<<std::endl;
	  }
	}else if (l1pathname.find("AND")!=std::string::npos){
	  std::cout<<"    L1Seeds(ANDed)\t"<< l1pathname<< std::endl;
	  std::vector<std::string> seeds=splitpathstr(l1pathname," AND ");
	  for(std::vector<std::string>::iterator i=seeds.begin();
	      i!=seeds.end();++i){
	    if(i->size()!=0)std::cout<<"\t\tseed: "<<*i<<std::endl;
	  }
	}else{
	  std::cout<<"    L1Seeds(ONE)\t"<< l1pathname<< std::endl;
	}
      }
    }
  }
}

// -----------------------------------------------------------------
void LumiCalculator::endJob(){
}

std::vector<std::string>
LumiCalculator::splitpathstr(const std::string& strValue,const std::string separator){
  std::vector<std::string> vecstrResult;
  boost::regex re(separator);
  boost::sregex_token_iterator p(strValue.begin(),strValue.end(),re,-1);
  boost::sregex_token_iterator end;
  while(p!=end){
    vecstrResult.push_back(*p++);
  }
  /**
  endpos = strValue.find_first_of(separator, startpos);
  while (endpos != std::string::npos){       
    vecstrResult.push_back(strValue.substr(startpos, endpos-startpos)); // add to vector
    startpos = endpos+separator.size(); //jump past sep
    endpos = strValue.find_first_of(separator, startpos); // find next
    if(endpos==std::string::npos){
      //lastone, so no 2nd param required to go to end of string
      vecstrResult.push_back(strValue.substr(startpos));
    }
  }
  **/
  return vecstrResult;
}

DEFINE_FWK_MODULE(LumiCalculator);
#endif
