#include "DQMOffline/Trigger/interface/EgHLTOfflineSummaryClient.h"


#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"



#include <boost/algorithm/string.hpp>

EgHLTOfflineSummaryClient::EgHLTOfflineSummaryClient(const edm::ParameterSet& iConfig):
  egHLTSumHistName_("egHLTTrigSum")
{  
  dirName_=iConfig.getParameter<std::string>("DQMDirName");
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("EgHLTOfflineSummaryClient") << "unable to get DQMStore service, no summary histograms will be produced";
  }else{
    if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
      dbe_->setVerbose(0);
    } 
    dbe_->setCurrentFolder(dirName_);
  }
 
  eleHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames"); 
  phoHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("phoHLTFilterNames");
  getEgHLTFiltersToMon_(egHLTFiltersToMon_);


  //std::vector<std::string> egHLTSumQTests = iConfig.getParameter<std::vector<std::string> >("egHLTSumQTests");
  // splitStringsToPairs_(egHLTSumQTests,egHLTSumHistXBins_);

  std::vector<edm::ParameterSet> qTestData = iConfig.getParameter<std::vector<edm::ParameterSet> >("egHLTSumQTests");
  egHLTSumHistXBins_.resize(qTestData.size());
  for(size_t sumHistBinNr=0;sumHistBinNr<qTestData.size();sumHistBinNr++){
    egHLTSumHistXBins_[sumHistBinNr].name = qTestData[sumHistBinNr].getParameter<std::string>("name");
    egHLTSumHistXBins_[sumHistBinNr].qTestPatterns = qTestData[sumHistBinNr].getParameter<std::vector<std::string> >("qTestsToCheck"); 
  }

  //egHLTSumHistXBins_.push_back(std::make_pair("Ele Rel Trig Eff",&EgHLTOfflineSummaryClient::eleTrigRelEffQTestResult_));
  //egHLTSumHistXBins_.push_back(std::make_pair("Pho Rel Trig Eff",&EgHLTOfflineSummaryClient::phoTrigRelEffQTestResult_));
  //egHLTSumHistXBins_.push_back(std::make_pair("Ele T&P Trig Eff",&EgHLTOfflineSummaryClient::eleTrigTPEffQTestResult_));
  //egHLTSumHistXBins_.push_back(std::make_pair("Triggered Ele",&EgHLTOfflineSummaryClient::trigEleQTestResult_));
  //egHLTSumHistXBins_.push_back(std::make_pair("Triggered Pho",&EgHLTOfflineSummaryClient::trigPhoQTestResult_)); 
}


EgHLTOfflineSummaryClient::~EgHLTOfflineSummaryClient()
{ 
  
}

void EgHLTOfflineSummaryClient::beginJob(const edm::EventSetup& iSetup)
{
 

}

void EgHLTOfflineSummaryClient::endJob() 
{

}

void EgHLTOfflineSummaryClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
 
}


void EgHLTOfflineSummaryClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  runClient_();
}

//dummy analysis function
void EgHLTOfflineSummaryClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  
}

void EgHLTOfflineSummaryClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 
  runClient_();
}

void EgHLTOfflineSummaryClient::runClient_()
{
  MonitorElement* egHLTSumME = getEgHLTSumHist_();

  for(size_t filterNr=0;filterNr<egHLTFiltersToMon_.size();filterNr++){
    for(size_t xBinNr=0;xBinNr<egHLTSumHistXBins_.size();xBinNr++){
      //egHLTSumHist->setBinContent(xBinNr+1,filterNr+1,(*egHLTSumHistXBins_[xBinNr].second)(egHLTFiltersToMon_[filterNr].c_str()));
      egHLTSumME->setBinContent(xBinNr+1,filterNr+1,
				getQTestResults_(egHLTFiltersToMon_[filterNr],egHLTSumHistXBins_[xBinNr].qTestPatterns)); 
    }
  }
   
}
void EgHLTOfflineSummaryClient::splitStringsToPairs_(const std::vector<std::string>& stringsToSplit,std::vector<std::pair<std::string,std::string> >& splitStrings)
{
  splitStrings.clear();
  splitStrings.reserve(stringsToSplit.size());
  for(size_t stringNr=0;stringNr<stringsToSplit.size();stringNr++){
    std::vector<std::string> tempSplitStrings;
    boost::split(tempSplitStrings,stringsToSplit[stringNr],boost::is_any_of(":"));
    if(tempSplitStrings.size()==2){
      splitStrings.push_back(std::make_pair(tempSplitStrings[0],tempSplitStrings[1]));
    }else{
      edm::LogWarning("EgHLTOfflineSummaryClient") <<" Error : entry "<<stringsToSplit[stringNr]<<" is not of form A:B, ignoring (ie this quailty test isnt being included in the sumamry hist) ";
    }
  }
}


MonitorElement* EgHLTOfflineSummaryClient::getEgHLTSumHist_()
{
  MonitorElement* egHLTSumHist = dbe_->get(dirName_+"/"+egHLTSumHistName_);
  if(egHLTSumHist==NULL){
    TH2F* hist = new TH2F(egHLTSumHistName_.c_str(),"E/g HLT Offline Summary",1,0.,1.,1,0.,1.);
    hist->SetBit(TH1::kCanRebin);
    for(size_t xBinNr=0;xBinNr<egHLTSumHistXBins_.size();xBinNr++){
      for(size_t yBinNr=0;yBinNr<egHLTFiltersToMon_.size();yBinNr++){
	hist->Fill(egHLTSumHistXBins_[xBinNr].name.c_str(),egHLTFiltersToMon_[yBinNr].c_str(),1.);
      }
    }
    hist->SetBit(TH1::kCanRebin,false);
    egHLTSumHist = dbe_->book2D(egHLTSumHistName_,hist);
  }
  return egHLTSumHist;

}

//this function puts every e/g trigger monitored in a std::vector
//this is *very* similar to EgHLTOfflineSource::getHLTFilterNamesUsed but 
//it differs in the fact it only gets the E/g primary triggers not the backups
//due to the design, to ensure we get every filter, filters will be inserted multiple times
//eg electron filters will contain photon triggers which are also in the photon filters
//but only want one copy in the vector
//this function is intended to be called once per job so some inefficiency can can be tolerated
//therefore we will use a std::set to ensure ensure that each filtername is only inserted once
//and then convert to a std::vector
void EgHLTOfflineSummaryClient::getEgHLTFiltersToMon_(std::vector<std::string>& filterNames)const
{ 
  std::set<std::string> filterNameSet;
  for(size_t i=0;i<eleHLTFilterNames_.size();i++) filterNameSet.insert(eleHLTFilterNames_[i]);
  for(size_t i=0;i<phoHLTFilterNames_.size();i++) filterNameSet.insert(phoHLTFilterNames_[i]);
 
  //right all the triggers are inserted once and only once in the set, convert to vector
  //very lazy, create a new vector so can use the constructor and then use swap to transfer
  std::vector<std::string>(filterNameSet.begin(),filterNameSet.end()).swap(filterNames);
}

//only returns 0 or 1, 0 is bad, one is good and if the test is not found defaults to good
//(this is because its a dumb algorithm, photon tests are run for electron triggers which unsurprisingly are not found)
int EgHLTOfflineSummaryClient::getQTestResults_(const std::string& filterName,const std::vector<std::string>& patterns)const
{
  int nrFail =0;
  for(size_t patternNr=0;patternNr<patterns.size();patternNr++){
    std::vector<MonitorElement*> monElems = dbe_->getMatchingContents(dirName_+"/"+filterName+patterns[patternNr]);
    
    for(size_t monElemNr=0;monElemNr<monElems.size();monElemNr++){
      if(monElems[monElemNr]->hasError()) nrFail++;
    }
  }
  if(nrFail==0) return 1;
  else return 0;
}

// int EgHLTOfflineSummaryClient::eleTrigRelEffQTestResult_(const std::string& filterName)const
// {
 

// }

// int EgHLTOfflineSummaryClient::phoTrigRelEffQTestResult_(const std::string& filterName)const
// {
  
// }

// int EgHLTOfflineSummaryClient::eleTrigTPEffQTestResult_(const std::string& filterName)const
// {
  
// }

// int EgHLTOfflineSummaryClient::trigEleQTestResult_(const std::string& filterName)const
// {
  
// }

// int EgHLTOfflineSummaryClient::trigPhoQTestResult_(const std::string& filterName)const
// {
  
// }
