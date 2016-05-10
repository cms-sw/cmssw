// -*- C++ -*-
//
// Package:   Alignment/CommonAlignment
// Class:     FilterOutLowPt
//
//
// Original Author:  Marco Musich

#include <memory>
#include <vector>
#include <map>
#include <set>
#include <utility>      // std::pair
#include "TMath.h"

// user include files

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class FilterOutLowPt : public edm::stream::EDFilter<> {
public:
  explicit FilterOutLowPt( const edm::ParameterSet & );
  ~FilterOutLowPt();
  
private:
  virtual void beginJob() ;
  virtual bool filter ( edm::Event &, const edm::EventSetup&); 
  virtual void endJob() ;

  bool applyfilter;
  bool debugOn;
  double thresh;
  unsigned int numtrack;
  double  ptmin;
  edm::InputTag tracks_;
  double trials;
  double passes;
  bool runControl_;
  std::vector<unsigned int> runControlNumbers_;
  std::map<unsigned int,std::pair<int,int>> eventsInRun_;

  reco::TrackBase::TrackQuality _trackQuality;
  edm::EDGetTokenT<reco::TrackCollection>  theTrackCollectionToken; 

};


FilterOutLowPt::FilterOutLowPt(const edm::ParameterSet& iConfig)
{
  std::vector<unsigned int> defaultRuns;
  defaultRuns.push_back(0);

  applyfilter = iConfig.getUntrackedParameter<bool>("applyfilter",true);
  debugOn     = iConfig.getUntrackedParameter<bool>("debugOn",false);
  thresh      = iConfig.getUntrackedParameter<int>("thresh",1);
  numtrack    = iConfig.getUntrackedParameter<unsigned int>("numtrack",0);
  ptmin       = iConfig.getUntrackedParameter<double>("ptmin",3);
  runControl_ = iConfig.getUntrackedParameter<bool>("runControl",false);
  runControlNumbers_ = iConfig.getUntrackedParameter<std::vector<unsigned int> >("runControlNumber",defaultRuns);

  edm::InputTag TrackCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("src",edm::InputTag("generalTracks"));
  theTrackCollectionToken = consumes<reco::TrackCollection>(TrackCollectionTag_);
  
}

FilterOutLowPt::~FilterOutLowPt()
{
}

void FilterOutLowPt::beginJob(){
  trials=0;
  passes=0;
}

bool FilterOutLowPt::filter( edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  bool passesRunControl = false;
  
  if(runControl_){
    if (debugOn){
      for(unsigned int i=0;i<runControlNumbers_.size();i++){
	edm::LogInfo("FilterOutLowPt")<<"run number:" <<iEvent.id().run()<<" keeping runs:"<<runControlNumbers_[i]<<std::endl;
      }
    }

    for(unsigned int j=0;j<runControlNumbers_.size();j++){
      if(iEvent.eventAuxiliary().run() == runControlNumbers_[j]){ 
	if (debugOn){
	  edm::LogInfo("FilterOutLowPt")<<"run number"<< runControlNumbers_[j] << " match!"<<std::endl;
	}
	passesRunControl = true;
	break;
      }
    }
    if (!passesRunControl) return false;
  }
  
  trials++;

  bool accepted = false;
  float fraction = 0;  
  // get GeneralTracks collection

  edm::Handle<reco::TrackCollection> tkRef;
  iEvent.getByToken(theTrackCollectionToken,tkRef);    
  const reco::TrackCollection* tkColl = tkRef.product();
 
  int numhighpurity=0;
  _trackQuality = reco::TrackBase::qualityByName("highPurity");

  if(tkColl->size()>numtrack){ 
    reco::TrackCollection::const_iterator itk = tkColl->begin();
    reco::TrackCollection::const_iterator itk_e = tkColl->end();
    for(;itk!=itk_e;++itk){
      if( itk->quality(_trackQuality) &&
	  (itk->pt() >= ptmin) 
	  ) numhighpurity++;
    }
    fraction = numhighpurity; //(float)tkColl->size();
    if(fraction>=thresh) accepted=true;
  }
    
  if (debugOn) {
    int ievt = iEvent.id().event();
    int irun = iEvent.id().run();
    int ils  = iEvent.luminosityBlock();
    int bx   = iEvent.bunchCrossing();
      
    edm::LogInfo("FilterOutLowPt")<<" Run " << irun << " Event " << ievt << " Lumi Block " << ils << " Bunch Crossing " << bx << " Fraction " << fraction << " NTracks " << tkColl->size() << " Accepted " << accepted << std::endl;

  }
 
  // count the trials and passes
  unsigned int iRun = iEvent.id().run(); 
  if (eventsInRun_.count(iRun)>0){
    eventsInRun_[iRun].first+=1;
    if(accepted) eventsInRun_[iRun].second+=1;
  } else {
    std::pair<int,int> mypass = std::make_pair(1,0);
    if(accepted) mypass.second = 1;
    eventsInRun_[iRun]= mypass;
  }

  if (applyfilter){
    if(accepted) passes++;
    return accepted;
  } else
    return true;
  
}

void FilterOutLowPt::endJob(){
  
  double eff =  passes/trials;
  double eff_err = TMath::Sqrt((eff*(1-eff))/trials);

  edm::LogVerbatim("FilterOutLowPt")
    <<"###################################### \n"			  
    <<"# FilterOutLowPt::endJob() report \n"			  
    <<"# Number of analyzed events: "<<trials<<" \n"			  
    <<"# Number of accpeted events: "<<passes<<" \n"			  
    <<"# Efficiency: "<< eff*100 << " +/- " << eff_err*100 << " %\n"  
    <<"######################################"; 
                    
  edm::LogVerbatim("FilterOutLowPt")<<"###################################### \n"
				    <<"# Filter Summary events accepted by run";
  for (std::map<unsigned int,std::pair<int,int> >::iterator it=eventsInRun_.begin(); it!=eventsInRun_.end(); ++it)			       
    edm::LogVerbatim("FilterOutLowPt")<<"# run:" << it->first << " => events tested: " << (it->second).first << " | events passed: " << (it->second).second;  
  edm::LogVerbatim("FilterOutLowPt")<<"###################################### \n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(FilterOutLowPt);
