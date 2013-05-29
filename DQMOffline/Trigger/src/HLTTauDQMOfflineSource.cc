#include "DQMOffline/Trigger/interface/HLTTauDQMOfflineSource.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
using namespace trigger;

//
// constructors and destructor
//
HLTTauDQMOfflineSource::HLTTauDQMOfflineSource( const edm::ParameterSet& ps ) :counterEvt_(0)
{
    //Get General Monitoring Parameters
    config_                 = ps.getParameter<std::vector<edm::ParameterSet> >("MonitorSetup");
    doRefAnalysis_          = ps.getParameter<bool>("doMatching");
    NPtBins_                = ps.getUntrackedParameter<int>("PtHistoBins",20);
    NEtaBins_               = ps.getUntrackedParameter<int>("EtaHistoBins",25);
    NPhiBins_               = ps.getUntrackedParameter<int>("PhiHistoBins",32);
    EtMax_                  = ps.getUntrackedParameter<double>("EtHistoMax",100);
    L1MatchDr_              = ps.getUntrackedParameter<double>("L1MatchDeltaR",0.5);
    HLTMatchDr_             = ps.getUntrackedParameter<double>("HLTMatchDeltaR",0.3);
    
    refObjects_             = ps.getUntrackedParameter<std::vector<edm::InputTag> >("refObjects");
    
    prescaleEvt_            = ps.getUntrackedParameter<int>("prescaleEvt", -1);
    
    //Read The Configuration
    for (unsigned int i=0;i<config_.size();++i) {
        if (config_[i].getUntrackedParameter<std::string>("ConfigType") == "L1") {
            HLTTauDQML1Plotter tmp(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,L1MatchDr_);
            l1Plotters.push_back(tmp);
        }
        
        else if (config_[i].getUntrackedParameter<std::string>("ConfigType") == "Calo") {
            HLTTauDQMCaloPlotter tmp(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_);
            caloPlotters.push_back(tmp);
        }
        
        else if (config_[i].getUntrackedParameter<std::string>("ConfigType") == "Track") {
            HLTTauDQMTrkPlotter tmp(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_);
            trackPlotters.push_back(tmp);
        }
        
        else if (config_[i].getUntrackedParameter<std::string>("ConfigType") == "Path") {
            HLTTauDQMPathPlotter tmp(config_[i],doRefAnalysis_);
            pathPlotters.push_back(tmp);
        }
        
        else if (config_[i].getUntrackedParameter<std::string>("ConfigType") == "LitePath") {
            HLTTauDQMLitePathPlotter tmp(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_);
            litePathPlotters.push_back(tmp);
        }
    }
}

HLTTauDQMOfflineSource::~HLTTauDQMOfflineSource()
{
    
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
    
}


//--------------------------------------------------------
void 
HLTTauDQMOfflineSource::beginJob(){
    
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::beginRun(const edm::Run& r, const EventSetup& iSetup) {
    
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                                  const EventSetup& context) {
    
}

// ----------------------------------------------------------
void 
HLTTauDQMOfflineSource::analyze(const Event& iEvent, const EventSetup& iSetup )
{  
    //Apply the prescaler
    if(counterEvt_ > prescaleEvt_)
    {
        //Do Analysis here
        
        //Create dummy Match Collections
        std::vector<LVColl> refC;
        
        if(doRefAnalysis_)
        {
            for(unsigned int i=0;i<refObjects_.size();++i)
            {
                Handle<LVColl> collHandle;
                if(iEvent.getByLabel(refObjects_[i],collHandle))
                {
                    refC.push_back(*collHandle);
                }
            }
        }
        
        
        
        //fill the empty slots with empty collections
        LVColl dummy;
        for(int k=refC.size();k<3;k++)
        {
            refC.push_back(dummy);
        }
        
        
        
        //Path Plotters
        for(unsigned int i=0;i<pathPlotters.size();++i)
            pathPlotters[i].analyze(iEvent,iSetup,refC);
        
        //Lite Path Plotters
        for(unsigned int i=0;i<litePathPlotters.size();++i)
            litePathPlotters[i].analyze(iEvent,iSetup,refC);
        
        //L1  Plotters
        for(unsigned int i=0;i<l1Plotters.size();++i)
            l1Plotters[i].analyze(iEvent,iSetup,refC);
        
        //Calo Plotters
        for(unsigned int i=0;i<caloPlotters.size();++i)
            caloPlotters[i].analyze(iEvent,iSetup,refC[0]);
        
        //Track Plotters
        for(unsigned int i=0;i<trackPlotters.size();++i)
            trackPlotters[i].analyze(iEvent,iSetup,refC[0]);
    }
    else
        counterEvt_++;
    
}




//--------------------------------------------------------
void HLTTauDQMOfflineSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                                const EventSetup& context) {
}
//--------------------------------------------------------
void HLTTauDQMOfflineSource::endRun(const Run& r, const EventSetup& context){
}
//--------------------------------------------------------
void HLTTauDQMOfflineSource::endJob(){
    return;
}



