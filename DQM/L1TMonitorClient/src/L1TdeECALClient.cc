#include "DQM/L1TMonitorClient/interface/L1TdeECALClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TRandom.h"
#include <TF1.h>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <TProfile.h>
#include <TProfile2D.h>
using namespace edm;
using namespace std;

L1TdeECALClient::L1TdeECALClient(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

L1TdeECALClient::~L1TdeECALClient(){
 if(verbose_) cout <<"[TriggerDQM]: ending... " << endl;
}

//--------------------------------------------------------
void L1TdeECALClient::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
  
  // base folder for the contents of this job
  verbose_ = parameters_.getUntrackedParameter<bool>("verbose", false);
  
  monitorDir_ = parameters_.getUntrackedParameter<string>("monitorDir","");
  if(verbose_) cout << "Monitor dir = " << monitorDir_ << endl;
    
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  if(verbose_) cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  if(verbose_) cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
  

      
}

//--------------------------------------------------------
void L1TdeECALClient::beginJob(const EventSetup& context){

  if(verbose_) cout <<"[TriggerDQM]: Begin Job" << endl;
  // get backendinterface  
  dbe_ = Service<DQMStore>().operator->();

  // do your thing
  dbe_->setCurrentFolder(monitorDir_);
  
     ecalEtMapDiff1D_proj = dbe_->book1D("ecalEtMapDiff1D_proj","ecalEtMapDiff1D_proj",2520,0,2520.);


//   bad channels from QTs results
     ecalEtMapDiff1D_proj_badChs = dbe_->book1D("ecalEtMapDiff1D_proj_badChs","ecalEtMapDiff1D_proj_badChs",2520,0,2520.);
     ecalEtMapDiff_badChs = dbe_->book2D("ecalEtMapDiff2D_badChs","ecalEtMapDiff2D_badChs",35, -17.5, 17.5,72, -10., 350.);

}

//--------------------------------------------------------
void L1TdeECALClient::beginRun(const Run& r, const EventSetup& context) {
}

//--------------------------------------------------------
void L1TdeECALClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void L1TdeECALClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c){

// retrieve all MEs in current dir
   vector<string> meVec = dbe_->getMEs();
   if(verbose_) cout << "meVec size = " << meVec.size() << endl;
   string currDir = dbe_->pwd();
   if(verbose_) cout << "currDir = " <<  currDir << endl;
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
     string full_path = currDir + "/" + (*it);
     if(verbose_) cout << "full path = " << full_path <<  endl;
     MonitorElement * me =dbe_->get(full_path);
     float me_entries=me->getEntries();

// for this MEs, get list of associated QTs

     std::vector<QReport *> Qtest_map = me->getQReports();
   
       if (Qtest_map.size() > 0) {
          for (std::vector<QReport *>::const_iterator it = Qtest_map.begin(); it != Qtest_map.end(); it++) {
             cout << endl;
             string qt_name = (*it)->getQRName();
             int qt_status = (*it)->getStatus();
             
             switch(qt_status){
               case dqm::qstatus::WARNING:
	       if(verbose_) cout << "****** QT name: " << qt_name << "; Status: WARNING; "<< " Message: " << (*it)->getMessage() <<endl;
	       break;
               
	       case dqm::qstatus::ERROR:
	       if(verbose_) cout << "****** QT name: " << qt_name << "; Status: ERROR; "<< " Message: " << (*it)->getMessage() <<endl;
	       break;
       
               case dqm::qstatus::DISABLED:
	       if(verbose_) cout << "****** QT name: " << qt_name << "; Status: DISABLED; "<< " Message: " << (*it)->getMessage() <<endl;
	       break;
       
               case dqm::qstatus::INVALID:
	       if(verbose_) cout << "****** QT name: " << qt_name << "; Status: INVALID; "<< " Message: " << (*it)->getMessage() <<endl;
	       break;
       
               case dqm::qstatus::INSUF_STAT:
	       if(verbose_) cout << "****** QT name: " << qt_name << "; Status: NOT ENOUGH STATISTICS; "<< " Message: " <<(*it)->getMessage() <<endl;
	       if(qt_status == dqm::qstatus::INSUF_STAT) cout <<  " entries = " << me_entries << endl;
	       break;
       
               default:
	       if(verbose_) cout << "****** Unknown QTest qith status="<<qt_status<< endl;
             }
       
//   get bad channel list

	          std::vector<dqm::me_util::Channel> badChannels=(*it)->getBadChannels();																		 
	          if(!badChannels.empty() && verbose_ ) cout << " Number of channels that failed test " <<qt_name <<  " = " << badChannels.size()<< "\n";																					 
	          vector<dqm::me_util::Channel>::iterator badchsit = badChannels.begin();  																		 
	          
		  while(badchsit != badChannels.end())																							 
	          {				
		    int ix = (*badchsit).getBinX();	
		    int iy = (*badchsit).getBinY();																					 
		   if(verbose_) cout <<" Bad channel ("<< ix<<"," << iy << ") with contents "<<(*badchsit).getContents() << endl;
		   if(qt_name=="testdeDiffInYRange") ecalEtMapDiff1D_proj_badChs->setBinContent(ix,(*badchsit).getContents());
		   if(qt_name=="testdeDiffInRange2DProfile") ecalEtMapDiff_badChs->setBinContent(ix,iy,(*badchsit).getContents());
		   ++badchsit;
	          }
	 
       }
     }      
       
   }

}

//--------------------------------------------------------
void L1TdeECALClient::analyze(const Event& e, const EventSetup& context){
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;

   if(verbose_) cout << "L1TdeECALClient::analyze" << endl;

// Example: get ROOT object 
    TProfile2D * ecalEtMapDiffRoot_;   
    ecalEtMapDiffRoot_ = this->get2DProfile("L1TEMU/xpert/Ecal/EcalEtMapDiff",dbe_);


    if (ecalEtMapDiffRoot_) {
     int lastBinX=(*ecalEtMapDiffRoot_).GetNbinsX();
     int lastBinY=(*ecalEtMapDiffRoot_).GetNbinsY();
     for(int i=0; i<lastBinX ; i++){   
         for(int j=0; j<lastBinY ; j++){  
	   int ibin=lastBinY*(i%lastBinX)+j; 
	   if(ecalEtMapDiffRoot_->GetBinContent(i,j))
           ecalEtMapDiff1D_proj->setBinContent(ibin,ecalEtMapDiffRoot_->GetBinContent(i,j));
         }
     }
    
    }
    
   
}

//--------------------------------------------------------
void L1TdeECALClient::endRun(const Run& r, const EventSetup& context){
}

//--------------------------------------------------------
void L1TdeECALClient::endJob(){
}



TH1F * L1TdeECALClient::get1DHisto(string meName, DQMStore * dbi)
{

  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH1F();
}

TH2F * L1TdeECALClient::get2DHisto(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTH2F();
}



TProfile2D *  L1TdeECALClient::get2DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
     if(verbose_) cout << "ME NOT FOUND." << endl;
   return NULL;
  }

  return me_->getTProfile2D();
}


TProfile *  L1TdeECALClient::get1DProfile(string meName, DQMStore * dbi)
{


  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    if(verbose_) cout << "ME NOT FOUND." << endl;
    return NULL;
  }

  return me_->getTProfile();
}








