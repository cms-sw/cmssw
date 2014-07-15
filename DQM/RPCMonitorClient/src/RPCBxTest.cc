#include <DQM/RPCMonitorClient/interface/RPCBxTest.h>
#include "DQM/RPCMonitorDigi/interface/utils.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// //Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
# include "TH1F.h"



RPCBxTest::RPCBxTest(const edm::ParameterSet& ps ){
  edm::LogVerbatim ("rpcbxtest") << "[RPCBxTest]: Constructor";
  
  prescaleFactor_ =  ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  
  //Nome della dir per gli istogrammi nuovi . Cominciare sempre con RPC/RecHits/
  globalFolder_ = ps.getUntrackedParameter<std::string>("RPCGlobalFolder", "RPC/RecHits/SummaryHistograms/");
  
  entriesCut_ = ps.getUntrackedParameter<int>("EntriesCut");
  rmsCut_ = ps.getUntrackedParameter<double>("RMSCut");
  distanceMean_ = ps.getUntrackedParameter<double>("DistanceFromZeroBx");

  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
}

RPCBxTest::~RPCBxTest(){ dbe_=0;}

void RPCBxTest::beginJob(DQMStore *  dbe){
  edm::LogVerbatim ("rpcbxtest") << "[RPCBxTest]: Begin job ";
  dbe_ = dbe;
}

//Qui puoi definitre gli istogrammi nuovi che vuoi riempire
void RPCBxTest::beginRun(const edm::Run& r, const edm::EventSetup& c){
  edm::LogVerbatim ("rpcbxtest") << "[RPCBxTest]: Begin run";
  
  MonitorElement* me;
  dbe_->setCurrentFolder(globalFolder_);

  std::stringstream histoName;

  histoName.str("");
  histoName<<"BX_Mean_Distribution_Barrel";  
  BXMeanBarrel = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 11, -5.5, 5.5);
  BXMeanBarrel->setAxisTitle("Bx",1);

  histoName.str("");
  histoName<<"BX_Mean_Distribution_EndcapP";  
  BXMeanEndcapP = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 11, -5.5, 5.5);

  histoName.str("");
  histoName<<"BX_Mean_Distribution_EndcapN";  
  BXMeanEndcapN = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 11, -5.5, 5.5);


  histoName.str("");
  histoName<<"BX_Entries_Distribution_Barrel";  
  BXEntriesBarrel = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 1000, -0.5, 999.5);

  histoName.str("");
  histoName<<"BX_Entries_Distribution_EndcapP";  
  BXEntriesEndcapP = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),1000, -0.5, 999.5); 

  histoName.str("");
  histoName<<"BX_Entries_Distribution_EndcapN";  
  BXEntriesEndcapN = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 1000, -0.5, 999.5);


  histoName.str("");
  histoName<<"BX_RMS_Distribution_Barrel"; 
  BXRmsBarrel = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 21, -0.1, 4.1);

  histoName.str("");
  histoName<<"BX_RMS_Distribution_EndcapP"; 
  BXRmsEndcapP = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 21, -0.1, 4.1);

  histoName.str("");
  histoName<<"BX_RMS_Distribution_EndcapN"; 
  BXRmsEndcapN = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 21, -0.1, 4.1);


  rpcdqm::utils rpcUtils;

  int limit = numberOfDisks_;
  if(numberOfDisks_ < 2) limit = 2;
  
  for (int w = -1 * limit; w<=limit;w++ ){//loop on wheels and disks
    if (w>-3 && w<3){//wheels
      histoName.str("");
      histoName<<"BX_Mean_Distribution_Wheel"<<w;     
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me ) {
	dbe_->removeElement(me->getName());
      }
      BXMeanWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 10, -0.5, 9.5);
  
      histoName.str("");
      histoName<<"BX_RMS_Distribution_Wheel"<<w;     
      me = 0;
      me = dbe_->get(globalFolder_ + histoName.str()) ;
      if ( 0!=me){
	dbe_->removeElement(me->getName());
      }
      BXRmsWheel[w+2] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 50, -0.5, 4.5);
    }//end loop on wheels

    if (w == 0 || w< (-1 * numberOfDisks_) || w > numberOfDisks_)continue;
    //Endcap
    int offset = numberOfDisks_;
    if (w>0) offset --; //used to skip case equale to zero
      
    histoName.str("");
    histoName<<"BX_Mean_Distribution_Disk"<<w;     
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me){
      dbe_->removeElement(me->getName());
    }
    BXMeanDisk[w+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),10, -0.5, 9.5);
    
   
    histoName.str("");
    histoName<<"BX_RMS_Distribution_Disk"<<w;    
    me = 0;
    me = dbe_->get(globalFolder_ + histoName.str()) ;
    if ( 0!=me){
      dbe_->removeElement(me->getName());
    }
    BXRmsDisk[w+offset] = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(), 50, -0.5, 4.5);
  }

}

void RPCBxTest::getMonitorElements(std::vector<MonitorElement *> & meVector, std::vector<RPCDetId> & detIdVector){

  //Qui prende gli isto del BX per roll definiti nel client
  //crea due vettore ordinati myBXMe_(istogrammi) e myDetIds_(rpcDetId)
 //Get  ME for each roll
 for (unsigned int i = 0 ; i<meVector.size(); i++){

   bool flag= false;
   
   DQMNet::TagList tagList;
   tagList = meVector[i]->getTags();
   DQMNet::TagList::iterator tagItr = tagList.begin();

   while (tagItr != tagList.end() && !flag ) {
     if((*tagItr) ==  rpcdqm::BX){ flag= true;}
     tagItr++;
   }
   
   if(flag){
     myBXMe_.push_back(meVector[i]);
     myDetIds_.push_back(detIdVector[i]);
   }
 }

}

void RPCBxTest::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context){} 

void RPCBxTest::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {}

void RPCBxTest::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) { }

void  RPCBxTest::endJob(void) {
  edm::LogVerbatim ("rpcbxtest") << "[RPCBxTest]: end job ";
}


//Loop sul vettore degli istogrammi (myBXMe_) e prendi le info
void  RPCBxTest::endRun(const edm::Run& r, const edm::EventSetup& c) {

  MonitorElement * myMe;
  RPCDetId detId;
  TH1F * myTH1F;

  MonitorElement * ENTRIES =NULL; 
  MonitorElement * MEAN =NULL;  
  MonitorElement * MEANRing =NULL;  
  MonitorElement * RMS =NULL; 
  MonitorElement * RMSRing =NULL;  

  for (unsigned int  i = 0 ; i<myBXMe_.size();i++){

    myMe = myBXMe_[i];
    detId = myDetIds_[i];

    //Prendi TH1F corrispondente al Monitor Element
    myTH1F = myMe->getTH1F();
   //  //Spegni Overflow
//     myTH1F->StatOverflows(false); // per accendere overflow mettere true
//     //Ricalcola la media e l'RMS  //commentare le 3 righe seguenti. Ricorda di ricompilare
//     myTH1F->GetXaxis()->SetRangeUser(-9.5,9.5);
//     Double_t stat[4];
//     myTH1F->GetStats(stat);


    float mean = myTH1F->GetMean();
    float rms = myTH1F->GetRMS();
    float entries = myTH1F->GetEntries();

    //Get Occupancy ME for roll
    RPCGeomServ RPCname(detId);
    //  if(rms==0) cout<<RPCname.name()<<endl;

    if(detId.region()== 0){
      ENTRIES =  BXEntriesBarrel;
      MEAN = BXMeanBarrel; //nome istogramma definito in beginRun
      MEANRing = BXMeanWheel[detId.ring()+2];
      RMS = BXRmsBarrel;
      RMSRing = BXRmsWheel[detId.ring()+2];
    }else if(detId.region()==1){
      ENTRIES =  BXEntriesEndcapP;
      MEAN = BXMeanEndcapP;
      MEANRing = BXMeanDisk[detId.station()+2];
      RMS = BXRmsEndcapP;
      RMSRing = BXRmsDisk[detId.station()+2];
    }else if(detId.region()==-1){
      ENTRIES =  BXEntriesEndcapN;
      MEAN = BXMeanEndcapN;
      MEANRing = BXMeanDisk[3-detId.station()];
      RMS = BXRmsEndcapN;
      RMSRing = BXRmsDisk[3-detId.station()];
    }

    ENTRIES->Fill(entries);

    if(entries  >= entriesCut_){
      RMSRing->Fill(rms);
      RMS->Fill(rms);
      
      if(rms <= rmsCut_){

	//if(mean> distanceMean_ || mean<-distanceMean_ )  cout<<RPCname.name()<<endl;

	MEAN->Fill(mean);
	MEANRing->Fill(mean);
      }
    }
    
  }
}
