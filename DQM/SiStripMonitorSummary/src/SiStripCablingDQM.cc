#include "DQM/SiStripMonitorSummary/interface/SiStripCablingDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TCanvas.h"
using namespace std;
// -----

SiStripCablingDQM::SiStripCablingDQM(const edm::EventSetup & eSetup,
				     edm::ParameterSet const& hPSet,
				     edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){

  // Build the Histo_TkMap:
  if(HistoMaps_On_ ) Tk_HM_ = new TkHistoMap("SiStrip/Histo_Map","Cabling_TkMap",0.);

}
// -----

// -----
SiStripCablingDQM::~SiStripCablingDQM(){}
// -----



// -----
void SiStripCablingDQM::getActiveDetIds(const edm::EventSetup & eSetup){
  
  // Get active and total detIds
  getConditionObject(eSetup);
  cablingHandle_->addActiveDetectorsRawIds(activeDetIds);
  cablingHandle_->addAllDetectorsRawIds(activeDetIds);


  //Initialize arrays for counting:
  int counterTIB[4];
  for(int i=0;i<4;i++) counterTIB[i]=0;
  int counterTID[2][3];
  for(int i=0;i<2;i++){
    for(int j=0;j<3;j++)counterTID[i][j]=0;
  }
  int counterTOB[6];
  for(int i=0;i<6;i++)counterTOB[i]=0;
  int counterTEC[2][9];
  for(int i=0;i<2;i++){
    for(int j=0;j<9;j++)counterTEC[i][j]=0;
  }


 
  std::vector<uint32_t>::const_iterator idet=activeDetIds.begin();

  //fill arrays for counting and fill Histo_Map with value for connected : 
  for(;idet!=activeDetIds.end();++idet){
    uint32_t detId = *idet;
    StripSubdetector subdet(detId);

    if(HistoMaps_On_ ) {Tk_HM_->fill(detId, cablingHandle_->nApvPairs(detId)*2);}
    if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")){
            fillTkMap(detId,cablingHandle_->getConnections(detId).size()*2.); //fill with numb of active APVs
      //          fillTkMap(detId,2.); //fill with numb of active APV
    }

    switch (subdet.subdetId()) 
      {
      case StripSubdetector::TIB:
	{
          TIBDetId tibId(detId);
          int i = tibId.layer() - 1;
	  counterTIB[i]++;
	  break;       
	}
      case StripSubdetector::TID:
	{
	  TIDDetId tidId(detId);
	  if (tidId.side() == 2) {
            int j = tidId.wheel() - 1;
	    counterTID[0][j]++;
	  }  else if (tidId.side() == 1) {
            int j = tidId.wheel() - 1;
	    counterTID[1][j]++;
	  }
	  break;       
	}
      case StripSubdetector::TOB:
	{
          TOBDetId tobId(detId);
          int i = tobId.layer() - 1;
	  counterTOB[i]++;
	  break;       
	}
      case StripSubdetector::TEC:
	{
	  TECDetId tecId(detId);
	  if (tecId.side() == 2) {
            int j = tecId.wheel() - 1;
	    counterTEC[0][j]++;
	  }  else if (tecId.side() == 1) {
            int j = tecId.wheel() - 1;
	    counterTEC[1][j]++;
	  }
	  break;       
	}
      }

  } // idet

  //obtained from tracker.dat and hard-coded
  int TIBDetIds[4]={672,864,540,648};
  int TIDDetIds[2][3]={{136,136,136},{136,136,136}};
  int TOBDetIds[6]={1008,1152,648,720,792,888};
  int TECDetIds[2][9]={{408,408,408,360,360,360,312,312,272},{408,408,408,360,360,360,312,312,272}};



  DQMStore* dqmStore_=edm::Service<DQMStore>().operator->();

  std::string FolderName=fPSet_.getParameter<std::string>("FolderName_For_QualityAndCabling_SummaryHistos");

  dqmStore_->setCurrentFolder(FolderName);

  //  dqmStore_->cd("SiStrip/MechanicalView/");
  MonitorElement *ME;
  ME = dqmStore_->book2D("SummaryOfCabling","SummaryOfCabling",6,0.5,6.5,9,0.5,9.5);
  ME->setAxisTitle("Sub Det",1);
  ME->setAxisTitle("Layer",2);


  ME->getTH1()->GetXaxis()->SetBinLabel(1,"TIB");
  ME->getTH1()->GetXaxis()->SetBinLabel(2,"TID F");
  ME->getTH1()->GetXaxis()->SetBinLabel(3,"TID B");
  ME->getTH1()->GetXaxis()->SetBinLabel(4,"TOB");
  ME->getTH1()->GetXaxis()->SetBinLabel(5,"TEC F");
  ME->getTH1()->GetXaxis()->SetBinLabel(6,"TEC B");

  for(int i=0;i<4;i++){
    ME->Fill(1,i+1,float(counterTIB[i])/TIBDetIds[i]);
  }
  
  for(int i=0;i<2;i++){
    for(int j=0;j<3;j++){
      ME->Fill(i+2,j+1,float(counterTID[i][j])/TIDDetIds[i][j]);
    }
  }

  for(int i=0;i<6;i++){
    ME->Fill(4,i+1,float(counterTOB[i])/TOBDetIds[i]);
  }
  
  for(int i=0;i<2;i++){
    for(int j=0;j<9;j++){
      ME->Fill(i+5,j+1,float(counterTEC[i][j])/TECDetIds[i][j]);
    }
  }

  if (fPSet_.getParameter<bool>("OutputSummaryAtLayerLevelAsImage")){

    TCanvas c1("c1");
    ME->getTH1()->Draw("TEXT");
    ME->getTH1()->SetStats(kFALSE);
    std::string name (ME->getTH1()->GetTitle());
    name+=".png";
    c1.Print(name.c_str());
  }

}


