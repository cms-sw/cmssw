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
  
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Get active and total detIds
  getConditionObject(eSetup);
  if(!cablingHandle_.isValid()) {
    edm::LogError("InvalidCablingHandle") << "Invalid Cabling Handle";
    return;
  }
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

    int32_t n_conn = 0;
      for(uint32_t connDet_i=0; connDet_i<cablingHandle_->getConnections(detId).size(); connDet_i++){
        if(cablingHandle_->getConnections(detId)[connDet_i]!=0 &&  cablingHandle_->getConnections(detId)[connDet_i]->isConnected()!=0) n_conn++;
      }
      fillTkMap(detId,n_conn*2.); 
    }
    switch (subdet.subdetId()) 
      {
      case StripSubdetector::TIB:
	{
          int i = tTopo->tibLayer(detId) - 1;
	  counterTIB[i]++;
	  break;       
	}
      case StripSubdetector::TID:
	{
          int j = tTopo->tidWheel(detId) - 1;
          int side = tTopo->tidSide(detId);
	  if (side == 2) {
	    counterTID[0][j]++;
	  } else if (side == 1) {
	    counterTID[1][j]++;
	  }
	  break;       
	}
      case StripSubdetector::TOB:
	{
          int i = tTopo->tobLayer(detId) - 1;
	  counterTOB[i]++;
	  break;       
	}
      case StripSubdetector::TEC:
	{
          int j = tTopo->tecWheel(detId) - 1;
          int side = tTopo->tecSide(detId);
	  if (side == 2) {
	    counterTEC[0][j]++;
	  } else if (side == 1) {
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


