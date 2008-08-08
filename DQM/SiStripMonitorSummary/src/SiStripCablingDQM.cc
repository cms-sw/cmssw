#include "DQM/SiStripMonitorSummary/interface/SiStripCablingDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"
using namespace std;
// -----

SiStripCablingDQM::SiStripCablingDQM(const edm::EventSetup & eSetup,
				     edm::ParameterSet const& hPSet,
				     edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){}
// -----

// -----
SiStripCablingDQM::~SiStripCablingDQM(){}
// -----


// -----
void SiStripCablingDQM::getActiveDetIds(const edm::EventSetup & eSetup){


  int counterTIB[4];
  for(int i=0;i<4;i++)counterTIB[i]=0;

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

  cablingHandle_->addActiveDetectorsRawIds(activeDetIds);
  selectModules(activeDetIds);

  std::vector<uint32_t>::const_iterator idet=activeDetIds.begin();

  for(;idet!=activeDetIds.end();++idet){
    std::string s;
    s=getLayerNameAndId(*idet).first; 
    LogDebug("SiStripCabling")<<"Sub Det and Layer "<<s;
    for(int i=0;i<4;i++){
      std::stringstream ss;
      ss<<"TIB__layer__"<<i+1;
      if(strstr(s.c_str(),ss.str().c_str())!=NULL)counterTIB[i]++;
    }

    for(int i=0;i<2;i++){
      for(int j=0;j<3;j++){
	std::stringstream ss;
	ss<<"TID__side__"<<i+1<<"__wheel__"<<j+1;
	if(strstr(s.c_str(),ss.str().c_str())!=NULL)counterTID[i][j]++;
      }
    }

    for(int i=0;i<6;i++){
      std::stringstream ss;
      ss<<"TOB__layer__"<<i+1;
      if(strstr(s.c_str(),ss.str().c_str())!=NULL)counterTOB[i]++;
    }

    for(int i=0;i<2;i++){
      for(int j=0;j<9;j++){
	std::stringstream ss;
	ss<<"TEC__side__"<<i+1<<"__wheel__"<<j+1;
	if(strstr(s.c_str(),ss.str().c_str())!=NULL)counterTEC[i][j]++;
      }
    }
  }

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

}
