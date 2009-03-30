#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"

//#define debug_TkHistoMap

TkHistoMap::TkHistoMap(std::string path, std::string MapName,float baseline, bool mechanicalView): 
  HistoNumber(35),
  MapName_(MapName)
{
  if(!edm::Service<DQMStore>().isAvailable()){
    edm::LogError("TkHistoMap") << 
      "\n------------------------------------------"
      "\nUnAvailable Service DQMStore: please insert in the configuration file an instance like"
      "\n\tprocess.load(\"DQMServices.Core.DQMStore_cfg\")"
      "\n------------------------------------------";
  }
  dqmStore_=edm::Service<DQMStore>().operator->();
  if(!edm::Service<TkDetMap>().isAvailable()){
    edm::LogError("TkHistoMap") << 
      "\n------------------------------------------"
      "\nUnAvailable Service TkHistoMap: please insert in the configuration file an instance like"
      "\n\tprocess.TkDetMap = cms.Service(\"TkDetMap\")"
      "\n------------------------------------------";
  }
  tkdetmap_=edm::Service<TkDetMap>().operator->();
  LogTrace("TkHistoMap") <<"TkHistoMap::constructor "; 
  createTkHistoMap(path,MapName, baseline, mechanicalView);
}

void TkHistoMap::save(std::string filename){
  dqmStore_->save(filename);
}

void TkHistoMap::createTkHistoMap(std::string& path, std::string& MapName, float& baseline, bool mechanicalView){
  
  std::string folder=path;
  dqmStore_->setCurrentFolder(folder);
  
  std::string name=MapName+std::string("_");
  std::string fullName;
  int nchX;
  int nchY;
  double lowX,highX;
  double lowY, highY;

  SiStripFolderOrganizer folderOrg;

  tkHistoMap_.resize(HistoNumber);  
  for(int layer=1;layer<HistoNumber;++layer){
    fullName=name+tkdetmap_->getLayerName(layer);
    if(mechanicalView){
      std::stringstream ss;
      
      SiStripDetId::SubDetector subDet;
      uint32_t subdetlayer, side;
      tkdetmap_->getSubDetLayerSide(layer,subDet,subdetlayer,side);
      folderOrg.getSubDetLayerFolderName(ss,subDet,subdetlayer,side);

      folder = ss.str();
      //folder=path+tkdetmap_->getLayerName(layer); //***FIXME with the mechanical view real code
      dqmStore_->setCurrentFolder(folder);
    }
    tkdetmap_->getComponents(layer,nchX,lowX,highX,nchY,lowY,highY);
    TProfile2D* h=new TProfile2D(fullName.c_str(),fullName.c_str(),
				 nchX,lowX,highX,
				 nchY,lowY,highY);

    //initialize bin content for the not assigned bins
    if(baseline!=0){
      for(size_t ix = 1; ix <= (unsigned int) nchX; ++ix)
	for(size_t iy = 1;iy <= (unsigned int) nchY; ++iy)
	  if(!tkdetmap_->getDetFromBin(layer,ix,iy))
	    h->Fill(1.*(lowX+ix-.5),1.*(lowY+iy-.5),baseline);	  
    }

    tkHistoMap_[layer]=dqmStore_->bookProfile2D(fullName,h);
    LogTrace("TkHistoMap")  << "[TkHistoMap::createTkHistoMap] histoName " << fullName << " layer " << layer << " ptr " << tkHistoMap_[layer];
  }

}

void TkHistoMap::fill(uint32_t& detid,float value){
  int16_t layer=tkdetmap_->FindLayer(detid);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid);
  LogTrace("TkHistoMap") << "[TkHistoMap::fill] Fill detid " << detid << " Layer " << layer << " value " << value << " ix,iy "  << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y << " " << tkHistoMap_[layer]->getTProfile2D()->GetName();
  tkHistoMap_[layer]->getTProfile2D()->Fill(xybin.x,xybin.y,value);

#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(xybin.ix,xybin.iy);
  for(size_t ii=0;ii<4;ii++)
    for(size_t jj=0;jj<11;jj++)
      LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << ii << " " << jj << " " << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(ii,jj);
#endif
}

void TkHistoMap::setBinContent(uint32_t& detid,float value){
  int16_t layer=tkdetmap_->FindLayer(detid);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid);
  tkHistoMap_[layer]->getTProfile2D()->SetBinEntries(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy),1);
  tkHistoMap_[layer]->getTProfile2D()->SetBinContent(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy),value);

  LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent]  setBinContent detid " << detid << " Layer " << layer << " value " << value << " ix,iy "  << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y << " " << tkHistoMap_[layer]->getTProfile2D()->GetName() << " bin " << tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy);

#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent] " << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(xybin.ix,xybin.iy);
  for(size_t ii=0;ii<4;ii++)
    for(size_t jj=0;jj<11;jj++){
      LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent] " << ii << " " << jj << " " << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(ii,jj);
    }
#endif
}

void TkHistoMap::add(uint32_t& detid,float value){
  LogTrace("TkHistoMap") << "[TkHistoMap::add]";
  int16_t layer=tkdetmap_->FindLayer(detid);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid);
  setBinContent(detid,tkHistoMap_[layer]->getTProfile2D()->GetBinContent(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy))+value);
  
}

#include "TCanvas.h"
#include "TFile.h"
void TkHistoMap::saveAsCanvas(std::string filename,std::string options,std::string mode){
  //  TCanvas C(MapName_,MapName_,200,10,900,700);
  TCanvas* CTIB=new TCanvas(std::string("Canvas_"+MapName_+"TIB").c_str(),std::string("Canvas_"+MapName_+"TIB").c_str());
  TCanvas* CTOB=new TCanvas(std::string("Canvas_"+MapName_+"TOB").c_str(),std::string("Canvas_"+MapName_+"TOB").c_str());
  TCanvas* CTIDP=new TCanvas(std::string("Canvas_"+MapName_+"TIDP").c_str(),std::string("Canvas_"+MapName_+"TIDP").c_str());
  TCanvas* CTIDM=new TCanvas(std::string("Canvas_"+MapName_+"TIDM").c_str(),std::string("Canvas_"+MapName_+"TIDM").c_str());
  TCanvas* CTECP=new TCanvas(std::string("Canvas_"+MapName_+"TECP").c_str(),std::string("Canvas_"+MapName_+"TECP").c_str());
  TCanvas* CTECM=new TCanvas(std::string("Canvas_"+MapName_+"TECM").c_str(),std::string("Canvas_"+MapName_+"TECM").c_str());
  CTIB->Divide(2,2);
  CTOB->Divide(2,3);
  CTIDP->Divide(1,3);
  CTIDM->Divide(1,3);
  CTECP->Divide(3,3);
  CTECM->Divide(3,3);


  int i;
  i=0;
  CTIB->cd(++i);tkHistoMap_[TkLayerMap::TIB_L1]->getTProfile2D()->Draw(options.c_str());
  CTIB->cd(++i);tkHistoMap_[TkLayerMap::TIB_L2]->getTProfile2D()->Draw(options.c_str());
  CTIB->cd(++i);tkHistoMap_[TkLayerMap::TIB_L3]->getTProfile2D()->Draw(options.c_str());
  CTIB->cd(++i);tkHistoMap_[TkLayerMap::TIB_L4]->getTProfile2D()->Draw(options.c_str());
  
  i=0;
  CTIDP->cd(++i);tkHistoMap_[TkLayerMap::TIDP_D1]->getTProfile2D()->Draw(options.c_str());
  CTIDP->cd(++i);tkHistoMap_[TkLayerMap::TIDP_D2]->getTProfile2D()->Draw(options.c_str());
  CTIDP->cd(++i);tkHistoMap_[TkLayerMap::TIDP_D3]->getTProfile2D()->Draw(options.c_str());

  i=0;
  CTIDM->cd(++i);tkHistoMap_[TkLayerMap::TIDM_D1]->getTProfile2D()->Draw(options.c_str());
  CTIDM->cd(++i);tkHistoMap_[TkLayerMap::TIDM_D2]->getTProfile2D()->Draw(options.c_str());
  CTIDM->cd(++i);tkHistoMap_[TkLayerMap::TIDM_D3]->getTProfile2D()->Draw(options.c_str());
 
  i=0;
  CTOB->cd(++i);tkHistoMap_[TkLayerMap::TOB_L1]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);tkHistoMap_[TkLayerMap::TOB_L2]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);tkHistoMap_[TkLayerMap::TOB_L3]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);tkHistoMap_[TkLayerMap::TOB_L4]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);tkHistoMap_[TkLayerMap::TOB_L5]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);tkHistoMap_[TkLayerMap::TOB_L6]->getTProfile2D()->Draw(options.c_str());

  i=0;
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W1]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W2]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W3]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W4]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W5]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W6]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W7]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W8]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);tkHistoMap_[TkLayerMap::TECP_W9]->getTProfile2D()->Draw(options.c_str());

  i=0;
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W1]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W2]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W3]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W4]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W5]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W6]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W7]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W8]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);tkHistoMap_[TkLayerMap::TECM_W9]->getTProfile2D()->Draw(options.c_str());
 
  TFile *f = new TFile(filename.c_str(),mode.c_str());
  CTIB->Write();
  CTIDP->Write();
  CTIDM->Write();
  CTOB->Write();
  CTECP->Write();
  CTECM->Write();
  f->Close();
  delete f;
}


