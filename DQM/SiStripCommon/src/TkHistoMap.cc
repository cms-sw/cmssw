#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"

//#define debug_TkHistoMap

TkHistoMap::TkHistoMap(const TkDetMap* tkDetMap):
  HistoNumber(35)
{
  cached_detid=0;
  cached_layer=0;
  
  LogTrace("TkHistoMap") <<"TkHistoMap::constructor without parameters"; 
  loadServices();
  tkdetmap_ = tkDetMap;
}

TkHistoMap::TkHistoMap(const TkDetMap* tkDetMap, const std::string& path, const std::string& MapName, float baseline, bool mechanicalView):
  HistoNumber(35), 
  MapName_(MapName)
{
  cached_detid=0;
  cached_layer=0;
  LogTrace("TkHistoMap") <<"TkHistoMap::constructor with parameters"; 
  loadServices();
  tkdetmap_ = tkDetMap;
  createTkHistoMap(path, MapName_, baseline, mechanicalView);
}

TkHistoMap::TkHistoMap(const TkDetMap* tkDetMap, const std::string& path, const std::string& MapName, float baseline, bool mechanicalView, bool isTH2F):
  HistoNumber(35),
  MapName_(MapName)
{
  cached_detid=0;
  cached_layer=0;
  LogTrace("TkHistoMap") <<"TkHistoMap::constructor with parameters"; 
  loadServices();
  tkdetmap_ = tkDetMap;
  isTH2F_ = isTH2F;
  createTkHistoMap(path, MapName_, baseline, mechanicalView);
}

void TkHistoMap::loadServices(){
  if(!edm::Service<DQMStore>().isAvailable()){
    edm::LogError("TkHistoMap") <<
      "\n------------------------------------------"
      "\nUnAvailable Service DQMStore: please insert in the configuration file an instance like"
      "\n\tprocess.load(\"DQMServices.Core.DQMStore_cfg\")"
      "\n------------------------------------------";
  }  
  dqmStore_ = edm::Service<DQMStore>().operator->();
  dqmStore_->meBookerGetter([this](DQMStore::IBooker &b, DQMStore::IGetter &g){
    this->ibooker_ = &b;
    this->igetter_ = &g;
  });
}

void TkHistoMap::save(const std::string& filename){
  // dqmStore_ only for saving
  dqmStore_->save(filename);
}

void TkHistoMap::loadTkHistoMap(const std::string& path, const std::string& MapName, bool mechanicalView){
  MapName_=MapName;
  std::string fullName, folder;
  tkHistoMap_.resize(HistoNumber);
  for(int layer=1;layer<HistoNumber;++layer){
    folder=folderDefinition(path, MapName_, layer, mechanicalView, fullName);

#ifdef debug_TkHistoMap
    LogTrace("TkHistoMap")  << "[TkHistoMap::loadTkHistoMap] folder " << folder << " histoName " << fullName << " find " << folder.find_last_of("/") << "  length " << folder.length();
#endif
    if(folder.find_last_of("/")!=folder.length()-1)
      folder+="/";
    tkHistoMap_[layer]=igetter_->get(folder+fullName);
#ifdef debug_TkHistoMap
    LogTrace("TkHistoMap")  << "[TkHistoMap::loadTkHistoMap] folder " << folder << " histoName " << fullName << " layer " << layer << " ptr " << tkHistoMap_[layer] << " find " << folder.find_last_of("/") << "  length " << folder.length();
#endif
  }
}

void TkHistoMap::createTkHistoMap(const std::string& path, const std::string& MapName, float baseline, bool mechanicalView){

  int nchX;
  int nchY;
  double lowX,highX;
  double lowY, highY;
  std::string fullName, folder;

  tkHistoMap_.resize(HistoNumber);
  const bool bookTH2F = isTH2F_;
  for(int layer=1;layer<HistoNumber;++layer){
    folder=folderDefinition(path, MapName,layer,mechanicalView,fullName);
    tkdetmap_->getComponents(layer,nchX,lowX,highX,nchY,lowY,highY);
    MonitorElement* me;
    if(bookTH2F==false){
	      me  = ibooker_->bookProfile2D(fullName.c_str(),fullName.c_str(),
                                    nchX,lowX,highX,
                                    nchY,lowY,highY,
                                    0.0, 0.0);
    }
    else{
        me  = ibooker_->book2D(fullName.c_str(),fullName.c_str(),
                              nchX,lowX,highX,
                              nchY,lowY,highY);
    }
    //initialize bin content for the not assigned bins
    if(baseline!=0){
      for(size_t ix = 1; ix <= (unsigned int) nchX; ++ix)
        for(size_t iy = 1;iy <= (unsigned int) nchY; ++iy)
          if(!tkdetmap_->getDetFromBin(layer,ix,iy))
            me->Fill(1.*(lowX+ix-.5),1.*(lowY+iy-.5),baseline);
    }

    tkHistoMap_[layer]=me;
#ifdef debug_TkHistoMap
    LogTrace("TkHistoMap")  << "[TkHistoMap::createTkHistoMap] folder " << folder << " histoName " << fullName << " layer " << layer << " ptr " << tkHistoMap_[layer];
#endif
  }
}

std::string TkHistoMap::folderDefinition(std::string folder, const std::string& MapName, int layer , bool mechanicalView,std::string& fullName ){

  std::string name = MapName+std::string("_");
  fullName=name+TkDetMap::getLayerName(layer);
  //  std::cout << "[TkHistoMap::folderDefinition] fullName: " << fullName << std::endl;

  if(mechanicalView){
    std::stringstream ss;

    SiStripFolderOrganizer folderOrg;
    folderOrg.setSiStripFolderName(folder);

    SiStripDetId::SubDetector subDet;
    uint32_t subdetlayer = 0, side = 0;
    TkDetMap::getSubDetLayerSide(layer,subDet,subdetlayer,side);
    folderOrg.getSubDetLayerFolderName(ss,subDet,subdetlayer,side);
    folder = ss.str();
    //    std::cout << "[TkHistoMap::folderDefinition] folder: " << folder << std::endl;
  }
  ibooker_->setCurrentFolder(folder);
  return folder;
}

#include <iostream>
void TkHistoMap::fillFromAscii(const std::string& filename){
  std::ifstream file;
  file.open(filename.c_str());
  float value;
  uint32_t detid;
  while (file.good()){
    file >> detid >> value;
    fill(detid,value);
  }
  file.close();
}

void TkHistoMap::fill(DetId detid,float value){
  int16_t layer=tkdetmap_->findLayer(detid , cached_detid , cached_layer , cached_XYbin);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid , cached_detid , cached_layer , cached_XYbin);
#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::fill] Fill detid " << detid.rawId() << " Layer " << layer << " value " << value << " ix,iy "  << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y << " " << tkHistoMap_[layer]->getTProfile2D()->GetName();
#endif
  if(tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TPROFILE2D)
    tkHistoMap_[layer]->getTProfile2D()->Fill(xybin.x,xybin.y,value);
  else if (tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TH2F)
    tkHistoMap_[layer]->getTH2F()->Fill(xybin.x,xybin.y,value);

#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(xybin.ix,xybin.iy);
  for(size_t ii=0;ii<4;ii++)
    for(size_t jj=0;jj<11;jj++){
      if(tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TPROFILE2D)
        LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << ii << " " << jj << " " << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(ii,jj);
      if(tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TH2F)
        LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << ii << " " << jj << " " << tkHistoMap_[layer]->getTH2F()->GetBinContent(ii,jj);
    }
#endif
}

void TkHistoMap::setBinContent(DetId detid,float value){
  int16_t layer=tkdetmap_->findLayer(detid , cached_detid , cached_layer , cached_XYbin);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid , cached_detid , cached_layer , cached_XYbin);
  if(tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TPROFILE2D){
    tkHistoMap_[layer]->getTProfile2D()->SetBinEntries(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy),1);
    tkHistoMap_[layer]->getTProfile2D()->SetBinContent(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy),value);
  }
  else if (tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TH2F){
    tkHistoMap_[layer]->getTH2F()->SetBinContent(xybin.ix,xybin.iy,value);
  }

#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent]  setBinContent detid " << detid.rawId() << " Layer " << layer << " value " << value << " ix,iy "  << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y << " " << tkHistoMap_[layer]->getTProfile2D()->GetName() << " bin " << tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy);

  LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent] " << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(xybin.ix,xybin.iy);
  for(size_t ii=0;ii<4;ii++)
    for(size_t jj=0;jj<11;jj++){
      LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent] " << ii << " " << jj << " " << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(ii,jj);
    }
#endif
}

void TkHistoMap::add(DetId detid,float value){
#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::add]";
#endif
  int16_t layer=tkdetmap_->findLayer(detid , cached_detid , cached_layer , cached_XYbin);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid , cached_detid , cached_layer , cached_XYbin);
  if(tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TPROFILE2D)
    setBinContent(detid,tkHistoMap_[layer]->getTProfile2D()->GetBinContent(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy))+value);
  else if (tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TH2F)
    setBinContent(detid,tkHistoMap_[layer]->getTH2F()->GetBinContent(tkHistoMap_[layer]->getTH2F()->GetBin(xybin.ix,xybin.iy))+value);
}

float TkHistoMap::getValue(DetId detid){
  int16_t layer=tkdetmap_->findLayer(detid , cached_detid , cached_layer , cached_XYbin);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid , cached_detid , cached_layer , cached_XYbin);

  if (tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TH2F)
    return tkHistoMap_[layer]->getTH2F()->GetBinContent(tkHistoMap_[layer]->getTH2F()->GetBin(xybin.ix,xybin.iy));
  else
    return tkHistoMap_[layer]->getTProfile2D()->GetBinContent(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy));
}
float TkHistoMap::getEntries(DetId detid){
  int16_t layer=tkdetmap_->findLayer(detid , cached_detid , cached_layer , cached_XYbin);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid , cached_detid , cached_layer , cached_XYbin);
  if (tkHistoMap_[layer]->kind() == MonitorElement::DQM_KIND_TH2F)
    return 1;
  else
    return tkHistoMap_[layer]->getTProfile2D()->GetBinEntries(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix,xybin.iy));
}

void TkHistoMap::dumpInTkMap(TrackerMap* tkmap,bool dumpEntries){
  for(int layer=1;layer<HistoNumber;++layer){
    // std::vector<uint32_t> dets;
    // tkdetmap_->getDetsForLayer(layer,dets);
    std::vector<DetId> dets = tkdetmap_->getDetsForLayer(layer);
    for(size_t i=0;i<dets.size();++i){
      if(dets[i]>0){
        if(getEntries(dets[i])>0) {
          tkmap->fill(dets[i],
                      dumpEntries ? getEntries(dets[i]) : getValue(dets[i]));
        }
      }
    }
  }
}

#include "TCanvas.h"
#include "TFile.h"
void TkHistoMap::saveAsCanvas(const std::string& filename, const std::string& options, const std::string& mode){
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