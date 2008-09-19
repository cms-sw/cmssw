#include "DQM/SiStripCommon/interface/TkHistoMap.h"


TkHistoMap::TkHistoMap(std::string path, std::string MapName): 
  dqmStore_(edm::Service<DQMStore>().operator->()),
  tkdetmap_(edm::Service<TkDetMap>().operator->())
{
  LogTrace("TkHistoMap") <<"TkHistoMap::constructor "; 
  createTkHistoMap(path,MapName);
}

void TkHistoMap::createTkHistoMap(std::string& path, std::string& MapName){

  std::string folder=path+std::string("/")+MapName;
  dqmStore_->setCurrentFolder(folder);
  
  std::string name=MapName+std::string("_");
  std::string fullName;
  int nchX;
  int nchY;
  double lowX,highX;
  double lowY, highY;

  int HistoNumber=23;
  tkHistoMap.resize(HistoNumber);  
  for(int layer=1;layer<HistoNumber;++layer){
    fullName=name+tkdetmap_->getLayerName(layer);
    LogTrace("TkHistoMap") << "histoName " << fullName;
    tkdetmap_->getComponents(layer,nchX,lowX,highX,nchY,lowY,highY);
    TProfile2D* h=new TProfile2D(fullName.c_str(),fullName.c_str(),
				 nchX,lowX,highX,
				 nchY,lowY,highY);
    LogTrace("TkHistoMap") << "h ptr " << h;

    tkHistoMap[layer]=dqmStore_->bookProfile2D(fullName,h);
    LogTrace("TkHistoMap") << "layer " << layer << " ptr " << tkHistoMap[layer];
  }

}

void TkHistoMap::fill(uint32_t& detid,float& value){
  TkLayerMap::TkLayerEnum layer=tkdetmap_->FindLayer(detid);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid);
  LogTrace("TkHistoMap") << "Fill detid " << detid << " Layer " << layer << "value " << value << " ix,iy "  << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y << " " << tkHistoMap[layer]->getTProfile2D()->GetName();
  tkHistoMap[layer]->getTProfile2D()->Fill(xybin.x,xybin.y,value);
  LogTrace("TkHistoMap") << tkHistoMap[layer]->getTProfile2D()->GetBinContent(xybin.ix,xybin.iy);
  for(size_t ii=0;ii<4;ii++)
  for(size_t jj=0;jj<11;jj++)
    LogTrace("TkHistoMap") << ii << " " << jj << " " << tkHistoMap[layer]->getTProfile2D()->GetBinContent(ii,jj);

}

void TkHistoMap::setBinContent(uint32_t& detid,float& value){
  TkLayerMap::TkLayerEnum layer=tkdetmap_->FindLayer(detid);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid);
  LogTrace("TkHistoMap") << " setBinContent detid " << detid << " Layer " << layer << "value " << value << " ix,iy "  << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y << " " << tkHistoMap[layer]->getTProfile2D()->GetName();
  TProfile2D* h=tkHistoMap[layer]->getTProfile2D();
  h->SetBinEntries(h->GetBin(xybin.ix,xybin.iy),1);
  h->SetBinContent(h->GetBin(xybin.ix,xybin.iy),value);

  LogTrace("TkHistoMap") << tkHistoMap[layer]->getTProfile2D()->GetBinContent(xybin.ix,xybin.iy);
  for(size_t ii=0;ii<4;ii++)
    for(size_t jj=0;jj<11;jj++){
      LogTrace("TkHistoMap") << ii << " " << jj << " " << h->GetBinContent(ii,jj);
    }
}


