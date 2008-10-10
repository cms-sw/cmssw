#include "DQM/SiStripCommon/interface/TkHistoMap.h"

//#define debug_TkHistoMap

TkHistoMap::TkHistoMap(std::string path, std::string MapName,float baseline): 
  dqmStore_(edm::Service<DQMStore>().operator->()),
  tkdetmap_(edm::Service<TkDetMap>().operator->()),
  HistoNumber(23)
{
  LogTrace("TkHistoMap") <<"TkHistoMap::constructor "; 
  createTkHistoMap(path,MapName, baseline);
}

void TkHistoMap::createTkHistoMap(std::string& path, std::string& MapName, float& baseline){

  std::string folder=path+std::string("/")+MapName;
  dqmStore_->setCurrentFolder(folder);
  
  std::string name=MapName+std::string("_");
  std::string fullName;
  int nchX;
  int nchY;
  double lowX,highX;
  double lowY, highY;

  tkHistoMap.resize(HistoNumber);  
  for(int layer=1;layer<HistoNumber;++layer){
    fullName=name+tkdetmap_->getLayerName(layer);
    tkdetmap_->getComponents(layer,nchX,lowX,highX,nchY,lowY,highY);
    TProfile2D* h=new TProfile2D(fullName.c_str(),fullName.c_str(),
				 nchX,lowX,highX,
				 nchY,lowY,highY);

    //initialize bin content for the not assigned bins
    if(baseline!=0){
      for(size_t ix=0;ix<nchX;++ix)
	for(size_t iy=0;iy<nchY;++iy)
	  if(!tkdetmap_->getDetFromBin(layer,ix,iy))
	    h->Fill(1.*(lowX+ix+.5),1.*(lowY+iy+.5),baseline);	  
    }

    tkHistoMap[layer]=dqmStore_->bookProfile2D(fullName,h);
    LogTrace("TkHistoMap")  << "[TkHistoMap::createTkHistoMap] histoName " << fullName << " layer " << layer << " ptr " << tkHistoMap[layer];
  }

}

void TkHistoMap::fill(uint32_t& detid,float value){
  int16_t layer=tkdetmap_->FindLayer(detid);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid);
  LogTrace("TkHistoMap") << "[TkHistoMap::fill] Fill detid " << detid << " Layer " << layer << " value " << value << " ix,iy "  << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y << " " << tkHistoMap[layer]->getTProfile2D()->GetName();
  tkHistoMap[layer]->getTProfile2D()->Fill(xybin.x,xybin.y,value);

#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << tkHistoMap[layer]->getTProfile2D()->GetBinContent(xybin.ix,xybin.iy);
  for(size_t ii=0;ii<4;ii++)
  for(size_t jj=0;jj<11;jj++)
    LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << ii << " " << jj << " " << tkHistoMap[layer]->getTProfile2D()->GetBinContent(ii,jj);
#endif
}

void TkHistoMap::setBinContent(uint32_t& detid,float value){
  int16_t layer=tkdetmap_->FindLayer(detid);
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid);
  TProfile2D* h=tkHistoMap[layer]->getTProfile2D();
  h->SetBinEntries(h->GetBin(xybin.ix,xybin.iy),1);
  h->SetBinContent(h->GetBin(xybin.ix,xybin.iy),value);
  LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent]  setBinContent detid " << detid << " Layer " << layer << " value " << value << " ix,iy "  << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y << " " << tkHistoMap[layer]->getTProfile2D()->GetName() << " bin " << h->GetBin(xybin.ix,xybin.iy);

#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent] " << tkHistoMap[layer]->getTProfile2D()->GetBinContent(xybin.ix,xybin.iy);
  for(size_t ii=0;ii<4;ii++)
    for(size_t jj=0;jj<11;jj++){
      LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent] " << ii << " " << jj << " " << h->GetBinContent(ii,jj);
    }
#endif
}


