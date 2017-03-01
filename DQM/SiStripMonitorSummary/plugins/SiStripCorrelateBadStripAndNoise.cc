#include "DQM/SiStripMonitorSummary/plugins/SiStripCorrelateBadStripAndNoise.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"



SiStripCorrelateBadStripAndNoise::SiStripCorrelateBadStripAndNoise(const edm::ParameterSet& iConfig):
  cacheID_quality(0xFFFFFFFF),cacheID_noise(0xFFFFFFFF)
{
  //now do what ever initialization is needed
  if(!edm::Service<SiStripDetInfoFileReader>().isAvailable()){
    edm::LogError("TkLayerMap") << 
      "\n------------------------------------------"
      "\nUnAvailable Service SiStripDetInfoFileReader: please insert in the configuration file an instance like"
      "\n\tprocess.SiStripDetInfoFileReader = cms.Service(\"SiStripDetInfoFileReader\")"
      "\n------------------------------------------";
  }
 
  fr=edm::Service<SiStripDetInfoFileReader>().operator->();
  file = new TFile("correlTest.root","RECREATE");
  tkmap = new TrackerMap();
}


SiStripCorrelateBadStripAndNoise::~SiStripCorrelateBadStripAndNoise()
{}

//

void
SiStripCorrelateBadStripAndNoise::beginRun(const edm::Run& run, const edm::EventSetup& es){

  if(getNoiseCache(es)==cacheID_noise && getQualityCache(es)==cacheID_quality)
    return;
  cacheID_noise=getNoiseCache(es);
  cacheID_quality=getQualityCache(es);
  
  edm::LogInfo("") << "[SiStripCorrelateBadStripAndNoise::beginRun] cacheID_quality " << cacheID_quality << " cacheID_noise " << cacheID_noise << std::endl; 
  
  es.get<SiStripQualityRcd>().get(qualityHandle_);
  es.get<SiStripNoisesRcd>().get(noiseHandle_);

  DoAnalysis(es);
}

void 
SiStripCorrelateBadStripAndNoise::DoAnalysis(const edm::EventSetup& es){

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //Loop on quality bad stirps
  //for each strip, look at the noise 
  // evalaute the mean apv noise and the ratio among strip noise and meanApvNoise
  // put the value in the histo in terms of ratio Vs percentage of badStrips per APV

  //Fill an histo per subdet and layer (and plus && minus for TEC/TID)
  edm::LogInfo("") << "[Doanalysis]";
  iterateOnDets(tTopo);

}

void 
SiStripCorrelateBadStripAndNoise::iterateOnDets(const TrackerTopology* tTopo){

  SiStripQuality::RegistryIterator rbegin = qualityHandle_->getRegistryVectorBegin();
  SiStripQuality::RegistryIterator rend   = qualityHandle_->getRegistryVectorEnd();

 
  for (SiStripBadStrip::RegistryIterator rp=rbegin; rp != rend; ++rp) {
    const uint32_t detid=rp->detid;
    
    SiStripQuality::Range sqrange = SiStripQuality::Range( qualityHandle_->getDataVectorBegin()+rp->ibegin , qualityHandle_->getDataVectorBegin()+rp->iend );
    iterateOnBadStrips(detid,tTopo,sqrange);
  }  
}

void 
SiStripCorrelateBadStripAndNoise::iterateOnBadStrips(const uint32_t & detid, const TrackerTopology* tTopo, SiStripQuality::Range& sqrange){

  float percentage=0;
  for(int it=0;it<sqrange.second-sqrange.first;it++){
    unsigned int firstStrip=qualityHandle_->decode( *(sqrange.first+it) ).firstStrip;
    unsigned int range=qualityHandle_->decode( *(sqrange.first+it) ).range;
    
    correlateWithNoise(detid,tTopo,firstStrip,range);

    edm::LogInfo("range")<< range;
    percentage+=range;
  }
  if(percentage!=0)
    percentage/=128.*fr->getNumberOfApvsAndStripLength(detid).first;
  if(percentage>1)
    edm::LogError("SiStripQualityStatistics") <<  "PROBLEM detid " << detid << " value " << percentage<< std::endl;
  
  //------- Global Statistics on percentage of bad components along the IOVs ------//
  if(percentage!=0)
    edm::LogInfo("")<< "percentage " << detid << " " << percentage;
}

void 
SiStripCorrelateBadStripAndNoise::correlateWithNoise(const uint32_t & detid, const TrackerTopology* tTopo, const uint32_t & firstStrip,  const uint32_t & range){

  std::vector<TH2F *>histos;

  SiStripNoises::Range noiseRange = noiseHandle_->getRange(detid);
  edm::LogInfo("Domenico") << "detid " << detid << " first " << firstStrip << " range " << range;
  float meanAPVNoise=getMeanNoise(noiseRange,firstStrip/128,128);

  //float meanNoiseHotStrips=getMeanNoise(noiseRange,firstStrip,range);
  for (size_t theStrip=firstStrip;theStrip<firstStrip+range;theStrip++){
    float meanNoiseHotStrips=getMeanNoise(noiseRange,theStrip,1);
  
    //Get the histogram for this detid
    getHistos(detid, tTopo, histos);
    float yvalue=range<21?1.*range:21;
    
    for(size_t i=0;i<histos.size();++i)
      histos[i]->Fill(meanNoiseHotStrips/meanAPVNoise-1.,yvalue);
    
    if(meanNoiseHotStrips/meanAPVNoise-1.<-0.3)
      tkmap->fillc(detid,0xFF0000);
    else
      tkmap->fillc(detid,0x0000FF);
  }
}

float  
SiStripCorrelateBadStripAndNoise::getMeanNoise(const SiStripNoises::Range& noiseRange,const uint32_t& firstStrip, const uint32_t& range){
  
  float mean=0;
  for (size_t istrip=firstStrip;istrip<firstStrip+range;istrip++){
    mean+=noiseHandle_->getNoise(istrip,noiseRange);
  }
  return mean/(1.*range);
}

void
SiStripCorrelateBadStripAndNoise::getHistos(const uint32_t& detid, const TrackerTopology* tTopo, std::vector<TH2F*>& histos){
  
  histos.clear();
  
  int subdet=-999; int component=-999;
  SiStripDetId a(detid);
  if ( a.subdetId() == 3 ){
    subdet=0;
    component=tTopo->tibLayer(detid);
  } else if ( a.subdetId() == 4 ) {
    subdet=1;
    component=tTopo->tidSide(detid)==2?tTopo->tidWheel(detid):tTopo->tidWheel(detid)+3;
  } else if ( a.subdetId() == 5 ) {
    subdet=2;
    component=tTopo->tobLayer(detid);
  } else if ( a.subdetId() == 6 ) {
    subdet=3;
    component=tTopo->tecSide(detid)==2?tTopo->tecWheel(detid):tTopo->tecWheel(detid)+9;
  } 
  
  int index=100+subdet*100+component;


  histos.push_back(getHisto(subdet));
  histos.push_back(getHisto(index));
  
}

TH2F*
SiStripCorrelateBadStripAndNoise::getHisto(const long unsigned int& index){
  if(vTH2.size()<index+1)
    vTH2.resize(index+1,0);
  
  if(vTH2[index]==0){
    char name[128];
    sprintf(name,"%lu",index);
    edm::LogInfo("")<<"[getHisto] creating index " << index << std::endl;
    vTH2[index]=new TH2F(name,name,50,-2.,2.,21,0.5,21.5);
  }
  
  return vTH2[index];
}

void 
SiStripCorrelateBadStripAndNoise::endJob() {
  for(size_t i=0;i<vTH2.size();i++)
    if(vTH2[i]!=0)
      vTH2[i]->Write();

  file->Write();
  file->Close();

  tkmap->save(true,0,0,"testTkMap.png");

}

