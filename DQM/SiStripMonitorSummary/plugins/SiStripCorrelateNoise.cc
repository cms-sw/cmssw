#include "DQM/SiStripMonitorSummary/plugins/SiStripCorrelateNoise.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/Run.h"
#include "TCanvas.h"


SiStripCorrelateNoise::SiStripCorrelateNoise(const edm::ParameterSet& iConfig):
  refNoise(0),oldGain(0),newGain(0),cacheID_noise(0xFFFFFFFF),cacheID_gain(0xFFFFFFFF)
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

  file->cd();
  tkmap = new TrackerMap();
}


SiStripCorrelateNoise::~SiStripCorrelateNoise()
{}

//

void
SiStripCorrelateNoise::beginRun(const edm::Run& run, const edm::EventSetup& es){

  if(getNoiseCache(es)==cacheID_noise )
    return;

  edm::LogInfo("") << "[SiStripCorrelateNoise::beginRun]  cacheID_noise " << cacheID_noise << std::endl; 
  
  es.get<SiStripNoisesRcd>().get(noiseHandle_);
  SiStripNoises * aNoise= new SiStripNoises(*noiseHandle_.product());
  
  //Check if gain is the same from one noise iov to the other, otherwise cache the new gain (and the old one) to rescale

  checkGainCache(es);

  if(cacheID_noise!=0xFFFFFFFF){
    char dir[128];
    theRun=run.run();
    sprintf(dir,"Run_%d",theRun);
    file->cd("");
    file->mkdir(dir);
    file->cd(dir);
    DoAnalysis(es,*noiseHandle_.product(),*refNoise);
    DoPlots(); 
  }

  cacheID_noise=getNoiseCache(es);  
  if(refNoise!=0)
    delete refNoise;
  refNoise=aNoise;
}

void 
SiStripCorrelateNoise::checkGainCache(const edm::EventSetup& es){
  equalGain=true;
  if(getGainCache(es)!=cacheID_gain ){
    es.get<SiStripApvGainRcd>().get(gainHandle_);
    if(oldGain!=0)
      delete oldGain;
    
    oldGain = newGain;
    newGain = new SiStripApvGain(*gainHandle_.product());

    if(cacheID_gain!=0xFFFFFFFF)
      equalGain=false;
   cacheID_gain=getGainCache(es);
   edm::LogInfo("") << "[SiStripCorrelateNoise::checkGainCache]  cacheID_gain " << cacheID_gain << std::endl; 
  }
}

void 
SiStripCorrelateNoise::DoPlots(){
  TCanvas *C=new TCanvas();
  C->Divide(2,2);
  
  char outName[128];
  sprintf(outName,"Run_%d.png",theRun);
  for(size_t i=0;i<vTH1.size();i++)
    if(vTH1[i]!=0){
      if(i%100==0){
	C->cd(i/100);
	vTH1[i]->SetLineColor(i/100);
	vTH1[i]->Draw();
	C->cd(i/100)->SetLogy();
      }
      vTH1[i]->Write();
    }
  
  C->Print(outName);
  delete C;
  
  vTH1.clear();
  file->cd("");

  char dir[128];
  sprintf(dir,"Run_%d_TkMap.png",theRun);
  tkmap->save(false,0,5,dir);
  delete tkmap;
  tkmap = new TrackerMap();
}

void 
SiStripCorrelateNoise::DoAnalysis(const edm::EventSetup& es, SiStripNoises Noise, SiStripNoises& refNoise){
  typedef std::vector<SiStripNoises::ratioData> collection; 
  collection divNoise=Noise/refNoise;

  edm::LogInfo("") << "[Doanalysis]";

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  std::vector<TH1F *>histos;

  collection::const_iterator iter=divNoise.begin();
  collection::const_iterator iterE=divNoise.end();

  float value;
  float gainRatio=1.;
  //Divide result by d
  for(;iter!=iterE;++iter){
    getHistos(iter->detid, tTopo, histos);
    
    size_t strip=0, stripE= iter->values.size();
    size_t apvNb=7;

    for (;strip<stripE;++strip){       
      if(!equalGain && strip/128!=apvNb){
	apvNb=strip/128;
	if(apvNb<6)
	  gainRatio=getGainRatio(iter->detid,apvNb);
	else
	  edm::LogInfo("") << "[Doanalysis] detid " << iter->detid << " strip " << strip << " apvNb " << apvNb;
      }
      //edm::LogInfo("") << "[Doanalysis] detid " << iter->detid << " strip " << strip << " value " << iter->values[strip];
      value=iter->values[strip]*gainRatio;
      tkmap->fill(iter->detid,value);
      for(size_t i=0;i<histos.size();++i)
	histos[i]->Fill(value);
    }
  }
}

float  
SiStripCorrelateNoise::getGainRatio(const uint32_t& detid, const uint16_t& apv){

  SiStripApvGain::Range oldRange=oldGain->getRange(detid); 
  SiStripApvGain::Range newRange=newGain->getRange(detid); 

  if(oldRange.first==oldRange.second ||
     newRange.first==newRange.second)
    return 1.;

  return oldGain->getApvGain(apv,oldRange)/newGain->getApvGain(apv,newRange);

}

float  
SiStripCorrelateNoise::getMeanNoise(const SiStripNoises::Range& noiseRange,const uint32_t& firstStrip, const uint32_t& range){
  
  float mean=0;
  for (size_t istrip=firstStrip;istrip<firstStrip+range;istrip++){
    mean+=noiseHandle_->getNoise(istrip,noiseRange);
  }
  return mean/(1.*range);
}

void
SiStripCorrelateNoise::getHistos(const uint32_t& detid, const TrackerTopology* tTopo, std::vector<TH1F*>& histos){
  
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
  

  histos.push_back(getHisto(100+100*subdet));
  histos.push_back(getHisto(index));
  
}

TH1F*
SiStripCorrelateNoise::getHisto(const long unsigned int& index){
  if(vTH1.size()<index+1)
    vTH1.resize(index+1,0);
  
  if(vTH1[index]==0){
    char name[128];
    std::string SubD;
    if(index<200)
      SubD="TIB";
    else if(index<300)
      SubD="TID";
    else if(index<400)
      SubD="TOB";
    else 
      SubD="TEC";
    sprintf(name,"%d_%lu__%s",theRun,index,SubD.c_str());
    edm::LogInfo("")<<"[getHisto] creating index " << index << std::endl;
    vTH1[index]=new TH1F(name,name,200,-0.5,10.5);
  }
  
  return vTH1[index];
}

void 
SiStripCorrelateNoise::endJob() {
  file->Write();
  file->Close();


}

