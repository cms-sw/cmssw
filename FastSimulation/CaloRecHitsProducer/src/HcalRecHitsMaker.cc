#include "FastSimulation/CaloRecHitsProducer/interface/HcalRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "CLHEP/GenericFunctions/Erf.hh"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLHcal.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "TFile.h"
#include "TGraph.h"
#include "TROOT.h"
#include <fstream>
#include <iostream>

class RandomEngine;

bool HcalRecHitsMaker::initialized_ = false; 
bool HcalRecHitsMaker::initializedHB_ = false; 
bool HcalRecHitsMaker::initializedHE_ = false; 
bool HcalRecHitsMaker::initializedHO_ = false; 
bool HcalRecHitsMaker::initializedHF_ = false; 
std::vector<float> HcalRecHitsMaker::peds_;
std::vector<int> HcalRecHitsMaker::fctoadc_;
std::vector<float> HcalRecHitsMaker::sat_;
std::vector<float> HcalRecHitsMaker::gains_;
std::vector<float> HcalRecHitsMaker::noisesigma_;
std::vector<float> HcalRecHitsMaker::TPGFactor_;
std::vector<float> HcalRecHitsMaker::miscalib_;
std::vector<HcalDetId> HcalRecHitsMaker::theDetIds_;
std::vector<int> HcalRecHitsMaker::hbhi_;
std::vector<int> HcalRecHitsMaker::hehi_;
std::vector<int> HcalRecHitsMaker::hohi_;
std::vector<int> HcalRecHitsMaker::hfhi_;
unsigned HcalRecHitsMaker::maxIndex_  = 0 ; 

HcalRecHitsMaker::HcalRecHitsMaker(edm::ParameterSet const & p, int det,
				   const RandomEngine * myrandom)
  :
  det_(det),
  doDigis_(false),
  random_(myrandom)
{
  edm::ParameterSet RecHitsParameters=p.getParameter<edm::ParameterSet>("HCAL");
  noise_ = RecHitsParameters.getParameter<std::vector<double> >("Noise");
  corrfac_ = RecHitsParameters.getParameter<std::vector<double> >("NoiseCorrectionFactor");
  threshold_ = RecHitsParameters.getParameter<std::vector<double> >("Threshold");
  doSaturation_ = RecHitsParameters.getParameter<bool>("EnableSaturation");
    
  refactor_ = RecHitsParameters.getParameter<double> ("Refactor");
  refactor_mean_ = RecHitsParameters.getParameter<double> ("Refactor_mean");
  hcalfileinpath_= RecHitsParameters.getParameter<std::string> ("fileNameHcal");  
  inputCol_=RecHitsParameters.getParameter<edm::InputTag>("MixedSimHits");
  nhbcells_=nhecells_=nhocells_=nhfcells_=0;

  if(det_==4)
    {
      hbhi_.reserve(2600);
      hehi_.reserve(2600);
    }
  else if (det_==5)
    hohi_.reserve(2200);
  else if (det_==6)
    hfhi_.reserve(1800);

  nnoise_=noise_.size();
  if(threshold_.size()!=noise_.size())
    {
      edm::LogWarning("CaloRecHitsProducer") << " WARNING : HCAL Noise simulation, the number of parameters should be the same for the noise and the thresholds. Disabling the noise simulation " << std::endl;
      noise_.resize(nnoise_,0);
    }

  //  edm::ParameterSet hcalparam = p2.getParameter<edm::ParameterSet>("hcalSimParam"); 
  //  myHcalSimParameterMap_ = new HcalSimParameterMap(hcalparam);

  // Computes the fraction of HCAL above the threshold
  Genfun::Erf myErf;
  noiseFromDb_.resize(nnoise_,false);
  hcalHotFraction_.resize(nnoise_,0.);
  myGaussianTailGenerators_.resize(nnoise_,0);
  if(noise_.size()>0) 
    {
      for(unsigned inoise=0;inoise<nnoise_;++inoise)
	{
	  if(noise_[inoise]==0) {
	    hcalHotFraction_[inoise]=0.;
	    continue;
	  } else if(noise_[inoise]==-1) {
	    noiseFromDb_[inoise]=true;
	    continue;
	  } else {
	    hcalHotFraction_[inoise] = 0.5-0.5*myErf(threshold_[inoise]/noise_[inoise]/sqrt(2.));
	    myGaussianTailGenerators_[inoise]=new GaussianTail(random_,noise_[inoise],threshold_[inoise]);
	  }
	}   
    }  
}

HcalRecHitsMaker::~HcalRecHitsMaker()
{
  clean();  
  if(myGaussianTailGenerators_.size()) 
    {
      for(unsigned igt=0; igt<myGaussianTailGenerators_.size();++igt)
	    delete myGaussianTailGenerators_[igt];
    }
  myGaussianTailGenerators_.clear();
  theDetIds_.clear();
  hbhi_.clear();
  hehi_.clear();
  hohi_.clear();
  hfhi_.clear();
    
}

void HcalRecHitsMaker::init(const edm::EventSetup &es,bool doDigis,bool doMiscalib)
{
  doDigis_=doDigis;
  doMiscalib_=doMiscalib;
// needed for the noise simulation
  edm::ESHandle<HcalDbService> conditions;
  es.get<HcalDbRecord>().get(conditions);
  const HcalDbService * theDbService=conditions.product();

  // get the correction factors
  edm::ESHandle<HcalRespCorrs> rchandle;
  es.get<HcalRespCorrsRcd>().get(rchandle);
  myRespCorr= rchandle.product();
  
  //general initialization
  if(!initialized_) 
    {
      theDetIds_.resize(9201);
      unsigned ncells=createVectorsOfCells(es);
      edm::LogInfo("CaloRecHitsProducer") << "Total number of cells in HCAL " << ncells << std::endl;
      hcalRecHits_.resize(maxIndex_+1,0.);
      edm::LogInfo("CaloRecHitsProducer") << "Largest HCAL hashedindex" << maxIndex_ << std::endl;

      peds_.resize(9201);
      gains_.resize(9201);
      sat_.resize(9201);
  	  noisesigma_.resize(9201);
      
      miscalib_.resize(maxIndex_+1,1.);
      // Read from file ( a la HcalRecHitsRecalib.cc)
      // here read them from xml (particular to HCAL)
      CaloMiscalibMapHcal mapHcal;
  
      edm::ESHandle<HcalTopology> topo;
      es.get<HcalRecNumberingRecord>().get( topo );
      mapHcal.prefillMap(*topo);
  
      edm::FileInPath hcalfiletmp("CalibCalorimetry/CaloMiscalibTools/data/"+hcalfileinpath_);      
      std::string hcalfile=hcalfiletmp.fullPath();            
      if(doMiscalib_ && !hcalfile.empty()) 
  	    {
  	      MiscalibReaderFromXMLHcal hcalreader_(mapHcal);
        
  	      hcalreader_.parseXMLMiscalibFile(hcalfile);
  	      //	  mapHcal.print();
  	      std::map<uint32_t,float>::const_iterator it=mapHcal.get().begin();
  	      std::map<uint32_t,float>::const_iterator itend=mapHcal.get().end();
  	      for(;it!=itend;++it)
  	        {
  	          HcalDetId myDetId(it->first);
  	          float icalconst=it->second;
  	          miscalib_[topo->detId2denseId(myDetId)]=refactor_mean_+(icalconst-refactor_mean_)*refactor_;
  	        }
  	    }
      
      // Open the histogram for the fC to ADC conversion
      gROOT->cd();
      edm::FileInPath myDataFile("FastSimulation/CaloRecHitsProducer/data/adcvsfc.root");
      TFile * myFile = new TFile(myDataFile.fullPath().c_str(),"READ");
      TGraph * myGraf = (TGraph*)myFile->Get("adcvsfc");
      unsigned size=myGraf->GetN();
      fctoadc_.resize(10000);
      unsigned p_index=0;
      fctoadc_[0]=0;
      int prev_nadc=0;
      int nadc=0;
      for(unsigned ibin=0;ibin<size;++ibin)
  	    {
  	      double x,y;
  	      myGraf->GetPoint(ibin,x,y);
  	      int index=(int)floor(x);
  	      if(index<0||index>=10000) continue;
  	      prev_nadc=nadc;
  	      nadc=(int)y;
  	      // Now fills the vector
  	      for(unsigned ivec=p_index;ivec<(unsigned)index;++ivec)
  	        {
  	          fctoadc_[ivec] = prev_nadc;
  	        }
  	      p_index = index;
  	    }
      myFile->Close();
      gROOT->cd();
      edm::FileInPath myTPGFilePath("CalibCalorimetry/HcalTPGAlgos/data/RecHit-TPG-calib.dat");
      TPGFactor_.resize(87,1.2);
      std::ifstream  myTPGFile(myTPGFilePath.fullPath().c_str(),ifstream::in);
      if(myTPGFile)
  	    {
  	      float gain;
  	      myTPGFile >> gain;
  	      for(unsigned i=0;i<86;++i)
  	        {
  	          myTPGFile >> TPGFactor_[i] ;
  	          //std::cout << TPGFactor_[i] << std::endl;
  	        }
  	    }
      else
  	    {
  	      std::cout << " Unable to open CalibCalorimetry/HcalTPGAlgos/data/RecHit-TPG-calib.dat" << std::endl;
  	      std::cout <<	" Using a constant 1.2 factor " << std::endl;
  	    }
		  
	  initialized_=true; 
	}
	
    if(!initializedHB_ && det_==4){
	  nhbcells_ = hbhi_.size();
      //HB
      for(unsigned ic=0;ic<nhbcells_;++ic)
	    {
	      float gain = theDbService->getGain(theDetIds_[hbhi_[ic]])->getValue(0);
	      float mgain=0.;
	      for(unsigned ig=0;ig<4;++ig)
	        mgain+=theDbService->getGain(theDetIds_[hbhi_[ic]])->getValue(ig);
	      if(noiseFromDb_[0])
	        noisesigma_[hbhi_[ic]]=noiseInfCfromDB(theDbService,theDetIds_[hbhi_[ic]])*mgain*0.25;
	      // std::cout << " NOISEHB " << theDetIds_[hbhi_[ic]].ieta() << " " << noisesigma_[hbhi_[ic]] << "  "<< std::endl;
	      // 10000 (ADC scale) / 4. (to compute the mean) / 0.92  ADC/fC
          // *1.25 (only ~80% in 1ts Digi, while saturation applied to 4ts RecHit) 
	      int ieta=theDetIds_[hbhi_[ic]].ieta();
	      float tpgfactor=TPGFactor_[(ieta>0)?ieta+43:-ieta];
	      mgain*=2500./0.92*tpgfactor ;// 10000 (ADC scale) / 4. (to compute the mean)
	      sat_[hbhi_[ic]]=(doSaturation_)?mgain:99999.;
          
	      peds_[hbhi_[ic]]=theDbService->getPedestal(theDetIds_[hbhi_[ic]])->getValue(0);
        
	      gain*=tpgfactor;
	      gains_[hbhi_[ic]]=gain;
	    }
	  
	  initializedHB_=true;
	}
	
    if(!initializedHE_ && det_==4){
	  nhecells_ = hehi_.size();
     //HE
      for(unsigned ic=0;ic<nhecells_;++ic)
	    {
	      float gain= theDbService->getGain(theDetIds_[hehi_[ic]])->getValue(0);
	      int ieta=theDetIds_[hehi_[ic]].ieta();
	      float mgain=0.;
	      for(unsigned ig=0;ig<4;++ig)
	        mgain+=theDbService->getGain(theDetIds_[hehi_[ic]])->getValue(ig);
	      if(noiseFromDb_[1])
	        noisesigma_[hehi_[ic]]=noiseInfCfromDB(theDbService,theDetIds_[hehi_[ic]])*mgain*0.25;
	      //      std::cout << " NOISEHE " << theDetIds_[hehi_[ic]].ieta() << " " << noisesigma_[hehi_[ic]] << "  "<< std::endl;
	      float tpgfactor=TPGFactor_[(ieta>0)?ieta+44:-ieta+1];
	      mgain*=2500./0.92*tpgfactor;
	      sat_[hehi_[ic]]=(doSaturation_)?mgain:99999.;
          
	      gain*=tpgfactor;
	      peds_[hehi_[ic]]=theDbService->getPedestal(theDetIds_[hehi_[ic]])->getValue(0);
	      gains_[hehi_[ic]]=gain;
	    }

      initializedHE_=true;
	}
	
    if(!initializedHO_ && det_==5){
	  nhocells_ = hohi_.size();
     //HO
      for(unsigned ic=0;ic<nhocells_;++ic)
	    {
	      float ped=theDbService->getPedestal(theDetIds_[hohi_[ic]])->getValue(0);
	      float gain=theDbService->getGain(theDetIds_[hohi_[ic]])->getValue(0);
	      float mgain=0.;
	      for(unsigned ig=0;ig<4;++ig)
	        mgain+=theDbService->getGain(theDetIds_[hohi_[ic]])->getValue(ig);
	      if(noiseFromDb_[0])
	        noisesigma_[hohi_[ic]]=noiseInfCfromDB(theDbService,theDetIds_[hohi_[ic]])*mgain*0.25;
	      //      std::cout << " NOISEHO " << theDetIds_[hohi_[ic]].ieta() << " " << noisesigma_[hohi_[ic]] << "  "<< std::endl;
	      int ieta=HcalDetId(hohi_[ic]).ieta();
	      float tpgfactor=TPGFactor_[(ieta>0)?ieta+43:-ieta];
	      mgain*=2500./0.92*tpgfactor;
	      sat_[hohi_[ic]]=(doSaturation_)?mgain:99999.;
        
	      gain*=tpgfactor;
	      peds_[hohi_[ic]]=ped;
	      gains_[hohi_[ic]]=gain;
	    }
    
      initializedHO_=true;
	}

    if(!initializedHF_ && det_==6){
	  nhfcells_ = hfhi_.size();
      //HF
      for(unsigned ic=0;ic<nhfcells_;++ic)
	    {
	      float ped=theDbService->getPedestal(theDetIds_[hfhi_[ic]])->getValue(0);
	      float gain=theDbService->getGain(theDetIds_[hfhi_[ic]])->getValue(0);
	      float mgain=0.;
	      for(unsigned ig=0;ig<4;++ig)
	        mgain+=theDbService->getGain(theDetIds_[hfhi_[ic]])->getValue(ig);
	      // additional 1/2 factor for the HF (Salavat)
	      if(noiseFromDb_[0])
	        {
	          // computation from when the noise was taken 
	          noisesigma_[hfhi_[ic]]=noiseInfCfromDB(theDbService,theDetIds_[hfhi_[ic]])*mgain*0.25;
	        }
	      //      std::cout << " NOISEHF " << theDetIds_[hfhi_[ic]].ieta() << " " << noisesigma_[hfhi_[ic]] << "  "<< std::endl;
          
	      mgain*=2500./0.36;
	      sat_[hfhi_[ic]]=(doSaturation_)?mgain:99999.;
	      int ieta=theDetIds_[hfhi_[ic]].ieta();
	      gain*=TPGFactor_[(ieta>0)?ieta+45:-ieta+2];
	      peds_[hfhi_[ic]]=ped;
	      gains_[hfhi_[ic]]=gain;
	    }

      initializedHF_=true;
	}
  
  // clear the vector we don't need. It is a bit stupid 
  hcalRecHits_.resize(maxIndex_+1,0.);

}


// Get the PCaloHits from the event. They have to be stored in a map, because when
// the pile-up is added thanks to the Mixing Module, the same cell can be present several times
void HcalRecHitsMaker::loadPCaloHits(const edm::Event & iEvent, const HcalTopology& topo)
{
  clean();

  edm::Handle<CrossingFrame<PCaloHit> > cf;
  iEvent.getByLabel(inputCol_,cf);
  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),std::pair<int,int>(0,0) ));

  MixCollection<PCaloHit>::iterator it=colcalo->begin();;
  MixCollection<PCaloHit>::iterator itend=colcalo->end();
  unsigned counter=0;
  for(;it!=itend;++it)
    {
      HcalDetId detid(it->id());
      int hashedindex=topo.detId2denseId(detid);

      // apply ToF correction
      int time_slice=0; // temporary
      double fTOF=1.;
      if (detid.subdet()==HcalForward) fTOF = (time_slice==0) ? 1. : 0.;	
      else fTOF = fractionOOT(time_slice);

      switch(detid.subdet())
	{
	case HcalBarrel: 
	  {
	    if(det_==4)
	      Fill(hashedindex,fTOF*(it->energy()),firedCells_,noise_[0],corrfac_[0]);
	  }
	  break;
	case HcalEndcap: 
	  {	  
	    if(det_==4)
	      Fill(hashedindex,fTOF*(it->energy()),firedCells_,noise_[1],corrfac_[1]);
	  }
	  break;
	case HcalOuter: 
	  {
	    if(det_==5)
	      Fill(hashedindex,fTOF*(it->energy()),firedCells_,noise_[0],corrfac_[0]);
	  }
	  break;		     
	case HcalForward: 
	  {
	    if(det_==6 && time_slice==0) // skip the HF hit if out-of-time
	      Fill(hashedindex,it->energy(),firedCells_,noise_[0],corrfac_[0]);
	  }
	  break;
	default:
	  edm::LogWarning("CaloRecHitsProducer") << "RecHit not registered\n";
	  ;
	}
      ++counter;
    }
}

// Fills the collections. 
void HcalRecHitsMaker::loadHcalRecHits(edm::Event &iEvent,const HcalTopology& topo, HBHERecHitCollection& hbheHits, HBHEDigiCollection& hbheDigis)
{
  loadPCaloHits(iEvent,topo);
  noisify();
  hbheHits.reserve(firedCells_.size());
  if(doDigis_)
    {
      hbheDigis.reserve(firedCells_.size());
    }
  static HcalQIESample zeroSample(0,0,0,0);
  unsigned nhits=firedCells_.size();
  // HB and HE

  for(unsigned ihit=0;ihit<nhits;++ihit)
    {
      unsigned cellhashedindex=firedCells_[ihit];
      const HcalDetId& detid  = theDetIds_[cellhashedindex];
      unsigned subdet=(detid.subdet()==HcalBarrel) ? 0: 1;	

      float energy=hcalRecHits_[cellhashedindex];
      // Check if it is above the threshold
      if(energy<threshold_[subdet]) continue; 
      // apply RespCorr only to the RecHit
      energy *= myRespCorr->getValues(theDetIds_[cellhashedindex])->getValue();
      // poor man saturation
      if(energy>sat_[cellhashedindex]) 
	{
	  //	  std::cout << " Saturation " << energy << " " << sat_[cellhashedindex] << " HB " << std::endl;
	  energy=sat_[cellhashedindex];
	}
      

      hbheHits.push_back(HBHERecHit(detid,energy,0.));      
      if(doDigis_)
	{
	  HBHEDataFrame myDataFrame(detid);
	  myDataFrame.setSize(2);
	  double nfc=hcalRecHits_[cellhashedindex]/gains_[cellhashedindex]+peds_[cellhashedindex];
	  int nadc=fCtoAdc(nfc);
	  HcalQIESample qie(nadc, 0, 0, 0) ;
	  myDataFrame.setSample(0,qie);
	  myDataFrame.setSample(1,zeroSample);
	  hbheDigis.push_back(myDataFrame);
	}
    }
}


// Fills the collections. 
void HcalRecHitsMaker::loadHcalRecHits(edm::Event &iEvent,const HcalTopology& topo, HORecHitCollection &hoHits, HODigiCollection & hoDigis)
{
  loadPCaloHits(iEvent,topo);
  noisify();
  hoHits.reserve(firedCells_.size());
  if(doDigis_)
    {
      hoDigis.reserve(firedCells_.size());
    }
  static HcalQIESample zeroSample(0,0,0,0);

  // HO
  unsigned nhits=firedCells_.size();
  for(unsigned ihit=0;ihit<nhits;++ihit)
    {

      unsigned cellhashedindex=firedCells_[ihit];
      // Check if it is above the threshold
      if(hcalRecHits_[cellhashedindex]<threshold_[0]) continue; 

      const HcalDetId&  detid=theDetIds_[cellhashedindex];
      int ieta = detid.ieta();
      
      // Force  only Ring#0 to remain
      if(ieta > 4 || ieta < -4 ) continue;

      float energy=hcalRecHits_[cellhashedindex];
      // apply RespCorr
      energy *= myRespCorr->getValues(theDetIds_[cellhashedindex])->getValue();

      // poor man saturation
      if(energy>sat_[cellhashedindex]) energy=sat_[cellhashedindex];

      hoHits.push_back(HORecHit(detid,energy,0));
    }
}

// Fills the collections. 
void HcalRecHitsMaker::loadHcalRecHits(edm::Event &iEvent,const HcalTopology& topo,HFRecHitCollection &hfHits, HFDigiCollection& hfDigis)
{
  loadPCaloHits(iEvent,topo);
  noisify();
  hfHits.reserve(firedCells_.size());
  if(doDigis_)
    {
      hfDigis.reserve(firedCells_.size());
    }
  static HcalQIESample zeroSample(0,0,0,0);

  unsigned nhits=firedCells_.size();
  for(unsigned ihit=0;ihit<nhits;++ihit)
    {
      unsigned cellhashedindex=firedCells_[ihit];
      // Check if it is above the threshold
      if(hcalRecHits_[cellhashedindex]<threshold_[0]) continue; 

      float energy=hcalRecHits_[cellhashedindex];

      // apply RespCorr
      energy *= myRespCorr->getValues(theDetIds_[cellhashedindex])->getValue();

      // poor man saturation
      if(energy>sat_[cellhashedindex]) energy=sat_[cellhashedindex];

      const HcalDetId & detid=theDetIds_[cellhashedindex];
      hfHits.push_back(HFRecHit(detid,energy,0.));      
      if(doDigis_)
	{
	  HFDataFrame myDataFrame(detid);
	  myDataFrame.setSize(1);

	  double nfc= hcalRecHits_[cellhashedindex]/gains_[cellhashedindex]+peds_[cellhashedindex];
	  int nadc=fCtoAdc(nfc/2.6);
	  HcalQIESample qie(nadc, 0, 0, 0) ;
	  myDataFrame.setSample(0,qie);
	  hfDigis.push_back(myDataFrame);
	}
    }
}


// For a fast injection of the noise: the list of cell ids is stored
unsigned HcalRecHitsMaker::createVectorsOfCells(const edm::EventSetup &es)
{
    edm::ESHandle<CaloGeometry> pG;
    es.get<CaloGeometryRecord>().get(pG);     
    edm::ESHandle<HcalTopology> topo;
    es.get<HcalRecNumberingRecord>().get( topo );

    nhbcells_ = createVectorOfSubdetectorCells(*pG, *topo, HcalBarrel,  hbhi_);    
    nhecells_ = createVectorOfSubdetectorCells(*pG, *topo, HcalEndcap,  hehi_);
    nhocells_ = createVectorOfSubdetectorCells(*pG, *topo, HcalOuter,   hohi_);
    nhfcells_ = createVectorOfSubdetectorCells(*pG, *topo, HcalForward, hfhi_);    

    return nhbcells_+nhecells_+nhocells_+nhfcells_;
}

// list of the cellids for a given subdetector
unsigned HcalRecHitsMaker::createVectorOfSubdetectorCells(const CaloGeometry& cg, const HcalTopology& topo, int subdetn,std::vector<int>& cellsvec ) 
{

  const CaloSubdetectorGeometry* geom=cg.getSubdetectorGeometry(DetId::Hcal,subdetn);  
  const std::vector<DetId>& ids=geom->getValidDetIds(DetId::Hcal,subdetn);  

  for (std::vector<DetId>::const_iterator i=ids.begin(); i!=ids.end(); i++) 
    {
      HcalDetId myDetId(*i);
      //      std::cout << myDetId << myHcalSimParameterMap_->simParameters(myDetId).simHitToPhotoelectrons() << std::endl;;
      //      std::cout << " hi " << hi << " " theDetIds_.size() << std::endl;
      unsigned int hi=topo.detId2denseId(myDetId);
      theDetIds_[hi]=myDetId;
      //      std::cout << myDetId << " " << hi <<  std::endl;
      cellsvec.push_back(hi);      

      if(hi>maxIndex_)
	maxIndex_=hi;
    }
  return cellsvec.size();
}

// Takes a hit (from a PSimHit) and fills a map 
void HcalRecHitsMaker::Fill(int id, float energy, std::vector<int>& theHits,float noise,float correctionfactor)
{
  if(doMiscalib_) 
    energy*=miscalib_[id];

  if(noise==-1)
    noise=noisesigma_[id]*correctionfactor;

  // Check if the RecHit exists
  if(hcalRecHits_[id]>0.)
    hcalRecHits_[id]+=energy;
  else
    {
      // the noise is injected only the first time
      hcalRecHits_[id]=energy + random_->gaussShoot(0.,noise);
      theHits.push_back(id);
    }
}

void HcalRecHitsMaker::noisify()
{
  unsigned total=0;
  switch(det_)
    {
    case 4:
      {
	// do the HB
	if(noise_[0] != 0.) {
	  total+=noisifySubdet(hcalRecHits_,firedCells_,hbhi_,nhbcells_,hcalHotFraction_[0],myGaussianTailGenerators_[0],noise_[0],threshold_[0],corrfac_[0]);
	}
	// do the HE
	if(noise_[1] != 0.) {
	  total+=noisifySubdet(hcalRecHits_,firedCells_,hehi_,nhecells_,hcalHotFraction_[1],myGaussianTailGenerators_[1],noise_[1],threshold_[1],corrfac_[1]);
	}
      }
      break;
    case 5:
      {
	// do the HO
	if(noise_[0] != 0.) {
	  total+=noisifySubdet(hcalRecHits_,firedCells_,hohi_,nhocells_,hcalHotFraction_[0],myGaussianTailGenerators_[0],noise_[0],threshold_[0],corrfac_[0]);
	}
      }
      break;
    case 6:
      {
	// do the HF
	if(noise_[0] != 0.) {
	  total+=noisifySubdet(hcalRecHits_,firedCells_,hfhi_,nhfcells_,hcalHotFraction_[0],myGaussianTailGenerators_[0],noise_[0],threshold_[0],corrfac_[0]);
	}
      }
      break;
    default:
      break;
    }
  edm::LogInfo("CaloRecHitsProducer") << "CaloRecHitsProducer : added noise in "<<  total << " HCAL cells "  << std::endl;
}

unsigned HcalRecHitsMaker::noisifySubdet(std::vector<float>& theMap, std::vector<int>& theHits, const std::vector<int>& thecells, unsigned ncells, double hcalHotFraction,const GaussianTail *myGT,double noise_,double threshold,double correctionfactor)
{
 // If the fraction of "hot " is small, use an optimized method to inject noise only in noisy cells. The 30% has not been tuned
  if(noise_!=-1 && hcalHotFraction==0.) return 0;
  if(hcalHotFraction<0.3 && noise_!=-1)
    {
      double mean = (double)(ncells-theHits.size())*hcalHotFraction;
      unsigned nhcal = random_->poissonShoot(mean);
  
      unsigned ncell=0;
      unsigned cellindex=0;
      uint32_t cellhashedindex=0;
      
      while(ncell < nhcal)
	{
	  cellindex = (unsigned)(random_->flatShoot()*ncells);
	  cellhashedindex = thecells[cellindex];

	  if(hcalRecHits_[cellhashedindex]==0.) // new cell
	    {
	      hcalRecHits_[cellhashedindex]=myGT->shoot();
	      theHits.push_back(cellhashedindex);
	      ++ncell;
	    }
	}
      return ncell;
    }
  else // otherwise, inject noise everywhere
    {
      uint32_t cellhashedindex=0;
      unsigned nhcal=thecells.size();

      double sigma = noise_;
      for(unsigned ncell=0;ncell<nhcal;++ncell)
	{
	  cellhashedindex = thecells[ncell];
	  if(hcalRecHits_[cellhashedindex]==0.) // new cell
	    { 
		  //only use noisesigma if it has been filled from the DB
		  //otherwise, leave sigma alone (default is user input value)
	      if(noise_==-1) sigma=noisesigma_[cellhashedindex]*correctionfactor;

	      double noise =random_->gaussShoot(0.,sigma);
	      if(noise>threshold)
		{
		  hcalRecHits_[cellhashedindex]=noise;		    
		  theHits.push_back(cellhashedindex);
		}
	    }
	}
      return nhcal;
    }
  return 0;
}

void HcalRecHitsMaker::clean()
{
  cleanSubDet(hcalRecHits_,firedCells_);
}

void HcalRecHitsMaker::cleanSubDet(std::vector<float>& hits,std::vector<int>& cells)
{
  unsigned size=cells.size();
  // Reset the energies
  for(unsigned ic=0;ic<size;++ic)
    {
      hits[cells[ic]] = 0.;
    }
  // Clear the list of fired cells 
  cells.clear();
}

// fC to ADC conversion
int HcalRecHitsMaker::fCtoAdc(double fc) const
{
  if(fc<0.) return 0;
  if(fc>9985.) return 127;
  return fctoadc_[(unsigned)floor(fc)];
}

double HcalRecHitsMaker::noiseInfCfromDB(const HcalDbService * conditions,const HcalDetId & detId)
{
  // method from Salavat
  // fetch pedestal widths (noise sigmas for all 4 CapID)
  const HcalPedestalWidth* pedWidth =
    conditions-> getPedestalWidth(detId); 
  double ssqq_1 = pedWidth->getSigma(0,0);
  double ssqq_2 = pedWidth->getSigma(1,1);
  double ssqq_3 = pedWidth->getSigma(2,2);
  double ssqq_4 = pedWidth->getSigma(3,3);

  // correction factors (hb,he,ho,hf)
  //  static float corrfac[4]={1.39,1.32,1.17,3.76};
  //  static float corrfac[4]={0.93,0.88,1.17,2.51}; // divided HB, HE, HF by 1.5 (to take into account the halving of the number of time slices), HO did not change

  int sub   = detId.subdet();

  // HO: only Ring#0 matters 
  int ieta  = detId.ieta();
  if (sub == 3 && abs (ieta) > 4) return 0.;   

  // effective RecHits (for this particular detId) noise calculation :
  
  double sig_sq_mean =  0.25 * ( ssqq_1 + ssqq_2 + ssqq_3 + ssqq_4);
  
  // f - external parameter (currently set to 0.5 in the FullSim) !!!
  double f=0.5;

  double term  = sqrt (1. + sqrt(1. - 2.*f*f));
  double alpha = sqrt (0.5) * term;
  double beta  = sqrt (0.5) * f / term;

  //  double RMS1   = sqrt(sig_sq_mean) * sqrt (2.*beta*beta + alpha*alpha) ;
  double RMS4   = sqrt(sig_sq_mean) * sqrt (2.*beta*beta + 2.*(alpha-beta)*(alpha-beta) + 2.*(alpha-2.*beta)*(alpha-2.*beta)) ;

  double noise_rms_fC;

  //  if(sub == 4)  noise_rms_fC = RMS1;
  //  else          noise_rms_fC = RMS4;
  noise_rms_fC = RMS4;


  // to convert from above fC to GeV - multiply by gain (GeV/fC)        
  //  const HcalGain*  gain = conditions->getGain(detId); 
  //  double noise_rms_GeV = noise_rms_fC * gain->getValue(0); // Noise RMS (GeV) for detId channel
  return noise_rms_fC;
}

// fraction of energy collected as a function of ToF (for out-of-time particles; use case is out-of-time pileup)
double HcalRecHitsMaker::fractionOOT(int time_slice)// in units of 25 ns; 0 means in-time
{
  double SF = 100./88.; // to normalize to in-time signal (88% is in the reco window)
  if (time_slice==-4) {
    return 0.02*SF;
  } else if (time_slice==-3) {
    return 0.06*SF;
  } else if (time_slice==-2) {
    return 0.19*SF;
  } else if (time_slice==-1) {
    return 0.24*SF;
  } else if (time_slice==0) {
    return 1.;
  } else if (time_slice==1) {
    return 0.70*SF;
  } else {
    return 0.;
  }

}
