#include "FastSimulation/CaloRecHitsProducer/interface/HcalRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
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
#include "TFile.h"
#include "TGraph.h"
#include "TROOT.h"
#include <fstream>

class RandomEngine;


HcalRecHitsMaker::HcalRecHitsMaker(edm::ParameterSet const & p1,edm::ParameterSet const & p2,
				   const RandomEngine * myrandom)
  :
  initialized_(false),
  random_(myrandom),
  myGaussianTailGeneratorHB_(0),  myGaussianTailGeneratorHE_(0),  myGaussianTailGeneratorHO_(0),  myGaussianTailGeneratorHF_(0),myHcalSimParameterMap_(0)
{
  edm::ParameterSet RecHitsParameters = p1.getParameter<edm::ParameterSet>("HCAL");
  noiseHB_ = RecHitsParameters.getParameter<double>("NoiseHB");
  noiseHE_ = RecHitsParameters.getParameter<double>("NoiseHE");
  noiseHO_ = RecHitsParameters.getParameter<double>("NoiseHO");
  noiseHF_ = RecHitsParameters.getParameter<double>("NoiseHF");
  thresholdHB_ = RecHitsParameters.getParameter<double>("ThresholdHB");
  thresholdHE_ = RecHitsParameters.getParameter<double>("ThresholdHE");
  thresholdHO_ = RecHitsParameters.getParameter<double>("ThresholdHO");
  thresholdHF_ = RecHitsParameters.getParameter<double>("ThresholdHF");

  
  satHB_ = RecHitsParameters.getParameter<double>("SaturationHB");
  satHE_ = RecHitsParameters.getParameter<double>("SaturationHE");
  satHO_ = RecHitsParameters.getParameter<double>("SaturationHO");
  satHF_ = RecHitsParameters.getParameter<double>("SaturationHF");

  refactor_ = RecHitsParameters.getParameter<double> ("Refactor");
  refactor_mean_ = RecHitsParameters.getParameter<double> ("Refactor_mean");
  hcalfileinpath_= RecHitsParameters.getParameter<std::string> ("fileNameHcal");  
  inputCol_=RecHitsParameters.getParameter<edm::InputTag>("MixedSimHits");
  maxIndex_=0;
  maxIndexDebug_=0;
  theDetIds_.resize(10000);

  peds_.resize(10000);
  gains_.resize(10000);
  hbhi_.reserve(2600);
  hehi_.reserve(2600);
  hohi_.reserve(2200);
  hfhi_.reserve(1800);
  doDigis_=false;

  //  edm::ParameterSet hcalparam = p2.getParameter<edm::ParameterSet>("hcalSimParam"); 
  //  myHcalSimParameterMap_ = new HcalSimParameterMap(hcalparam);

  // Computes the fraction of HCAL above the threshold
  Genfun::Erf myErf;

  if(noiseHB_>0.) {
    hcalHotFractionHB_ = 0.5-0.5*myErf(thresholdHB_/noiseHB_/sqrt(2.));
    myGaussianTailGeneratorHB_ = new GaussianTail(random_,noiseHB_,thresholdHB_);
  } else {
    hcalHotFractionHB_ =0.;
  }

  if(noiseHO_>0.) {
    hcalHotFractionHO_ = 0.5-0.5*myErf(thresholdHO_/noiseHO_/sqrt(2.));
    myGaussianTailGeneratorHO_ = new GaussianTail(random_,noiseHO_,thresholdHO_);
  } else {
    hcalHotFractionHO_ =0.;
  }

  if(noiseHE_>0.) {
    hcalHotFractionHE_ = 0.5-0.5*myErf(thresholdHE_/noiseHE_/sqrt(2.));
    myGaussianTailGeneratorHE_ = new GaussianTail(random_,noiseHE_,thresholdHE_);
  } else {
    hcalHotFractionHE_ =0.;
  }

  if(noiseHF_>0.) {
    hcalHotFractionHF_ = 0.5-0.5*myErf(thresholdHF_/noiseHF_/sqrt(2.));
    myGaussianTailGeneratorHF_ = new GaussianTail(random_,noiseHF_,thresholdHF_);
  } else {
    hcalHotFractionHF_ =0.;
  }

  
}

HcalRecHitsMaker::~HcalRecHitsMaker()
{
  clean();  
  if(myGaussianTailGeneratorHB_) delete myGaussianTailGeneratorHB_;
  if(myGaussianTailGeneratorHE_) delete myGaussianTailGeneratorHE_;
  if(myGaussianTailGeneratorHO_) delete myGaussianTailGeneratorHO_;
  if(myGaussianTailGeneratorHF_) delete myGaussianTailGeneratorHF_;
  if(myHcalSimParameterMap_) delete myHcalSimParameterMap_;
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
  if(initialized_) return;


  unsigned ncells=createVectorsOfCells(es);
  edm::LogInfo("CaloRecHitsProducer") << "Total number of cells in HCAL " << ncells << std::endl;
  hcalRecHits_.resize(maxIndex_+1,0.);
  edm::LogInfo("CaloRecHitsProducer") << "Largest HCAL hashedindex" << maxIndex_ << std::endl;

  if(doMiscalib_)
    {
      miscalib_.resize(maxIndex_+1,1.);
      // Read from file ( a la HcalRecHitsRecalib.cc)
      // here read them from xml (particular to HCAL)
      CaloMiscalibMapHcal mapHcal;
      mapHcal.prefillMap();

      edm::FileInPath hcalfiletmp("CalibCalorimetry/CaloMiscalibTools/data/"+hcalfileinpath_);      
      std::string hcalfile=hcalfiletmp.fullPath();            
      MiscalibReaderFromXMLHcal hcalreader_(mapHcal);
      if(!hcalfile.empty()) 
	{
	  hcalreader_.parseXMLMiscalibFile(hcalfile);
	  mapHcal.print();
	  std::map<uint32_t,float>::const_iterator it=mapHcal.get().begin();
	  std::map<uint32_t,float>::const_iterator itend=mapHcal.get().end();
	  for(;it!=itend;++it)
	    {
	      HcalDetId myDetId(it->first);
	      float icalconst=it->second;
	      miscalib_[myDetId.hashed_index()]=refactor_mean_+(icalconst-refactor_mean_)*refactor_;
	    }
	}
    }




  if(!doDigis_)
    {
      initialized_=true;
      return;
    }

// Will be needed for the DB-based miscalibration
  std::cout << " Getting HcalDb service " ;
  edm::ESHandle<HcalDbService> conditions;
  es.get<HcalDbRecord>().get(conditions);
  const HcalDbService * theDbService=conditions.product();
  //  myHcalSimParameterMap_->setDbService(theDbService);
  std::cout << " - done " << std::endl;
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
	  //	  std::cout << TPGFactor_[i] << std::endl;
	}
    }
  else
    {
      std::cout << " Unable to open CalibCalorimetry/HcalTPGAlgos/data/RecHit-TPG-calib.dat" << std::endl;
      std::cout <<	" Using a constant 1.2 factor " << std::endl;
    }
  
  //HB
  for(unsigned ic=0;ic<nhbcells_;++ic)
    {
      float gain = theDbService->getGain(theDetIds_[hbhi_[ic]])->getValue(0);
      peds_[hbhi_[ic]]=theDbService->getPedestal(theDetIds_[hbhi_[ic]])->getValue(0);
      int ieta=theDetIds_[hbhi_[ic]].ieta();
      gain*=TPGFactor_[(ieta>0)?ieta+43:-ieta];
      gains_[hbhi_[ic]]=gain;
    }
  //HE
  for(unsigned ic=0;ic<nhecells_;++ic)
    {
      float gain= theDbService->getGain(theDetIds_[hehi_[ic]])->getValue(0);
      int ieta=theDetIds_[hehi_[ic]].ieta();

      gain*=TPGFactor_[(ieta>0)?ieta+44:-ieta+1];
//      if(abs(ieta)>=21&&abs(ieta)<=26)
//	gain*=2.;
//      if(abs(ieta)>=27&&abs(ieta)<=29)
//	gain*=5.;
      peds_[hehi_[ic]]=theDbService->getPedestal(theDetIds_[hehi_[ic]])->getValue(0);
      gains_[hehi_[ic]]=gain;
    }
//HO
  for(unsigned ic=0;ic<nhocells_;++ic)
    {
      float ped=theDbService->getPedestal(theDetIds_[hohi_[ic]])->getValue(0);
      float gain=theDbService->getGain(theDetIds_[hohi_[ic]])->getValue(0);
      int ieta=HcalDetId(hohi_[ic]).ieta();
      gain*=TPGFactor_[(ieta>0)?ieta+43:-ieta];
      peds_[hohi_[ic]]=ped;
      gains_[hohi_[ic]]=gain;
    }
  //HF
  for(unsigned ic=0;ic<nhfcells_;++ic)
    {
      float ped=theDbService->getPedestal(theDetIds_[hfhi_[ic]])->getValue(0);
      float gain=theDbService->getGain(theDetIds_[hfhi_[ic]])->getValue(0);
      int ieta=theDetIds_[hfhi_[ic]].ieta();
      gain*=TPGFactor_[(ieta>0)?ieta+45:-ieta+2];
      peds_[hfhi_[ic]]=ped;
      gains_[hfhi_[ic]]=gain;
    }
  initialized_=true; 
  
}


// Get the PCaloHits from the event. They have to be stored in a map, because when
// the pile-up is added thanks to the Mixing Module, the same cell can be present several times
void HcalRecHitsMaker::loadPCaloHits(const edm::Event & iEvent)
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
      int hashedindex=detid.hashed_index();
      double noise_ = 0.;
      switch(detid.subdet())
	{
	case HcalBarrel: 
	  {
	    noise_ = noiseHB_; 
	    Fill(hashedindex,it->energy(),firedCellsHB_,noise_);
	  }
	  break;
	case HcalEndcap: 
	  {	  
	    noise_ = noiseHE_; 
	    Fill(hashedindex,it->energy(),firedCellsHE_,noise_);
	  }
	  break;
	case HcalOuter: 
	  {
	    noise_ = noiseHO_; 
	    Fill(hashedindex,it->energy(),firedCellsHO_,noise_);
	  }
	  break;		     
	case HcalForward: 
	  {
	    noise_ = noiseHF_; 
	    Fill(hashedindex,it->energy(),firedCellsHF_,noise_);
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
void HcalRecHitsMaker::loadHcalRecHits(edm::Event &iEvent,HBHERecHitCollection& hbheHits, HORecHitCollection &hoHits,HFRecHitCollection &hfHits, HBHEDigiCollection& hbheDigis, HODigiCollection & hoDigis, HFDigiCollection& hfDigis)
{
  loadPCaloHits(iEvent);
  noisify();
  hbheHits.reserve(firedCellsHB_.size()+firedCellsHE_.size());
  hoHits.reserve(firedCellsHO_.size());
  hfHits.reserve(firedCellsHF_.size());
  if(doDigis_)
    {
      hbheDigis.reserve(firedCellsHB_.size()+firedCellsHE_.size());
      hfDigis.reserve(firedCellsHF_.size());
      hoDigis.reserve(firedCellsHO_.size());
    }
  static HcalQIESample zeroSample(0,0,0,0);
  unsigned nhits=firedCellsHB_.size();
  // HB
  for(unsigned ihit=0;ihit<nhits;++ihit)
    {
      unsigned cellhashedindex=firedCellsHB_[ihit];
      // Check if it is above the threshold
      if(hcalRecHits_[cellhashedindex]<thresholdHB_) continue; 
      float energy=hcalRecHits_[cellhashedindex];
      // poor man saturation
      if(energy>satHB_) energy=satHB_;

      const HcalDetId& detid  = theDetIds_[cellhashedindex];
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
      
  // HE
  nhits=firedCellsHE_.size();
  for(unsigned ihit=0;ihit<nhits;++ihit)
    {
      unsigned cellhashedindex=firedCellsHE_[ihit];
      // Check if it is above the threshold
      if(hcalRecHits_[cellhashedindex]<thresholdHE_) continue; 
      float energy=hcalRecHits_[cellhashedindex];
      // poor man saturation
      if(energy>satHE_) energy=satHE_;

      const HcalDetId & detid= theDetIds_[cellhashedindex];
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


  // HO
  nhits=firedCellsHO_.size();
  for(unsigned ihit=0;ihit<nhits;++ihit)
    {
      unsigned cellhashedindex=firedCellsHO_[ihit];
      // Check if it is above the threshold
      if(hcalRecHits_[cellhashedindex]<thresholdHO_) continue; 

      float energy=hcalRecHits_[cellhashedindex];
      // poor man saturation
      if(energy>satHO_) energy=satHO_;

      const HcalDetId&  detid=theDetIds_[cellhashedindex];
      hoHits.push_back(HORecHit(detid,energy,0));
    }
  
  // HF
  nhits=firedCellsHF_.size();
  for(unsigned ihit=0;ihit<nhits;++ihit)
    {
      unsigned cellhashedindex=firedCellsHF_[ihit];
      // Check if it is above the threshold
      if(hcalRecHits_[cellhashedindex]<thresholdHF_) continue; 

      float energy=hcalRecHits_[cellhashedindex];
      // poor man saturation
      if(energy>satHF_) energy=satHF_;

      const HcalDetId & detid=theDetIds_[cellhashedindex];
      hfHits.push_back(HFRecHit(detid,energy,0.));      
      if(doDigis_)
	{
	  HFDataFrame myDataFrame(detid);
	  myDataFrame.setSize(1);
	  double nfc=hcalRecHits_[cellhashedindex]/gains_[cellhashedindex]+peds_[cellhashedindex];
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
    es.get<IdealGeometryRecord>().get(pG);     
    nhbcells_ = createVectorOfSubdetectorCells(*pG, HcalBarrel,  hbhi_);
    nhecells_ = createVectorOfSubdetectorCells(*pG, HcalEndcap,  hehi_);
    nhocells_ = createVectorOfSubdetectorCells(*pG, HcalOuter,   hohi_);
    nhfcells_ = createVectorOfSubdetectorCells(*pG, HcalForward, hfhi_);    
    return nhbcells_+nhecells_+nhocells_+nhfcells_;
}

// list of the cellids for a given subdetector
unsigned HcalRecHitsMaker::createVectorOfSubdetectorCells(const CaloGeometry& cg,int subdetn,std::vector<int>& cellsvec ) 
{
  const CaloSubdetectorGeometry* geom=cg.getSubdetectorGeometry(DetId::Hcal,subdetn);  
  std::vector<DetId> ids=geom->getValidDetIds(DetId::Hcal,subdetn);  
  for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) 
    {
      HcalDetId myDetId(*i);
      //      std::cout << myDetId << myHcalSimParameterMap_->simParameters(myDetId).simHitToPhotoelectrons() << std::endl;;
      unsigned hi=myDetId.hashed_index();
      theDetIds_[hi]=myDetId;
      //      std::cout << myDetId << " " << hi <<  std::endl;
      cellsvec.push_back(hi);      

      if(hi>maxIndex_)
	maxIndex_=hi;
    }
  return cellsvec.size();
}

// Takes a hit (from a PSimHit) and fills a map 
void HcalRecHitsMaker::Fill(int id, float energy, std::vector<int>& theHits,float noise)
{
  if(doMiscalib_) 
    energy*=miscalib_[id];
  // Check if the RecHit exists
  if(hcalRecHits_[id]>0.)
    hcalRecHits_[id]+=energy;
  else
    {
      // the noise is injected only in the first cell 
      hcalRecHits_[id]=energy + random_->gaussShoot(0.,noise);
      theHits.push_back(id);
    }
}

void HcalRecHitsMaker::noisify()
{
  unsigned total=0;
  if(noiseHB_ > 0.) {
    if(firedCellsHB_.size()<nhbcells_)
      {
	// No need to do it anymore. The noise on the signal has been added 
	// when loading the PCaloHits
	// noisifySignal(hbheRecHits_);      
	total+=noisifySubdet(hcalRecHits_,firedCellsHB_,hbhi_,nhbcells_,hcalHotFractionHB_,myGaussianTailGeneratorHB_);
      }
    else
      edm::LogWarning("CaloRecHitsProducer") << "All HCAL (HB) cells on ! " << std::endl;
  }


  if(noiseHE_ > 0.) {
    if(firedCellsHE_.size()<nhecells_)
      {
	// No need to do it anymore. The noise on the signal has been added 
	// when loading the PCaloHits
	// noisifySignal(hbheRecHits_);      
	total+=noisifySubdet(hcalRecHits_,firedCellsHE_,hehi_,nhecells_,hcalHotFractionHE_,myGaussianTailGeneratorHE_);
      }
    else
      edm::LogWarning("CaloRecHitsProducer") << "All HCAL (HE) cells on ! " << std::endl;
  }

  if(noiseHO_ > 0.) {
    if( firedCellsHO_.size()<nhocells_)
      {
	//      noisifySignal(hoRecHits_);
	total+=noisifySubdet(hcalRecHits_,firedCellsHO_,hohi_,nhocells_,hcalHotFractionHO_,myGaussianTailGeneratorHO_);
      }
    else
      edm::LogWarning("CaloRecHitsProducer") << "All HCAL(HO) cells on ! " << std::endl;
  }
   
  if(noiseHF_ > 0.) {
    if(firedCellsHF_.size()<nhfcells_)
      {
	//      noisifySignal(hfRecHits_);
	total+=noisifySubdet(hcalRecHits_,firedCellsHF_,hfhi_,nhfcells_,hcalHotFractionHF_,myGaussianTailGeneratorHF_);
      }
    else
      edm::LogWarning("CaloRecHitsProducer") << "All HCAL(HF) cells on ! " << std::endl;
  }

   edm::LogInfo("CaloRecHitsProducer") << "CaloRecHitsProducer : added noise in "<<  total << " HCAL cells "  << std::endl;
}

unsigned HcalRecHitsMaker::noisifySubdet(std::vector<float>& theMap, std::vector<int>& theHits, const std::vector<int>& thecells, unsigned ncells, double hcalHotFraction,const GaussianTail *myGT)
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
  return nhcal;
}

void HcalRecHitsMaker::clean()
{
  cleanSubDet(hcalRecHits_,firedCellsHB_);
  cleanSubDet(hcalRecHits_,firedCellsHE_);
  cleanSubDet(hcalRecHits_,firedCellsHO_);
  cleanSubDet(hcalRecHits_,firedCellsHF_);
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
