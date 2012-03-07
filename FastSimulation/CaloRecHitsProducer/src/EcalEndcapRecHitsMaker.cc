#include "FastSimulation/CaloRecHitsProducer/interface/EcalEndcapRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/Utilities/interface/GaussianTail.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "CLHEP/GenericFunctions/Erf.hh"

#include <algorithm>

EcalEndcapRecHitsMaker::EcalEndcapRecHitsMaker(edm::ParameterSet const & p, 

					       const RandomEngine * myrandom) 
  : random_(myrandom)
{
  edm::ParameterSet RecHitsParameters=p.getParameter<edm::ParameterSet>("ECALEndcap");
  inputCol_=RecHitsParameters.getParameter<edm::InputTag>("MixedSimHits");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
  SRThreshold_ = RecHitsParameters.getParameter<double> ("SRThreshold");
  refactor_ = RecHitsParameters.getParameter<double> ("Refactor");
  refactor_mean_ = RecHitsParameters.getParameter<double> ("Refactor_mean");
  noiseADC_ = RecHitsParameters.getParameter<double>("NoiseADC");
  highNoiseParameters_ = RecHitsParameters.getParameter<std::vector<double> > ("HighNoiseParameters");

  theCalorimeterHits_.resize(EEDetId::kSizeForDenseIndexing,0.);
  applyZSCells_.resize(EEDetId::kSizeForDenseIndexing,true);
  towerOf_.resize(EEDetId::kSizeForDenseIndexing);
  theTTDetIds_.resize(1440);
  SCofTT_.resize(1440);
  SCHighInterest_.resize(633,0);
  treatedSC_.resize(633,false);
  TTofSC_.resize(633);
  TTTEnergy_.resize(1440,0.);
  CrystalsinSC_.resize(633);
  sinTheta_.resize(EEDetId::kEEhalf,0.);
  doCustomHighNoise_=false;
  
  noisified_ = (noise_==0.);
  edm::ParameterSet CalibParameters=RecHitsParameters.getParameter<edm::ParameterSet>("ContFact"); 
  double c1 = CalibParameters.getParameter<double>("EEs25notContainment");
  calibfactor_= 1./c1;
}
  

EcalEndcapRecHitsMaker::~EcalEndcapRecHitsMaker()
{;
}

void EcalEndcapRecHitsMaker::clean()
{

  unsigned size=theFiredCells_.size();
  for(unsigned ic=0;ic<size;++ic)
    {
      theCalorimeterHits_[theFiredCells_[ic]] = 0.;
      applyZSCells_[theFiredCells_[ic]] = false;
    }
  theFiredCells_.clear();
  // If the noise is set to 0. No need to simulate it. 
  noisified_ = (noise_==0.);

  size=theFiredTTs_.size();

  for(unsigned itt=0;itt<size;++itt)
    {
      //      std::cout << " TT " << theFiredTTs_[itt] << " " << TTTEnergy_[theFiredTTs_[itt]] << std::endl;
      TTTEnergy_[theFiredTTs_[itt]]=0.;
    }  
  theFiredTTs_.clear();
 
  size=theFiredSC_.size();
  for(unsigned isc=0;isc<size;++isc)
    {
      SCHighInterest_[theFiredSC_[isc]]=0;
      treatedSC_[theFiredSC_[isc]]=false;
    }
  theFiredSC_.clear();



//  for(unsigned ic=0;ic<TTTEnergy_.size();++ic)
//    if(TTTEnergy_[ic]!=0.) std::cout << " TT " << ic << " not cleaned " << std::endl;
//  for(unsigned ic=0;ic<SCHighInterest_.size();++ic)
//    if(SCHighInterest_[ic]!=0) std::cout << " SCHighInterest " << ic << SCHighInterest_[ic] << " not cleaned " << std::endl;
//  for(unsigned ic=0;ic<treatedSC_.size();++ic)
//    if(treatedSC_[ic]) std::cout << " treatedSC " << ic << treatedSC_[ic] << " not cleaned " << std::endl;
}


void EcalEndcapRecHitsMaker::loadEcalEndcapRecHits(edm::Event &iEvent,EERecHitCollection & ecalHits,EEDigiCollection & ecalDigis)
{
  clean();
  loadPCaloHits(iEvent);

  unsigned nhit=theFiredCells_.size();
  unsigned gain, adc;
  ecalDigis.reserve(nhit);
  ecalHits.reserve(nhit);
  for(unsigned ihit=0;ihit<nhit;++ihit)
    {      
      unsigned icell = theFiredCells_[ihit];

      EEDetId myDetId(endcapRawId_[icell]);
      if(doDigis_)
	{
	   ecalDigis.push_back( myDetId );
	   EEDataFrame myDataFrame( ecalDigis.back() );
	   // myDataFrame.setSize(1); // now useless - by construction fixed at 1 frame - FIXME
	   //  The real work is in the following line
	   geVtoGainAdc(theCalorimeterHits_[icell],gain,adc);
	   myDataFrame.setSample(0,EcalMGPASample(adc,gain));
	   //ecalDigis.push_back(myDataFrame);
	}

      // If the energy+noise is below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
      // is afterwards put in this cell which would not be correct. 
      float energy=theCalorimeterHits_[icell];
      
      // the threshold is in amplitude, so the intercalibration constant should be injected 
      if ( energy<threshold_*((*ICMC_)[icell]) && !isHighInterest(myDetId) && applyZSCells_[icell]) 
	{
	  theCalorimeterHits_[icell]=0.;
	  //	  int TThashedindex=towerOf_[icell];
	  //	  std::cout << " SR " << TTTEnergy_[TThashedindex] << " Cell energy " << energy << " 0 "<< std::endl;
	  energy=0.;
	}
      else 
	if( energy > sat_)
	  {
	    energy=sat_;
	    theCalorimeterHits_[icell]=sat_;
	  }
      if(energy!=0.)
	{
	  ecalHits.push_back(EcalRecHit(myDetId,energy,0.));
	}
    }
  noisified_ = true;

}

void EcalEndcapRecHitsMaker::loadPCaloHits(const edm::Event & iEvent)
{
  //  std::cout << " loadPCaloHits " << std::endl;
  edm::Handle<CrossingFrame<PCaloHit> > cf;
  iEvent.getByLabel(inputCol_,cf);
  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),std::pair<int,int>(0,0) ));

  theFiredCells_.reserve(colcalo->size());
  MixCollection<PCaloHit>::iterator cficalo;
  MixCollection<PCaloHit>::iterator cficaloend=colcalo->end();
  for (cficalo=colcalo->begin(); cficalo!=cficaloend;cficalo++) 
    {

      unsigned hashedindex = EEDetId(cficalo->id()).hashedIndex();      
      // Check if the hit already exists
      float calib=(doMisCalib_) ? calibfactor_*theCalibConstants_[hashedindex]:calibfactor_;
      if(theCalorimeterHits_[hashedindex]==0.)
	{
	  theFiredCells_.push_back(hashedindex); 
	  float noise=(noise_==-1.) ? noisesigma_[hashedindex] : noise_ ;
	  if (!noisified_ ) {
	    theCalorimeterHits_[hashedindex] += random_->gaussShoot(0.,noise*calib); 
	  }
	}
      // the famous 1/0.97 calibration factor is applied here ! 
      // the miscalibration is applied here:

      // cficalo->energy can be 0 (a 7x7 grid is always built), in this case, one should not kill the cell (for later noise injection), but it should
      // be added only once.  This is a dirty trick. 
      float energy=(cficalo->energy()==0.) ? 0.000001 : cficalo->energy() ;
      energy*=calib;
      theCalorimeterHits_[hashedindex]+=energy;   

      // Now deal with the TTs
      int TThashedindex=towerOf_[hashedindex];

      if(TTTEnergy_[TThashedindex]==0.)
	{
	  theFiredTTs_.push_back(TThashedindex);
	}
      // the same dirty trick as before. sinTheta is stored only for one endcap
      TTTEnergy_[TThashedindex]+=energy*sinTheta_[(hashedindex<EEDetId::kEEhalf)? hashedindex : hashedindex-EEDetId::kEEhalf];

    }
  //  std::cout << " Noisifying the TT " << std::endl;
  noisifyTriggerTowers();
  randomNoisifier();
}

void EcalEndcapRecHitsMaker::noisifyTriggerTowers()
{
  if(noise_==0.) return;

  // noise should be injected in all the Super-crystals contained in each trigger tower 
  unsigned nTT=theFiredTTs_.size();
  for(unsigned itt=0;itt<nTT;++itt)
    {      
      // shoot noise in the trigger tower 
      //      std::cout << " Treating " << theFiredTTs_[itt] << " " << theTTDetIds_[theFiredTTs_[itt]] << " " << TTTEnergy_[theFiredTTs_[itt]] << std::endl;       
	noisifySuperCrystals(theFiredTTs_[itt]);
    }
}

// inject noise in all the supercrystals of a given trigger tower
void EcalEndcapRecHitsMaker::noisifySuperCrystals(int tthi)
{
  unsigned size=SCofTT_[tthi].size();
  // get the list of Supercrytals
  for(unsigned isc=0;isc<size;++isc)
    {
      //      std::cout << " Looking at SC " << isc << " " << SCofTT_[tthi][isc] << std::endl;
      // should not do it twice
      if(treatedSC_[SCofTT_[tthi][isc]]) continue;
      
      const std::vector<int> & xtals(CrystalsinSC_[SCofTT_[tthi][isc]]);
      unsigned nxtals=xtals.size();
      unsigned count=0;
//      if (nxtals!=25)
//	{
//	  std::cout << " This SC has " << nxtals << " crystals " << std::endl;
//	}
      for(unsigned ix=0;ix<nxtals;++ix)
	{
	  unsigned hashedindex=xtals[ix];
	  // check if the crystal has already something or not 
	  //	  std::cout << " Looking at crystal " << EEDetId(endcapRawId_[hashedindex]) << std::endl;
	  if(theCalorimeterHits_[hashedindex]==0.)
	    {
	      float calib = (doMisCalib_) ? calibfactor_*theCalibConstants_[hashedindex]:calibfactor_;
	      float noise = (noise_==-1.) ? noisesigma_[hashedindex]:noise_;
	      float energy = calib*random_->gaussShoot(0.,noise);
	      theCalorimeterHits_[hashedindex]=energy;
	      theFiredCells_.push_back(hashedindex);
	      // the corresponding trigger tower should be updated 
	      int newtthi=towerOf_[hashedindex];
	      //	      std::cout << " Updatung TT " << newtthi << std::endl;
	      if(TTTEnergy_[newtthi]==0.)
		{
		  theFiredTTs_.push_back(newtthi);
		}
	      TTTEnergy_[newtthi]+=energy*sinTheta_[(hashedindex<EEDetId::kEEhalf)? hashedindex : hashedindex-EEDetId::kEEhalf];
	      ++count;
	    }
	}
      treatedSC_[SCofTT_[tthi][isc]]=true;
      //      std::cout << "SC " << SCofTT_[tthi][isc] << " done ; injected " << count << std::endl;
    }
}

// injects high noise fluctuations cells. It is assumed that these cells cannot 
// make of the tower a high interest one. 
// Two different cases:
// 1) noise flat in energy -> use a GaussianTail generator
// 2) noise from database (~flat in pT) -> inject noise everywhere
void EcalEndcapRecHitsMaker::randomNoisifier()
{
  // first of cells where some noise will be injected
  double mean = (double)(EEDetId::kSizeForDenseIndexing-theFiredCells_.size())*EEHotFraction_;
  unsigned ncells= random_->poissonShoot(mean);

  // for debugging
  //  std::vector<int> listofNewTowers;
  
  if(noise_==-1. && !doCustomHighNoise_)
    ncells=EEDetId::kSizeForDenseIndexing;

  unsigned icell=0;
  while(icell < ncells)
    {
      unsigned cellindex= (noise_!=-1. || doCustomHighNoise_) ? 
	(unsigned)(floor(random_->flatShoot()*EEDetId::kSizeForDenseIndexing)): icell ;
      
      if(theCalorimeterHits_[cellindex]==0.)
	{
	  // if noise_>0 the following is really an energy, if -1. it is a transverse energy
	  double energy=0.;
	  if(noise_>0.) 
	    energy=myGaussianTailGenerator_->shoot();
	  if(noise_==-1.) 
	    {
	      // in this case the generated noise might be below the threshold but it 
	      // does not matter, the threshold will be applied anyway
	      //	      energy/=sinTheta_[(cellindex<EEDetId::kEEhalf)?cellindex : cellindex-EEDetId::kEEhalf];	 
	      float noisemean  = (doCustomHighNoise_)? highNoiseParameters_[0]*(*ICMC_)[cellindex]*adcToGeV_: 0.;
	      float noisesigma = (doCustomHighNoise_)? highNoiseParameters_[1]*(*ICMC_)[cellindex]*adcToGeV_ : noisesigma_[cellindex]; 
	      energy=random_->gaussShoot(noisemean,noisesigma); 

	      // in the case of high noise fluctuation, the ZS should not be applied later 
	      if(doCustomHighNoise_) applyZSCells_[cellindex]=false;
	    }
	  float calib = (doMisCalib_) ? calibfactor_*theCalibConstants_[cellindex]:calibfactor_;
	  energy *= calib;
	  theCalorimeterHits_[cellindex]=energy;	  
	  theFiredCells_.push_back(cellindex);
	  EEDetId myDetId(EEDetId::unhashIndex(cellindex));
	  // now get the TT 	  
	  int TThashedindex=towerOf_[cellindex];
	  //	  std::cout << " myDetIds " << myDetId << " "TTHI " << TThashedindex<< std::endl;
	  if(TTTEnergy_[TThashedindex]==0.)
	    {
	      theFiredTTs_.push_back(TThashedindex);
	      TTTEnergy_[TThashedindex]+=energy*sinTheta_[(cellindex<EEDetId::kEEhalf)?cellindex : cellindex-EEDetId::kEEhalf];	 
	      //	      listofNewTowers.push_back(TThashedindex);
	    }
//	  else
//	    {
//	      std::vector<int>::const_iterator itcheck=std::find(listofNewTowers.begin(),listofNewTowers.end(),
//								 TThashedindex);
//	      if(itcheck==listofNewTowers.end())
//		{
//		  const std::vector<int> & scxtals=SCofTT_[TThashedindex];
//		  for(unsigned isc=0;isc<scxtals.size();++isc)
//		    {
//		      const std::vector<int> & xtals(CrystalsinSC_[SCofTT_[TThashedindex][isc]]);
//		      for(unsigned ic=0;ic<xtals.size();++ic)
//			std::cout << isc << " " << EEDetId::unhashIndex(xtals[ic]) << " " << theCalorimeterHits_[xtals[ic]] << std::endl;
//		    }
//		}
//	    }
	  if(noise_>0.)
	    ++icell;
	}
      if(noise_==-1.)
	++icell;
    }
  //  std::cout << " Injected random noise in " << ncells << " cells " << std::endl;
}


void EcalEndcapRecHitsMaker::init(const edm::EventSetup &es,bool doDigis,bool domiscalib)  
{
  doDigis_=doDigis;
  doMisCalib_=domiscalib;

  
  edm::ESHandle<EcalADCToGeVConstant> agc;
  es.get<EcalADCToGeVConstantRcd>().get(agc);

  adcToGeV_=   agc->getEEValue() ; // ~0.06 
  minAdc_ = 200;
  maxAdc_ = 4085;
  
  geVToAdc1_ = 1./adcToGeV_;
  geVToAdc2_ = geVToAdc1_/2.;
  geVToAdc3_ = geVToAdc1_/12.;
  
  t1_ = ((int)maxAdc_-(int)minAdc_)*adcToGeV_;
  t2_ = 2.* t1_ ; 

  sat_ = 12.*t1_*calibfactor_;

  endcapRawId_.resize(EEDetId::kSizeForDenseIndexing);

  if (doMisCalib_) theCalibConstants_.resize(EEDetId::kSizeForDenseIndexing);

  edm::ESHandle<CaloGeometry> pG;
  es.get<CaloGeometryRecord>().get(pG);   
  
  edm::ESHandle<EcalTrigTowerConstituentsMap> hetm;
  es.get<IdealGeometryRecord>().get(hetm);
  eTTmap_ = &(*hetm);

  const EcalEndcapGeometry * myEcalEndcapGeometry = dynamic_cast<const EcalEndcapGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalEndcap));
  const std::vector<DetId>& vec(myEcalEndcapGeometry->getValidDetIds(DetId::Ecal,EcalEndcap));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      EEDetId myDetId(vec[ic]);
      int cellhashedindex=myDetId.hashedIndex();
      endcapRawId_[cellhashedindex]=vec[ic].rawId();
      // trick to save a bit of memory. sin Theta is identical in EE+/-
      if (cellhashedindex< EEDetId::kEEhalf)
	{
	  float sintheta=std::sin(myEcalEndcapGeometry->getGeometry(myDetId)->getPosition().theta());
	  sinTheta_[cellhashedindex]=sintheta;
	}
      // a bit of trigger tower and SuperCrystals algebra
      // first get the trigger tower 
      EcalTrigTowerDetId towid1= eTTmap_->towerOf(vec[ic]);
      int tthashedindex=TThashedIndexforEE(towid1);
      towerOf_[cellhashedindex]=tthashedindex;

      // get the SC of the cell
      int schi=SChashedIndex(EEDetId(vec[ic]));
      if(schi<0) 
	{
	  //	  std::cout << " OOps " << schi << std::endl;
	  EEDetId myID(vec[ic]);
	  //	  std::cout << " DetId " << myID << " " << myID.isc() << " " <<  myID.zside() <<  " " << myID.isc()+(myID.zside()+1)*158 << std::endl;
	}
      
      theTTDetIds_[tthashedindex]=towid1;

      // check if this SC is already in the list of the corresponding TT
      std::vector<int>::const_iterator itcheck=find(SCofTT_[tthashedindex].begin(),
						    SCofTT_[tthashedindex].end(),
						    schi);
      if(itcheck==SCofTT_[tthashedindex].end())
	SCofTT_[tthashedindex].push_back(schi);
      
      // check if this crystal is already in the list of crystals per sc
      itcheck=find(CrystalsinSC_[schi].begin(),CrystalsinSC_[schi].end(),cellhashedindex);
      if(itcheck==CrystalsinSC_[schi].end())
	CrystalsinSC_[schi].push_back(cellhashedindex);

      // check if the TT is already in the list of sc
      //      std::cout << " SCHI " << schi << " " << TTofSC_.size() << std::endl;
      //      std::cout << TTofSC_[schi].size() << std::endl;
      itcheck=find(TTofSC_[schi].begin(),TTofSC_[schi].end(),tthashedindex);
      if(itcheck==TTofSC_[schi].end())
	TTofSC_[schi].push_back(tthashedindex);
    }
  //  std::cout << " Made the array " << std::endl;
  // Stores the miscalibration constants
  if(doMisCalib_||noise_==-1.)
    {
      double rms=0.;
      double mean=0.;
      unsigned ncells=0;
      
      if(noise_==-1.)
	noisesigma_.resize(EEDetId::kSizeForDenseIndexing);

      // Intercalib constants IC_MC_i
      edm::ESHandle<EcalIntercalibConstantsMC> pJcal;
      es.get<EcalIntercalibConstantsMCRcd>().get(pJcal); 
      const EcalIntercalibConstantsMC* jcal = pJcal.product(); 
      const std::vector<float>& ICMC = jcal->endcapItems();

      // should be saved, used by the zero suppression
      ICMC_ = &ICMC;

      // Intercalib constants IC_i 
      // IC = IC_MC * (1+delta)
      // where delta is the miscalib
      edm::ESHandle<EcalIntercalibConstants> pIcal;
      es.get<EcalIntercalibConstantsRcd>().get(pIcal);
      const EcalIntercalibConstants* ical = pIcal.product();
      const std::vector<float>& IC = ical->endcapItems();


      unsigned nic = IC.size();
      meanNoiseSigmaEt_ = 0.;
      for(unsigned ic=0;ic<nic;++ic)
	{
	  // the miscalibration factor is 
	  float factor = IC[ic]/ICMC[ic];
	  // Apply Refactor & refactor_mean
	  if(doMisCalib_) theCalibConstants_[ic] = refactor_mean_+(factor-1.)*refactor_;	  
	  rms+=(factor-1.)*(factor-1.);
	  mean+=(factor-1.);

	  // the miscalibration on the noise will be applied later one; so it is really ICMC here
	  if(noise_==-1.)
	    { 
	      // the calibfactor will be applied later on 
	      noisesigma_[ic]=noiseADC_*adcToGeV_*ICMC[ic]/calibfactor_;
	      meanNoiseSigmaEt_ += noisesigma_[ic] * sinTheta_[(ic<EEDetId::kEEhalf)? ic : ic-EEDetId::kEEhalf];
	      //	      EEDetId myDetId(EEDetId::unhashIndex(ic));
	      //	      std::cout << " DDD " <<  myDetId << " " << myDetId.ix() << " " << myDetId.iy() <<  " " << ic << " " << noiseADC_ << " " << agc->getEEValue() << " " << ICMC[ic] << " " << noisesigma_[ic] << std::endl;
	    }
	  ++ncells;	
	}

      mean/=(float)ncells;
      rms/=(float)ncells;

      rms=sqrt(rms-mean*mean);

      meanNoiseSigmaEt_ /=(float)ncells;

      edm::LogInfo("CaloRecHitsProducer") << "Found  " << ncells << " cells in the endcap calibration map. RMS is " << rms << std::endl;
      //      std::cout << "Found  " << ncells << " cells in the endcap calibration map. RMS is " << rms << std::endl;
    }  
  
  // Initialize the Gaussian tail generator
  // Two options : noise is set by the user (to a positive value). In this case, this value is taken
  // or the user chose to use the noise from DB. In this case, the noise is flat in pT and not in energy
  // but the threshold is in energy and not in pT. 

  doCustomHighNoise_=highNoiseParameters_.size()==4;
  Genfun::Erf myErf; 
  if(  noise_>0. ) {
    EEHotFraction_ = 0.5-0.5*myErf(threshold_/noise_/sqrt(2.));
    myGaussianTailGenerator_ = new GaussianTail(random_, noise_, threshold_);
    edm::LogInfo("CaloRecHitsProducer") <<"Uniform noise simulation selected";
  }
  else  if( noise_==-1 && doCustomHighNoise_)
    {
      // computes the hot fraction from the threshold applied on the online amplitude reconstruction
      // 
      EEHotFraction_ = 0.5-0.5*myErf(highNoiseParameters_[3]/highNoiseParameters_[2]/sqrt(2.));
      edm::LogInfo("CaloRecHitsProducer")<< " The gaussian model for high noise fluctuation cells after ZS is selected (best model), hot fraction " << EEHotFraction_  << std::endl;
    }
}


void EcalEndcapRecHitsMaker::geVtoGainAdc(float e,unsigned & gain, unsigned &adc) const
{
  if(e<t1_)
    {
      gain = 1; // x1 
      //      std::cout << " E " << e << std::endl;
      adc = minAdc_ + (unsigned)(e*geVToAdc1_);
      //      std::cout << " e*geVtoAdc1_ " << e*geVToAdc1_ << " " <<(unsigned)(e*geVToAdc1_) << std::endl;
    } 
  else if (e<t2_)
    {
      gain = 2; // x6
      adc = minAdc_ + (unsigned)(e*geVToAdc2_);
    }
  else 
    {
      gain = 3; // x12
      adc = std::min(minAdc_+(unsigned)(e*geVToAdc3_),maxAdc_);
    }
}

bool EcalEndcapRecHitsMaker::isHighInterest(const EEDetId& detid)
{
  //  std::cout << " is HI " ;
  int schi=SChashedIndex(detid);
  //  std::cout <<  detid << "  " << schi << " ";
  // check if it has already been treated or not
  // 0 <=> not treated
  // 1 <=> high interest
  // -1 <=> low interest
  //  std::cout << SCHighInterest_[schi] << std::endl;
  if(SCHighInterest_[schi]!=0) return (SCHighInterest_[schi]>0);
  
  // now look if a TT contributing is of high interest
  const std::vector<int> & tts(TTofSC_[schi]);
  unsigned size=tts.size();
  bool result=false;
  for(unsigned itt=0;itt<size&&!result;++itt)
    {
      //      std::cout << " Checking TT " << tts[itt] << std::endl;
      if(TTTEnergy_[tts[itt]]>SRThreshold_) result=true;
    }
  SCHighInterest_[schi]=(result)? 1:-1;
  theFiredSC_.push_back(schi);
  return result;
}
