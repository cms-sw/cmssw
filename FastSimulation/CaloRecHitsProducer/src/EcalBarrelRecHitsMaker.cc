#include "FastSimulation/CaloRecHitsProducer/interface/EcalBarrelRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

EcalBarrelRecHitsMaker::EcalBarrelRecHitsMaker(edm::ParameterSet const & p,
					       const RandomEngine* myrandom)
  : random_(myrandom)
{
  edm::ParameterSet RecHitsParameters=p.getParameter<edm::ParameterSet>("ECALBarrel");
  inputCol_=RecHitsParameters.getParameter<edm::InputTag>("MixedSimHits");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
  refactor_ = RecHitsParameters.getParameter<double> ("Refactor");
  refactor_mean_ = RecHitsParameters.getParameter<double> ("Refactor_mean");
  SRThreshold_ = RecHitsParameters.getParameter<double> ("SRThreshold");
  SREtaSize_ = RecHitsParameters.getUntrackedParameter<int> ("SREtaSize",1);
  SRPhiSize_ = RecHitsParameters.getUntrackedParameter<int> ("SRPhiSize",1);

  
  theCalorimeterHits_.resize(62000,0.);
  crystalsinTT_.resize(2448);
  TTTEnergy_.resize(2448,0.);
  TTHighInterest_.resize(2448,0);
  treatedTTs_.resize(2448,false);
  theTTDetIds_.resize(2448);
  neighboringTTs_.resize(2448);
  sinTheta_.resize(86,0.);
  
  noisified_ = (noise_==0.);
  edm::ParameterSet CalibParameters = RecHitsParameters.getParameter<edm::ParameterSet>("ContFact"); 
  double c1=CalibParameters.getParameter<double>("EBs25notContainment"); 
  calibfactor_=1./c1;
  adcToGeV_= 0.035;
  minAdc_ = 200;
  maxAdc_ = 4085;

  geVToAdc1_ = 1./adcToGeV_;
  geVToAdc2_ = geVToAdc1_/2.;
  geVToAdc3_ = geVToAdc1_/12.;
  
  t1_ = ((int)maxAdc_-(int)minAdc_)*adcToGeV_;
  t2_ = 2.* t1_ ; 

  // Saturation value. Not needed in the digitization
  sat_ = 12.*t1_*calibfactor_;
}

EcalBarrelRecHitsMaker::~EcalBarrelRecHitsMaker()
{;
}


void EcalBarrelRecHitsMaker::clean()
{
  //  std::cout << " clean " << std::endl;
  unsigned size=theFiredCells_.size();
  for(unsigned ic=0;ic<size;++ic)
    {
      theCalorimeterHits_[theFiredCells_[ic]] = 0.;
    }
  theFiredCells_.clear();
  // If the noise is set to 0. No need to simulate it. 
  noisified_ = (noise_==0.);
  //  std::cout << " Finished to clean "  << std::endl;
  
  size=theFiredTTs_.size();
  //  std::cout << " Number of barrel TT " << size << std::endl;
  for(unsigned itt=0;itt<size;++itt)
    {
      //      std::cout << " TT " << theFiredTTs_[itt] << " " << TTTEnergy_[theFiredTTs_[itt]] << std::endl;
      TTTEnergy_[theFiredTTs_[itt]]=0.;
      treatedTTs_[theFiredTTs_[itt]]=false;
    }  
  theFiredTTs_.clear();
  
  size=theTTofHighInterest_.size();
  for(unsigned itt=0;itt<size;++itt)
    TTHighInterest_[theTTofHighInterest_[itt]]=0;
  theTTofHighInterest_.clear();

//  std::cout << " Check cleaning " << std::endl;
//  for(unsigned ic=0;ic<TTTEnergy_.size();++ic)
//    if(TTTEnergy_[ic]!=0.) std::cout << " TT " << ic << " not cleaned " << std::endl;
//  for(unsigned ic=0;ic<TTHighInterest_.size();++ic)
//    if(TTHighInterest_[ic]!=0) std::cout << " TTHighInterest " << ic << TTHighInterest_[ic] << " not cleaned " << std::endl;
//  for(unsigned ic=0;ic<treatedTTs_.size();++ic)
//    if(treatedTTs_[ic]) std::cout << " treatedTT " << ic << treatedTTs_[ic] << " not cleaned " << std::endl;
  
}

void EcalBarrelRecHitsMaker::loadEcalBarrelRecHits(edm::Event &iEvent,EBRecHitCollection & ecalHits,EBDigiCollection & ecalDigis)
{

  clean();
  loadPCaloHits(iEvent);
  
  unsigned nhit=theFiredCells_.size();
  //  std::cout << " loadEcalBarrelRecHits " << nhit << std::endl;
  unsigned gain, adc;
  ecalDigis.reserve(nhit);
  ecalHits.reserve(nhit);
  for(unsigned ihit=0;ihit<nhit;++ihit)
    {      
      unsigned icell = theFiredCells_[ihit];

      EBDetId myDetId(barrelRawId_[icell]);
      EcalTrigTowerDetId towid= eTTmap_->towerOf(myDetId);
      int TThashedindex=towid.hashedIndex();      

      if(doDigis_)
	{
          ecalDigis.push_back( myDetId );
	  EBDataFrame myDataFrame( ecalDigis.back() );
	  // myDataFrame.setSize(1);  // now useless - by construction fixed at 1 frame - FIXME
	  //  The real work is in the following line
	  geVtoGainAdc(theCalorimeterHits_[icell],gain,adc);
	  myDataFrame.setSample(0,EcalMGPASample(adc,gain));
	  
	  //      std::cout << "myDataFrame" << myDataFrame.sample(0).raw() << std::endl;
	  //ecalDigis.push_back(myDataFrame);
	}
      
      // If the energy+noise is below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
      // is afterwards put in this cell which would not be correct. 
      float energy=theCalorimeterHits_[icell];
      //      std::cout << myDetId << " Energy " << theCalorimeterHits_[icell] << " " << TTTEnergy_[TThashedindex] << " " << isHighInterest(TThashedindex) << std::endl;
      if ( SRThreshold_ && energy < threshold_  && !isHighInterest(TThashedindex))
	{
	  //	  std::cout << " Killed " << std::endl;
	  theCalorimeterHits_[icell]=0.;
	  energy=0.;
	} 
      //      else
	//	std::cout << " SR " <<  TTTEnergy_[TThashedindex] << " Cell energy " << energy << " 1" << std::endl;
//      if( TTTEnergy_[TThashedindex] < SRThreshold_ && energy > threshold_)
//	std::cout << " SR " << TTTEnergy_[TThashedindex] << " Cell energy " << energy << std::endl;
      if (energy > sat_)
	{
	  theCalorimeterHits_[icell]=sat_;
	  energy=sat_;
	}
//      std::cout << " Threshold ok " << std::endl;
//      std::cout << " Raw Id " << barrelRawId_[icell] << std::endl;
//      std::cout << " Adding " << icell << " " << barrelRawId_[icell] << " " << energy << std::endl;
      if(energy!=0.)
	ecalHits.push_back(EcalRecHit(myDetId,energy,0.));
      //      std::cout << " Hit stored " << std::endl;
    }
  //  std::cout << " Done " << std::endl;

}

void EcalBarrelRecHitsMaker::loadPCaloHits(const edm::Event & iEvent)
{

  edm::Handle<CrossingFrame<PCaloHit> > cf;
  iEvent.getByLabel(inputCol_,cf);
  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),std::pair<int,int>(0,0) ));


  theFiredCells_.reserve(colcalo->size());

  MixCollection<PCaloHit>::iterator cficalo;
  MixCollection<PCaloHit>::iterator cficaloend=colcalo->end();

  for (cficalo=colcalo->begin(); cficalo!=cficaloend;cficalo++) 
    {
      unsigned hashedindex = EBDetId(cficalo->id()).hashedIndex();      
      // Check if the hit already exists
      if(theCalorimeterHits_[hashedindex]==0.)
	{
	  theFiredCells_.push_back(hashedindex);
	  if (!noisified_ )  theCalorimeterHits_[hashedindex] += random_->gaussShoot(0.,noise_); 
	}
      // the famous 1/0.97 calibration factor is applied here ! 
      // the miscalibration is applied here:
      float calib = (doMisCalib_) ? calibfactor_*theCalibConstants_[hashedindex]:calibfactor_;
      // cficalo->energy can be 0 (a 7x7 grid is always built) if there is no noise simulated, in this case the cells should
      // not be added several times. 
      float energy=(cficalo->energy()==0.) ? 0.000001 : cficalo->energy() ;
      energy*=calib;
      theCalorimeterHits_[hashedindex]+=energy;         

      // Now deal with the TTs. 
      EBDetId myDetId(EBDetId(cficalo->id()));
      EcalTrigTowerDetId towid= eTTmap_->towerOf(myDetId);
//      std::cout << " Added " << energy << " in " << EBDetId(cficalo->id()) << std::endl;
      int TThashedindex=towid.hashedIndex();
      if(TTTEnergy_[TThashedindex]==0.)
	{
	  theFiredTTs_.push_back(TThashedindex);	   
//	  std::cout << " Creating " ;
	}
      //      std::cout << " Updating " << TThashedindex << " " << energy << " " << sinTheta_[myDetId.ietaAbs()] <<std::endl;
       TTTEnergy_[TThashedindex]+=energy*sinTheta_[myDetId.ietaAbs()];
       //       std::cout << " TT " << TThashedindex  << " updated, now contains " << TTTEnergy_[TThashedindex] << std::endl;
    }
  noisifyTriggerTowers();
  noisified_ = true;
}

void EcalBarrelRecHitsMaker::noisifyTriggerTowers()
{
  if(noise_==0.) return;
//  std::cout << " List of TT before" << std::endl;
//  for(unsigned itt=0;itt<theFiredTTs_.size();++itt)
//    std::cout << " TT " << theFiredTTs_[itt] << " " << TTTEnergy_[theFiredTTs_[itt]] << std::endl;

  //  std::cout << " Starting to noisify the trigger towers " << std::endl;
  unsigned nTT=theFiredTTs_.size();
  for(unsigned itt=0;itt<nTT;++itt)
    {      
      //      std::cout << "Treating " << theFiredTTs_[itt] << " " << theTTDetIds_[theFiredTTs_[itt]].ieta() << " " <<  theTTDetIds_[theFiredTTs_[itt]].iphi() << " " << TTTEnergy_[theFiredTTs_[itt]] << std::endl;
      // shoot noise in the trigger tower 
      noisifyTriggerTower(theFiredTTs_[itt]);
      // get the neighboring TTs
      const std::vector<int>& neighbors=neighboringTTs_[theFiredTTs_[itt]];
      unsigned nneighbors=neighbors.size();
      // inject noise in those towers only if they have not been looked at yet
      //      std::cout << " Now looking at the neighbours " << std::endl;
      for(unsigned in=0;in<nneighbors;++in)
	{
	  //	  std::cout << " TT " << neighbors[in] << " " << theTTDetIds_[neighbors[in]] << " has been treated " << treatedTTs_[neighbors[in]] << std::endl;
	  if(!treatedTTs_[neighbors[in]])
	    {
	      noisifyTriggerTower(neighbors[in]);
	      if(TTTEnergy_[neighbors[in]]==0.)
		theFiredTTs_.push_back(neighbors[in]);
	      //	      std::cout << " Added " << neighbors[in] << " in theFiredTTs_ " << std::endl;;
	    }
	}
    }
//  std::cout << " List of TT after" << std::endl;
//  for(unsigned itt=0;itt<theFiredTTs_.size();++itt)
//    std::cout << " TT " << theFiredTTs_[itt] << " " << TTTEnergy_[theFiredTTs_[itt]] << std::endl;
}

bool EcalBarrelRecHitsMaker::noisifyTriggerTower(unsigned tthi)
{
  // check if the TT has already been treated or not
  if(treatedTTs_[tthi]) return false;
  // get the crystals in the TT (this info might need to be cached)
  //  const std::vector<DetId> & xtals=eTTmap_->constituentsOf(theTTDetIds_[tthi]);
  const std::vector<int> & xtals(crystalsinTT_[tthi]);
  unsigned nxtals=xtals.size();
  unsigned counter =0 ; 
  float energy=0.;
  for(unsigned ic=0;ic<nxtals;++ic)
    {
      unsigned hashedindex=xtals[ic];
      // check if the crystal has been already hit
//      std::cout << " Checking " << EBDetId(barrelRawId_[xtals[ic]]) << " " << theCalorimeterHits_[hashedindex] << std::endl;
      if(theCalorimeterHits_[hashedindex]==0)
	{
	  float calib = (doMisCalib_) ? calibfactor_*theCalibConstants_[hashedindex]:calibfactor_;
	  float energy = calib*random_->gaussShoot(0.,noise_);
	  theCalorimeterHits_[hashedindex]=energy;
	  //	  std::cout << " Updating with noise " << tthi << " " << energy << " " << sinTheta_[EBDetId(barrelRawId_[hashedindex]).ietaAbs()] << std::endl;
	  if(TTTEnergy_[tthi]==0.)
	    theFiredTTs_.push_back(tthi);
	  TTTEnergy_[tthi]+=energy*sinTheta_[EBDetId(barrelRawId_[hashedindex]).ietaAbs()];
	  
	  theFiredCells_.push_back(hashedindex);
	  ++counter;
	}
      else
	energy+=theCalorimeterHits_[hashedindex];
    }
//  std::cout << " Energy " << energy  << " Added noise in " << counter << " cells" << std::endl;
  treatedTTs_[tthi]=true;
  return true;
}

void EcalBarrelRecHitsMaker::init(const edm::EventSetup &es,bool doDigis,bool doMiscalib)
{
  //  std::cout << " Initializing EcalBarrelRecHitsMaker " << std::endl;
  doDigis_=doDigis;
  doMisCalib_=doMiscalib;
  barrelRawId_.resize(62000);
  if (doMisCalib_) theCalibConstants_.resize(62000);
  edm::ESHandle<CaloGeometry> pG;
  es.get<CaloGeometryRecord>().get(pG);   

//  edm::ESHandle<CaloTopology> theCaloTopology;
//  es.get<CaloTopologyRecord>().get(theCaloTopology);     

  edm::ESHandle<EcalTrigTowerConstituentsMap> hetm;
  es.get<IdealGeometryRecord>().get(hetm);
  eTTmap_ = &(*hetm);
  
  const EcalBarrelGeometry * myEcalBarrelGeometry = dynamic_cast<const EcalBarrelGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalBarrel));
  //  std::cout << " Got the geometry " << myEcalBarrelGeometry << std::endl;
  const std::vector<DetId>& vec(myEcalBarrelGeometry->getValidDetIds(DetId::Ecal,EcalBarrel));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      EBDetId myDetId(vec[ic]);
      int crystalHashedIndex=myDetId.hashedIndex();
      barrelRawId_[crystalHashedIndex]=vec[ic].rawId();
      // save the Trigger tower DetIds
      EcalTrigTowerDetId towid= eTTmap_->towerOf(EBDetId(vec[ic]));
      int TThashedindex=towid.hashedIndex();      
      theTTDetIds_[TThashedindex]=towid;                  
      crystalsinTT_[TThashedindex].push_back(crystalHashedIndex);
      int ietaAbs=myDetId.ietaAbs();
      if(sinTheta_[ietaAbs]==0.)
	{
	  sinTheta_[ietaAbs]=std::sin(myEcalBarrelGeometry->getGeometry(myDetId)->getPosition().theta());
	  //	  std::cout << " Ieta abs " << ietaAbs << " " << sinTheta_[ietaAbs] << std::endl;
	}
    }


  unsigned nTTs=theTTDetIds_.size();

//  EBDetId myDetId(-58,203);
////  std::cout << " CellID " << myDetId << std::endl;
//  EcalTrigTowerDetId towid= eTTmap_->towerOf(myDetId);
////  std::cout << " EcalTrigTowerDetId ieta, iphi" << towid.ieta() << " , " << towid.iphi() << std::endl;
////  std::cout << " Constituents of this tower " <<towid.hashedIndex() << std::endl;
//  const std::vector<int> & xtals(crystalsinTT_[towid.hashedIndex()]);
//  unsigned Size=xtals.size();
//  for(unsigned i=0;i<Size;++i)
//    {
//      std::cout << EBDetId(barrelRawId_[xtals[i]]) << std::endl;
//    }

  // now loop on each TT and save its neighbors. 
  for(unsigned iTT=0;iTT<nTTs;++iTT)
    {
      int ietaPivot=theTTDetIds_[iTT].ieta();
      int iphiPivot=theTTDetIds_[iTT].iphi();
      int TThashedIndex=theTTDetIds_[iTT].hashedIndex();
      //      std::cout << " TT Pivot " << TThashedIndex << " " << ietaPivot << " " << iphiPivot << " iz " << theTTDetIds_[iTT].zside() << std::endl;
      int ietamin=std::max(ietaPivot-SREtaSize_,-17);
      if(ietamin==0) ietamin=-1;
      int ietamax=std::min(ietaPivot+SREtaSize_,17);
      if(ietamax==0) ietamax=1;
      int iphimin=iphiPivot-SRPhiSize_;
      int iphimax=iphiPivot+SRPhiSize_;
      for(int ieta=ietamin;ieta<=ietamax;)
	{
	  int iz=(ieta>0)? 1 : -1; 
	  for(int iphi=iphimin;iphi<=iphimax;)
	    {
	      int riphi=iphi;
	      if(riphi<1) riphi+=72;
	      else if(riphi>72) riphi-=72;
	      EcalTrigTowerDetId neighborTTDetId(iz,EcalBarrel,abs(ieta),riphi);
	      //      std::cout << " Voisin " << ieta << " " << riphi << " " <<neighborTTDetId.hashedIndex()<< " " << neighborTTDetId.ieta() << " " << neighborTTDetId.iphi() << std::endl;
	      if(ieta!=ietaPivot||riphi!=iphiPivot)
		{
		  neighboringTTs_[TThashedIndex].push_back(neighborTTDetId.hashedIndex());
		}
	      ++iphi;

	    }
	  ++ieta;
	  if(ieta==0) ieta=1;
	}
    }

  //  std::cout << " Made the array " << std::endl;

  // Stores the miscalibration constants
  if(doMisCalib_)
    {
      float rms=0.;
      unsigned ncells=0;
      // Intercalib constants
      edm::ESHandle<EcalIntercalibConstants> pIcal;
      es.get<EcalIntercalibConstantsRcd>().get(pIcal);
      const EcalIntercalibConstants* ical = pIcal.product();

      theCalibConstants_ = ical->barrelItems();
      std::vector<float>::iterator it=theCalibConstants_.begin();
      std::vector<float>::iterator itend=theCalibConstants_.end();
      
      for ( ; it != itend; ++it ) {

	// Apply Refactor & refactor_mean
	*it= refactor_mean_+(*it-1.)*refactor_;
	rms+=(*it-1.)*(*it-1.);
	++ncells;
      }
      rms = std::sqrt(rms) / (float)ncells;
      // The following should be on LogInfo
      //std::cout << " Found " << ncells << " cells in the barrel calibration map. RMS is " << rms << std::endl;
    }  
}

void EcalBarrelRecHitsMaker::geVtoGainAdc(float e,unsigned & gain, unsigned &adc) const
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
      gain = 2; 
      adc = minAdc_ + (unsigned)(e*geVToAdc2_);
    }
  else 
    {
      gain = 3; 
      adc = std::min(minAdc_+(unsigned)(e*geVToAdc3_),maxAdc_);
    }
}

bool EcalBarrelRecHitsMaker::isHighInterest(int tthi)
{

  if(TTHighInterest_[tthi]!=0) return (TTHighInterest_[tthi]>0);

  TTHighInterest_[tthi]=(TTTEnergy_[tthi] > SRThreshold_) ? 1:-1;
  // if high interest, can leave ; otherwise look at the neighbours)
  if( TTHighInterest_[tthi]==1) 
    {
      theTTofHighInterest_.push_back(tthi);
      return true;
    }

  // now look if a neighboring TT is of high interest
  const std::vector<int> & tts(neighboringTTs_[tthi]);
  // a tower is of high interest if it or one of its neighbour is above the SR threshold
  unsigned size=tts.size();
  bool result=false;
  for(unsigned itt=0;itt<size&&!result;++itt)
    {
      if(TTTEnergy_[tts[itt]] > SRThreshold_) result=true;
    }
  TTHighInterest_[tthi]=(result)? 1:-1;
  theTTofHighInterest_.push_back(tthi);
  return result;
}
