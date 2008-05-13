#include "FastSimulation/CaloRecHitsProducer/interface/EcalEndcapRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalEndcapAlgo/interface/EcalEndcapGeometry.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"


EcalEndcapRecHitsMaker::EcalEndcapRecHitsMaker(edm::ParameterSet const & p, 
					       edm::ParameterSet const & pcalib,
					       const RandomEngine * myrandom) 
  : random_(myrandom)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("ECALEndcap");
  inputCol_=RecHitsParameters.getParameter<edm::InputTag>("MixedSimHits");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
  refactor_ = RecHitsParameters.getParameter<double> ("Refactor");
  refactor_mean_ = RecHitsParameters.getParameter<double> ("Refactor_mean");
  theCalorimeterHits_.resize(20000,0.);
  noisified_ = (noise_==0.);
  double c1 = pcalib.getParameter<double>("EEs25notContainment");
  calibfactor_= 1./c1;

  adcToGeV_= 0.060;
  minAdc_ = 200;
  maxAdc_ = 4085;
  
  geVToAdc1_ = 1./adcToGeV_;
  geVToAdc2_ = geVToAdc1_/2.;
  geVToAdc3_ = geVToAdc1_/12.;
  
  t1_ = ((int)maxAdc_-(int)minAdc_)*adcToGeV_;
  t2_ = 2.* t1_ ; 

  sat_ = 12.*t1_*calibfactor_;
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
    }
  theFiredCells_.clear();
  // If the noise is set to 0. No need to simulate it. 
  noisified_ = (noise_==0.);
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

      // It is safer to update the orignal array in case this methods is called several times
      if (!noisified_ )  theCalorimeterHits_[icell] += random_->gaussShoot(0.,noise_);
      
      // If the energy+noise is below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
      // is afterwards put in this cell which would not be correct. 
      float energy=theCalorimeterHits_[icell];
      if ( energy<threshold_ ) 
	{
	  theCalorimeterHits_[icell]=0.;
	  energy=0.;
	}
      else 
	if( energy > sat_)
	  {
	    energy=sat_;
	    theCalorimeterHits_[icell]=sat_;
	  }

      ecalHits.push_back(EcalRecHit(myDetId,energy,0.));
    }
  noisified_ = true;

}

void EcalEndcapRecHitsMaker::loadPCaloHits(const edm::Event & iEvent)
{

  edm::Handle<CrossingFrame<PCaloHit> > cf;
  iEvent.getByLabel(inputCol_,cf);
  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),std::pair<int,int>(0,0) ));

  theFiredCells_.reserve(colcalo->size());

  MixCollection<PCaloHit>::iterator cficalo;
  MixCollection<PCaloHit>::iterator cficaloend=colcalo->end();
  float calib=1.;
  for (cficalo=colcalo->begin(); cficalo!=cficaloend;cficalo++) 
    {
      unsigned hashedindex = EEDetId(cficalo->id()).hashedIndex();      
      // Check if the hit already exists
      if(theCalorimeterHits_[hashedindex]==0.)
	{
	  theFiredCells_.push_back(hashedindex); 
	}
      // the famous 1/0.97 calibration factor is applied here ! 
      calib=calibfactor_;
      // the miscalibration is applied here:
      if(doMisCalib_) calib*=theCalibConstants_[hashedindex];
      theCalorimeterHits_[hashedindex]+=cficalo->energy()*calib;   
    }
}


void EcalEndcapRecHitsMaker::init(const edm::EventSetup &es,bool doDigis,bool domiscalib)  
{
  doDigis_=doDigis;
  doMisCalib_=domiscalib;
  endcapRawId_.resize(20000);
  if (doMisCalib_) theCalibConstants_.resize(20000);
  edm::ESHandle<CaloGeometry> pG;
  es.get<CaloGeometryRecord>().get(pG);   
  
  const EcalEndcapGeometry * myEcalEndcapGeometry = dynamic_cast<const EcalEndcapGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalEndcap));
  std::vector<DetId> vec(myEcalEndcapGeometry->getValidDetIds(DetId::Ecal,EcalEndcap));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      endcapRawId_[EEDetId(vec[ic]).hashedIndex()]=vec[ic].rawId();
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


      theCalibConstants_ = ical->endcapItems();
      std::vector<float>::iterator it=theCalibConstants_.begin();
      std::vector<float>::iterator itend=theCalibConstants_.end();
      for ( ; it != itend; ++it ) {	
	if(!EEDetId::validHashIndex(ncells)) continue;
	*it= refactor_mean_+(*it-1.)*refactor_;
	rms+=(*it-1.)*(*it-1.);
	++ncells;
      }
      rms = std::sqrt(rms) / (float)ncells;
      std::cout << " Found " << ncells << " cells in the endcap calibration map. RMS is " << rms << std::endl;
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
