//
// HcalNoiseInfoProducer.cc
//
//   description: Implementation of the producer for the HCAL noise information
//
//   author: J.P. Chou, Brown
//
//


#include "RecoMET/METProducers/interface/HcalNoiseInfoProducer.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

using namespace reco;

//
// constructors and destructor
//

HcalNoiseInfoProducer::HcalNoiseInfoProducer(const edm::ParameterSet& iConfig)
{
  // set the parameters
  fillDigis_      = iConfig.getParameter<bool>("fillDigis");
  fillRecHits_    = iConfig.getParameter<bool>("fillRecHits");
  fillCaloTowers_ = iConfig.getParameter<bool>("fillCaloTowers");

  HPDEnergyThreshold_ = iConfig.getParameter<double>("HPDEnergyThreshold");
  RBXEnergyThreshold_ = iConfig.getParameter<double>("RBXEnergyThreshold");

  recHitEnergyThreshold_     = iConfig.getParameter<double>("recHitEnergyThreshold");
  recHitTimeEnergyThreshold_ = iConfig.getParameter<double>("recHitTimeEnergyThreshold");

  digiCollName_      = iConfig.getParameter<std::string>("digiCollName");
  recHitCollName_    = iConfig.getParameter<std::string>("recHitCollName");
  caloTowerCollName_ = iConfig.getParameter<std::string>("caloTowerCollName");

  requirePedestals_ = iConfig.getParameter<bool>("requirePedestals");
  nominalPedestal_  = iConfig.getParameter<double>("nominalPedestal");

  // if digis are filled, then rechits must also be filled
  if(fillDigis_ && !fillRecHits_) {
    fillRecHits_=true;
    edm::LogWarning("HCalNoiseInfoProducer") << " forcing fillRecHits to be true if fillDigis is true.\n";
  }

  // we produce a vector of HcalNoiseRBXs
  produces<HcalNoiseRBXCollection>();
}


HcalNoiseInfoProducer::~HcalNoiseInfoProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HcalNoiseInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // define an empty HcalNoiseRBXArray that we're going to fill
  HcalNoiseRBXArray rbxarray;

  // fill them with the various components
  // digi assumes that rechit information is available
  if(fillRecHits_)    fillrechits(iEvent, iSetup, rbxarray);
  if(fillDigis_)      filldigis(iEvent, iSetup, rbxarray);
  if(fillCaloTowers_) fillcalotwrs(iEvent, iSetup, rbxarray);

  // this is what we're going to actually write to the EDM
  std::auto_ptr<HcalNoiseRBXCollection> result(new HcalNoiseRBXCollection);
  
  // select those RBXs which are interesting
  for(HcalNoiseRBXArray::const_iterator it = rbxarray.begin(); it!=rbxarray.end(); ++it) {
    const HcalNoiseRBX &rbx=(*it);
    
    // select certain RBXs to be written

    // the total energy in an RBX has to be above some energy threshold
    // and there has to be at least one rechit
    if(rbx.rechitEnergy()>RBXEnergyThreshold_
       && rbx.numHitsAboveThreshold()>=1) {
      result->push_back(rbx);
      continue;
    }
    
    // alternatively, the highest energy HPD can also be over some energy threshold
    // where it has at least one rechit
    if(rbx.maxHPD()->rechitEnergy()>HPDEnergyThreshold_
       && rbx.maxHPD()->numHitsAboveThreshold()>=1) {
      result->push_back(rbx);
      continue;
    }
  }

  // put the rbxarray into the EDM
  iEvent.put(result);
  
  return;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalNoiseInfoProducer::beginJob(const edm::EventSetup&)
{
  return;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalNoiseInfoProducer::endJob()
{
  return;
}


// ------------ method called once each run just before starting event loop  ------------
// ------------ fills the pedestals
void 
HcalNoiseInfoProducer::beginRun(edm::Run&, const edm::EventSetup& iSetup)
{
  // clear pedestals
  pedestalmap_.clear();

  // get pedestals
  edm::ESHandle<HcalPedestals> refPeds;
  try {
    iSetup.get<HcalPedestalsRcd>().get(refPeds);
  } catch(cms::Exception& ex) {
    if(requirePedestals_) {
      throw ex << " could not find HcalPedestals.\n";
    } else {
      edm::LogWarning("HCalNoiseInfoProducer") << " No pedestals found.  Using nominal pedestal of " << nominalPedestal_ << " fC.\n";
      return;
    }
  }
  const HcalPedestals *myped = refPeds.product();

  // create HcalDetId for all channels in HB and HE
  for(int det = 1; det <= 2; det++) // only consider HB and HE
    for(int eta = -29; eta <= 29; eta++)
      for(int phi = 1; phi <= 72; phi++)
	for(int depth = 1; depth <= 3; depth++) {
	  
	  HcalDetId hcaldetid(static_cast<HcalSubdetector>(det), eta, phi, depth);
	  if(!HcalHPDRBXMap::isValid(hcaldetid)) continue;

	  const float *values = (myped->getValues(hcaldetid))->getValues();
	  double pedestal = 0.;
	  
	  for (int icap = 0; icap < 4; ++icap)
	    pedestal += (0.25 * values[icap]);
	  
	  // fill the pedestsal map here with the average over all capacitors
	  pedestalmap_[hcaldetid] = pedestal;
	}
  return;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalNoiseInfoProducer::endRun(edm::Run&, const edm::EventSetup&)
{
  return;
}


// ------------ fill the array with digi information
void
HcalNoiseInfoProducer::filldigis(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array) const
{
  // get the digis
  edm::Handle<HBHEDigiCollection> handle;
  iEvent.getByLabel(digiCollName_, handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HBHEDigiCollection named " << digiCollName_ << "\n.";
    return;
  }

  // loop over all of the digi information
  for(HBHEDigiCollection::const_iterator it=handle->begin(); it!=handle->end(); ++it) {
    const HBHEDataFrame &digi=(*it);
    HcalNoiseHPDArray::iterator hpditer=array.findHPD(digi);
    HcalNoiseHPD &hpd=(*hpditer);

    // get the pedestal
    pedestalmap_t::const_iterator pedestalit = pedestalmap_.find(digi.id());
    double pedestal = (pedestalit != pedestalmap_.end()) ? pedestalit->second : nominalPedestal_;

    // determine if the digi is one the highest energy hits in the HPD
    bool isBig=false, isBig5=false;

    // if the first (highest energy) rechit has the same id as the digi
    if(hpd.recHits_.size()>0 && hpd.recHits_.begin()->id() == digi.id())
      isBig=isBig5=true;

    // loop over the top five highest energy rechits
    for(EnergySortedHBHERecHits::iterator it=hpd.recHits_.begin();
	it!=hpd.recHits_.end(); ++it) {
      if(it->id() == digi.id()) isBig5=true;
    }

    // loop over each of the digi's time slices
    int totalzeros=0;
    for(int ts=0; ts<HBHEDataFrame::MAXSAMPLES; ts++) {

      // require a good digi
      if(!digi[ts].dv() || digi[ts].er()) continue;      

      // count zero's
      if(digi[ts].adc()==0) {
	++hpd.totalZeros_;
	++totalzeros;
      }

      // get the fC's
      double corrfc = digi[ts].nominal_fC()-pedestal;

      // fill the relevant digi arrays
      if(isBig)  hpd.bigDigi_[ts]+=corrfc;
      if(isBig5) hpd.big5Digi_[ts]+=corrfc;
      hpd.allDigi_[ts]+=corrfc;
    }

    // record the maximum number of zero's found
    if(totalzeros>hpd.maxZeros_)
      hpd.maxZeros_=totalzeros;
  }

  return;
}

// ------------ fill the array with rec hit information
void
HcalNoiseInfoProducer::fillrechits(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array) const
{
  // get the rechits
  edm::Handle<HBHERecHitCollection> handle;
  iEvent.getByLabel(recHitCollName_, handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find HBHERecHitCollection named " << recHitCollName_ << "\n.";
    return;
  }

  // loop over all of the digi information
  for(HBHERecHitCollection::const_iterator it=handle->begin(); it!=handle->end(); ++it) {
    const HBHERecHit &rechit=(*it);
    HcalNoiseHPD& hpd=(*array.findHPD(rechit));
    
    // fill the info
    hpd.rechitEnergy_+=rechit.energy();
    ++hpd.numHits_;

    // only consider rechits with enough energy
    // to contribute to the min/max time
    if(rechit.energy()>recHitTimeEnergyThreshold_) {
      double time=rechit.time();
      if(time>hpd.maxTime_) hpd.maxTime_=time;
      if(time<hpd.minTime_) hpd.minTime_=time;
    }

    // consider here only rechits above a certain threshold
    if(rechit.energy()>=recHitEnergyThreshold_) {
      ++hpd.numHitsAboveThreshold_;

      // store and sort the rechits here
      hpd.recHits_.insert(rechit);
      // if we have too many rec hits, delete the lowest energy one
      // we keep only the top HcalNoiseHPD::MAXRECHITS
      if(hpd.recHits_.size()>static_cast<unsigned int>(HcalNoiseHPD::MAXRECHITS))
	hpd.recHits_.erase(--hpd.recHits_.end());
    }

  } // end loop over rechits

  return;
}

// ------------ fill the array with calo tower information
void
HcalNoiseInfoProducer::fillcalotwrs(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array) const
{
  // get the calotowers
  edm::Handle<CaloTowerCollection> handle;
  iEvent.getByLabel(caloTowerCollName_, handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find CaloTowerCollection named " << caloTowerCollName_ << "\n.";
    return;
  }

  // loop over all of the calotower information
  for(CaloTowerCollection::const_iterator it = handle->begin(); it!=handle->end(); ++it) {
    const CaloTower& twr=(*it);

    // get all of the hpd's and rbx's that are pointed to by the calotower
    std::vector<HcalNoiseHPDArray::iterator> hpditervec;
    std::vector<HcalNoiseRBXArray::iterator> rbxitervec;
    array.findHPD(twr, hpditervec);
    array.findRBX(twr, rbxitervec);

    // loop over the hpd's
    for(std::vector<HcalNoiseHPDArray::iterator>::iterator it=hpditervec.begin();
	it!=hpditervec.end(); ++it) {

      // de-reference twice
      HcalNoiseHPD &hpd = **it;
      hpd.twrHadE_ += twr.hadEnergy();
      hpd.twrEmE_ += twr.emEnergy();
    }

    // loop over the rbx's
    for(std::vector<HcalNoiseRBXArray::iterator>::iterator it=rbxitervec.begin();
	it!=rbxitervec.end(); ++it) {

      // de-reference twice
      HcalNoiseRBX &rbx = **it;
      rbx.twrHadE_ += twr.hadEnergy();
      rbx.twrEmE_ += twr.emEnergy();
    }
  }

  return;
}





//define this as a plug-in
DEFINE_FWK_MODULE(HcalNoiseInfoProducer);
