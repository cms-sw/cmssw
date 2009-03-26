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
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
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
  fillJets_       = iConfig.getParameter<bool>("fillJets");
  fillTracks_     = iConfig.getParameter<bool>("fillTracks");
  dropRefVectors_ = iConfig.getParameter<bool>("dropRefVectors");
  refillRefVectors_ = iConfig.getParameter<bool>("refillRefVectors");

  RBXEnergyThreshold_ = iConfig.getParameter<double>("RBXEnergyThreshold");
  minHPDEnergy_    = iConfig.getParameter<double>("minHPDEnergy");
  minRBXEnergy_    = iConfig.getParameter<double>("minRBXEnergy");
  minRecHitEnergy_ = iConfig.getParameter<double>("minRecHitEnergy");
  minHPDNumRecHit_ = iConfig.getParameter<int>("minHPDNumRecHit");
  minRBXNumZeros_  = iConfig.getParameter<int>("minRBXNumZeros");
  minRBXMaxZeros_  = iConfig.getParameter<int>("minRBXMaxZeros");
  minRBXTime_      = iConfig.getParameter<double>("minRBXTime");
  maxRBXTime_      = iConfig.getParameter<double>("maxRBXTime");
  minHPDRatio_     = iConfig.getParameter<double>("minHPDRatio");
  maxHPDRatio_     = iConfig.getParameter<double>("maxHPDRatio");
  maxProblemRBXs_  = iConfig.getParameter<int>("maxProblemRBXs");

  maxJetEmFraction_   = iConfig.getParameter<double>("maxJetEmFraction");
  maxJetEta_          = iConfig.getParameter<double>("maxJetEta");
  maxCaloTowerIEta_   = iConfig.getParameter<int>("maxCaloTowerIEta");
  maxTrackEta_        = iConfig.getParameter<double>("maxTrackEta");
  minTrackPt_         = iConfig.getParameter<double>("minTrackPt");

  digiCollName_      = iConfig.getParameter<std::string>("digiCollName");
  recHitCollName_    = iConfig.getParameter<std::string>("recHitCollName");
  caloTowerCollName_ = iConfig.getParameter<std::string>("caloTowerCollName");
  caloJetCollName_   = iConfig.getParameter<std::string>("caloJetCollName");
  trackCollName_     = iConfig.getParameter<std::string>("trackCollName");
  hcalNoiseRBXCollName_ = iConfig.getParameter<std::string>("hcalNoiseRBXCollName");

  requirePedestals_ = iConfig.getParameter<bool>("requirePedestals");
  nominalPedestal_  = iConfig.getParameter<double>("nominalPedestal");

  // if digis are filled, then rechits must also be filled
  if(fillDigis_ && !fillRecHits_) {
    fillRecHits_=true;
    edm::LogWarning("HCalNoiseInfoProducer") << " forcing fillRecHits to be true if fillDigis is true.\n";
  }

  // we produce a vector of HcalNoiseRBXs
  produces<HcalNoiseRBXCollection>();
  // we also produce a noise summary
  produces<HcalNoiseSummary>();
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
  if(!refillRefVectors_) {
    // we're creating HcalNoiseRBXs for the first time

    // this is what we're going to actually write to the EDM
    std::auto_ptr<HcalNoiseRBXCollection> result1(new HcalNoiseRBXCollection);
    std::auto_ptr<HcalNoiseSummary> result2(new HcalNoiseSummary);

    // define an empty HcalNoiseRBXArray that we're going to fill
    HcalNoiseRBXArray rbxarray;
    HcalNoiseSummary &summary=*result2;
    
    // fill them with the various components
    // digi assumes that rechit information is available
    if(fillRecHits_)    fillrechits(iEvent, iSetup, rbxarray, summary);
    if(fillDigis_)      filldigis(iEvent, iSetup, rbxarray);
    if(fillCaloTowers_) fillcalotwrs(iEvent, iSetup, rbxarray, summary);
    if(fillJets_)       filljets(iEvent, iSetup, summary);    
    if(fillTracks_)     filltracks(iEvent, iSetup, summary);

    // select those RBXs which are interesting
    // also look for the highest energy RBX
    HcalNoiseRBXArray::iterator maxit=rbxarray.begin();
    double maxenergy=maxit->recHitEnergy();
    bool maxwritten=false;
    for(HcalNoiseRBXArray::iterator rit = rbxarray.begin(); rit!=rbxarray.end(); ++rit) {
      HcalNoiseRBX &rbx=(*rit);
      
      // calculate certain quantities once and once only
      double hpdenergy=rbx.maxHPD()->recHitEnergy();
      double rbxenergy=rbx.recHitEnergy();
      double bigratio = rbx.maxHPD()->bigChargeHighest2TS()/rbx.maxHPD()->bigChargeTotal();

      // find the highest energy rbx
      if(rbxenergy>maxenergy) {
	maxenergy=rbxenergy;
	maxit=rit;
	maxwritten=false;
      }

      // skip RBXs unless they meet some basic threshold energy
      if(rbxenergy<RBXEnergyThreshold_) continue;

      // select certain RBXs to be written
      if(rbxenergy>=minRBXEnergy_ ||
         hpdenergy>=minHPDEnergy_ ||
	 bigratio<minHPDRatio_ ||
	 bigratio>maxHPDRatio_ ||
	 rbx.maxHPD()->numRecHits(minRecHitEnergy_) >=minHPDNumRecHit_ ||
	 rbx.totalZeros()>=minRBXNumZeros_ ||
	 rbx.maxZeros()>=minRBXMaxZeros_ ||
	 rbx.minRecHitTime()<minRBXTime_ ||
	 rbx.maxRecHitTime()>maxRBXTime_) {

	// drop the ref vectors if we need to
	if(dropRefVectors_) {
	  for(std::vector<HcalNoiseHPD>::iterator hit = rbx.hpds_.begin(); hit != rbx.hpds_.end(); ++hit) {
	    hit->rechits_.clear();
	    hit->calotowers_.clear();
	  }
	}

	// add the RBX to the event
	summary.nproblemRBXs_++;
	if(summary.nproblemRBXs_<=maxProblemRBXs_) {
	  result1->push_back(rbx);
	  if(maxit==rit) maxwritten=true;
	}
      }
    } // end loop over rbxs

    // if we still haven't written the maximum energy rbx, write it now
    if(!maxwritten) {
      HcalNoiseRBX &rbx=(*maxit);
      
      // drop the ref vectors if we need to
      if(dropRefVectors_) {
	for(std::vector<HcalNoiseHPD>::iterator hit = rbx.hpds_.begin(); hit != rbx.hpds_.end(); ++hit) {
	  hit->rechits_.clear();
	  hit->calotowers_.clear();
	}
      }

      // add the RBX to the event
      result1->push_back(rbx);
    }
  
    // determine if the event is noisy
    summary.filterstatus_=0;
    
    // put the rbxcollection and summary into the EDM
    iEvent.put(result1);
    iEvent.put(result2);


  } else {

    // we're taking HcalNoiseRBXs that are already present, and creating a new set with RefVector information stored

    // define an empty HcalNoiseRBXArray that we're going to fill
    // the summary object is a dummy placeholder
    HcalNoiseRBXArray rbxarray;
    HcalNoiseSummary summary;
    
    // fill them with the various components
    if(fillRecHits_)    fillrechits(iEvent, iSetup, rbxarray, summary);
    if(fillCaloTowers_) fillcalotwrs(iEvent, iSetup, rbxarray, summary);

    // this is what we're going to actually write to the EDM
    std::auto_ptr<HcalNoiseRBXCollection> result(new HcalNoiseRBXCollection);

    // get the old HcalNoiseRBX's
    edm::Handle<HcalNoiseRBXCollection> handle;
    iEvent.getByLabel(hcalNoiseRBXCollName_, handle);
    if(!handle.isValid()) {
      throw edm::Exception(edm::errors::ProductNotFound) << " could not find HcalNoiseRBXCollection named " << hcalNoiseRBXCollName_ << "\n.";
      return;
    }
    
    // loop over the old HcalNoiseRBX's and match them to the recently filled RBXs
    for(HcalNoiseRBXCollection::const_iterator rit=handle->begin(); rit!=handle->end(); ++rit) {
      const HcalNoiseRBX &oldrbx=(*rit);
      HcalNoiseRBX &newrbx=rbxarray[oldrbx.idnumber()];

      // copy over the Digi Information
      newrbx.allCharge_ = oldrbx.allCharge_;
      std::vector<HcalNoiseHPD>::iterator hit1 = newrbx.hpds_.begin();
      std::vector<HcalNoiseHPD>::const_iterator hit2 = oldrbx.hpds_.begin(); 
      for( ; hit1 != newrbx.hpds_.end() && hit2 != oldrbx.hpds_.end(); ++hit1, ++hit2) {
	hit1->totalZeros_ = hit2->totalZeros_;
	hit1->maxZeros_ = hit2->maxZeros_;
	hit1->bigCharge_ = hit2->bigCharge_;
	hit1->big5Charge_ = hit2->big5Charge_;
      }

      result->push_back(newrbx);
    }

    // put the rbxcollection into the EDM
    iEvent.put(result);
  }
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
    HcalNoiseRBX &rbx=(*array.findRBX(digi));
    HcalNoiseHPD &hpd=(*array.findHPD(digi));
    edm::RefVector<HBHERecHitCollection> &rechits=hpd.rechits_;

    // get the pedestal
    pedestalmap_t::const_iterator pedestalit = pedestalmap_.find(digi.id());
    double pedestal = (pedestalit != pedestalmap_.end()) ? pedestalit->second : nominalPedestal_;

    // determine if the digi is one the highest energy hits in the HPD
    bool isBig=false, isBig5=false;

    // see if the digi has the same id as the highest energy rechit
    // this assumes that the rechits are properly sorted
    if(rechits.begin()!=rechits.end() && (*rechits.begin())->id() == digi.id())
      isBig=isBig5=true;

    // loop over the five highest E rechits and see if the digi is there
    for(edm::RefVector<HBHERecHitCollection>::const_iterator rit=rechits.begin();
	rit!=rechits.end() && rit!=rechits.begin()+5; ++rit) {
      if((*rit)->id() == digi.id()) {
	isBig5=true;
	break;
      }
    }

    // loop over each of the digi's time slices
    int totalzeros=0;
    for(int ts=0; ts<digi.size(); ++ts) {

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
      if(isBig)  hpd.bigCharge_[ts]+=corrfc;
      if(isBig5) hpd.big5Charge_[ts]+=corrfc;
      rbx.allCharge_[ts]+=corrfc;
    }

    // record the maximum number of zero's found
    if(totalzeros>hpd.maxZeros_)
      hpd.maxZeros_=totalzeros;
  }

  return;
}

// ------------ fill the array with rec hit information
void
HcalNoiseInfoProducer::fillrechits(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array, HcalNoiseSummary& summary) const
{
  // get the rechits
  edm::Handle<HBHERecHitCollection> handle;
  iEvent.getByLabel(recHitCollName_, handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find HBHERecHitCollection named " << recHitCollName_ << "\n.";
    return;
  }

  summary.min10_=summary.min25_=99999.;
  summary.max10_=summary.max25_=-99999.;
  summary.rms10_=summary.rms25_=0.0;
  int cnt10=0, cnt25=0;

  // loop over all of the rechit information
  for(HBHERecHitCollection::const_iterator it=handle->begin(); it!=handle->end(); ++it) {
    const HBHERecHit &rechit=(*it);

    // find the hpd that the rechit is in
    HcalNoiseHPD& hpd=(*array.findHPD(rechit));

    // create a persistent reference to the rechit
    edm::Ref<HBHERecHitCollection> myRef(handle, it-handle->begin());
    
    // store it in a place so that it remains sorted by energy
    hpd.refrechitset_.insert(myRef);

    // calculate summary objects
    float energy=rechit.energy();
    float time=rechit.time();
    if(energy>=10. && summary.min10_>time) summary.min10_=time;
    if(energy>=25. && summary.min25_>time) summary.min25_=time;
    if(energy>=10. && summary.max10_<time) summary.max10_=time;
    if(energy>=25. && summary.max25_<time) summary.max25_=time;
    if(energy>=10.) {
      summary.rms10_ += time*time;
      ++cnt10;
    }
    if(energy>=25.) {
      summary.rms25_ += time*time;
      ++cnt25;
    }

  } // end loop over rechits

  // finish calculation of rms
  summary.rms10_= cnt10>0 ? sqrt(summary.rms10_/cnt10) : -999.;
  summary.rms25_= cnt25>0 ? sqrt(summary.rms25_/cnt25) : -999.;

  // now loop over all HPDs and transfer the information from refrechitset_ to rechits_;
  for(HcalNoiseRBXArray::iterator rbxit=array.begin(); rbxit!=array.end(); ++rbxit) {
    for(std::vector<HcalNoiseHPD>::iterator hpdit=rbxit->hpds_.begin(); hpdit!=rbxit->hpds_.end(); ++hpdit) {
      
      // now loop over all of the entries in the set
      // and add them to rechits_
      for(std::set<edm::Ref<HBHERecHitCollection>, RefHBHERecHitEnergyComparison>::const_iterator
	    it=hpdit->refrechitset_.begin(); it!=hpdit->refrechitset_.end(); ++it) {
	hpdit->rechits_.push_back(*it);
      }
    }
  }

  return;
}

// ------------ fill the array with calo tower information
void
HcalNoiseInfoProducer::fillcalotwrs(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array, HcalNoiseSummary& summary) const
{
  // get the calotowers
  edm::Handle<CaloTowerCollection> handle;
  iEvent.getByLabel(caloTowerCollName_, handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find CaloTowerCollection named " << caloTowerCollName_ << "\n.";
    return;
  }

  summary.emenergy_ = summary.hadenergy_ = 0.0;

  // loop over all of the calotower information
  for(CaloTowerCollection::const_iterator it = handle->begin(); it!=handle->end(); ++it) {
    const CaloTower& twr=(*it);

    // skip over anything with |ieta|>maxCaloTowerIEta
    if(twr.ietaAbs()>maxCaloTowerIEta_) continue;

    // create a persistent reference to the tower
    edm::Ref<CaloTowerCollection> myRef(handle, it-handle->begin());

    // get all of the hpd's that are pointed to by the calotower
    std::vector<std::vector<HcalNoiseHPD>::iterator> hpditervec;
    array.findHPD(twr, hpditervec);

    // loop over the hpd's and add the reference to the RefVectors
    for(std::vector<std::vector<HcalNoiseHPD>::iterator>::iterator it=hpditervec.begin();
	it!=hpditervec.end(); ++it)
      (*it)->calotowers_.push_back(myRef);

    // include the emenergy and hadenergy
    summary.emenergy_ += twr.emEnergy();
    summary.hadenergy_ += twr.hadEnergy();
  }

  return;
}

// ------------ fill the array with calo tower information
void
HcalNoiseInfoProducer::filljets(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseSummary& summary) const
{
  edm::Handle<reco::CaloJetCollection> handle;
  iEvent.getByLabel(caloJetCollName_, handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find jetCollection named " << caloJetCollName_ << "\n.";
    return;
  }
  
  for(reco::CaloJetCollection::const_iterator iJet = handle->begin(); iJet!=handle->end(); ++iJet) {
    reco::CaloJet jet=*iJet;
    if(jet.eta()>maxJetEta_) continue;
    
    // create a persistent reference to the jet
    edm::Ref<CaloJetCollection> myRef(handle, iJet-handle->begin());

    // calculate em fraction ignoring HF and HO component
    double hadE = jet.hadEnergyInHB() + jet.hadEnergyInHE();
    double emeE = jet.emEnergyInEB() + jet.emEnergyInEE();

    // if the emEnergy fraction is very small, keep this jet
    if(emeE+hadE>0 && emeE/(emeE+hadE)<maxJetEmFraction_)
      summary.problemjets_.push_back(myRef);
    
  }


  return;
}

void
HcalNoiseInfoProducer::filltracks(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseSummary& summary) const
{
  edm::Handle<reco::TrackCollection> handle;
  iEvent.getByLabel(trackCollName_, handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find trackCollection named " << trackCollName_ << "\n.";
    return;
  }

  summary.trackenergy_=0.0;
  for(reco::TrackCollection::const_iterator iTrack = handle->begin(); iTrack!=handle->end(); ++iTrack) {
    reco::Track trk=*iTrack;
    if(trk.pt()<minTrackPt_ || fabs(trk.eta())>maxTrackEta_) continue;

    summary.trackenergy_ += trk.p();
  }

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalNoiseInfoProducer);
