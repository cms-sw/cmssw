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
  minRecHitEnergy_ = iConfig.getParameter<double>("minRecHitEnergy");
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
  // this is what we're going to actually write to the EDM
  std::auto_ptr<HcalNoiseRBXCollection> result1(new HcalNoiseRBXCollection);
  std::auto_ptr<HcalNoiseSummary> result2(new HcalNoiseSummary);
  
  // define an empty HcalNoiseRBXArray that we're going to fill
  HcalNoiseRBXArray rbxarray;
  HcalNoiseSummary &summary=*result2;


  // we're creating HcalNoiseRBXs for the first time
  if(!refillRefVectors_) {

    // fill them with the various components
    // digi assumes that rechit information is available
    if(fillRecHits_)    fillrechits(iEvent, iSetup, rbxarray);
    if(fillDigis_)      filldigis(iEvent, iSetup, rbxarray);
    if(fillCaloTowers_) fillcalotwrs(iEvent, iSetup, rbxarray, summary);
    if(fillJets_)       filljets(iEvent, iSetup, summary);    
    if(fillTracks_)     filltracks(iEvent, iSetup, summary);

    // select those RBXs which are interesting
    // also look for the highest energy RBX
    HcalNoiseRBXArray::iterator maxit=rbxarray.begin();
    double maxenergy=maxit->recHitEnergy(minRecHitEnergy_);
    bool maxwritten=false;
    for(HcalNoiseRBXArray::iterator rit = rbxarray.begin(); rit!=rbxarray.end(); ++rit) {
      HcalNoiseRBX &rbx=(*rit);

      // require that the RBX have a minimum energy
      double rbxenergy=rit->recHitEnergy(minRecHitEnergy_);
      if(rbxenergy<RBXEnergyThreshold_) continue;

      // find the highest energy rbx
      if(rbxenergy>maxenergy) {
	maxenergy=rbxenergy;
	maxit=rit;
	maxwritten=false;
      }

      // find out if the rbx is problematic/noisy/interesting
      bool problem=isProblematicRBX(rbx);

      // fill variables in the summary object not filled elsewhere
      fillOtherSummaryVariables(summary, rbx);

      // drop the ref vectors if we need to
      // make sure we do this after we calculate quantities with the RBX
      if(dropRefVectors_) {
	for(std::vector<HcalNoiseHPD>::iterator hit = rbx.hpds_.begin(); hit != rbx.hpds_.end(); ++hit) {
	  hit->rechits_.clear();
	  hit->calotowers_.clear();
	}
      }

      // if the rbx is problematic write it
      if(problem) {
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
      
      // add the RBX to the event
      result1->push_back(rbx);
    }
  
    
  } else {
    // we're taking HcalNoiseRBXs that are already present, and creating a new set with RefVector information stored

    // fill them with the various components
    if(fillRecHits_)    fillrechits(iEvent, iSetup, rbxarray);
    if(fillCaloTowers_) fillcalotwrs(iEvent, iSetup, rbxarray, summary);
    if(fillJets_)       filljets(iEvent, iSetup, summary);    
    if(fillTracks_)     filltracks(iEvent, iSetup, summary);

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

      // copy over the Digi-level Information
      newrbx.allCharge_ = oldrbx.allCharge_;
      std::vector<HcalNoiseHPD>::iterator hit1 = newrbx.hpds_.begin();
      std::vector<HcalNoiseHPD>::const_iterator hit2 = oldrbx.hpds_.begin(); 
      for( ; hit1 != newrbx.hpds_.end() && hit2 != oldrbx.hpds_.end(); ++hit1, ++hit2) {
	hit1->totalZeros_ = hit2->totalZeros_;
	hit1->maxZeros_ = hit2->maxZeros_;
	hit1->bigCharge_ = hit2->bigCharge_;
	hit1->big5Charge_ = hit2->big5Charge_;
      }

      // fill variables in the summary object not filled elsewhere
      fillOtherSummaryVariables(summary, newrbx);

      result1->push_back(newrbx);
    }
  }

  // put the rbxcollection and summary into the EDM
  iEvent.put(result1);
  iEvent.put(result2);

  return;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalNoiseInfoProducer::beginJob()
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

// ------------ returns whether the rbx is problematic
bool
HcalNoiseInfoProducer::isProblematicRBX(const HcalNoiseRBX& rbx) const
{
  double allcharge2ts = rbx.allChargeHighest2TS();
  double allchargetotal = rbx.allChargeTotal();
  double allratio = allchargetotal==0 ? 999 : allcharge2ts/allchargetotal;

  if(rbx.recHitEnergy(minRecHitEnergy_)>=200.) return true;
  if(rbx.maxHPD()->recHitEnergy(minRecHitEnergy_)>=100.) return true;
  if(allratio<0.7 && allchargetotal!=0) return true;
  if(rbx.maxHPD()->numRecHits(minRecHitEnergy_) >=10) return true;
  if(rbx.totalZeros()>=5) return true;
  if(rbx.maxZeros()>=4) return true;
  if(rbx.minRecHitTime()<-6.) return true;
  if(rbx.maxRecHitTime()>6.) return true;
  return false;
}


// ------------ here we fill specific variables in the summary object not already accounted for earlier
void 
HcalNoiseInfoProducer::fillOtherSummaryVariables(HcalNoiseSummary& summary, const HcalNoiseRBX& rbx) const
{
  // E2TS/E10TS for the RBX
  double allcharge2ts = rbx.allChargeHighest2TS();
  double allchargetotal = rbx.allChargeTotal();
  double allratio = allchargetotal==0 ? 999 : allcharge2ts/allchargetotal;
  if(allratio<summary.minE2Over10TS()) {
    summary.mine2ts_=allcharge2ts;
    summary.mine10ts_=allchargetotal;
  }

  // # of zeros in RBX
  int maxzeros=rbx.totalZeros();
  if(maxzeros>summary.maxZeros()) {
    summary.maxzeros_=maxzeros;
  }

  // # of RBX hits
  int nrbxhits=rbx.numRecHits(minRecHitEnergy_);
  if(nrbxhits>summary.maxRBXHits()) summary.maxrbxhits_=nrbxhits;
  
  // RBX EMF
  float rbxemf=rbx.caloTowerEmFraction();
  if(rbxemf<summary.minRBXEMF()) summary.minrbxemf_=rbxemf;

  // # of HPD hits, HPD EMF, and timing: done together for speed
  // loop over the HPDs in the RBX
  for(std::vector<HcalNoiseHPD>::const_iterator it1=rbx.hpds_.begin(); it1!=rbx.hpds_.end(); ++it1) {
    int nhits=it1->numRecHits(minRecHitEnergy_);
    float emf=it1->caloTowerEmFraction();
    if(nhits>summary.maxHPDHits()) summary.maxhpdhits_=nhits;
    if(emf<summary.minHPDEMF()) summary.minhpdemf_=emf;
    
    // loop over the hits in the HPD
    for(edm::RefVector<HBHERecHitCollection>::const_iterator it2=it1->rechits_.begin(); it2!=it1->rechits_.end(); ++it2) {
      float energy=(*it2)->energy();
      float time=(*it2)->time();
      if(energy>=10.) {
	if(time<summary.min10GeVHitTime()) summary.min10_=time;
	if(time>summary.max10GeVHitTime()) summary.max10_=time;
	summary.rms10_ += time*time;
	summary.cnthit10_++;
      }
      if(energy>=25.) {
	if(time<summary.min25GeVHitTime()) summary.min25_=time;
	if(time>summary.max25GeVHitTime()) summary.max25_=time;
	summary.rms25_ += time*time;
	summary.cnthit25_++;
      }
    }
  }

  // loose cuts
  if(summary.minE2Over10TS()<0.7)   summary.filterstatus_ |= 0x1;
  if(summary.min25GeVHitTime()<-7.) summary.filterstatus_ |= 0x2;
  if(summary.max25GeVHitTime()>6.)  summary.filterstatus_ |= 0x4;
  if(summary.maxZeros()>8)          summary.filterstatus_ |= 0x8;

  // tight cuts
  if(summary.minE2Over10TS()<0.8)   summary.filterstatus_ |= 0x100;
  if(summary.min25GeVHitTime()<-5.) summary.filterstatus_ |= 0x200;
  if(summary.max25GeVHitTime()>4.)  summary.filterstatus_ |= 0x400;
  if(summary.maxZeros()>7)          summary.filterstatus_ |= 0x800;
  if(summary.maxHPDHits()>16)       summary.filterstatus_ |= 0x1000;

  // high level cuts
  if(summary.minHPDEMF()<0.01)      summary.filterstatus_ |= 0x10000;
  if(summary.minRBXEMF()<0.01)      summary.filterstatus_ |= 0x20000;

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
    // this works because the rechits are sorted by energy (see fillrechits() below)
    bool isBig=false, isBig5=false, isRBX=false;
    int counter=0;
    for(edm::RefVector<HBHERecHitCollection>::const_iterator rit=rechits.begin();
	rit!=rechits.end(); ++rit, ++counter) {
      if((*rit)->id() == digi.id()) {
	if(counter==0) isBig=isBig5=true;  // digi is also the highest energy rechit
	if(counter<5) isBig5=true;         // digi is one of 5 highest energy rechits
	isRBX=true;                        // digi has rechit energy>minRecHitEnergy_ GeV
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
      if(isRBX)  rbx.allCharge_[ts]+=corrfc;
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

  // loop over all of the rechit information
  for(HBHERecHitCollection::const_iterator it=handle->begin(); it!=handle->end(); ++it) {
    const HBHERecHit &rechit=(*it);

    // calculate energy, time
    float energy=rechit.energy();

    // if the energy is too low, we skip it
    if(energy<minRecHitEnergy_) continue;

    // find the hpd that the rechit is in
    HcalNoiseHPD& hpd=(*array.findHPD(rechit));

    // create a persistent reference to the rechit
    edm::Ref<HBHERecHitCollection> myRef(handle, it-handle->begin());
    
    // store it in a place so that it remains sorted by energy
    hpd.refrechitset_.insert(myRef);

  } // end loop over rechits

  // loop over all HPDs and transfer the information from refrechitset_ to rechits_;
  for(HcalNoiseRBXArray::iterator rbxit=array.begin(); rbxit!=array.end(); ++rbxit) {
    for(std::vector<HcalNoiseHPD>::iterator hpdit=rbxit->hpds_.begin(); hpdit!=rbxit->hpds_.end(); ++hpdit) {
      
      // loop over all of the entries in the set and add them to rechits_
      for(std::set<edm::Ref<HBHERecHitCollection>, RefHBHERecHitEnergyComparison>::const_iterator
	    it=hpdit->refrechitset_.begin(); it!=hpdit->refrechitset_.end(); ++it) {
	hpdit->rechits_.push_back(*it);
      }
    }
  }
  // now the rechits in all the HPDs are sorted by energy!

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

    // include the emenergy and hadenergy if it points to an HPD
    if(hpditervec.size()>0) {
      summary.emenergy_ += twr.emEnergy();
      summary.hadenergy_ += twr.hadEnergy();
    }
  }

  return;
}

// ------------ fill the summary with jet information
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

// ------------ fill the summary with track information
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
