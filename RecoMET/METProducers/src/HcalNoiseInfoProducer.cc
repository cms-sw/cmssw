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
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"


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
  fillTracks_     = iConfig.getParameter<bool>("fillTracks");

  RBXEnergyThreshold_ = iConfig.getParameter<double>("RBXEnergyThreshold");
  minRecHitEnergy_ = iConfig.getParameter<double>("minRecHitEnergy");
  maxProblemRBXs_  = iConfig.getParameter<int>("maxProblemRBXs");
  writeAllRBXs_       = iConfig.getParameter<bool>("writeAllRBXs");

  maxCaloTowerIEta_   = iConfig.getParameter<int>("maxCaloTowerIEta");
  maxTrackEta_        = iConfig.getParameter<double>("maxTrackEta");
  minTrackPt_         = iConfig.getParameter<double>("minTrackPt");

  digiCollName_      = iConfig.getParameter<std::string>("digiCollName");
  recHitCollName_    = iConfig.getParameter<std::string>("recHitCollName");
  caloTowerCollName_ = iConfig.getParameter<std::string>("caloTowerCollName");
  trackCollName_     = iConfig.getParameter<std::string>("trackCollName");
  hcalNoiseRBXCollName_ = iConfig.getParameter<std::string>("hcalNoiseRBXCollName");

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

  // fill them with the various components
  // digi assumes that rechit information is available
  if(fillRecHits_)    fillrechits(iEvent, iSetup, rbxarray);
  if(fillDigis_)      filldigis(iEvent, iSetup, rbxarray);
  if(fillCaloTowers_) fillcalotwrs(iEvent, iSetup, rbxarray, summary);
  if(fillTracks_)     filltracks(iEvent, iSetup, summary);

  // select those RBXs which are interesting
  // also look for the highest energy RBX
  HcalNoiseRBXArray::iterator maxit=rbxarray.begin();
  double maxenergy=maxit->caloTowerHadE();
  bool maxwritten=false;
  for(HcalNoiseRBXArray::iterator rit = rbxarray.begin(); rit!=rbxarray.end(); ++rit) {
    HcalNoiseRBX &rbx=(*rit);
    
    // require that the RBX have a minimum energy
    double rbxenergy=rit->caloTowerHadE();
    if(rbxenergy<RBXEnergyThreshold_) continue;
    
    // find the highest energy rbx
    if(rbxenergy>maxenergy) {
      maxenergy=rbxenergy;
      maxit=rit;
      maxwritten=false;
    }
    
    // find out if the rbx is problematic/noisy/interesting
    bool problem=isProblematicRBX(rbx) || writeAllRBXs_;

    // fill variables in the summary object not filled elsewhere
    fillOtherSummaryVariables(summary, rbx);

    // if the rbx is problematic write it
    if(problem) {
      summary.nproblemRBXs_++;
      if(summary.nproblemRBXs_<=maxProblemRBXs_ || writeAllRBXs_) {
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
HcalNoiseInfoProducer::beginRun(edm::Run&, const edm::EventSetup&)
{
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
  double energy = rbx.recHitEnergy(minRecHitEnergy_);

  if(energy>200.) return true;
  if((allratio<0.7 || allratio>0.9) && energy>25.) return true;
  if(rbx.numRecHits(minRecHitEnergy_) >=17) return true;
  if(rbx.totalZeros()>=4) return true;
  if(fabs(rbx.minRecHitTime(20.0))>4.0) return true;
  if(rbx.caloTowerEmFraction()<0.01 && energy>25.) return true;
  return false;
}


// ------------ here we fill specific variables in the summary object not already accounted for earlier
void 
HcalNoiseInfoProducer::fillOtherSummaryVariables(HcalNoiseSummary& summary, const HcalNoiseRBX& rbx) const
{
  double rbxenergy = rbx.caloTowerHadE();

  // E2TS/E10TS for the RBX
  double allcharge2ts = rbx.allChargeHighest2TS();
  double allchargetotal = rbx.allChargeTotal();
  if(rbxenergy>50.0 && allchargetotal!=0) {
    double allratio = allcharge2ts/allchargetotal;
    if(allratio<summary.minE2Over10TS()) {
      summary.mine2ts_=allcharge2ts;
      summary.mine10ts_=allchargetotal;
    }
    if(allratio>summary.maxE2Over10TS()) {
      summary.maxe2ts_=allcharge2ts;
      summary.maxe10ts_=allchargetotal;
    }
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
  if(rbxenergy>50.0 && rbxemf<summary.minRBXEMF()) summary.minrbxemf_=rbxemf;

  // # of HPD hits, HPD EMF, and timing: done together for speed
  // loop over the HPDs in the RBX
  for(std::vector<HcalNoiseHPD>::const_iterator it1=rbx.hpds_.begin(); it1!=rbx.hpds_.end(); ++it1) {
    int nhpdhits=it1->numRecHits(minRecHitEnergy_);
    double emf=it1->caloTowerEmFraction();
    double ene=it1->caloTowerHadE();
    if(nhpdhits>summary.maxHPDHits()) summary.maxhpdhits_=nhpdhits;
    if(ene>50. && emf<summary.minHPDEMF()) summary.minhpdemf_=emf;
    if(nhpdhits==nrbxhits && nhpdhits>summary.maxHPDNoOtherHits()) summary.maxhpdhitsnoother_=nhpdhits;
    
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
  bool failloose=false;
  if(summary.minE2Over10TS()<0.70)    { summary.filterstatus_ |= 0x1;  failloose=true; }
  if(summary.maxE2Over10TS()>0.90)    { summary.filterstatus_ |= 0x2;  failloose=true; }
  if(summary.min25GeVHitTime()<-7.)   { summary.filterstatus_ |= 0x4;  failloose=true; }
  if(summary.max25GeVHitTime()>6.)    { summary.filterstatus_ |= 0x8;  failloose=true; }
  if(summary.maxZeros()>=9)           { summary.filterstatus_ |= 0x10; failloose=true; }
  if(summary.maxHPDHits()>=17)        { summary.filterstatus_ |= 0x20; failloose=true; }
  if(summary.maxHPDNoOtherHits()>=10) { summary.filterstatus_ |= 0x40; failloose=true; }

  // tight cuts
  bool failtight=false;
  if(summary.minE2Over10TS()<0.75)    { summary.filterstatus_ |= 0x100;  failtight=true; }
  if(summary.maxE2Over10TS()>0.90)    { summary.filterstatus_ |= 0x200;  failtight=true; }
  if(summary.min25GeVHitTime()<-5.)   { summary.filterstatus_ |= 0x400;  failtight=true; }
  if(summary.max25GeVHitTime()>4.)    { summary.filterstatus_ |= 0x800;  failtight=true; }
  if(summary.maxZeros()>=8)           { summary.filterstatus_ |= 0x1000; failtight=true; }
  if(summary.maxHPDHits()>=16)        { summary.filterstatus_ |= 0x2000; failtight=true; }
  if(summary.maxHPDNoOtherHits()>=9)  { summary.filterstatus_ |= 0x4000; failtight=true; }

  // high level cuts
  bool failhl=false;
  if(summary.minRBXEMF()<0.01)        { summary.filterstatus_ |= 0x10000; failhl=true; }

  // get the calotowers associated with the RBX which failed
  if(failtight || failloose || failhl) {

    // loop over all of the HPDs in the RBX
    for(std::vector<HcalNoiseHPD>::const_iterator it1=rbx.hpds_.begin(); it1!=rbx.hpds_.end(); ++it1) {
      edm::RefVector<CaloTowerCollection> twrsref=it1->caloTowers();
      for(edm::RefVector<CaloTowerCollection>::const_iterator it2=twrsref.begin(); it2!=twrsref.end(); ++it2) {
	if(failloose) summary.loosenoisetwrs_.push_back(*it2);
	if(failtight) summary.tightnoisetwrs_.push_back(*it2);
	if(failhl)    summary.hlnoisetwrs_.push_back(*it2);
      }
    }
  }
  
  return;
}


// ------------ fill the array with digi information
void
HcalNoiseInfoProducer::filldigis(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array) const
{
  // get the conditions and channel quality
  edm::ESHandle<HcalDbService> conditions;
  iSetup.get<HcalDbRecord>().get(conditions);
  const HcalQIEShape* shape = conditions->getHcalShape();
  edm::ESHandle<HcalChannelQuality> qualhandle;
  iSetup.get<HcalChannelQualityRcd>().get(qualhandle);
  HcalChannelQuality* myqual = new HcalChannelQuality(*qualhandle.product());
  edm::ESHandle<HcalSeverityLevelComputer> mycomputer;
  iSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);
  const HcalSeverityLevelComputer* mySeverity = mycomputer.product();

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
    HcalDetId cell = digi.id();
    DetId detcell=(DetId)cell;

    // check on cells to be ignored and dropped
    const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());
    if(mySeverity->dropChannel(mydigistatus->getValue())) continue;
    if(digi.zsMarkAndPass()) continue;

    // get the calibrations and coder
    const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
    const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
    HcalCoderDb coder (*channelCoder, *shape);

    // match the digi to an rbx and hpd
    HcalNoiseRBX &rbx=(*array.findRBX(digi));
    HcalNoiseHPD &hpd=(*array.findHPD(digi));

    // determine if the digi is one the highest energy hits in the HPD
    // this works because the rechits are sorted by energy (see fillrechits() below)
    bool isBig=false, isBig5=false, isRBX=false;
    int counter=0;
    edm::RefVector<HBHERecHitCollection> &rechits=hpd.rechits_;
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
    CaloSamples tool;
    coder.adc2fC(digi,tool);
    for(int ts=0; ts<tool.size(); ++ts) {

      // count zero's
      if(digi[ts].adc()==0) {
	++hpd.totalZeros_;
	++totalzeros;
      }

      // get the fC's
      double corrfc = tool[ts]-calibrations.pedestal(digi[ts].capid());

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

    // create a persistent reference to the tower
    edm::Ref<CaloTowerCollection> myRef(handle, it-handle->begin());

    // get all of the hpd's that are pointed to by the calotower
    std::vector<std::vector<HcalNoiseHPD>::iterator> hpditervec;
    array.findHPD(twr, hpditervec);

    // loop over the hpd's and add the reference to the RefVectors
    for(std::vector<std::vector<HcalNoiseHPD>::iterator>::iterator it=hpditervec.begin();
	it!=hpditervec.end(); ++it)
      (*it)->calotowers_.push_back(myRef);

    // skip over anything with |ieta|>maxCaloTowerIEta
    if(twr.ietaAbs()>maxCaloTowerIEta_) {
      summary.emenergy_ += twr.emEnergy();
      summary.hadenergy_ += twr.hadEnergy();
    }
  }

  return;
}

// ------------ fill the summary with track information
void
HcalNoiseInfoProducer::filltracks(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseSummary& summary) const
{
  edm::Handle<reco::TrackCollection> handle;
  iEvent.getByLabel(trackCollName_, handle);

  // don't throw exception, just return quietly
  if(!handle.isValid()) {
    //    throw edm::Exception(edm::errors::ProductNotFound)
    //      << " could not find trackCollection named " << trackCollName_ << "\n.";
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
