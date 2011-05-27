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
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

using namespace reco;

//
// constructors and destructor
//

HcalNoiseInfoProducer::HcalNoiseInfoProducer(const edm::ParameterSet& iConfig) : algo_(iConfig)
{
  // set the parameters
  fillDigis_         = iConfig.getParameter<bool>("fillDigis");
  fillRecHits_       = iConfig.getParameter<bool>("fillRecHits");
  fillCaloTowers_    = iConfig.getParameter<bool>("fillCaloTowers");
  fillTracks_        = iConfig.getParameter<bool>("fillTracks");

  maxProblemRBXs_    = iConfig.getParameter<int>("maxProblemRBXs");

  maxCaloTowerIEta_  = iConfig.getParameter<int>("maxCaloTowerIEta");
  maxTrackEta_       = iConfig.getParameter<double>("maxTrackEta");
  minTrackPt_        = iConfig.getParameter<double>("minTrackPt");

  digiCollName_      = iConfig.getParameter<std::string>("digiCollName");
  recHitCollName_    = iConfig.getParameter<std::string>("recHitCollName");
  caloTowerCollName_ = iConfig.getParameter<std::string>("caloTowerCollName");
  trackCollName_     = iConfig.getParameter<std::string>("trackCollName");

  minRecHitE_        = iConfig.getParameter<double>("minRecHitE");
  minLowHitE_        = iConfig.getParameter<double>("minLowHitE");
  minHighHitE_       = iConfig.getParameter<double>("minHighHitE");

  HcalAcceptSeverityLevel_ = iConfig.getParameter<uint32_t>("HcalAcceptSeverityLevel");

  TS4TS5EnergyThreshold_ = iConfig.getParameter<double>("TS4TS5EnergyThreshold");

  std::vector<double> TS4TS5UpperThresholdTemp = iConfig.getParameter<std::vector<double> >("TS4TS5UpperThreshold");
  std::vector<double> TS4TS5UpperCutTemp = iConfig.getParameter<std::vector<double> >("TS4TS5UpperCut");
  std::vector<double> TS4TS5LowerThresholdTemp = iConfig.getParameter<std::vector<double> >("TS4TS5LowerThreshold");
  std::vector<double> TS4TS5LowerCutTemp = iConfig.getParameter<std::vector<double> >("TS4TS5LowerCut");

  for(int i = 0; i < (int)TS4TS5UpperThresholdTemp.size() && i < (int)TS4TS5UpperCutTemp.size(); i++)
     TS4TS5UpperCut_.push_back(std::pair<double, double>(TS4TS5UpperThresholdTemp[i], TS4TS5UpperCutTemp[i]));
  sort(TS4TS5UpperCut_.begin(), TS4TS5UpperCut_.end());

  for(int i = 0; i < (int)TS4TS5LowerThresholdTemp.size() && i < (int)TS4TS5LowerCutTemp.size(); i++)
     TS4TS5LowerCut_.push_back(std::pair<double, double>(TS4TS5LowerThresholdTemp[i], TS4TS5LowerCutTemp[i]));
  sort(TS4TS5LowerCut_.begin(), TS4TS5LowerCut_.end());

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
  if(fillRecHits_)    fillrechits(iEvent, iSetup, rbxarray, summary);
  if(fillDigis_)      filldigis(iEvent, iSetup, rbxarray);
  if(fillCaloTowers_) fillcalotwrs(iEvent, iSetup, rbxarray, summary);
  if(fillTracks_)     filltracks(iEvent, iSetup, summary);

  // select those RBXs which are interesting
  // also look for the highest energy RBX
  HcalNoiseRBXArray::iterator maxit=rbxarray.begin();
  double maxenergy=-999;
  bool maxwritten=false;
  for(HcalNoiseRBXArray::iterator rit = rbxarray.begin(); rit!=rbxarray.end(); ++rit) {
    HcalNoiseRBX &rbx=(*rit);
    CommonHcalNoiseRBXData data(rbx, minRecHitE_, minLowHitE_, minHighHitE_, TS4TS5EnergyThreshold_,
      TS4TS5UpperCut_, TS4TS5LowerCut_);

    // find the highest energy rbx
    if(data.energy()>maxenergy) {
      maxenergy=data.energy();
      maxit=rit;
      maxwritten=false;
    }
    
    // find out if the rbx is problematic/noisy/etc.
    bool writerbx = algo_.isProblematic(data) || !algo_.passLooseNoiseFilter(data) ||
      !algo_.passTightNoiseFilter(data) || !algo_.passHighLevelNoiseFilter(data);
    
    // fill variables in the summary object not filled elsewhere
    fillOtherSummaryVariables(summary, data);

    if(writerbx) {
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

// ------------ here we fill specific variables in the summary object not already accounted for earlier
void 
HcalNoiseInfoProducer::fillOtherSummaryVariables(HcalNoiseSummary& summary, const CommonHcalNoiseRBXData& data) const
{
  // charge ratio
  if(algo_.passRatioThreshold(data) && data.validRatio()) {
    if(data.ratio()<summary.minE2Over10TS()) {
      summary.mine2ts_ = data.e2ts();
      summary.mine10ts_ = data.e10ts();    }
    if(data.ratio()>summary.maxE2Over10TS()) {
      summary.maxe2ts_ = data.e2ts();
      summary.maxe10ts_ = data.e10ts();
    }
  }

  // ADC zeros
  if(algo_.passZerosThreshold(data)) {
    if(data.numZeros()>summary.maxZeros()) {
      summary.maxzeros_ = data.numZeros();
    }
  }

  // hits count
  if(data.numHPDHits() > summary.maxHPDHits()) {
    summary.maxhpdhits_ = data.numHPDHits();
  }
  if(data.numRBXHits() > summary.maxRBXHits()) {
    summary.maxrbxhits_ = data.numRBXHits();
  }
  if(data.numHPDNoOtherHits() > summary.maxHPDNoOtherHits()) {
    summary.maxhpdhitsnoother_ = data.numHPDNoOtherHits();
  }

  // TS4TS5
  if(data.PassTS4TS5() == false)
     summary.hasBadRBXTS4TS5_ = true;

  // hit timing
  if(data.minLowEHitTime()<summary.min10GeVHitTime()) {
    summary.min10_ = data.minLowEHitTime();
  }
  if(data.maxLowEHitTime()>summary.max10GeVHitTime()) {
    summary.max10_ = data.maxLowEHitTime();
  }
  summary.rms10_ += data.lowEHitTimeSqrd();
  summary.cnthit10_ += data.numLowEHits();
  if(data.minHighEHitTime()<summary.min25GeVHitTime()) {
    summary.min25_ = data.minHighEHitTime();
  }
  if(data.maxHighEHitTime()>summary.max25GeVHitTime()) {
    summary.max25_ = data.maxHighEHitTime();
  }
  summary.rms25_ += data.highEHitTimeSqrd();
  summary.cnthit25_ += data.numHighEHits();

  // EMF
  if(algo_.passEMFThreshold(data)) {
    if(summary.minHPDEMF() > data.HPDEMF()) {
      summary.minhpdemf_ = data.HPDEMF();
    }
    if(summary.minRBXEMF() > data.RBXEMF()) {
      summary.minrbxemf_ = data.RBXEMF();
    }
  }

  // summary flag
  if(!algo_.passLooseRatio(data))  summary.filterstatus_ |= 0x1;
  if(!algo_.passLooseHits(data))   summary.filterstatus_ |= 0x2;
  if(!algo_.passLooseZeros(data))  summary.filterstatus_ |= 0x4;
  if(!algo_.passLooseTiming(data)) summary.filterstatus_ |= 0x8;

  if(!algo_.passTightRatio(data))  summary.filterstatus_ |= 0x100;
  if(!algo_.passTightHits(data))   summary.filterstatus_ |= 0x200;
  if(!algo_.passTightZeros(data))  summary.filterstatus_ |= 0x400;
  if(!algo_.passTightTiming(data)) summary.filterstatus_ |= 0x800;

  if(!algo_.passHighLevelNoiseFilter(data)) summary.filterstatus_ |= 0x10000;

  // summary refvectors
  JoinCaloTowerRefVectorsWithoutDuplicates join;
  if(!algo_.passLooseNoiseFilter(data))
    join(summary.loosenoisetwrs_, data.rbxTowers());
  if(!algo_.passTightNoiseFilter(data))
    join(summary.tightnoisetwrs_, data.rbxTowers());
  if(!algo_.passHighLevelNoiseFilter(data))
    join(summary.hlnoisetwrs_, data.rbxTowers());

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
  const HcalChannelQuality* myqual = qualhandle.product();
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
	isRBX=true;
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
HcalNoiseInfoProducer::fillrechits(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array, HcalNoiseSummary& summary) const
{
  // get the HCAL channel status map
  edm::ESHandle<HcalChannelQuality> hcalChStatus;    
  iSetup.get<HcalChannelQualityRcd>().get( hcalChStatus );
  const HcalChannelQuality* dbHcalChStatus = hcalChStatus.product();

  // get the severity level computer
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
  iSetup.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComputerHndl);
  const HcalSeverityLevelComputer* hcalSevLvlComputer = hcalSevLvlComputerHndl.product();

  // get the calo geometry
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

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

    // skip bad rechits (other than those flagged by the isolated noise algorithm)
    const DetId id = rechit.detid();
    uint32_t recHitFlag = rechit.flags();    
    uint32_t noisebitset = (1 << HcalCaloFlagLabels::HBHEIsolatedNoise);
    uint32_t flatbitset = (1 << HcalCaloFlagLabels::HBHEFlatNoise);
    uint32_t spikebitset = (1 << HcalCaloFlagLabels::HBHESpikeNoise);
    uint32_t trianglebitset = (1 << HcalCaloFlagLabels::HBHETriangleNoise);
    uint32_t ts4ts5bitset = (1 << HcalCaloFlagLabels::HBHETS4TS5Noise);
    recHitFlag = (recHitFlag & noisebitset) ? recHitFlag-noisebitset : recHitFlag;
    const uint32_t dbStatusFlag = dbHcalChStatus->getValues(id)->getValue();
    int severityLevel = hcalSevLvlComputer->getSeverityLevel(id, recHitFlag, dbStatusFlag);
    bool isRecovered  = hcalSevLvlComputer->recoveredRecHit(id, recHitFlag);
    if(severityLevel!=0 && !isRecovered && severityLevel>static_cast<int>(HcalAcceptSeverityLevel_)) continue;

    // if it was ID'd as isolated noise, update the summary object
    if(rechit.flags() & noisebitset) {
      summary.nisolnoise_++;
      summary.isolnoisee_ += rechit.energy();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.energy()*gp.perp()/gp.mag();
      summary.isolnoiseet_ += et;
    }

    if(rechit.flags() & flatbitset) {
      summary.nflatnoise_++;
      summary.flatnoisee_ += rechit.energy();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.energy()*gp.perp()/gp.mag();
      summary.flatnoiseet_ += et;
    }

    if(rechit.flags() & spikebitset) {
      summary.nspikenoise_++;
      summary.spikenoisee_ += rechit.energy();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.energy()*gp.perp()/gp.mag();
      summary.spikenoiseet_ += et;
    }

    if(rechit.flags() & trianglebitset) {
      summary.ntrianglenoise_++;
      summary.trianglenoisee_ += rechit.energy();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.energy()*gp.perp()/gp.mag();
      summary.trianglenoiseet_ += et;
    }

    if(rechit.flags() & ts4ts5bitset) {
      summary.nts4ts5noise_++;
      summary.ts4ts5noisee_ += rechit.energy();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.energy()*gp.perp()/gp.mag();
      summary.ts4ts5noiseet_ += et;
    }

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
