#include "RecoMET/METProducers/interface/HcalNoiseInfoProducer.h"

//
// HcalNoiseInfoProducer.cc
//
//   description: Implementation of the producer for the HCAL noise information
//
//   author: J.P. Chou, Brown
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

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
  fillLaserMonitor_  = iConfig.getParameter<bool>("fillLaserMonitor");

  maxProblemRBXs_    = iConfig.getParameter<int>("maxProblemRBXs");

  maxCaloTowerIEta_  = iConfig.getParameter<int>("maxCaloTowerIEta");
  maxTrackEta_       = iConfig.getParameter<double>("maxTrackEta");
  minTrackPt_        = iConfig.getParameter<double>("minTrackPt");

  digiCollName_      = iConfig.getParameter<std::string>("digiCollName");
  recHitCollName_    = iConfig.getParameter<std::string>("recHitCollName");
  caloTowerCollName_ = iConfig.getParameter<std::string>("caloTowerCollName");
  trackCollName_     = iConfig.getParameter<std::string>("trackCollName");

  if (iConfig.existsAs<std::string>("jetCollName"))
  {
      jetCollName_   = iConfig.getParameter<std::string>("jetCollName");
      maxNHF_        = iConfig.getParameter<double>("maxNHF");
      maxjetindex_   = iConfig.getParameter<int>("maxjetindex");
      jet_token_     = consumes<reco::PFJetCollection>(edm::InputTag(jetCollName_));
  }

  minRecHitE_        = iConfig.getParameter<double>("minRecHitE");
  minLowHitE_        = iConfig.getParameter<double>("minLowHitE");
  minHighHitE_       = iConfig.getParameter<double>("minHighHitE");
  if(iConfig.existsAs<double>("minR45HitE"))
     minR45HitE_        = iConfig.getParameter<double>("minR45HitE");

  HcalAcceptSeverityLevel_ = iConfig.getParameter<uint32_t>("HcalAcceptSeverityLevel");
  if (iConfig.exists("HcalRecHitFlagsToBeExcluded"))
      HcalRecHitFlagsToBeExcluded_ = iConfig.getParameter<std::vector<int> >("HcalRecHitFlagsToBeExcluded");
  else{
    edm::LogWarning("MisConfiguration")<<"the module is missing the parameter HcalAcceptSeverityLevel. created empty.";
    HcalRecHitFlagsToBeExcluded_.resize(0);
  }

  // Digi threshold and time slices to use for HBHE and HF calibration digis
  useCalibDigi_ = true;
  if(iConfig.existsAs<double>("calibdigiHBHEthreshold") == false)               useCalibDigi_ = false;
  if(iConfig.existsAs<double>("calibdigiHFthreshold") == false)                 useCalibDigi_ = false;
  if(iConfig.existsAs<std::vector<int> >("calibdigiHBHEtimeslices") == false)   useCalibDigi_ = false;
  if(iConfig.existsAs<std::vector<int> >("calibdigiHFtimeslices") == false)     useCalibDigi_ = false;

  if(useCalibDigi_ == true)
  {
    calibdigiHBHEthreshold_   = iConfig.getParameter<double>("calibdigiHBHEthreshold");
    calibdigiHBHEtimeslices_  = iConfig.getParameter<std::vector<int> >("calibdigiHBHEtimeslices");
    calibdigiHFthreshold_   = iConfig.getParameter<double>("calibdigiHFthreshold");
    calibdigiHFtimeslices_  = iConfig.getParameter<std::vector<int> >("calibdigiHFtimeslices");
  }
  else
  {
     calibdigiHBHEthreshold_ = 0;
     calibdigiHBHEtimeslices_ = std::vector<int>();
     calibdigiHFthreshold_ = 0;
     calibdigiHFtimeslices_ = std::vector<int>();
  }

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

  // get the fiber configuration vectors
  std::vector<int> TmpLaserMonDetTypeList = iConfig.getParameter<std::vector<int> >("LaserMonDetTypeList");
  std::vector<int> TmpLaserMonIPhiList = iConfig.getParameter<std::vector<int> >("LaserMonIPhiList");
  std::vector<int> TmpLaserMonIEtaList = iConfig.getParameter<std::vector<int> >("LaserMonIEtaList");

  // the transfer of data from python to c seems to have issues
  // this can be fixed by explicitly filling the vector
  for( std::vector<int>::const_iterator itr = TmpLaserMonDetTypeList.begin();
          itr != TmpLaserMonDetTypeList.end(); ++itr ) {
      LaserMonDetTypeList_.push_back( *itr );
  }

  for( std::vector<int>::const_iterator itr = TmpLaserMonIPhiList.begin();
          itr != TmpLaserMonIPhiList.end(); ++itr ) {
      LaserMonIPhiList_.push_back( *itr );
  }

  for( std::vector<int>::const_iterator itr = TmpLaserMonIEtaList.begin();
          itr != TmpLaserMonIEtaList.end(); ++itr ) {
      LaserMonIEtaList_.push_back( *itr );
  }

  // check that the vectors have the same size, if not
  // disable the laser monitor
  if( !( (LaserMonDetTypeList_.size() == LaserMonIEtaList_.size() ) && 
         (LaserMonDetTypeList_.size() == LaserMonIPhiList_.size() ) ) ) { 
    edm::LogWarning("MisConfiguration")<<"Must provide equally sized lists for LaserMonDetTypeList, LaserMonIEtaList, and LaserMonIPhiList.  Will not fill LaserMon\n";
    fillLaserMonitor_=false;
  }

  // get the integration region with defaults
  if( iConfig.existsAs<int>("LaserMonitorTSStart" ) ) {
    LaserMonitorTSStart_ = iConfig.getParameter<int>("LaserMonitorTSStart");
  } else {
    LaserMonitorTSStart_ = 0;
  }

  if( iConfig.existsAs<int>("LaserMonitorTSEnd") ) {
    LaserMonitorTSEnd_   = iConfig.getParameter<int>("LaserMonitorTSEnd");
  } else {
    LaserMonitorTSEnd_ = -1;
  }

  const float adc2fCTemp[128]={-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,
     13.5,15.,17.,19.,21.,23.,25.,27.,29.5,32.5,35.5,38.5,42.,46.,50.,54.5,59.5,
     64.5,59.5,64.5,69.5,74.5,79.5,84.5,89.5,94.5,99.5,104.5,109.5,114.5,119.5,
     124.5,129.5,137.,147.,157.,167.,177.,187.,197.,209.5,224.5,239.5,254.5,272.,
     292.,312.,334.5,359.5,384.5,359.5,384.5,409.5,434.5,459.5,484.5,509.5,534.5,
     559.5,584.5,609.5,634.5,659.5,684.5,709.5,747.,797.,847.,897.,947.,997.,
     1047.,1109.5,1184.5,1259.5,1334.5,1422.,1522.,1622.,1734.5,1859.5,1984.5,
     1859.5,1984.5,2109.5,2234.5,2359.5,2484.5,2609.5,2734.5,2859.5,2984.5,
     3109.5,3234.5,3359.5,3484.5,3609.5,3797.,4047.,4297.,4547.,4797.,5047.,
     5297.,5609.5,5984.5,6359.5,6734.5,7172.,7672.,8172.,8734.5,9359.5,9984.5};
  for(int i = 0; i < 128; i++)
     adc2fC[i] = adc2fCTemp[i];

  // adc -> fC for qie8 with PMT input, for laser monitor
  const float adc2fCTempHF[128]={-3,-0.4,2.2,4.8,7.4,10,12.6,15.2,17.8,20.4,23,25.6,28.2,30.8,33.4,
                                 36,41.2,46.4,51.6,56.8,62,67.2,73,80.8,88.6,96.4,104,114.4,124.8,135,
                                 148,161,150,163,176,189,202,215,228,241,254,267,280,293,306,319,332,
                                 343,369,395,421,447,473,499,525,564,603,642,681,733,785,837,902,967,
                                 902,967,1032,1097,1162,1227,1292,1357,1422,1487,1552,1617,1682,1747,
                                 1812,1877,2007,2137,2267,2397,2527,2657,2787,2982,3177,3372,3567,
                                 3827,4087,4347,4672,4997,4672,4997,5322,5647,5972,6297,6622,6947,
                                 7272,7597,7922,8247,8572,8897,9222,9547,10197,10847,11497,12147,
                                 12797,13447,14097,15072,16047,17022,17997,19297,20597,21897,23522,25147};
  for(int i = 0; i < 128; i++)
     adc2fCHF[i] = adc2fCTempHF[i];

  hbhedigi_token_      = consumes<HBHEDigiCollection>(edm::InputTag(digiCollName_));
  hcalcalibdigi_token_ = consumes<HcalCalibDigiCollection>(edm::InputTag("hcalDigis"));
  hbherechit_token_    = consumes<HBHERecHitCollection>(edm::InputTag(recHitCollName_));
  calotower_token_     = consumes<CaloTowerCollection>(edm::InputTag(caloTowerCollName_));
  track_token_         = consumes<reco::TrackCollection>(edm::InputTag(trackCollName_));

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
  auto result1 = std::make_unique<HcalNoiseRBXCollection>();
  auto result2 = std::make_unique<HcalNoiseSummary>();

  // define an empty HcalNoiseRBXArray that we're going to fill
  HcalNoiseRBXArray rbxarray;
  HcalNoiseSummary &summary=*result2;

  // Get topology class to use later
  edm::ESHandle<HcalTopology> topo;
  iSetup.get<HcalRecNumberingRecord>().get(topo);
  theHcalTopology_ = topo.product();
  
  // fill them with the various components
  // digi assumes that rechit information is available
  if(fillRecHits_)    fillrechits(iEvent, iSetup, rbxarray, summary);
  if(fillDigis_)      filldigis(iEvent, iSetup, rbxarray, summary);
  if(fillCaloTowers_) fillcalotwrs(iEvent, iSetup, rbxarray, summary);
  if(fillTracks_)     filltracks(iEvent, iSetup, summary);

  filljetinfo(iEvent, iSetup, summary);

  // Why is this here?  Shouldn't it have been in the filldigis method? Any reason for TotalCalibCharge to be defined outside filldigis(...) ?-- Jeff, 7/2/12
  //if(fillDigis_)      summary.calibCharge_ = TotalCalibCharge;

  // select those RBXs which are interesting
  // also look for the highest energy RBX
  HcalNoiseRBXArray::iterator maxit=rbxarray.begin();
  double maxenergy=-999;
  bool maxwritten=false;
  for(HcalNoiseRBXArray::iterator rit = rbxarray.begin(); rit!=rbxarray.end(); ++rit) {
    HcalNoiseRBX &rbx=(*rit);
    CommonHcalNoiseRBXData data(rbx, minRecHitE_, minLowHitE_, minHighHitE_, TS4TS5EnergyThreshold_,
      TS4TS5UpperCut_, TS4TS5LowerCut_, minR45HitE_);

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
  iEvent.put(std::move(result1));
  iEvent.put(std::move(result2));
  
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

  if(algo_.passLooseRBXRechitR45(data) == false)
     summary.hasBadRBXRechitR45Loose_ = true;
  if(algo_.passTightRBXRechitR45(data) == false)
     summary.hasBadRBXRechitR45Tight_ = true;

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
HcalNoiseInfoProducer::filldigis(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array, HcalNoiseSummary& summary)
{
  // Some initialization
  TotalCalibCharge = 0;
  TotalLasmonCharge = 0;

  // Starting with this version (updated by Jeff Temple on Dec. 6, 2012), the "TS45" names in the variables are mis-nomers.  The actual time slices used are determined from the digiTimeSlices_ variable, which may not be limited to only time slices 4 and 5.  For now, "TS45" name kept, because that is what is used in HcalNoiseSummary object (in GetCalibCountTS45, etc.).  Likewise, the charge value in 'gt15' is now configurable, though the name remains the same.  For HBHE, we track both the number of calibration channels (NcalibTS45) and the number of calibration channels above threshold (NcalibTS45gt15).  For HF, we track only the number of channels above the given threshold in the given time window (NcalibHFgtX).  Default for HF in 2012 is to use the full time sample with effectively no threshold (threshold=-999)
  int NcalibTS45=0;
  int NcalibTS45gt15=0;
  int NcalibHFgtX=0;

  double chargecalibTS45=0;
  double chargecalibgt15TS45=0;

  // get the conditions and channel quality
  edm::ESHandle<HcalDbService> conditions;
  iSetup.get<HcalDbRecord>().get(conditions);
  edm::ESHandle<HcalChannelQuality> qualhandle;
  iSetup.get<HcalChannelQualityRcd>().get("withTopo",qualhandle);
  const HcalChannelQuality* myqual = qualhandle.product();
  edm::ESHandle<HcalSeverityLevelComputer> mycomputer;
  iSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);
  const HcalSeverityLevelComputer* mySeverity = mycomputer.product();

  // get the digis
  edm::Handle<HBHEDigiCollection> handle;
  //  iEvent.getByLabel(digiCollName_, handle);
  iEvent.getByToken(hbhedigi_token_, handle);

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
    // Drop if exclude bit set
    if ((mydigistatus->getValue() & (1 <<HcalChannelStatus::HcalCellExcludeFromHBHENoiseSummary))==1) continue;
      
    // get the calibrations and coder
    const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
    const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
    const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
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
      const HcalDetId & detid = (*rit)->idFront();
      if(DetId(detid) == digi.id()) {
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

  // get the calibration digis
  edm::Handle<HcalCalibDigiCollection> hCalib;
  //  iEvent.getByLabel("hcalDigis", hCalib);
  iEvent.getByToken(hcalcalibdigi_token_, hCalib);

  // get total charge in calibration channels
  if(hCalib.isValid() == true)
  {

     // store the data from the lasermon fibers
     std::map<int, std::vector<int> > lasmon_adcs;
     std::map<int, std::vector<int> > lasmon_capids;

     // we may find the fibers in different orders, initialize them here
     if( fillLaserMonitor_ ) {
         for( unsigned i = 0; i < LaserMonDetTypeList_.size() ; ++i ) {
             lasmon_adcs[i] = std::vector<int>();
             lasmon_capids[i] = std::vector<int>();
         }
     }

     for(HcalCalibDigiCollection::const_iterator digi = hCalib->begin(); digi != hCalib->end(); digi++)
     {
        if(digi->id().hcalSubdet() == 0)
           continue;

        /*
        HcalCalibDetId cell = digi->id();
        DetId detcell = (DetId)cell;
        
        const HcalChannelStatus* mydigistatus=myqual->getValues(detcell.rawId());

        if(mySeverity->dropChannel(mydigistatus->getValue()))
           continue;
        if(digi->zsMarkAndPass())
           continue;

        const HcalQIECoder *channelCoder = conditions->getHcalCoder(cell);
	const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
        HcalCoderDb coder(*channelCoder, *shape);

        CaloSamples tool;
        coder.adc2fC(*digi, tool);

        for(int i = 0; i < (int)digi->size(); i++)
           TotalCalibCharge = TotalCalibCharge + tool[i];
        */

	// Original code computes total calib charge over all digis.  While I think it would be more useful to skip
	// zs mark-and-pass channels, I keep this computation as is.  Individual HBHE and HF variables do skip
	// the m-p channels.  -- Jeff Temple, 6 December 2012
        
        // Fill the lasermonitor channels
        if( fillLaserMonitor_ ) {
          int dettype = digi->id().hcalSubdet( );
          int iphi    = digi->id().iphi();
          int ieta    = digi->id().ieta();
         
          // check that we have the correct dettype
          if (std::find( LaserMonDetTypeList_.begin(), LaserMonDetTypeList_.end(),
                         dettype ) != LaserMonDetTypeList_.end() ) {
              // check that we have a contained IPhi
            if( std::find( LaserMonIPhiList_.begin(), LaserMonIPhiList_.end(),
                           iphi ) != LaserMonIPhiList_.end() ) {
              // check that we have a contained IEta
              if( std::find( LaserMonIEtaList_.begin(), LaserMonIEtaList_.end(),
                             ieta ) != LaserMonIEtaList_.end() ) {
                // we have a lasmon channel, find the index in the list of inputs
                for( unsigned idx = 0; idx < LaserMonDetTypeList_.size(); ++idx ) {
                  if( dettype == LaserMonDetTypeList_[idx] &&
                        iphi  == LaserMonIPhiList_[idx] && 
                        ieta  == LaserMonIEtaList_[idx] ) {

                    // now get the digis
                    int ts_size = int(digi->size());
	            for(int i = 0; i < ts_size; i++) {
                      lasmon_adcs[idx].push_back( digi->sample(i).adc() );
                      lasmon_capids[idx].push_back( digi->sample(i).capid() );
                    } // end digi loop
                  } // end matching channel if
                } // end fiber order loop
              } // end ieta check 
            } // end iphi check
          } // end dettype check
        } // end filllasmon check


	for(int i = 0; i < (int)digi->size(); i++)
	  TotalCalibCharge = TotalCalibCharge + adc2fC[digi->sample(i).adc()&0xff];
	

	HcalCalibDetId myid=(HcalCalibDetId)digi->id();
	if ( myid.calibFlavor()==HcalCalibDetId::HOCrosstalk)
	  continue; // ignore HOCrosstalk channels
	if(digi->zsMarkAndPass()) continue;  // skip "mark-and-pass" channels when computing charge in calib channels


	if (digi->id().hcalSubdet()==HcalForward) // check HF
	  {
	    double sumChargeHF=0;
	    for (unsigned int i=0;i<calibdigiHFtimeslices_.size();++i)
	      {
		// skip unphysical time slices
		if (calibdigiHFtimeslices_[i]<0 || calibdigiHFtimeslices_[i]>digi->size())
		  continue;
		sumChargeHF+=adc2fC[digi->sample(calibdigiHFtimeslices_[i]).adc()&0xff];
	      }
	    if (sumChargeHF>calibdigiHFthreshold_) ++NcalibHFgtX;
	  } // end of HF check
	else if (digi->id().hcalSubdet()==HcalBarrel || digi->id().hcalSubdet()==HcalEndcap) // now check HBHE
	  {
            double sumChargeHBHE=0;
            for (unsigned int i=0;i<calibdigiHBHEtimeslices_.size();++i)
              {
                // skip unphysical time slices
                if (calibdigiHBHEtimeslices_[i]<0 || calibdigiHBHEtimeslices_[i]>digi->size())
                  continue;
                sumChargeHBHE+=adc2fC[digi->sample(calibdigiHBHEtimeslices_[i]).adc()&0xff];
              }
	    ++NcalibTS45;
	    chargecalibTS45+=sumChargeHBHE;
            if (sumChargeHBHE>calibdigiHBHEthreshold_) 
	      {
		++NcalibTS45gt15;
		chargecalibgt15TS45+=sumChargeHBHE;
	      }
          } // end of HBHE check
     } // loop on HcalCalibDigiCollection

     // now match the laser monitor data by fiber (in time) 
     if( fillLaserMonitor_ ) {
       // check for any fibers without data and fill
       // them so we dont run into problems later
       for( unsigned idx = 0; idx < LaserMonDetTypeList_.size(); ++idx ) {
           if( lasmon_adcs[idx].size() == 0 ) {
               for( int i = 0; i < 10; ++i ) {
                   lasmon_adcs[idx].push_back( -1 );
               }
           }
           if( lasmon_capids[idx].size() == 0 ) {
               for( int i = 0; i < 10; ++i ) {
                   lasmon_capids[idx].push_back( -1 );
               }
           }
       }
       int nFibers = LaserMonIEtaList_.size();
       // for each fiber we need to find the index at with the 
       // data from the next fiber matches in order to stitch them together.
       // When there is an overlap, the data from the end of the
       // earlier fiber is removed.  There is no removal of the last fiber
       std::map<int, int> matching_idx; 
       // we assume that the list of fibers was given in time order
       // (if this was not the case, then we just end up using 
       // all data from all fibers )
       for( int fidx = 0; fidx < (nFibers - 1); ++fidx ) {
         // start with the last TS and loop backwards
         // on each iteration check if all capId and ADCs from this
         // TS forward match the beginning entries of 
         // the next fiber
         int last_ts = lasmon_capids[fidx].size()-1;  // last TS
         int start_ts = last_ts; // start_ts will be decrimented on each loop
         // in the case that our stringent check below doesn't work 
         // store the latest capID that has a match
         int latest_cap_match = -1;
         // reverse loop over TSs
         while( start_ts >= 0 ) {
           int ncheck = 0; // count the number of TS that were checked
           int nmatch_cap = 0; // count the number of TS where capID matched
           int nmatch_adc = 0; // count the number of TS where ADC matched

           for( int cidx = start_ts, nidx = 0; cidx <= last_ts; cidx++, nidx++ ) { 
             ncheck++;
             // if we get an invald value, move on
             if( lasmon_capids[fidx][cidx] == -1 ) continue;

             if( lasmon_capids[fidx][cidx] == lasmon_capids[fidx+1][nidx] ) {
               nmatch_cap++;
             }
             // check the data values as well
             if( lasmon_adcs[fidx][cidx] == lasmon_adcs[fidx+1][nidx] ) {
               nmatch_adc++;
             }
           }
           bool cap_match = (ncheck == nmatch_cap);
           bool adc_match = (ncheck == nmatch_adc);
           if( cap_match && start_ts > latest_cap_match ) {
             latest_cap_match = start_ts;
           }
           if( cap_match && adc_match ) {
             // end the loop and we'll take the current start_ts
             // as the end of the data for this fiber
             break;
           }
           else {
             // if we don't have a match, then decrement the 
             // starting TS and check again
             start_ts--;
           }
         }

         // now make some sanity checks on the determined overlap index

         if( start_ts == -1 ) {
           // if we didn't find any match, use the capID only to compare
           if( latest_cap_match < 0 ) {
             //this shouldn't happen, in this case use all the data from the fiber
             start_ts = lasmon_capids[fidx].size();
           }
           else {
             // its possible that the timing of the fibers
             // is shifted such that they do not overlap
             // and we just want to stitch the fibers
             // together with no removal.
             // In this case the capIDs will match at the
             // N-4 spot (and the ADCs will not)
             // if this is not the case, then we just take
             // the value of latest match
             if( latest_cap_match == (last_ts - 3) ) {
               start_ts = lasmon_capids[fidx].size();
             } else {
               start_ts = latest_cap_match;
             }
           }
         }

         // now store as the matching index
         matching_idx[fidx] = start_ts;
       }

       // for the last fiber we always use all of the data
       matching_idx[nFibers - 1] = 10;

       // now loop over the time slices of each fiber and make the sum
       int icombts = -1;
       for( int fidx = 0 ; fidx < nFibers; ++fidx ) {
         for( int its = 0; its < matching_idx[fidx]; ++its ) {
           icombts++;

           // apply integration limits
           if( icombts < LaserMonitorTSStart_ ) continue;
           if( LaserMonitorTSEnd_ > 0 && icombts > LaserMonitorTSEnd_ ) continue;

           int adc = lasmon_adcs[fidx][its];

           if( adc >= 0 ) { // skip invalid data
             float fc = adc2fCHF[adc];
             TotalLasmonCharge += fc;
           }

         } 
       }
     } // if( fillLaserMonitor_ )
  } // if (hCalib.isValid()==true)

  summary.calibCharge_ = TotalCalibCharge;
  summary.lasmonCharge_ = TotalLasmonCharge;
  summary.calibCountTS45_=NcalibTS45;
  summary.calibCountgt15TS45_=NcalibTS45gt15;
  summary.calibChargeTS45_=chargecalibTS45;
  summary.calibChargegt15TS45_=chargecalibgt15TS45;
  summary.calibCountHF_=NcalibHFgtX;

  return;
}

// ------------ fill the array with rec hit information
void
HcalNoiseInfoProducer::fillrechits(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseRBXArray& array, HcalNoiseSummary& summary) const
{
  // get the HCAL channel status map
  edm::ESHandle<HcalChannelQuality> hcalChStatus;
  iSetup.get<HcalChannelQualityRcd>().get( "withTopo", hcalChStatus );
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
  //  iEvent.getByLabel(recHitCollName_, handle);
  iEvent.getByToken(hbherechit_token_, handle);

  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find HBHERecHitCollection named " << recHitCollName_ << "\n.";
    return;
  }

  summary.rechitCount_ = 0;
  summary.rechitCount15_ = 0;
  summary.rechitEnergy_ = 0;
  summary.rechitEnergy15_ = 0;

  summary.hitsInLaserRegion_=0;
  summary.hitsInNonLaserRegion_=0;
  summary.energyInLaserRegion_=0;
  summary.energyInNonLaserRegion_=0;



  // loop over all of the rechit information
  for(HBHERecHitCollection::const_iterator it=handle->begin(); it!=handle->end(); ++it) {
    const HBHERecHit &rechit=(*it);

    // skip bad rechits (other than those flagged by the isolated noise, triangle, flat, and spike algorithms)
    const DetId id = rechit.idFront();

    uint32_t recHitFlag = rechit.flags();
    uint32_t isolbitset = (1 << HcalCaloFlagLabels::HBHEIsolatedNoise);
    uint32_t flatbitset = (1 << HcalCaloFlagLabels::HBHEFlatNoise);
    uint32_t spikebitset = (1 << HcalCaloFlagLabels::HBHESpikeNoise);
    uint32_t trianglebitset = (1 << HcalCaloFlagLabels::HBHETriangleNoise);
    uint32_t ts4ts5bitset = (1 << HcalCaloFlagLabels::HBHETS4TS5Noise);
    uint32_t negativebitset = (1 << HcalCaloFlagLabels::HBHENegativeNoise);
    for(unsigned int i=0; i<HcalRecHitFlagsToBeExcluded_.size(); i++) {
      uint32_t bitset = (1 << HcalRecHitFlagsToBeExcluded_[i]);
      recHitFlag = (recHitFlag & bitset) ? recHitFlag-bitset : recHitFlag;
    }
    const uint32_t dbStatusFlag = dbHcalChStatus->getValues(id)->getValue();

    // Ignore rechit if exclude bit set, regardless of severity of other bits
    if ((dbStatusFlag & (1 <<HcalChannelStatus::HcalCellExcludeFromHBHENoiseSummary))==1) continue;
      
    int severityLevel = hcalSevLvlComputer->getSeverityLevel(id, recHitFlag, dbStatusFlag);
    bool isRecovered  = hcalSevLvlComputer->recoveredRecHit(id, recHitFlag);
    if(severityLevel!=0 && !isRecovered && severityLevel>static_cast<int>(HcalAcceptSeverityLevel_)) continue;

    // do some rechit counting and energies
    summary.rechitCount_ = summary.rechitCount_ + 1;
    summary.rechitEnergy_ = summary.rechitEnergy_ + rechit.eraw();
    if ((dbStatusFlag & (1 <<HcalChannelStatus::HcalBadLaserSignal))==1) // hit comes from a region where no laser calibration pulse is normally seen
      {
	++summary.hitsInNonLaserRegion_;
	summary.energyInNonLaserRegion_+=rechit.eraw();
      }
    else // hit comes from region where laser calibration pulse is seen
      {
	++summary.hitsInLaserRegion_;
	summary.energyInLaserRegion_+=rechit.eraw();
      }

    if(rechit.eraw() > 1.5)
    {
      summary.rechitCount15_ = summary.rechitCount15_ + 1;
      summary.rechitEnergy15_ = summary.rechitEnergy15_ + rechit.eraw();
    }

    // if it was ID'd as isolated noise, update the summary object
    if(rechit.flags() & isolbitset) {
      summary.nisolnoise_++;
      summary.isolnoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw()*gp.perp()/gp.mag();
      summary.isolnoiseet_ += et;
    }

    if(rechit.flags() & flatbitset) {
      summary.nflatnoise_++;
      summary.flatnoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw()*gp.perp()/gp.mag();
      summary.flatnoiseet_ += et;
    }

    if(rechit.flags() & spikebitset) {
      summary.nspikenoise_++;
      summary.spikenoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw()*gp.perp()/gp.mag();
      summary.spikenoiseet_ += et;
    }

    if(rechit.flags() & trianglebitset) {
      summary.ntrianglenoise_++;
      summary.trianglenoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw()*gp.perp()/gp.mag();
      summary.trianglenoiseet_ += et;
    }

    if(rechit.flags() & ts4ts5bitset) {
      if ((dbStatusFlag & (1 <<HcalChannelStatus::HcalCellExcludeFromHBHENoiseSummaryR45))==0)  // only add to TS4TS5 if the bit is not marked as "HcalCellExcludeFromHBHENoiseSummaryR45"
	{
	  summary.nts4ts5noise_++;
	  summary.ts4ts5noisee_ += rechit.eraw();
	  GlobalPoint gp = geo->getPosition(rechit.id());
	  double et = rechit.eraw()*gp.perp()/gp.mag();
	  summary.ts4ts5noiseet_ += et;
	}
    }
    
    if(rechit.flags() & negativebitset) {
	  summary.nnegativenoise_++;
	  summary.negativenoisee_ += rechit.eraw();
	  GlobalPoint gp = geo->getPosition(rechit.id());
	  double et = rechit.eraw()*gp.perp()/gp.mag();
	  summary.negativenoiseet_ += et;
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
  //  iEvent.getByLabel(caloTowerCollName_, handle);
  iEvent.getByToken(calotower_token_, handle);

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

// ------------ fill the summary info from jets
void
HcalNoiseInfoProducer::filljetinfo(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseSummary& summary) const
{
    bool goodJetFoundInLowBVRegion = false; // checks whether a jet is in
                                            // a low BV region, where false
                                            // noise flagging rate is higher.
    if (!jetCollName_.empty())
    {
        edm::Handle<reco::PFJetCollection> pfjet_h;
        iEvent.getByToken(jet_token_, pfjet_h);

        if (pfjet_h.isValid())
        {
            int jetindex=0;
            for(reco::PFJetCollection::const_iterator jet = pfjet_h->begin();
                jet != pfjet_h->end(); ++jet)
            {
                if (jetindex>maxjetindex_) break; // only look at jets with
                                                  // indices up to maxjetindex_

                // Check whether jet is in low-BV region (0<eta<1.4, -1.8<phi<-1.4)
                if (jet->eta()>0.0 && jet->eta()<1.4 &&
                    jet->phi()>-1.8 && jet->phi()<-1.4)
                {
                    // Look for a good jet in low BV region;
                    // if found, we will keep event
                    if  (maxNHF_<0.0 || jet->neutralHadronEnergyFraction()<maxNHF_)
                    {
                        goodJetFoundInLowBVRegion=true;
                        break;
                    }
                }
                ++jetindex;
            }
        }
    }

    summary.goodJetFoundInLowBVRegion_ = goodJetFoundInLowBVRegion;
}

// ------------ fill the summary with track information
void
HcalNoiseInfoProducer::filltracks(edm::Event& iEvent, const edm::EventSetup& iSetup, HcalNoiseSummary& summary) const
{
  edm::Handle<reco::TrackCollection> handle;
  //  iEvent.getByLabel(trackCollName_, handle);
  iEvent.getByToken(track_token_, handle);

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
