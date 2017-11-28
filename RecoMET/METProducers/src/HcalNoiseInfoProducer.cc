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

  jetCollName_   = iConfig.getParameter<std::string>("jetCollName");
  maxNHF_        = iConfig.getParameter<double>("maxNHF");
  maxjetindex_   = iConfig.getParameter<int>("maxjetindex");
  jet_token_     = consumes<reco::PFJetCollection>(edm::InputTag(jetCollName_));

  minRecHitE_        = iConfig.getParameter<double>("minRecHitE");
  minLowHitE_        = iConfig.getParameter<double>("minLowHitE");
  minHighHitE_       = iConfig.getParameter<double>("minHighHitE");
  
  minR45HitE_        = iConfig.getParameter<double>("minR45HitE");

  HcalAcceptSeverityLevel_ = iConfig.getParameter<uint32_t>("HcalAcceptSeverityLevel");
  HcalRecHitFlagsToBeExcluded_ = iConfig.getParameter<std::vector<int> >("HcalRecHitFlagsToBeExcluded");

  // Digi threshold and time slices to use for HBHE and HF calibration digis
  calibdigiHBHEthreshold_ = 0;
  calibdigiHBHEtimeslices_ = std::vector<int>();
  calibdigiHFthreshold_ = 0;
  calibdigiHFtimeslices_ = std::vector<int>();

  calibdigiHBHEthreshold_   = iConfig.getParameter<double>("calibdigiHBHEthreshold");
  calibdigiHBHEtimeslices_  = iConfig.getParameter<std::vector<int> >("calibdigiHBHEtimeslices");
  calibdigiHFthreshold_   = iConfig.getParameter<double>("calibdigiHFthreshold");
  calibdigiHFtimeslices_  = iConfig.getParameter<std::vector<int> >("calibdigiHFtimeslices");

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
  laserMonCBoxList_ = iConfig.getParameter<std::vector<int> >("laserMonCBoxList");
  laserMonIPhiList_ = iConfig.getParameter<std::vector<int> >("laserMonIPhiList");
  laserMonIEtaList_ = iConfig.getParameter<std::vector<int> >("laserMonIEtaList");

  // check that the vectors have the same size, if not
  // disable the laser monitor
  if( !( (laserMonCBoxList_.size() == laserMonIEtaList_.size() ) && 
         (laserMonCBoxList_.size() == laserMonIPhiList_.size() ) ) ) { 
    edm::LogWarning("MisConfiguration")<<"Must provide equally sized lists for laserMonCBoxList, laserMonIEtaList, and laserMonIPhiList.  Will not fill LaserMon\n";
    fillLaserMonitor_=false;
  }

  // get the integration region with defaults
  laserMonitorTSStart_ = iConfig.getParameter<int>("laserMonTSStart");
  laserMonitorTSEnd_   = iConfig.getParameter<int>("laserMonTSEnd");

  adc2fC= std::vector<float> {-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,
     13.5,15.,17.,19.,21.,23.,25.,27.,29.5,32.5,35.5,38.5,42.,46.,50.,54.5,59.5,
     64.5,59.5,64.5,69.5,74.5,79.5,84.5,89.5,94.5,99.5,104.5,109.5,114.5,119.5,
     124.5,129.5,137.,147.,157.,167.,177.,187.,197.,209.5,224.5,239.5,254.5,272.,
     292.,312.,334.5,359.5,384.5,359.5,384.5,409.5,434.5,459.5,484.5,509.5,534.5,
     559.5,584.5,609.5,634.5,659.5,684.5,709.5,747.,797.,847.,897.,947.,997.,
     1047.,1109.5,1184.5,1259.5,1334.5,1422.,1522.,1622.,1734.5,1859.5,1984.5,
     1859.5,1984.5,2109.5,2234.5,2359.5,2484.5,2609.5,2734.5,2859.5,2984.5,
     3109.5,3234.5,3359.5,3484.5,3609.5,3797.,4047.,4297.,4547.,4797.,5047.,
     5297.,5609.5,5984.5,6359.5,6734.5,7172.,7672.,8172.,8734.5,9359.5,9984.5};

  // adc -> fC for qie8 with PMT input, for laser monitor
  // Taken from Table 2 in 
  // https://cms-docdb.cern.ch/cgi-bin/DocDB/RetrieveFile?docid=3275&filename=qie_spec.pdf&version=1
  adc2fCHF = std::vector<float> {-3,-0.4,2.2,4.8,7.4,10,12.6,15.2,17.8,20.4,23,25.6,28.2,30.8,33.4,
                                 36,41.2,46.4,51.6,56.8,62,67.2,73,80.8,88.6,96.4,104,114.4,124.8,135,
                                 148,161,150,163,176,189,202,215,228,241,254,267,280,293,306,319,332,
                                 343,369,395,421,447,473,499,525,564,603,642,681,733,785,837,902,967,
                                 902,967,1032,1097,1162,1227,1292,1357,1422,1487,1552,1617,1682,1747,
                                 1812,1877,2007,2137,2267,2397,2527,2657,2787,2982,3177,3372,3567,
                                 3827,4087,4347,4672,4997,4672,4997,5322,5647,5972,6297,6622,6947,
                                 7272,7597,7922,8247,8572,8897,9222,9547,10197,10847,11497,12147,
                                 12797,13447,14097,15072,16047,17022,17997,19297,20597,21897,23522,25147};

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

void HcalNoiseInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // define hit energy thesholds
  desc.add<double>("minRecHitE", 1.5);
  desc.add<double>("minLowHitE", 10.0);
  desc.add<double>("minHighHitE", 25.0);
  desc.add<double>("minR45HitE", 5.0);

  // define energy threshold for "problematic" cuts
  desc.add<double>("pMinERatio", 25.0);
  desc.add<double>("pMinEZeros", 5.0);
  desc.add<double>("pMinEEMF", 10.0);

  // define energy threshold for loose/tight/high level cuts
  desc.add<double>("minERatio", 50.0);
  desc.add<double>("minEZeros", 10.0);
  desc.add<double>("minEEMF", 50.0);

  // define problematic RBX
  desc.add<double>("pMinE", 40.0);
  desc.add<double>("pMinRatio", 0.75);
  desc.add<double>("pMaxRatio", 0.85);
  desc.add<int>("pMinHPDHits", 10);
  desc.add<int>("pMinRBXHits", 20);
  desc.add<int>("pMinHPDNoOtherHits", 7);
  desc.add<int>("pMinZeros", 4);
  desc.add<double>("pMinLowEHitTime", -6.0);
  desc.add<double>("pMaxLowEHitTime", 6.0);
  desc.add<double>("pMinHighEHitTime", -4.0);
  desc.add<double>("pMaxHighEHitTime", 5.0);
  desc.add<double>("pMaxHPDEMF", -0.02);
  desc.add<double>("pMaxRBXEMF", 0.02);
  desc.add<int>("pMinRBXRechitR45Count", 1);
  desc.add<double>("pMinRBXRechitR45Fraction", 0.1);
  desc.add<double>("pMinRBXRechitR45EnergyFraction", 0.1);

  // define loose noise cuts
  desc.add<double>("lMinRatio", -999.0);
  desc.add<double>("lMaxRatio", 999.0);
  desc.add<int>("lMinHPDHits", 17);
  desc.add<int>("lMinRBXHits", 999);
  desc.add<int>("lMinHPDNoOtherHits", 10);
  desc.add<int>("lMinZeros", 10);
  desc.add<double>("lMinLowEHitTime", -9999.0);
  desc.add<double>("lMaxLowEHitTime", 9999.0);
  desc.add<double>("lMinHighEHitTime", -9999.0);
  desc.add<double>("lMaxHighEHitTime", 9999.0);

  // define tight noise cuts
  desc.add<double>("tMinRatio", -999.0);
  desc.add<double>("tMaxRatio", 999.0);
  desc.add<int>("tMinHPDHits", 16);
  desc.add<int>("tMinRBXHits", 50);
  desc.add<int>("tMinHPDNoOtherHits", 9);
  desc.add<int>("tMinZeros", 8);
  desc.add<double>("tMinLowEHitTime", -9999.0);
  desc.add<double>("tMaxLowEHitTime", 9999.0);
  desc.add<double>("tMinHighEHitTime", -7.0);
  desc.add<double>("tMaxHighEHitTime", 6.0);

  // define high level noise cuts
  desc.add<double>("hlMaxHPDEMF", -9999.0);
  desc.add<double>("hlMaxRBXEMF", 0.01);
  
  // Calibration digi noise variables (used for finding laser noise events)
  desc.add<double>("calibdigiHBHEthreshold", 15)->
      setComment("minimum threshold in fC of any HBHE  \
              calib digi to be counted in summary");
  desc.add<std::vector<int>>("calibdigiHBHEtimeslices", {3,4,5,6,})->
      setComment("time slices to use when determining charge of HBHE calib digis");
  desc.add<double>("calibdigiHFthreshold", -999)->
      setComment("minimum threshold in fC of any HF calib digi to be counted in summary");
  desc.add<std::vector<int>>("calibdigiHFtimeslices", {0,1,2,3,4,5,6,7,8,9,})->
      setComment("time slices to use when determining charge of HF calib digis");

  // RBX-wide TS4TS5 variable
  desc.add<double>("TS4TS5EnergyThreshold", 50);
  desc.add<std::vector<double>>("TS4TS5UpperThreshold", {70,90,100,400,4000,});
  desc.add<std::vector<double>>("TS4TS5UpperCut", {1,0.8,0.75,0.72,0.72,});
  desc.add<std::vector<double>>("TS4TS5LowerThreshold", {100,120,150,200,300,400,500,});
  desc.add<std::vector<double>>("TS4TS5LowerCut", {-1,-0.7,-0.4,-0.2,-0.08,0,0.1,});

  // rechit R45 population filter variables
  // this comes in groups of four: (a_Count, a_Fraction, a_EnergyFraction, const)
  // flag as noise if (count * a_count + fraction * a_fraction + energyfraction * a_energyfraction + const) > 0
  desc.add<std::vector<double>>("lRBXRecHitR45Cuts", 
                                {0.0,1.0,0.0,-0.5,0.0,0.0,1.0,-0.5,})->
      setComment("first 4 entries : equivalent to 'fraction > 0.5'  \
                  last 4 entries : equivalent to 'energy fraction > 0.5'");
  desc.add<std::vector<double>>("tRBXRecHitR45Cuts", 
                                {0.0,1.0,0.0,-0.2,0.0,0.0,1.0,-0.2,})->
      setComment("first 4 entries : equivalent to 'fraction > 0.2' \
                  last 4 entries : equivalent to 'energy fraction > 0.2'" );

  // define the channels used for laser monitoring
  // note that the order here indicates the time order
  // of the channels
  desc.add<std::vector<int>>("laserMonCBoxList", {6,6,6,6,6,6,6,6,})->
      setComment("time ordered list of the cBox values of laser monitor channels");
  desc.add<std::vector<int>>("laserMonIPhiList", {23,17,11,5,29,35,41,47,})->
      setComment("time ordered list of the iPhi values of laser monitor channels");
  desc.add<std::vector<int>>("laserMonIEtaList", {0,0,0,0,0,0,0,0,})->
      setComment("time ordered list of the iEta values of laser monitor channels");

  // boundaries for total charge integration
  desc.add<int>("laserMonTSStart", 0)->
      setComment("lower bound of laser monitor charge integration window");
  desc.add<int>("laserMonTSEnd", -1)->
      setComment("upper bound of laser monitor charge integration window (-1 = no bound)");

  // what to fill
  desc.add<bool>("fillDigis", true);
  desc.add<bool>("fillRecHits", true);
  desc.add<bool>("fillCaloTowers", true);
  desc.add<bool>("fillTracks", true);
  desc.add<bool>("fillLaserMonitor", true);

  // maximum number of RBXs to fill
  // if you want to record all RBXs above some energy threshold,
  // change maxProblemRBXs to 999 and pMinE (above) to the threshold you want
  desc.add<int>("maxProblemRBXs", 72)->
      setComment("maximum number of RBXs to fill.  if you want to record  \
              all RBXs above some energy threshold,change maxProblemRBXs to  \
              999 and pMinE (above) to the threshold you want");
;

  // parameters for calculating summary variables
  desc.add<int>("maxCaloTowerIEta", 20);
  desc.add<double>("maxTrackEta", 2.0);
  desc.add<double>("minTrackPt", 1.0);
  desc.add<double>("maxNHF", 0.9);
  desc.add<int>("maxjetindex", 0);

  // collection names
  desc.add<std::string>("digiCollName", "hcalDigis");
  desc.add<std::string>("recHitCollName", "hbhereco");
  desc.add<std::string>("caloTowerCollName", "towerMaker");
  desc.add<std::string>("trackCollName", "generalTracks");
  desc.add<std::string>("jetCollName", "ak4PFJets");

  // severity level
  desc.add<unsigned int>("HcalAcceptSeverityLevel", 9);

  // which hcal calo flags to mask 
  // (HBHEIsolatedNoise=11, HBHEFlatNoise=12, HBHESpikeNoise=13, 
  // HBHETriangleNoise=14, HBHETS4TS5Noise=15, HBHENegativeNoise=27)
  desc.add<std::vector<int>>("HcalRecHitFlagsToBeExcluded", {11,12,13,14,15,27,})->
      setComment("which hcal calo flags to mask (HBHEIsolatedNoise=11, \
                  HBHEFlatNoise=12, HBHESpikeNoise=13, \
                  HBHETriangleNoise=14, HBHETS4TS5Noise=15, HBHENegativeNoise=27)");
;

  descriptions.add("hcalnoise", desc);
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

  // Why is this here?  Shouldn't it have been in the filldigis method? Any reason for totalCalibCharge to be defined outside filldigis(...) ?-- Jeff, 7/2/12
  //if(fillDigis_)      summary.calibCharge_ = totalCalibCharge;

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
  totalCalibCharge = 0;
  totalLasmonCharge = 0;

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


     std::vector<std::vector<int> > lasmon_adcs(laserMonCBoxList_.size(), std::vector<int>());
     std::vector<std::vector<int> > lasmon_capids(laserMonCBoxList_.size(), std::vector<int>());

     for(HcalCalibDigiCollection::const_iterator digi = hCalib->begin(); digi != hCalib->end(); digi++)
     {
        if(digi->id().hcalSubdet() == 0)
           continue;

        // Fill the lasermonitor channels
        if( fillLaserMonitor_ ) {
          int cboxch  = digi->id().cboxChannel( );
          int iphi    = digi->id().iphi();
          int ieta    = digi->id().ieta();
         
          // only check channels having the requested cboxch
          if (std::find( laserMonCBoxList_.begin(), laserMonCBoxList_.end(),
                         cboxch ) != laserMonCBoxList_.end() ) {
            // find the index of this channel by matching cBox, iEta, iPhi
            for( unsigned idx = 0; idx < laserMonCBoxList_.size(); ++idx ) {
              if( cboxch == laserMonCBoxList_[idx] &&
                iphi  == laserMonIPhiList_[idx] && 
                ieta  == laserMonIEtaList_[idx] ) {

                // now get the digis
                unsigned ts_size = digi->size();
                for(unsigned i = 0; i < ts_size; i++) {
                  lasmon_adcs[idx].push_back( digi->sample(i).adc() );
                  lasmon_capids[idx].push_back( digi->sample(i).capid() );
                } // end digi loop
              } // end matching channel if
            } // end fiber order loop
          } // end cboxch check
        } // end filllasmon check


	for(unsigned i = 0; i < (unsigned)digi->size(); i++)
	  totalCalibCharge = totalCalibCharge + adc2fC[digi->sample(i).adc()&0xff];
	

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
       for( unsigned idx = 0; idx < laserMonCBoxList_.size(); ++idx ) {
           if( lasmon_adcs[idx].empty() ) {
               lasmon_adcs[idx] = std::vector<int>(10, -1);
           }
           if( lasmon_capids[idx].empty() ) {
               lasmon_capids[idx] = std::vector<int>(10, -1);
           }
       }
       unsigned nFibers = laserMonIEtaList_.size();
       // for each fiber we need to find the index at with the 
       // data from the next fiber matches in order to stitch them together.
       // When there is an overlap, the data from the end of the
       // earlier fiber is removed.  There is no removal of the last fiber
       std::vector<unsigned> matching_idx; 
       // we assume that the list of fibers was given in time order
       // (if this was not the case, then we just end up using 
       // all data from all fibers )
       for( unsigned fidx = 0; nFibers > 0 && (fidx < (nFibers - 1)); ++fidx ) {

         unsigned nts = lasmon_capids[fidx].size();  // number of time slices

         // start by checking just the last TS of the earlier fiber
         // against the first TS of the later fiber
         // on each iteration, check one additional TS
         // moving back in time on the earlier fiber and
         // forward in time in the later fiber
         
         int start_ts = nts - 1; // start_ts will be decrimented on each loop where a match is not found

         // in the case that our stringent check below doesn't work 
         // store the latest capID that has a match
         int latest_cap_match = -1;

         // loop over the number of checks to make
         for( unsigned ncheck = 1; ncheck <= nts ; ncheck++ ) {
           bool cap_match = true; //will be set to false if at least one check fails below
           bool adc_match = true; //will be set to false if at least one check fails below

           // loop over the channel TS, this is for the later fiber in time
           for( unsigned lidx = 0; lidx < ncheck; lidx++) {
             // we are looping over the TS of the later fiber in time
             // the TS of the earlier fiber starts from the end
             unsigned eidx = nts-ncheck+lidx;
             // if we get an invald value, this fiber has no data
             // the check and match will fail, so the start_ts will 
             // be decrimented
             if( lasmon_capids[fidx][eidx] == -1 || lasmon_capids[fidx+1][lidx] == -1 ) {
               cap_match = false;
               adc_match = false;
               break;
             }

             if( lasmon_capids[fidx][eidx] != lasmon_capids[fidx+1][lidx] ) {
               cap_match = false;
             }
             // check the data values as well
             if( lasmon_adcs[fidx][eidx] != lasmon_adcs[fidx+1][lidx] ) {
               adc_match = false;
             }
           }
           if( cap_match && (start_ts > latest_cap_match) ) {
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
             start_ts = nts;
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
             if( latest_cap_match == int(nts - 4) ) {
               start_ts = nts;
             } else {
               start_ts = latest_cap_match;
             }
           }
         }

         // now store as the matching index
         matching_idx.push_back(start_ts);
       }

       // for the last fiber we always use all of the data
       matching_idx.push_back(10);

       // now loop over the time slices of each fiber and make the sum
       int icombts = -1;
       for( unsigned fidx = 0 ; fidx < nFibers; ++fidx ) {
         for( unsigned its = 0; its < matching_idx[fidx]; ++its ) {
           icombts++;

           // apply integration limits
           if( icombts < laserMonitorTSStart_ ) continue;
           if( laserMonitorTSEnd_ > 0 && icombts > laserMonitorTSEnd_ ) continue;

           int adc = lasmon_adcs[fidx][its];

           if( adc >= 0 ) { // skip invalid data
             float fc = adc2fCHF[adc];
             totalLasmonCharge += fc;
           }

         } 
       }
     } // if( fillLaserMonitor_ )
  } // if (hCalib.isValid()==true)

  summary.calibCharge_ = totalCalibCharge;
  summary.lasmonCharge_ = totalLasmonCharge;
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
  CaloGeometry* geo = const_cast<CaloGeometry*>(pG.product());

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
