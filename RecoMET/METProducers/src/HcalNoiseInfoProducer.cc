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
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHitAuxSetter.h"
#include "DataFormats/HcalRecHit/interface/CaloRecHitAuxSetter.h"

using namespace reco;

//
// constructors and destructor
//

HcalNoiseInfoProducer::HcalNoiseInfoProducer(const edm::ParameterSet& iConfig) : algo_(iConfig) {
  // set the parameters
  fillDigis_ = iConfig.getParameter<bool>("fillDigis");
  fillRecHits_ = iConfig.getParameter<bool>("fillRecHits");
  fillCaloTowers_ = iConfig.getParameter<bool>("fillCaloTowers");
  fillTracks_ = iConfig.getParameter<bool>("fillTracks");
  fillLaserMonitor_ = iConfig.getParameter<bool>("fillLaserMonitor");

  maxProblemRBXs_ = iConfig.getParameter<int>("maxProblemRBXs");

  maxCaloTowerIEta_ = iConfig.getParameter<int>("maxCaloTowerIEta");
  maxTrackEta_ = iConfig.getParameter<double>("maxTrackEta");
  minTrackPt_ = iConfig.getParameter<double>("minTrackPt");

  digiCollName_ = iConfig.getParameter<std::string>("digiCollName");
  recHitCollName_ = iConfig.getParameter<std::string>("recHitCollName");
  caloTowerCollName_ = iConfig.getParameter<std::string>("caloTowerCollName");
  trackCollName_ = iConfig.getParameter<std::string>("trackCollName");

  jetCollName_ = iConfig.getParameter<std::string>("jetCollName");
  maxNHF_ = iConfig.getParameter<double>("maxNHF");
  maxjetindex_ = iConfig.getParameter<int>("maxjetindex");
  jet_token_ = consumes<reco::PFJetCollection>(edm::InputTag(jetCollName_));

  minRecHitE_ = iConfig.getParameter<double>("minRecHitE");
  minLowHitE_ = iConfig.getParameter<double>("minLowHitE");
  minHighHitE_ = iConfig.getParameter<double>("minHighHitE");

  minR45HitE_ = iConfig.getParameter<double>("minR45HitE");

  HcalAcceptSeverityLevel_ = iConfig.getParameter<uint32_t>("HcalAcceptSeverityLevel");
  HcalRecHitFlagsToBeExcluded_ = iConfig.getParameter<std::vector<int>>("HcalRecHitFlagsToBeExcluded");

  // Digi threshold and time slices to use for HBHE and HF calibration digis
  calibdigiHBHEthreshold_ = 0;
  calibdigiHBHEtimeslices_ = std::vector<int>();
  calibdigiHFthreshold_ = 0;
  calibdigiHFtimeslices_ = std::vector<int>();

  calibdigiHBHEthreshold_ = iConfig.getParameter<double>("calibdigiHBHEthreshold");
  calibdigiHBHEtimeslices_ = iConfig.getParameter<std::vector<int>>("calibdigiHBHEtimeslices");
  calibdigiHFthreshold_ = iConfig.getParameter<double>("calibdigiHFthreshold");
  calibdigiHFtimeslices_ = iConfig.getParameter<std::vector<int>>("calibdigiHFtimeslices");

  TS4TS5EnergyThreshold_ = iConfig.getParameter<double>("TS4TS5EnergyThreshold");

  std::vector<double> TS4TS5UpperThresholdTemp = iConfig.getParameter<std::vector<double>>("TS4TS5UpperThreshold");
  std::vector<double> TS4TS5UpperCutTemp = iConfig.getParameter<std::vector<double>>("TS4TS5UpperCut");
  std::vector<double> TS4TS5LowerThresholdTemp = iConfig.getParameter<std::vector<double>>("TS4TS5LowerThreshold");
  std::vector<double> TS4TS5LowerCutTemp = iConfig.getParameter<std::vector<double>>("TS4TS5LowerCut");

  for (int i = 0; i < (int)TS4TS5UpperThresholdTemp.size() && i < (int)TS4TS5UpperCutTemp.size(); i++)
    TS4TS5UpperCut_.push_back(std::pair<double, double>(TS4TS5UpperThresholdTemp[i], TS4TS5UpperCutTemp[i]));
  sort(TS4TS5UpperCut_.begin(), TS4TS5UpperCut_.end());

  for (int i = 0; i < (int)TS4TS5LowerThresholdTemp.size() && i < (int)TS4TS5LowerCutTemp.size(); i++)
    TS4TS5LowerCut_.push_back(std::pair<double, double>(TS4TS5LowerThresholdTemp[i], TS4TS5LowerCutTemp[i]));
  sort(TS4TS5LowerCut_.begin(), TS4TS5LowerCut_.end());

  // if digis are filled, then rechits must also be filled
  if (fillDigis_ && !fillRecHits_) {
    fillRecHits_ = true;
    edm::LogWarning("HCalNoiseInfoProducer") << " forcing fillRecHits to be true if fillDigis is true.\n";
  }

  // get the fiber configuration vectors
  laserMonCBoxList_ = iConfig.getParameter<std::vector<int>>("laserMonCBoxList");
  laserMonIPhiList_ = iConfig.getParameter<std::vector<int>>("laserMonIPhiList");
  laserMonIEtaList_ = iConfig.getParameter<std::vector<int>>("laserMonIEtaList");

  // check that the vectors have the same size, if not
  // disable the laser monitor
  if (!((laserMonCBoxList_.size() == laserMonIEtaList_.size()) &&
        (laserMonCBoxList_.size() == laserMonIPhiList_.size()))) {
    edm::LogWarning("MisConfiguration") << "Must provide equally sized lists for laserMonCBoxList, laserMonIEtaList, "
                                           "and laserMonIPhiList.  Will not fill LaserMon\n";
    fillLaserMonitor_ = false;
  }

  // get the integration region with defaults
  laserMonitorTSStart_ = iConfig.getParameter<int>("laserMonTSStart");
  laserMonitorTSEnd_ = iConfig.getParameter<int>("laserMonTSEnd");
  laserMonitorSamples_ = iConfig.getParameter<unsigned>("laserMonSamples");

  adc2fC = std::vector<float>{
      -0.5,   0.5,    1.5,    2.5,    3.5,    4.5,    5.5,    6.5,    7.5,    8.5,    9.5,    10.5,   11.5,
      12.5,   13.5,   15.,    17.,    19.,    21.,    23.,    25.,    27.,    29.5,   32.5,   35.5,   38.5,
      42.,    46.,    50.,    54.5,   59.5,   64.5,   59.5,   64.5,   69.5,   74.5,   79.5,   84.5,   89.5,
      94.5,   99.5,   104.5,  109.5,  114.5,  119.5,  124.5,  129.5,  137.,   147.,   157.,   167.,   177.,
      187.,   197.,   209.5,  224.5,  239.5,  254.5,  272.,   292.,   312.,   334.5,  359.5,  384.5,  359.5,
      384.5,  409.5,  434.5,  459.5,  484.5,  509.5,  534.5,  559.5,  584.5,  609.5,  634.5,  659.5,  684.5,
      709.5,  747.,   797.,   847.,   897.,   947.,   997.,   1047.,  1109.5, 1184.5, 1259.5, 1334.5, 1422.,
      1522.,  1622.,  1734.5, 1859.5, 1984.5, 1859.5, 1984.5, 2109.5, 2234.5, 2359.5, 2484.5, 2609.5, 2734.5,
      2859.5, 2984.5, 3109.5, 3234.5, 3359.5, 3484.5, 3609.5, 3797.,  4047.,  4297.,  4547.,  4797.,  5047.,
      5297.,  5609.5, 5984.5, 6359.5, 6734.5, 7172.,  7672.,  8172.,  8734.5, 9359.5, 9984.5};

  // adc -> fC for qie10, for laser monitor
  // Taken from page 3 in
  // https://cms-docdb.cern.ch/cgi-bin/DocDB/RetrieveFile?docid=12570&filename=QIE10_final.pdf&version=5
  adc2fCHF = std::vector<float>{// - - - - - - - range 0 - - - - - - - -
                                //subrange0
                                1.58,
                                4.73,
                                7.88,
                                11.0,
                                14.2,
                                17.3,
                                20.5,
                                23.6,
                                26.8,
                                29.9,
                                33.1,
                                36.2,
                                39.4,
                                42.5,
                                45.7,
                                48.8,
                                //subrange1
                                53.6,
                                60.1,
                                66.6,
                                73.0,
                                79.5,
                                86.0,
                                92.5,
                                98.9,
                                105,
                                112,
                                118,
                                125,
                                131,
                                138,
                                144,
                                151,
                                //subrange2
                                157,
                                164,
                                170,
                                177,
                                186,
                                199,
                                212,
                                225,
                                238,
                                251,
                                264,
                                277,
                                289,
                                302,
                                315,
                                328,
                                //subrange3
                                341,
                                354,
                                367,
                                380,
                                393,
                                406,
                                418,
                                431,
                                444,
                                464,
                                490,
                                516,
                                542,
                                568,
                                594,
                                620,

                                // - - - - - - - range 1 - - - - - - - -
                                //subrange0
                                569,
                                594,
                                619,
                                645,
                                670,
                                695,
                                720,
                                745,
                                771,
                                796,
                                821,
                                846,
                                871,
                                897,
                                922,
                                947,
                                //subrange1
                                960,
                                1010,
                                1060,
                                1120,
                                1170,
                                1220,
                                1270,
                                1320,
                                1370,
                                1430,
                                1480,
                                1530,
                                1580,
                                1630,
                                1690,
                                1740,
                                //subrange2
                                1790,
                                1840,
                                1890,
                                1940,
                                2020,
                                2120,
                                2230,
                                2330,
                                2430,
                                2540,
                                2640,
                                2740,
                                2850,
                                2950,
                                3050,
                                3150,
                                //subrange3
                                3260,
                                3360,
                                3460,
                                3570,
                                3670,
                                3770,
                                3880,
                                3980,
                                4080,
                                4240,
                                4450,
                                4650,
                                4860,
                                5070,
                                5280,
                                5490,

                                // - - - - - - - range 2 - - - - - - - -
                                //subrange0
                                5080,
                                5280,
                                5480,
                                5680,
                                5880,
                                6080,
                                6280,
                                6480,
                                6680,
                                6890,
                                7090,
                                7290,
                                7490,
                                7690,
                                7890,
                                8090,
                                //subrange1
                                8400,
                                8810,
                                9220,
                                9630,
                                10000,
                                10400,
                                10900,
                                11300,
                                11700,
                                12100,
                                12500,
                                12900,
                                13300,
                                13700,
                                14100,
                                14500,
                                //subrange2
                                15000,
                                15400,
                                15800,
                                16200,
                                16800,
                                17600,
                                18400,
                                19300,
                                20100,
                                20900,
                                21700,
                                22500,
                                23400,
                                24200,
                                25000,
                                25800,
                                //subrange3
                                26600,
                                27500,
                                28300,
                                29100,
                                29900,
                                30700,
                                31600,
                                32400,
                                33200,
                                34400,
                                36100,
                                37700,
                                39400,
                                41000,
                                42700,
                                44300,

                                // - - - - - - - range 3 - - - - - - - - -
                                //subrange0
                                41100,
                                42700,
                                44300,
                                45900,
                                47600,
                                49200,
                                50800,
                                52500,
                                54100,
                                55700,
                                57400,
                                59000,
                                60600,
                                62200,
                                63900,
                                65500,
                                //subrange1
                                68000,
                                71300,
                                74700,
                                78000,
                                81400,
                                84700,
                                88000,
                                91400,
                                94700,
                                98100,
                                101000,
                                105000,
                                108000,
                                111000,
                                115000,
                                118000,
                                //subrange2
                                121000,
                                125000,
                                128000,
                                131000,
                                137000,
                                145000,
                                152000,
                                160000,
                                168000,
                                176000,
                                183000,
                                191000,
                                199000,
                                206000,
                                214000,
                                222000,
                                //subrange3
                                230000,
                                237000,
                                245000,
                                253000,
                                261000,
                                268000,
                                276000,
                                284000,
                                291000,
                                302000,
                                316000,
                                329000,
                                343000,
                                356000,
                                370000,
                                384000};

  hbhedigi_token_ = consumes<HBHEDigiCollection>(edm::InputTag(digiCollName_));
  hcalcalibdigi_token_ = consumes<HcalCalibDigiCollection>(edm::InputTag("hcalDigis"));
  lasermondigi_token_ = consumes<QIE10DigiCollection>(iConfig.getParameter<edm::InputTag>("lasermonDigis"));
  hbherechit_token_ = consumes<HBHERecHitCollection>(edm::InputTag(recHitCollName_));
  calotower_token_ = consumes<CaloTowerCollection>(edm::InputTag(caloTowerCollName_));
  track_token_ = consumes<reco::TrackCollection>(edm::InputTag(trackCollName_));
  hcaltopo_token_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  service_token_ = esConsumes<HcalDbService, HcalDbRecord>();
  quality_token_ = esConsumes<HcalChannelQuality, HcalChannelQualityRcd>(edm::ESInputTag("", "withTopo"));
  severitycomputer_token_ = esConsumes<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd>();
  calogeometry_token_ = esConsumes<CaloGeometry, CaloGeometryRecord>();

  // we produce a vector of HcalNoiseRBXs
  produces<HcalNoiseRBXCollection>();
  // we also produce a noise summary
  produces<HcalNoiseSummary>();
}

HcalNoiseInfoProducer::~HcalNoiseInfoProducer() {}

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
  desc.add<double>("calibdigiHBHEthreshold", 15)
      ->setComment(
          "minimum threshold in fC of any HBHE  \
              calib digi to be counted in summary");
  desc.add<std::vector<int>>("calibdigiHBHEtimeslices",
                             {
                                 3,
                                 4,
                                 5,
                                 6,
                             })
      ->setComment("time slices to use when determining charge of HBHE calib digis");
  desc.add<double>("calibdigiHFthreshold", -999)
      ->setComment("minimum threshold in fC of any HF calib digi to be counted in summary");
  desc.add<std::vector<int>>("calibdigiHFtimeslices",
                             {
                                 0,
                                 1,
                                 2,
                                 3,
                                 4,
                                 5,
                                 6,
                                 7,
                                 8,
                                 9,
                             })
      ->setComment("time slices to use when determining charge of HF calib digis");

  // RBX-wide TS4TS5 variable
  desc.add<double>("TS4TS5EnergyThreshold", 50);
  desc.add<std::vector<double>>("TS4TS5UpperThreshold",
                                {
                                    70,
                                    90,
                                    100,
                                    400,
                                    4000,
                                });
  desc.add<std::vector<double>>("TS4TS5UpperCut",
                                {
                                    1,
                                    0.8,
                                    0.75,
                                    0.72,
                                    0.72,
                                });
  desc.add<std::vector<double>>("TS4TS5LowerThreshold",
                                {
                                    100,
                                    120,
                                    150,
                                    200,
                                    300,
                                    400,
                                    500,
                                });
  desc.add<std::vector<double>>("TS4TS5LowerCut",
                                {
                                    -1,
                                    -0.7,
                                    -0.4,
                                    -0.2,
                                    -0.08,
                                    0,
                                    0.1,
                                });

  // rechit R45 population filter variables
  // this comes in groups of four: (a_Count, a_Fraction, a_EnergyFraction, const)
  // flag as noise if (count * a_count + fraction * a_fraction + energyfraction * a_energyfraction + const) > 0
  desc.add<std::vector<double>>("lRBXRecHitR45Cuts",
                                {
                                    0.0,
                                    1.0,
                                    0.0,
                                    -0.5,
                                    0.0,
                                    0.0,
                                    1.0,
                                    -0.5,
                                })
      ->setComment(
          "first 4 entries : equivalent to 'fraction > 0.5'  \
                  last 4 entries : equivalent to 'energy fraction > 0.5'");
  desc.add<std::vector<double>>("tRBXRecHitR45Cuts",
                                {
                                    0.0,
                                    1.0,
                                    0.0,
                                    -0.2,
                                    0.0,
                                    0.0,
                                    1.0,
                                    -0.2,
                                })
      ->setComment(
          "first 4 entries : equivalent to 'fraction > 0.2' \
                  last 4 entries : equivalent to 'energy fraction > 0.2'");

  // define the channels used for laser monitoring
  // note that the order here indicates the time order
  // of the channels
  desc.add<std::vector<int>>("laserMonCBoxList",
                             {
                                 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                             })
      ->setComment("time ordered list of the cBox values of laser monitor channels");
  desc.add<std::vector<int>>("laserMonIPhiList",
                             {23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0})
      ->setComment("time ordered list of the iPhi values of laser monitor channels");
  desc.add<std::vector<int>>("laserMonIEtaList",
                             {
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             })
      ->setComment("time ordered list of the iEta values of laser monitor channels");

  // boundaries for total charge integration
  desc.add<int>("laserMonTSStart", 0)->setComment("lower bound of laser monitor charge integration window");
  desc.add<int>("laserMonTSEnd", -1)
      ->setComment("upper bound of laser monitor charge integration window (-1 = no bound)");
  desc.add<unsigned>("laserMonSamples", 4)->setComment("Number of laser monitor samples to take per channel");

  // what to fill
  desc.add<bool>("fillDigis", true);
  desc.add<bool>("fillRecHits", true);
  desc.add<bool>("fillCaloTowers", true);
  desc.add<bool>("fillTracks", true);
  desc.add<bool>("fillLaserMonitor", true);

  // maximum number of RBXs to fill
  // if you want to record all RBXs above some energy threshold,
  // change maxProblemRBXs to 999 and pMinE (above) to the threshold you want
  desc.add<int>("maxProblemRBXs", 72)
      ->setComment(
          "maximum number of RBXs to fill.  if you want to record  \
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
  desc.add<edm::InputTag>("lasermonDigis", edm::InputTag("hcalDigis", "LASERMON"));

  // severity level
  desc.add<unsigned int>("HcalAcceptSeverityLevel", 9);

  // which hcal calo flags to mask
  // (HBHEIsolatedNoise=11, HBHEFlatNoise=12, HBHESpikeNoise=13,
  // HBHETriangleNoise=14, HBHETS4TS5Noise=15, HBHENegativeNoise=27)
  desc.add<std::vector<int>>("HcalRecHitFlagsToBeExcluded",
                             {
                                 11,
                                 12,
                                 13,
                                 14,
                                 15,
                                 27,
                             })
      ->setComment(
          "which hcal calo flags to mask (HBHEIsolatedNoise=11, \
                  HBHEFlatNoise=12, HBHESpikeNoise=13, \
                  HBHETriangleNoise=14, HBHETS4TS5Noise=15, HBHENegativeNoise=27)");
  ;

  descriptions.add("hcalnoise", desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HcalNoiseInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // this is what we're going to actually write to the EDM
  auto result1 = std::make_unique<HcalNoiseRBXCollection>();
  auto result2 = std::make_unique<HcalNoiseSummary>();

  // define an empty HcalNoiseRBXArray that we're going to fill
  HcalNoiseRBXArray rbxarray;
  HcalNoiseSummary& summary = *result2;

  // Get topology class to use later
  edm::ESHandle<HcalTopology> topo = iSetup.getHandle(hcaltopo_token_);
  theHcalTopology_ = topo.product();

  // fill them with the various components
  // digi assumes that rechit information is available
  if (fillRecHits_)
    fillrechits(iEvent, iSetup, rbxarray, summary);
  if (fillDigis_)
    filldigis(iEvent, iSetup, rbxarray, summary);
  if (fillCaloTowers_)
    fillcalotwrs(iEvent, iSetup, rbxarray, summary);
  if (fillTracks_)
    filltracks(iEvent, iSetup, summary);

  filljetinfo(iEvent, iSetup, summary);

  // select those RBXs which are interesting
  // also look for the highest energy RBX
  HcalNoiseRBXArray::iterator maxit = rbxarray.begin();
  double maxenergy = -999;
  bool maxwritten = false;
  for (HcalNoiseRBXArray::iterator rit = rbxarray.begin(); rit != rbxarray.end(); ++rit) {
    HcalNoiseRBX& rbx = (*rit);
    CommonHcalNoiseRBXData data(rbx,
                                minRecHitE_,
                                minLowHitE_,
                                minHighHitE_,
                                TS4TS5EnergyThreshold_,
                                TS4TS5UpperCut_,
                                TS4TS5LowerCut_,
                                minR45HitE_);

    // find the highest energy rbx
    if (data.energy() > maxenergy) {
      maxenergy = data.energy();
      maxit = rit;
      maxwritten = false;
    }

    // find out if the rbx is problematic/noisy/etc.
    bool writerbx = algo_.isProblematic(data) || !algo_.passLooseNoiseFilter(data) ||
                    !algo_.passTightNoiseFilter(data) || !algo_.passHighLevelNoiseFilter(data);

    // fill variables in the summary object not filled elsewhere
    fillOtherSummaryVariables(summary, data);

    if (writerbx) {
      summary.nproblemRBXs_++;
      if (summary.nproblemRBXs_ <= maxProblemRBXs_) {
        result1->push_back(rbx);
        if (maxit == rit)
          maxwritten = true;
      }
    }
  }  // end loop over rbxs

  // if we still haven't written the maximum energy rbx, write it now
  if (!maxwritten) {
    HcalNoiseRBX& rbx = (*maxit);

    // add the RBX to the event
    result1->push_back(rbx);
  }

  // put the rbxcollection and summary into the EDM
  iEvent.put(std::move(result1));
  iEvent.put(std::move(result2));

  return;
}

// ------------ here we fill specific variables in the summary object not already accounted for earlier
void HcalNoiseInfoProducer::fillOtherSummaryVariables(HcalNoiseSummary& summary,
                                                      const CommonHcalNoiseRBXData& data) const {
  // charge ratio
  if (algo_.passRatioThreshold(data) && data.validRatio()) {
    if (data.ratio() < summary.minE2Over10TS()) {
      summary.mine2ts_ = data.e2ts();
      summary.mine10ts_ = data.e10ts();
    }
    if (data.ratio() > summary.maxE2Over10TS()) {
      summary.maxe2ts_ = data.e2ts();
      summary.maxe10ts_ = data.e10ts();
    }
  }

  // ADC zeros
  if (algo_.passZerosThreshold(data)) {
    if (data.numZeros() > summary.maxZeros()) {
      summary.maxzeros_ = data.numZeros();
    }
  }

  // hits count
  if (data.numHPDHits() > summary.maxHPDHits()) {
    summary.maxhpdhits_ = data.numHPDHits();
  }
  if (data.numRBXHits() > summary.maxRBXHits()) {
    summary.maxrbxhits_ = data.numRBXHits();
  }
  if (data.numHPDNoOtherHits() > summary.maxHPDNoOtherHits()) {
    summary.maxhpdhitsnoother_ = data.numHPDNoOtherHits();
  }

  // TS4TS5
  if (data.PassTS4TS5() == false)
    summary.hasBadRBXTS4TS5_ = true;

  if (algo_.passLooseRBXRechitR45(data) == false)
    summary.hasBadRBXRechitR45Loose_ = true;
  if (algo_.passTightRBXRechitR45(data) == false)
    summary.hasBadRBXRechitR45Tight_ = true;

  // hit timing
  if (data.minLowEHitTime() < summary.min10GeVHitTime()) {
    summary.min10_ = data.minLowEHitTime();
  }
  if (data.maxLowEHitTime() > summary.max10GeVHitTime()) {
    summary.max10_ = data.maxLowEHitTime();
  }
  summary.rms10_ += data.lowEHitTimeSqrd();
  summary.cnthit10_ += data.numLowEHits();
  if (data.minHighEHitTime() < summary.min25GeVHitTime()) {
    summary.min25_ = data.minHighEHitTime();
  }
  if (data.maxHighEHitTime() > summary.max25GeVHitTime()) {
    summary.max25_ = data.maxHighEHitTime();
  }
  summary.rms25_ += data.highEHitTimeSqrd();
  summary.cnthit25_ += data.numHighEHits();

  // EMF
  if (algo_.passEMFThreshold(data)) {
    if (summary.minHPDEMF() > data.HPDEMF()) {
      summary.minhpdemf_ = data.HPDEMF();
    }
    if (summary.minRBXEMF() > data.RBXEMF()) {
      summary.minrbxemf_ = data.RBXEMF();
    }
  }

  // summary flag
  if (!algo_.passLooseRatio(data))
    summary.filterstatus_ |= 0x1;
  if (!algo_.passLooseHits(data))
    summary.filterstatus_ |= 0x2;
  if (!algo_.passLooseZeros(data))
    summary.filterstatus_ |= 0x4;
  if (!algo_.passLooseTiming(data))
    summary.filterstatus_ |= 0x8;

  if (!algo_.passTightRatio(data))
    summary.filterstatus_ |= 0x100;
  if (!algo_.passTightHits(data))
    summary.filterstatus_ |= 0x200;
  if (!algo_.passTightZeros(data))
    summary.filterstatus_ |= 0x400;
  if (!algo_.passTightTiming(data))
    summary.filterstatus_ |= 0x800;

  if (!algo_.passHighLevelNoiseFilter(data))
    summary.filterstatus_ |= 0x10000;

  // summary refvectors
  JoinCaloTowerRefVectorsWithoutDuplicates join;
  if (!algo_.passLooseNoiseFilter(data))
    join(summary.loosenoisetwrs_, data.rbxTowers());
  if (!algo_.passTightNoiseFilter(data))
    join(summary.tightnoisetwrs_, data.rbxTowers());
  if (!algo_.passHighLevelNoiseFilter(data))
    join(summary.hlnoisetwrs_, data.rbxTowers());

  return;
}

// ------------ fill the array with digi information
void HcalNoiseInfoProducer::filldigis(edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      HcalNoiseRBXArray& array,
                                      HcalNoiseSummary& summary) {
  // Some initialization
  totalCalibCharge = 0;
  totalLasmonCharge = 0;

  // Starting with this version (updated by Jeff Temple on Dec. 6, 2012), the "TS45" names in the variables are mis-nomers.  The actual time slices used are determined from the digiTimeSlices_ variable, which may not be limited to only time slices 4 and 5.  For now, "TS45" name kept, because that is what is used in HcalNoiseSummary object (in GetCalibCountTS45, etc.).  Likewise, the charge value in 'gt15' is now configurable, though the name remains the same.  For HBHE, we track both the number of calibration channels (NcalibTS45) and the number of calibration channels above threshold (NcalibTS45gt15).  For HF, we track only the number of channels above the given threshold in the given time window (NcalibHFgtX).  Default for HF in 2012 is to use the full time sample with effectively no threshold (threshold=-999)
  int NcalibTS45 = 0;
  int NcalibTS45gt15 = 0;
  int NcalibHFgtX = 0;

  double chargecalibTS45 = 0;
  double chargecalibgt15TS45 = 0;

  // get the conditions and channel quality
  edm::ESHandle<HcalDbService> conditions = iSetup.getHandle(service_token_);
  edm::ESHandle<HcalChannelQuality> qualhandle = iSetup.getHandle(quality_token_);
  const HcalChannelQuality* myqual = qualhandle.product();
  edm::ESHandle<HcalSeverityLevelComputer> mycomputer = iSetup.getHandle(severitycomputer_token_);
  const HcalSeverityLevelComputer* mySeverity = mycomputer.product();

  // get the digis
  edm::Handle<HBHEDigiCollection> handle;
  //  iEvent.getByLabel(digiCollName_, handle);
  iEvent.getByToken(hbhedigi_token_, handle);

  if (!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
        << " could not find HBHEDigiCollection named " << digiCollName_ << "\n.";
    return;
  }

  // loop over all of the digi information
  for (HBHEDigiCollection::const_iterator it = handle->begin(); it != handle->end(); ++it) {
    const HBHEDataFrame& digi = (*it);
    HcalDetId cell = digi.id();
    DetId detcell = (DetId)cell;

    // check on cells to be ignored and dropped
    const HcalChannelStatus* mydigistatus = myqual->getValues(detcell.rawId());
    if (mySeverity->dropChannel(mydigistatus->getValue()))
      continue;
    if (digi.zsMarkAndPass())
      continue;
    // Drop if exclude bit set
    if (mydigistatus->isBitSet(HcalChannelStatus::HcalCellExcludeFromHBHENoiseSummary))
      continue;

    // get the calibrations and coder
    const HcalCalibrations& calibrations = conditions->getHcalCalibrations(cell);
    const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
    const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
    HcalCoderDb coder(*channelCoder, *shape);

    // match the digi to an rbx and hpd
    HcalNoiseRBX& rbx = (*array.findRBX(digi));
    HcalNoiseHPD& hpd = (*array.findHPD(digi));

    // determine if the digi is one the highest energy hits in the HPD
    // this works because the rechits are sorted by energy (see fillrechits() below)
    bool isBig = false, isBig5 = false, isRBX = false;
    int counter = 0;
    edm::RefVector<HBHERecHitCollection>& rechits = hpd.rechits_;
    for (edm::RefVector<HBHERecHitCollection>::const_iterator rit = rechits.begin(); rit != rechits.end();
         ++rit, ++counter) {
      const HcalDetId& detid = (*rit)->idFront();
      if (DetId(detid) == digi.id()) {
        if (counter == 0)
          isBig = isBig5 = true;  // digi is also the highest energy rechit
        if (counter < 5)
          isBig5 = true;  // digi is one of 5 highest energy rechits
        isRBX = true;
      }
    }

    // loop over each of the digi's time slices
    int totalzeros = 0;
    CaloSamples tool;
    coder.adc2fC(digi, tool);
    for (int ts = 0; ts < tool.size(); ++ts) {
      // count zero's
      if (digi[ts].adc() == 0) {
        ++hpd.totalZeros_;
        ++totalzeros;
      }

      // get the fC's
      double corrfc = tool[ts] - calibrations.pedestal(digi[ts].capid());

      // fill the relevant digi arrays
      if (isBig)
        hpd.bigCharge_[ts] += corrfc;
      if (isBig5)
        hpd.big5Charge_[ts] += corrfc;
      if (isRBX)
        rbx.allCharge_[ts] += corrfc;
    }

    // record the maximum number of zero's found
    if (totalzeros > hpd.maxZeros_)
      hpd.maxZeros_ = totalzeros;
  }

  // get the calibration digis
  edm::Handle<HcalCalibDigiCollection> hCalib;
  //  iEvent.getByLabel("hcalDigis", hCalib);
  iEvent.getByToken(hcalcalibdigi_token_, hCalib);

  // get the lasermon digis
  edm::Handle<QIE10DigiCollection> hLasermon;
  iEvent.getByToken(lasermondigi_token_, hLasermon);

  // get total charge in calibration channels
  if (hCalib.isValid() == true) {
    for (HcalCalibDigiCollection::const_iterator digi = hCalib->begin(); digi != hCalib->end(); digi++) {
      if (digi->id().hcalSubdet() == 0)
        continue;

      for (unsigned i = 0; i < (unsigned)digi->size(); i++)
        totalCalibCharge = totalCalibCharge + adc2fC[digi->sample(i).adc() & 0xff];

      HcalCalibDetId myid = (HcalCalibDetId)digi->id();
      if (myid.calibFlavor() == HcalCalibDetId::HOCrosstalk)
        continue;  // ignore HOCrosstalk channels
      if (digi->zsMarkAndPass())
        continue;  // skip "mark-and-pass" channels when computing charge in calib channels

      if (digi->id().hcalSubdet() == HcalForward)  // check HF
      {
        double sumChargeHF = 0;
        for (unsigned int i = 0; i < calibdigiHFtimeslices_.size(); ++i) {
          // skip unphysical time slices
          if (calibdigiHFtimeslices_[i] < 0 || calibdigiHFtimeslices_[i] > digi->size())
            continue;
          sumChargeHF += adc2fC[digi->sample(calibdigiHFtimeslices_[i]).adc() & 0xff];
        }
        if (sumChargeHF > calibdigiHFthreshold_)
          ++NcalibHFgtX;
      }                                                                                         // end of HF check
      else if (digi->id().hcalSubdet() == HcalBarrel || digi->id().hcalSubdet() == HcalEndcap)  // now check HBHE
      {
        double sumChargeHBHE = 0;
        for (unsigned int i = 0; i < calibdigiHBHEtimeslices_.size(); ++i) {
          // skip unphysical time slices
          if (calibdigiHBHEtimeslices_[i] < 0 || calibdigiHBHEtimeslices_[i] > digi->size())
            continue;
          sumChargeHBHE += adc2fC[digi->sample(calibdigiHBHEtimeslices_[i]).adc() & 0xff];
        }
        ++NcalibTS45;
        chargecalibTS45 += sumChargeHBHE;
        if (sumChargeHBHE > calibdigiHBHEthreshold_) {
          ++NcalibTS45gt15;
          chargecalibgt15TS45 += sumChargeHBHE;
        }
      }  // end of HBHE check
    }    // loop on HcalCalibDigiCollection

  }  // if (hCalib.isValid()==true)
  if (fillLaserMonitor_ && (hLasermon.isValid() == true)) {
    int icombts = -1;
    float max_charge = 0;
    int max_ts = -1;
    std::vector<float> comb_charge;

    unsigned nch = laserMonCBoxList_.size();
    // loop over channels in the order provided
    for (unsigned ich = 0; ich < nch; ++ich) {
      int cboxch = laserMonCBoxList_[ich];
      int iphi = laserMonIPhiList_[ich];
      int ieta = laserMonIEtaList_[ich];

      // loop over digis, find the digi that matches this channel
      for (const QIE10DataFrame df : (*hLasermon)) {
        HcalCalibDetId calibId(df.id());

        int ch_cboxch = calibId.cboxChannel();
        int ch_iphi = calibId.iphi();
        int ch_ieta = calibId.ieta();

        if (cboxch == ch_cboxch && iphi == ch_iphi && ieta == ch_ieta) {
          unsigned ts_size = df.samples();

          // loop over time slices
          for (unsigned its = 0; its < ts_size; ++its) {
            // if we are on the last channel, use all the data
            // otherwise only take the unique samples
            if (((ich + 1) < nch) && its >= laserMonitorSamples_)
              continue;

            bool ok = df[its].ok();
            int adc = df[its].adc();

            icombts++;
            // apply integration limits
            if (icombts < laserMonitorTSStart_)
              continue;
            if (laserMonitorTSEnd_ > 0 && icombts > laserMonitorTSEnd_)
              continue;

            if (ok && adc >= 0) {  // protection against QIE reset or bad ADC values

              float charge = adc2fCHF[adc];
              if (charge > max_charge) {
                max_charge = charge;
                max_ts = icombts;
              }

              comb_charge.push_back(charge);
            }  // if( ok && adc >= 0 )
          }    // loop over time slices
        }      // if( cboxch == ch_cboxch && iphi == ch_iphi && ieta == ch_ieta )
      }        // loop over digi collection
    }          // loop over channel list

    // do not continue with the calculation
    // if the vector was not filled
    if (comb_charge.empty()) {
      totalLasmonCharge = -1;
    } else {
      // integrate from +- 3 TS around the time sample
      // having the maximum charge
      int start_ts = max_ts - 3;
      int end_ts = max_ts + 3;

      // Change the integration limits
      // if outside of the range
      if (start_ts < 0)
        start_ts = 0;
      if (end_ts >= int(comb_charge.size()))
        end_ts = comb_charge.size() - 1;

      for (int isum = start_ts; isum <= end_ts; ++isum) {
        totalLasmonCharge += comb_charge[isum];
      }
    }
  }  // if( fillLaserMonitor_  && (hLasermon.isValid() == true) )

  summary.calibCharge_ = totalCalibCharge;
  summary.lasmonCharge_ = totalLasmonCharge;
  summary.calibCountTS45_ = NcalibTS45;
  summary.calibCountgt15TS45_ = NcalibTS45gt15;
  summary.calibChargeTS45_ = chargecalibTS45;
  summary.calibChargegt15TS45_ = chargecalibgt15TS45;
  summary.calibCountHF_ = NcalibHFgtX;

  return;
}

// ------------ fill the array with rec hit information
void HcalNoiseInfoProducer::fillrechits(edm::Event& iEvent,
                                        const edm::EventSetup& iSetup,
                                        HcalNoiseRBXArray& array,
                                        HcalNoiseSummary& summary) const {
  // get the HCAL channel status map
  edm::ESHandle<HcalChannelQuality> hcalChStatus = iSetup.getHandle(quality_token_);
  const HcalChannelQuality* dbHcalChStatus = hcalChStatus.product();

  // get the severity level computer
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl = iSetup.getHandle(severitycomputer_token_);
  const HcalSeverityLevelComputer* hcalSevLvlComputer = hcalSevLvlComputerHndl.product();

  // get the calo geometry
  edm::ESHandle<CaloGeometry> pG = iSetup.getHandle(calogeometry_token_);
  const CaloGeometry* geo = pG.product();

  // get the rechits
  edm::Handle<HBHERecHitCollection> handle;
  //  iEvent.getByLabel(recHitCollName_, handle);
  iEvent.getByToken(hbherechit_token_, handle);

  if (!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
        << " could not find HBHERecHitCollection named " << recHitCollName_ << "\n.";
    return;
  }

  summary.rechitCount_ = 0;
  summary.rechitCount15_ = 0;
  summary.rechitEnergy_ = 0;
  summary.rechitEnergy15_ = 0;

  summary.hitsInLaserRegion_ = 0;
  summary.hitsInNonLaserRegion_ = 0;
  summary.energyInLaserRegion_ = 0;
  summary.energyInNonLaserRegion_ = 0;

  // loop over all of the rechit information
  for (HBHERecHitCollection::const_iterator it = handle->begin(); it != handle->end(); ++it) {
    const HBHERecHit& rechit = (*it);

    // skip bad rechits (other than those flagged by the isolated noise, triangle, flat, and spike algorithms)
    const DetId id = rechit.idFront();

    uint32_t recHitFlag = rechit.flags();
    uint32_t isolbitset = (1 << HcalCaloFlagLabels::HBHEIsolatedNoise);
    uint32_t flatbitset = (1 << HcalCaloFlagLabels::HBHEFlatNoise);
    uint32_t spikebitset = (1 << HcalCaloFlagLabels::HBHESpikeNoise);
    uint32_t trianglebitset = (1 << HcalCaloFlagLabels::HBHETriangleNoise);
    uint32_t ts4ts5bitset = (1 << HcalCaloFlagLabels::HBHETS4TS5Noise);
    uint32_t negativebitset = (1 << HcalCaloFlagLabels::HBHENegativeNoise);
    for (unsigned int i = 0; i < HcalRecHitFlagsToBeExcluded_.size(); i++) {
      uint32_t bitset = (1 << HcalRecHitFlagsToBeExcluded_[i]);
      recHitFlag = (recHitFlag & bitset) ? recHitFlag - bitset : recHitFlag;
    }
    const HcalChannelStatus* dbStatus = dbHcalChStatus->getValues(id);

    // Ignore rechit if exclude bit set, regardless of severity of other bits
    if (dbStatus->isBitSet(HcalChannelStatus::HcalCellExcludeFromHBHENoiseSummary))
      continue;

    int severityLevel = hcalSevLvlComputer->getSeverityLevel(id, recHitFlag, dbStatus->getValue());
    bool isRecovered = hcalSevLvlComputer->recoveredRecHit(id, recHitFlag);
    if (severityLevel != 0 && !isRecovered && severityLevel > static_cast<int>(HcalAcceptSeverityLevel_))
      continue;

    // do some rechit counting and energies
    summary.rechitCount_ = summary.rechitCount_ + 1;
    summary.rechitEnergy_ = summary.rechitEnergy_ + rechit.eraw();
    if (dbStatus->isBitSet(
            HcalChannelStatus::
                HcalBadLaserSignal))  // hit comes from a region where no laser calibration pulse is normally seen
    {
      ++summary.hitsInNonLaserRegion_;
      summary.energyInNonLaserRegion_ += rechit.eraw();
    } else  // hit comes from region where laser calibration pulse is seen
    {
      ++summary.hitsInLaserRegion_;
      summary.energyInLaserRegion_ += rechit.eraw();
    }

    if (rechit.eraw() > 1.5) {
      summary.rechitCount15_ = summary.rechitCount15_ + 1;
      summary.rechitEnergy15_ = summary.rechitEnergy15_ + rechit.eraw();
    }

    // Exclude uncollapsed QIE11 channels
    if (CaloRecHitAuxSetter::getBit(rechit.auxPhase1(), HBHERecHitAuxSetter::OFF_TDC_TIME) &&
        !CaloRecHitAuxSetter::getBit(rechit.auxPhase1(), HBHERecHitAuxSetter::OFF_COMBINED))
      continue;

    // if it was ID'd as isolated noise, update the summary object
    if (rechit.flags() & isolbitset) {
      summary.nisolnoise_++;
      summary.isolnoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw() * gp.perp() / gp.mag();
      summary.isolnoiseet_ += et;
    }

    if (rechit.flags() & flatbitset) {
      summary.nflatnoise_++;
      summary.flatnoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw() * gp.perp() / gp.mag();
      summary.flatnoiseet_ += et;
    }

    if (rechit.flags() & spikebitset) {
      summary.nspikenoise_++;
      summary.spikenoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw() * gp.perp() / gp.mag();
      summary.spikenoiseet_ += et;
    }

    if (rechit.flags() & trianglebitset) {
      summary.ntrianglenoise_++;
      summary.trianglenoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw() * gp.perp() / gp.mag();
      summary.trianglenoiseet_ += et;
    }

    if (rechit.flags() & ts4ts5bitset) {
      // only add to TS4TS5 if the bit is not marked as "HcalCellExcludeFromHBHENoiseSummaryR45"
      if (not dbStatus->isBitSet(HcalChannelStatus::HcalCellExcludeFromHBHENoiseSummaryR45)) {
        summary.nts4ts5noise_++;
        summary.ts4ts5noisee_ += rechit.eraw();
        GlobalPoint gp = geo->getPosition(rechit.id());
        double et = rechit.eraw() * gp.perp() / gp.mag();
        summary.ts4ts5noiseet_ += et;
      }
    }

    if (rechit.flags() & negativebitset) {
      summary.nnegativenoise_++;
      summary.negativenoisee_ += rechit.eraw();
      GlobalPoint gp = geo->getPosition(rechit.id());
      double et = rechit.eraw() * gp.perp() / gp.mag();
      summary.negativenoiseet_ += et;
    }

    // find the hpd that the rechit is in
    HcalNoiseHPD& hpd = (*array.findHPD(rechit));

    // create a persistent reference to the rechit
    edm::Ref<HBHERecHitCollection> myRef(handle, it - handle->begin());

    // store it in a place so that it remains sorted by energy
    hpd.refrechitset_.insert(myRef);

  }  // end loop over rechits

  // loop over all HPDs and transfer the information from refrechitset_ to rechits_;
  for (HcalNoiseRBXArray::iterator rbxit = array.begin(); rbxit != array.end(); ++rbxit) {
    for (std::vector<HcalNoiseHPD>::iterator hpdit = rbxit->hpds_.begin(); hpdit != rbxit->hpds_.end(); ++hpdit) {
      // loop over all of the entries in the set and add them to rechits_
      for (std::set<edm::Ref<HBHERecHitCollection>, RefHBHERecHitEnergyComparison>::const_iterator it =
               hpdit->refrechitset_.begin();
           it != hpdit->refrechitset_.end();
           ++it) {
        hpdit->rechits_.push_back(*it);
      }
    }
  }
  // now the rechits in all the HPDs are sorted by energy!

  return;
}

// ------------ fill the array with calo tower information
void HcalNoiseInfoProducer::fillcalotwrs(edm::Event& iEvent,
                                         const edm::EventSetup& iSetup,
                                         HcalNoiseRBXArray& array,
                                         HcalNoiseSummary& summary) const {
  // get the calotowers
  edm::Handle<CaloTowerCollection> handle;
  //  iEvent.getByLabel(caloTowerCollName_, handle);
  iEvent.getByToken(calotower_token_, handle);

  if (!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
        << " could not find CaloTowerCollection named " << caloTowerCollName_ << "\n.";
    return;
  }

  summary.emenergy_ = summary.hadenergy_ = 0.0;

  // loop over all of the calotower information
  for (CaloTowerCollection::const_iterator it = handle->begin(); it != handle->end(); ++it) {
    const CaloTower& twr = (*it);

    // create a persistent reference to the tower
    edm::Ref<CaloTowerCollection> myRef(handle, it - handle->begin());

    // get all of the hpd's that are pointed to by the calotower
    std::vector<std::vector<HcalNoiseHPD>::iterator> hpditervec;
    array.findHPD(twr, hpditervec);

    // loop over the hpd's and add the reference to the RefVectors
    for (std::vector<std::vector<HcalNoiseHPD>::iterator>::iterator it = hpditervec.begin(); it != hpditervec.end();
         ++it)
      (*it)->calotowers_.push_back(myRef);

    // skip over anything with |ieta|>maxCaloTowerIEta
    if (twr.ietaAbs() > maxCaloTowerIEta_) {
      summary.emenergy_ += twr.emEnergy();
      summary.hadenergy_ += twr.hadEnergy();
    }
  }

  return;
}

// ------------ fill the summary info from jets
void HcalNoiseInfoProducer::filljetinfo(edm::Event& iEvent,
                                        const edm::EventSetup& iSetup,
                                        HcalNoiseSummary& summary) const {
  bool goodJetFoundInLowBVRegion = false;  // checks whether a jet is in
                                           // a low BV region, where false
                                           // noise flagging rate is higher.
  if (!jetCollName_.empty()) {
    edm::Handle<reco::PFJetCollection> pfjet_h;
    iEvent.getByToken(jet_token_, pfjet_h);

    if (pfjet_h.isValid()) {
      int jetindex = 0;
      for (reco::PFJetCollection::const_iterator jet = pfjet_h->begin(); jet != pfjet_h->end(); ++jet) {
        if (jetindex > maxjetindex_)
          break;  // only look at jets with
                  // indices up to maxjetindex_

        // Check whether jet is in low-BV region (0<eta<1.4, -1.8<phi<-1.4)
        if (jet->eta() > 0.0 && jet->eta() < 1.4 && jet->phi() > -1.8 && jet->phi() < -1.4) {
          // Look for a good jet in low BV region;
          // if found, we will keep event
          if (maxNHF_ < 0.0 || jet->neutralHadronEnergyFraction() < maxNHF_) {
            goodJetFoundInLowBVRegion = true;
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
void HcalNoiseInfoProducer::filltracks(edm::Event& iEvent,
                                       const edm::EventSetup& iSetup,
                                       HcalNoiseSummary& summary) const {
  edm::Handle<reco::TrackCollection> handle;
  //  iEvent.getByLabel(trackCollName_, handle);
  iEvent.getByToken(track_token_, handle);

  // don't throw exception, just return quietly
  if (!handle.isValid()) {
    //    throw edm::Exception(edm::errors::ProductNotFound)
    //      << " could not find trackCollection named " << trackCollName_ << "\n.";
    return;
  }

  summary.trackenergy_ = 0.0;
  for (reco::TrackCollection::const_iterator iTrack = handle->begin(); iTrack != handle->end(); ++iTrack) {
    const reco::Track& trk = *iTrack;
    if (trk.pt() < minTrackPt_ || fabs(trk.eta()) > maxTrackEta_)
      continue;

    summary.trackenergy_ += trk.p();
  }

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalNoiseInfoProducer);
