#include "CalibTracker/SiStripChannelGain/interface/SiStripGainsPCLWorker.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>

//********************************************************************************//
SiStripGainsPCLWorker::SiStripGainsPCLWorker(const edm::ParameterSet& iConfig) {
  MinTrackMomentum = iConfig.getUntrackedParameter<double>("minTrackMomentum", 3.0);
  MaxTrackMomentum = iConfig.getUntrackedParameter<double>("maxTrackMomentum", 99999.0);
  MinTrackEta = iConfig.getUntrackedParameter<double>("minTrackEta", -5.0);
  MaxTrackEta = iConfig.getUntrackedParameter<double>("maxTrackEta", 5.0);
  MaxNrStrips = iConfig.getUntrackedParameter<unsigned>("maxNrStrips", 2);
  MinTrackHits = iConfig.getUntrackedParameter<unsigned>("MinTrackHits", 8);
  MaxTrackChiOverNdf = iConfig.getUntrackedParameter<double>("MaxTrackChiOverNdf", 3);
  MaxTrackingIteration = iConfig.getUntrackedParameter<int>("MaxTrackingIteration", 7);
  AllowSaturation = iConfig.getUntrackedParameter<bool>("AllowSaturation", false);
  FirstSetOfConstants = iConfig.getUntrackedParameter<bool>("FirstSetOfConstants", true);
  Validation = iConfig.getUntrackedParameter<bool>("Validation", false);
  OldGainRemoving = iConfig.getUntrackedParameter<bool>("OldGainRemoving", false);
  useCalibration = iConfig.getUntrackedParameter<bool>("UseCalibration", false);
  doChargeMonitorPerPlane = iConfig.getUntrackedParameter<bool>("doChargeMonitorPerPlane", false);
  m_DQMdir = iConfig.getUntrackedParameter<std::string>("DQMdir", "AlCaReco/SiStripGains");
  m_calibrationMode = iConfig.getUntrackedParameter<std::string>("calibrationMode", "StdBunch");
  VChargeHisto = iConfig.getUntrackedParameter<std::vector<std::string>>("ChargeHisto");

  // fill in the mapping between the histogram indices and the (id,side,plane) tuple
  std::vector<std::pair<std::string, std::string>> hnames =
      APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    int id = APVGain::subdetectorId((hnames[i]).first);
    int side = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    int thick = APVGain::thickness((hnames[i]).first);
    std::string s = hnames[i].first;

    auto loc = APVloc(thick, id, side, plane, s);
    theTopologyMap.insert(std::make_pair(i, loc));
  }

  //Set the monitoring element tag and store
  dqm_tag_.reserve(7);
  dqm_tag_.clear();
  dqm_tag_.push_back("StdBunch");    // statistic collection from Standard Collision Bunch @ 3.8 T
  dqm_tag_.push_back("StdBunch0T");  // statistic collection from Standard Collision Bunch @ 0 T
  dqm_tag_.push_back("AagBunch");    // statistic collection from First Collision After Abort Gap @ 3.8 T
  dqm_tag_.push_back("AagBunch0T");  // statistic collection from First Collision After Abort Gap @ 0 T
  dqm_tag_.push_back("IsoMuon");     // statistic collection from Isolated Muon @ 3.8 T
  dqm_tag_.push_back("IsoMuon0T");   // statistic collection from Isolated Muon @ 0 T
  dqm_tag_.push_back("Harvest");     // statistic collection: Harvest

  // configure token for gathering the ntuple variables
  edm::ParameterSet swhallowgain_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("gain");

  std::string label = swhallowgain_pset.getUntrackedParameter<std::string>("label");
  CalibPrefix_ = swhallowgain_pset.getUntrackedParameter<std::string>("prefix");
  CalibSuffix_ = swhallowgain_pset.getUntrackedParameter<std::string>("suffix");

  trackindex_token_ = consumes<std::vector<int>>(edm::InputTag(label, CalibPrefix_ + "trackindex" + CalibSuffix_));
  rawid_token_ = consumes<std::vector<unsigned int>>(edm::InputTag(label, CalibPrefix_ + "rawid" + CalibSuffix_));
  localdirx_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "localdirx" + CalibSuffix_));
  localdiry_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "localdiry" + CalibSuffix_));
  localdirz_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "localdirz" + CalibSuffix_));
  firststrip_token_ =
      consumes<std::vector<unsigned short>>(edm::InputTag(label, CalibPrefix_ + "firststrip" + CalibSuffix_));
  nstrips_token_ = consumes<std::vector<unsigned short>>(edm::InputTag(label, CalibPrefix_ + "nstrips" + CalibSuffix_));
  saturation_token_ = consumes<std::vector<bool>>(edm::InputTag(label, CalibPrefix_ + "saturation" + CalibSuffix_));
  overlapping_token_ = consumes<std::vector<bool>>(edm::InputTag(label, CalibPrefix_ + "overlapping" + CalibSuffix_));
  farfromedge_token_ = consumes<std::vector<bool>>(edm::InputTag(label, CalibPrefix_ + "farfromedge" + CalibSuffix_));
  charge_token_ = consumes<std::vector<unsigned int>>(edm::InputTag(label, CalibPrefix_ + "charge" + CalibSuffix_));
  path_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "path" + CalibSuffix_));
#ifdef ExtendedCALIBTree
  chargeoverpath_token_ =
      consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "chargeoverpath" + CalibSuffix_));
#endif
  amplitude_token_ =
      consumes<std::vector<unsigned char>>(edm::InputTag(label, CalibPrefix_ + "amplitude" + CalibSuffix_));
  gainused_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "gainused" + CalibSuffix_));
  gainusedTick_token_ =
      consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "gainusedTick" + CalibSuffix_));

  edm::ParameterSet evtinfo_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("evtinfo");
  label = evtinfo_pset.getUntrackedParameter<std::string>("label");
  EventPrefix_ = evtinfo_pset.getUntrackedParameter<std::string>("prefix");
  EventSuffix_ = evtinfo_pset.getUntrackedParameter<std::string>("suffix");
  TrigTech_token_ = consumes<std::vector<bool>>(edm::InputTag(label, EventPrefix_ + "TrigTech" + EventSuffix_));

  edm::ParameterSet track_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("tracks");
  label = track_pset.getUntrackedParameter<std::string>("label");
  TrackPrefix_ = track_pset.getUntrackedParameter<std::string>("prefix");
  TrackSuffix_ = track_pset.getUntrackedParameter<std::string>("suffix");

  trackchi2ndof_token_ = consumes<std::vector<double>>(edm::InputTag(label, TrackPrefix_ + "chi2ndof" + TrackSuffix_));
  trackp_token_ = consumes<std::vector<float>>(edm::InputTag(label, TrackPrefix_ + "momentum" + TrackSuffix_));
  trackpt_token_ = consumes<std::vector<float>>(edm::InputTag(label, TrackPrefix_ + "pt" + TrackSuffix_));
  tracketa_token_ = consumes<std::vector<double>>(edm::InputTag(label, TrackPrefix_ + "eta" + TrackSuffix_));
  trackphi_token_ = consumes<std::vector<double>>(edm::InputTag(label, TrackPrefix_ + "phi" + TrackSuffix_));
  trackhitsvalid_token_ =
      consumes<std::vector<unsigned int>>(edm::InputTag(label, TrackPrefix_ + "hitsvalid" + TrackSuffix_));
  trackalgo_token_ = consumes<std::vector<int>>(edm::InputTag(label, TrackPrefix_ + "algo" + TrackSuffix_));

  tTopoToken_ = esConsumes();
  tkGeomToken_ = esConsumes<edm::Transition::BeginRun>();
  gainToken_ = esConsumes<edm::Transition::BeginRun>();
  qualityToken_ = esConsumes<edm::Transition::BeginRun>();
}

//********************************************************************************//
void SiStripGainsPCLWorker::dqmBeginRun(edm::Run const& run,
                                        edm::EventSetup const& iSetup,
                                        APVGain::APVGainHistograms& histograms) const {
  using namespace edm;
  static constexpr float defaultGainTick = 690. / 640.;

  // fills the APV collections at each begin run
  const TrackerGeometry* bareTkGeomPtr = &iSetup.getData(tkGeomToken_);
  checkBookAPVColls(bareTkGeomPtr, histograms);

  const auto gainHandle = iSetup.getHandle(gainToken_);
  if (!gainHandle.isValid()) {
    edm::LogError("SiStripGainPCLWorker") << "gainHandle is not valid\n";
    exit(0);
  }

  const auto& siStripQuality = iSetup.getData(qualityToken_);

  for (unsigned int a = 0; a < histograms.APVsCollOrdered.size(); a++) {
    std::shared_ptr<stAPVGain> APV = histograms.APVsCollOrdered[a];

    if (APV->SubDet == PixelSubdetector::PixelBarrel || APV->SubDet == PixelSubdetector::PixelEndcap)
      continue;

    APV->isMasked = siStripQuality.IsApvBad(APV->DetId, APV->APVId);

    if (gainHandle->getNumberOfTags() != 2) {
      edm::LogError("SiStripGainPCLWorker") << "NUMBER OF GAIN TAG IS EXPECTED TO BE 2\n";
      fflush(stdout);
      exit(0);
    };
    float newPreviousGain = gainHandle->getApvGain(APV->APVId, gainHandle->getRange(APV->DetId, 1), 1);
    if (APV->PreviousGain != 1 and newPreviousGain != APV->PreviousGain)
      edm::LogWarning("SiStripGainPCLWorker") << "WARNING: ParticleGain in the global tag changed\n";
    APV->PreviousGain = newPreviousGain;

    float newPreviousGainTick =
        APV->isMasked ? defaultGainTick : gainHandle->getApvGain(APV->APVId, gainHandle->getRange(APV->DetId, 0), 0);
    if (APV->PreviousGainTick != 1 and newPreviousGainTick != APV->PreviousGainTick) {
      edm::LogWarning("SiStripGainPCLWorker")
          << "WARNING: TickMarkGain in the global tag changed\n"
          << std::endl
          << " APV->SubDet: " << APV->SubDet << " APV->APVId:" << APV->APVId << std::endl
          << " APV->PreviousGainTick: " << APV->PreviousGainTick << " newPreviousGainTick: " << newPreviousGainTick
          << std::endl;
    }
    APV->PreviousGainTick = newPreviousGainTick;
  }
}

//********************************************************************************//
// ------------ method called for each event  ------------
void SiStripGainsPCLWorker::dqmAnalyze(edm::Event const& iEvent,
                                       edm::EventSetup const& iSetup,
                                       APVGain::APVGainHistograms const& histograms) const {
  using namespace edm;

  unsigned int eventnumber = iEvent.id().event();
  unsigned int runnumber = iEvent.id().run();

  edm::LogInfo("SiStripGainsPCLWorker") << "Processing run " << runnumber << " and event " << eventnumber << std::endl;

  const TrackerTopology* topo = &iSetup.getData(tTopoToken_);

  // *****************************
  // * Event data handles
  // *****************************

  //Event data

  // Track data
  Handle<const std::vector<double>> handle01;
  iEvent.getByToken(trackchi2ndof_token_, handle01);
  auto trackchi2ndof = handle01.product();

  Handle<const std::vector<float>> handle02;
  iEvent.getByToken(trackp_token_, handle02);
  auto trackp = handle02.product();

  Handle<const std::vector<double>> handle03;
  iEvent.getByToken(tracketa_token_, handle03);
  auto tracketa = handle03.product();

  Handle<const std::vector<unsigned int>> handle04;
  iEvent.getByToken(trackhitsvalid_token_, handle04);
  auto trackhitsvalid = handle04.product();

  Handle<const std::vector<int>> handle05;
  iEvent.getByToken(trackalgo_token_, handle05);
  auto trackalgo = handle05.product();

  // CalibTree data
  Handle<const std::vector<int>> handle06;
  iEvent.getByToken(trackindex_token_, handle06);
  auto trackindex = handle06.product();

  Handle<const std::vector<unsigned int>> handle07;
  iEvent.getByToken(rawid_token_, handle07);
  auto rawid = handle07.product();

  Handle<const std::vector<unsigned short>> handle08;
  iEvent.getByToken(firststrip_token_, handle08);
  auto firststrip = handle08.product();

  Handle<const std::vector<unsigned short>> handle09;
  iEvent.getByToken(nstrips_token_, handle09);
  auto nstrips = handle09.product();

  Handle<const std::vector<bool>> handle10;
  iEvent.getByToken(saturation_token_, handle10);
  auto saturation = handle10.product();

  Handle<const std::vector<bool>> handle11;
  iEvent.getByToken(overlapping_token_, handle11);
  auto overlapping = handle11.product();

  Handle<const std::vector<bool>> handle12;
  iEvent.getByToken(farfromedge_token_, handle12);
  auto farfromedge = handle12.product();

  Handle<const std::vector<unsigned int>> handle13;
  iEvent.getByToken(charge_token_, handle13);
  auto charge = handle13.product();

  Handle<const std::vector<double>> handle14;
  iEvent.getByToken(path_token_, handle14);
  auto path = handle14.product();

#ifdef ExtendedCALIBTree
  Handle<const std::vector<double>> handle15;
  iEvent.getByToken(chargeoverpath_token_, handle15);
  auto chargeoverpath = handle15.product();
#endif

  Handle<const std::vector<unsigned char>> handle16;
  iEvent.getByToken(amplitude_token_, handle16);
  auto amplitude = handle16.product();

  Handle<const std::vector<double>> handle17;
  iEvent.getByToken(gainused_token_, handle17);
  auto gainused = handle17.product();

  Handle<const std::vector<double>> handle18;
  iEvent.getByToken(gainusedTick_token_, handle18);
  auto gainusedTick = handle18.product();

  for (const auto& elem : theTopologyMap) {
    LogDebug("SiStripGainsPCLWorker") << elem.first << " - " << elem.second.m_string << " "
                                      << elem.second.m_subdetectorId << " " << elem.second.m_subdetectorSide << " "
                                      << elem.second.m_subdetectorPlane << std::endl;
  }

  LogDebug("SiStripGainsPCLWorker") << "for mode" << m_calibrationMode << std::endl;

  int elepos = statCollectionFromMode(m_calibrationMode.c_str());

  histograms.EventStats->Fill(0., 0., 1);
  histograms.EventStats->Fill(1., 0., trackp->size());
  histograms.EventStats->Fill(2., 0., charge->size());

  unsigned int FirstAmplitude = 0;
  for (unsigned int i = 0; i < charge->size(); i++) {
    FirstAmplitude += (*nstrips)[i];
    int TI = (*trackindex)[i];

    if ((*tracketa)[TI] < MinTrackEta)
      continue;
    if ((*tracketa)[TI] > MaxTrackEta)
      continue;
    if ((*trackp)[TI] < MinTrackMomentum)
      continue;
    if ((*trackp)[TI] > MaxTrackMomentum)
      continue;
    if ((*trackhitsvalid)[TI] < MinTrackHits)
      continue;
    if ((*trackchi2ndof)[TI] > MaxTrackChiOverNdf)
      continue;
    if ((*trackalgo)[TI] > MaxTrackingIteration)
      continue;

    std::shared_ptr<stAPVGain> APV = histograms.APVsColl.at(
        ((*rawid)[i] << 4) |
        ((*firststrip)[i] /
         128));  //works for both strip and pixel thanks to firstStrip encoding for pixel in the calibTree

    if (APV->SubDet > 2 && (*farfromedge)[i] == false)
      continue;
    if (APV->SubDet > 2 && (*overlapping)[i] == true)
      continue;
    if (APV->SubDet > 2 && (*saturation)[i] && !AllowSaturation)
      continue;
    if (APV->SubDet > 2 && (*nstrips)[i] > MaxNrStrips)
      continue;

    int Charge = 0;
    if (APV->SubDet > 2 && (useCalibration || !FirstSetOfConstants)) {
      bool Saturation = false;
      for (unsigned int s = 0; s < (*nstrips)[i]; s++) {
        int StripCharge = (*amplitude)[FirstAmplitude - (*nstrips)[i] + s];
        if (useCalibration && !FirstSetOfConstants) {
          StripCharge = (int)(StripCharge * (APV->PreviousGain / APV->CalibGain));
        } else if (useCalibration) {
          StripCharge = (int)(StripCharge / APV->CalibGain);
        } else if (!FirstSetOfConstants) {
          StripCharge = (int)(StripCharge * APV->PreviousGain);
        }
        if (StripCharge > 1024) {
          StripCharge = 255;
          Saturation = true;
        } else if (StripCharge > 254) {
          StripCharge = 254;
          Saturation = true;
        }
        Charge += StripCharge;
      }
      if (Saturation && !AllowSaturation)
        continue;
    } else if (APV->SubDet > 2) {
      Charge = (*charge)[i];
    } else {
      Charge = (*charge)[i] / 265.0;  //expected scale factor between pixel and strip charge
    }

    double ClusterChargeOverPath = ((double)Charge) / (*path)[i];
    if (APV->SubDet > 2) {
      if (Validation) {
        ClusterChargeOverPath /= (*gainused)[i];
      }
      if (OldGainRemoving) {
        ClusterChargeOverPath *= (*gainused)[i];
      }
    }

    // keep processing of pixel cluster charge until here
    if (APV->SubDet <= 2)
      continue;

    // real histogram for calibration
    histograms.Charge_Vs_Index[elepos]->Fill(APV->Index, ClusterChargeOverPath);
    LogDebug("SiStripGainsPCLWorker") << " for mode " << m_calibrationMode << "\n"
                                      << " i " << i << " useCalibration " << useCalibration << " FirstSetOfConstants "
                                      << FirstSetOfConstants << " APV->PreviousGain " << APV->PreviousGain
                                      << " APV->CalibGain " << APV->CalibGain << " APV->DetId " << APV->DetId
                                      << " APV->Index " << APV->Index << " Charge " << Charge << " Path " << (*path)[i]
                                      << " ClusterChargeOverPath " << ClusterChargeOverPath << std::endl;

    // Fill monitoring histograms
    int mCharge1 = 0;
    int mCharge2 = 0;
    int mCharge3 = 0;
    int mCharge4 = 0;
    if (APV->SubDet > 2) {
      for (unsigned int s = 0; s < (*nstrips)[i]; s++) {
        int StripCharge = (*amplitude)[FirstAmplitude - (*nstrips)[i] + s];
        if (StripCharge > 1024)
          StripCharge = 255;
        else if (StripCharge > 254)
          StripCharge = 254;
        mCharge1 += StripCharge;
        mCharge2 += StripCharge;
        mCharge3 += StripCharge;
        mCharge4 += StripCharge;
      }
      // Revome gains for monitoring
      mCharge2 *= (*gainused)[i];                         // remove G2
      mCharge3 *= (*gainusedTick)[i];                     // remove G1
      mCharge4 *= ((*gainused)[i] * (*gainusedTick)[i]);  // remove G1 and G2
    }

    LogDebug("SiStripGainsPCLWorker") << " full charge " << mCharge1 << " remove G2 " << mCharge2 << " remove G1 "
                                      << mCharge3 << " remove G1*G2 " << mCharge4 << std::endl;

    auto indices = APVGain::FetchIndices(theTopologyMap, (*rawid)[i], topo);

    for (auto m : indices)
      histograms.Charge_1[elepos][m]->Fill(((double)mCharge1) / (*path)[i]);
    for (auto m : indices)
      histograms.Charge_2[elepos][m]->Fill(((double)mCharge2) / (*path)[i]);
    for (auto m : indices)
      histograms.Charge_3[elepos][m]->Fill(((double)mCharge3) / (*path)[i]);
    for (auto m : indices)
      histograms.Charge_4[elepos][m]->Fill(((double)mCharge4) / (*path)[i]);

    if (APV->SubDet == StripSubdetector::TIB) {
      histograms.Charge_Vs_PathlengthTIB[elepos]->Fill((*path)[i], Charge);  // TIB

    } else if (APV->SubDet == StripSubdetector::TOB) {
      histograms.Charge_Vs_PathlengthTOB[elepos]->Fill((*path)[i], Charge);  // TOB

    } else if (APV->SubDet == StripSubdetector::TID) {
      if (APV->Eta < 0) {
        histograms.Charge_Vs_PathlengthTIDM[elepos]->Fill((*path)[i], Charge);
      }  // TID minus
      else if (APV->Eta > 0) {
        histograms.Charge_Vs_PathlengthTIDP[elepos]->Fill((*path)[i], Charge);
      }  // TID plus

    } else if (APV->SubDet == StripSubdetector::TEC) {
      if (APV->Eta < 0) {
        if (APV->Thickness < 0.04) {
          histograms.Charge_Vs_PathlengthTECM1[elepos]->Fill((*path)[i], Charge);
        }  // TEC minus, type 1
        else if (APV->Thickness > 0.04) {
          histograms.Charge_Vs_PathlengthTECM2[elepos]->Fill((*path)[i], Charge);
        }  // TEC minus, type 2
      } else if (APV->Eta > 0) {
        if (APV->Thickness < 0.04) {
          histograms.Charge_Vs_PathlengthTECP1[elepos]->Fill((*path)[i], Charge);
        }  // TEC plus, type 1
        else if (APV->Thickness > 0.04) {
          histograms.Charge_Vs_PathlengthTECP2[elepos]->Fill((*path)[i], Charge);
        }  // TEC plus, type 2
      }
    }

  }  // END OF ON-CLUSTER LOOP

  //LogDebug("SiStripGainsPCLWorker")<<" for mode"<< m_calibrationMode
  //				   <<" entries in histogram:"<< histograms.Charge_Vs_Index[elepos].getEntries()
  //				   <<std::endl;
}

//********************************************************************************//
void SiStripGainsPCLWorker::beginJob() {}

//********************************************************************************//
// ------------ method called once each job just before starting event loop  ------------
void SiStripGainsPCLWorker::checkBookAPVColls(const TrackerGeometry* bareTkGeomPtr,
                                              APVGain::APVGainHistograms& histograms) const {
  if (bareTkGeomPtr) {  // pointer not yet set: called the first time => fill the APVColls
    auto const& Det = bareTkGeomPtr->dets();

    edm::LogInfo("SiStripGainsPCLWorker") << " Resetting APV struct" << std::endl;

    unsigned int Index = 0;

    for (unsigned int i = 0; i < Det.size(); i++) {
      DetId Detid = Det[i]->geographicalId();
      int SubDet = Detid.subdetId();

      if (SubDet == StripSubdetector::TIB || SubDet == StripSubdetector::TID || SubDet == StripSubdetector::TOB ||
          SubDet == StripSubdetector::TEC) {
        auto DetUnit = dynamic_cast<const StripGeomDetUnit*>(Det[i]);
        if (!DetUnit)
          continue;

        const StripTopology& Topo = DetUnit->specificTopology();
        unsigned int NAPV = Topo.nstrips() / 128;

        for (unsigned int j = 0; j < NAPV; j++) {
          auto APV = std::make_shared<stAPVGain>();
          APV->Index = Index;
          APV->Bin = -1;
          APV->DetId = Detid.rawId();
          APV->APVId = j;
          APV->SubDet = SubDet;
          APV->FitMPV = -1;
          APV->FitMPVErr = -1;
          APV->FitWidth = -1;
          APV->FitWidthErr = -1;
          APV->FitChi2 = -1;
          APV->FitNorm = -1;
          APV->Gain = -1;
          APV->PreviousGain = 1;
          APV->PreviousGainTick = 1;
          APV->x = DetUnit->position().basicVector().x();
          APV->y = DetUnit->position().basicVector().y();
          APV->z = DetUnit->position().basicVector().z();
          APV->Eta = DetUnit->position().basicVector().eta();
          APV->Phi = DetUnit->position().basicVector().phi();
          APV->R = DetUnit->position().basicVector().transverse();
          APV->Thickness = DetUnit->surface().bounds().thickness();
          APV->NEntries = 0;
          APV->isMasked = false;

          histograms.APVsCollOrdered.push_back(APV);
          histograms.APVsColl[(APV->DetId << 4) | APV->APVId] = APV;
          Index++;
          histograms.NStripAPVs++;
        }  // loop on APVs
      }    // if is Strips
    }      // loop on dets

    for (unsigned int i = 0; i < Det.size();
         i++) {  //Make two loop such that the Pixel information is added at the end --> make transition simpler
      DetId Detid = Det[i]->geographicalId();
      int SubDet = Detid.subdetId();
      if (SubDet == PixelSubdetector::PixelBarrel || SubDet == PixelSubdetector::PixelEndcap) {
        auto DetUnit = dynamic_cast<const PixelGeomDetUnit*>(Det[i]);
        if (!DetUnit)
          continue;

        const PixelTopology& Topo = DetUnit->specificTopology();
        unsigned int NROCRow = Topo.nrows() / (80.);
        unsigned int NROCCol = Topo.ncolumns() / (52.);

        for (unsigned int j = 0; j < NROCRow; j++) {
          for (unsigned int i = 0; i < NROCCol; i++) {
            auto APV = std::make_shared<stAPVGain>();
            APV->Index = Index;
            APV->Bin = -1;
            APV->DetId = Detid.rawId();
            APV->APVId = (j << 3 | i);
            APV->SubDet = SubDet;
            APV->FitMPV = -1;
            APV->FitMPVErr = -1;
            APV->FitWidth = -1;
            APV->FitWidthErr = -1;
            APV->FitChi2 = -1;
            APV->Gain = -1;
            APV->PreviousGain = 1;
            APV->PreviousGainTick = 1;
            APV->x = DetUnit->position().basicVector().x();
            APV->y = DetUnit->position().basicVector().y();
            APV->z = DetUnit->position().basicVector().z();
            APV->Eta = DetUnit->position().basicVector().eta();
            APV->Phi = DetUnit->position().basicVector().phi();
            APV->R = DetUnit->position().basicVector().transverse();
            APV->Thickness = DetUnit->surface().bounds().thickness();
            APV->isMasked = false;  //SiPixelQuality_->IsModuleBad(Detid.rawId());
            APV->NEntries = 0;

            histograms.APVsCollOrdered.push_back(APV);
            histograms.APVsColl[(APV->DetId << 4) | APV->APVId] = APV;
            Index++;
            histograms.NPixelDets++;

          }  // loop on ROC cols
        }    // loop on ROC rows
      }      // if Pixel
    }        // loop on Dets
  }          //if (!bareTkGeomPtr_) ...
}

//********************************************************************************//
void SiStripGainsPCLWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//********************************************************************************//
void SiStripGainsPCLWorker::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& run,
                                           edm::EventSetup const& setup,
                                           APVGain::APVGainHistograms& histograms) const {
  ibooker.cd();
  std::string dqm_dir = m_DQMdir;
  const char* tag = dqm_tag_[statCollectionFromMode(m_calibrationMode.c_str())].c_str();

  edm::LogInfo("SiStripGainsPCLWorker") << "Setting " << dqm_dir << " in DQM and booking histograms for tag " << tag
                                        << std::endl;

  ibooker.setCurrentFolder(dqm_dir);

  // this MonitorElement is created to log the number of events / tracks and clusters used
  // by the calibration algorithm

  histograms.EventStats = ibooker.book2S("EventStats", "Statistics", 3, -0.5, 2.5, 1, 0, 1);
  histograms.EventStats->setBinLabel(1, "events count", 1);
  histograms.EventStats->setBinLabel(2, "tracks count", 1);
  histograms.EventStats->setBinLabel(3, "clusters count", 1);

  std::string stag(tag);
  if (!stag.empty() && stag[0] != '_')
    stag.insert(0, 1, '_');

  std::string cvi = std::string("Charge_Vs_Index") + stag;
  std::string cvpTIB = std::string("Charge_Vs_PathlengthTIB") + stag;
  std::string cvpTOB = std::string("Charge_Vs_PathlengthTOB") + stag;
  std::string cvpTIDP = std::string("Charge_Vs_PathlengthTIDP") + stag;
  std::string cvpTIDM = std::string("Charge_Vs_PathlengthTIDM") + stag;
  std::string cvpTECP1 = std::string("Charge_Vs_PathlengthTECP1") + stag;
  std::string cvpTECP2 = std::string("Charge_Vs_PathlengthTECP2") + stag;
  std::string cvpTECM1 = std::string("Charge_Vs_PathlengthTECM1") + stag;
  std::string cvpTECM2 = std::string("Charge_Vs_PathlengthTECM2") + stag;

  int elepos = statCollectionFromMode(tag);

  histograms.Charge_Vs_Index.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTIB.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTOB.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTIDP.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTIDM.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTECP1.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTECP2.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTECM1.reserve(dqm_tag_.size());
  histograms.Charge_Vs_PathlengthTECM2.reserve(dqm_tag_.size());

  // The cluster charge is stored by exploiting a non uniform binning in order
  // reduce the histogram memory size. The bin width is relaxed with a falling
  // exponential function and the bin boundaries are stored in the binYarray.
  // The binXarray is used to provide as many bins as the APVs.
  //
  // More details about this implementations are here:
  // https://indico.cern.ch/event/649344/contributions/2672267/attachments/1498323/2332518/OptimizeChHisto.pdf

  std::vector<float> binXarray;
  binXarray.reserve(histograms.NStripAPVs + 1);
  for (unsigned int a = 0; a <= histograms.NStripAPVs; a++) {
    binXarray.push_back((float)a);
  }

  std::array<float, 688> binYarray;
  double p0 = 5.445;
  double p1 = 0.002113;
  double p2 = 69.01576;
  double y = 0.;
  for (int b = 0; b < 687; b++) {
    binYarray[b] = y;
    if (y <= 902.)
      y = y + 2.;
    else
      y = (p0 - log(exp(p0 - p1 * y) - p2 * p1)) / p1;
  }
  binYarray[687] = 4000.;

  histograms.Charge_1[elepos].clear();
  histograms.Charge_2[elepos].clear();
  histograms.Charge_3[elepos].clear();
  histograms.Charge_4[elepos].clear();

  auto it = histograms.Charge_Vs_Index.begin();
  histograms.Charge_Vs_Index.insert(
      it + elepos,
      ibooker.book2S(cvi.c_str(), cvi.c_str(), histograms.NStripAPVs, &binXarray[0], 687, binYarray.data()));

  it = histograms.Charge_Vs_PathlengthTIB.begin();
  histograms.Charge_Vs_PathlengthTIB.insert(it + elepos,
                                            ibooker.book2S(cvpTIB.c_str(), cvpTIB.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTOB.begin();
  histograms.Charge_Vs_PathlengthTOB.insert(it + elepos,
                                            ibooker.book2S(cvpTOB.c_str(), cvpTOB.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTIDP.begin();
  histograms.Charge_Vs_PathlengthTIDP.insert(
      it + elepos, ibooker.book2S(cvpTIDP.c_str(), cvpTIDP.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTIDM.begin();
  histograms.Charge_Vs_PathlengthTIDM.insert(
      it + elepos, ibooker.book2S(cvpTIDM.c_str(), cvpTIDM.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTECP1.begin();
  histograms.Charge_Vs_PathlengthTECP1.insert(
      it + elepos, ibooker.book2S(cvpTECP1.c_str(), cvpTECP1.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTECP2.begin();
  histograms.Charge_Vs_PathlengthTECP2.insert(
      it + elepos, ibooker.book2S(cvpTECP2.c_str(), cvpTECP2.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTECM1.begin();
  histograms.Charge_Vs_PathlengthTECM1.insert(
      it + elepos, ibooker.book2S(cvpTECM1.c_str(), cvpTECM1.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  it = histograms.Charge_Vs_PathlengthTECM2.begin();
  histograms.Charge_Vs_PathlengthTECM2.insert(
      it + elepos, ibooker.book2S(cvpTECM2.c_str(), cvpTECM2.c_str(), 20, 0.3, 1.3, 250, 0, 2000));

  std::vector<std::pair<std::string, std::string>> hnames =
      APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    histograms.Charge_1[elepos].push_back(ibooker.book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG2");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    histograms.Charge_2[elepos].push_back(ibooker.book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG1");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    histograms.Charge_3[elepos].push_back(ibooker.book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG1G2");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    histograms.Charge_4[elepos].push_back(ibooker.book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.));
  }
}
