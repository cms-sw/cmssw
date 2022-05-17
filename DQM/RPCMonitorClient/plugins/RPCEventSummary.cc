/*  \author Anna Cimmino*/
#include "DQM/RPCMonitorClient/interface/RPCEventSummary.h"
#include "DQM/RPCMonitorClient/interface/RPCSummaryMapHisto.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

RPCEventSummary::RPCEventSummary(const edm::ParameterSet& ps) {
  edm::LogVerbatim("rpceventsummary") << "[RPCEventSummary]: Constructor";

  enableReportSummary_ = ps.getUntrackedParameter<bool>("EnableSummaryReport", true);
  prescaleFactor_ = ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  eventInfoPath_ = ps.getUntrackedParameter<std::string>("EventInfoPath", "RPC/EventInfo");

  std::string subsystemFolder = ps.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  std::string recHitTypeFolder = ps.getUntrackedParameter<std::string>("RecHitTypeFolder", "AllHits");
  std::string summaryFolder = ps.getUntrackedParameter<std::string>("SummaryFolder", "SummaryHistograms");

  globalFolder_ = subsystemFolder + "/" + recHitTypeFolder + "/" + summaryFolder;
  prefixFolder_ = subsystemFolder + "/" + recHitTypeFolder;

  minimumEvents_ = ps.getUntrackedParameter<int>("MinimumRPCEvents", 10000);
  numberDisk_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  doEndcapCertification_ = ps.getUntrackedParameter<bool>("EnableEndcapSummary", false);

  FEDRange_.first = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);

  NumberOfFeds_ = FEDRange_.second - FEDRange_.first + 1;

  offlineDQM_ = ps.getUntrackedParameter<bool>("OfflineDQM", true);

  runInfoToken_ = esConsumes<edm::Transition::EndLuminosityBlock>();
}

RPCEventSummary::~RPCEventSummary() { edm::LogVerbatim("rpceventsummary") << "[RPCEventSummary]: Destructor "; }

void RPCEventSummary::beginJob() {
  edm::LogVerbatim("rpceventsummary") << "[RPCEventSummary]: Begin job ";
  init_ = false;
}

void RPCEventSummary::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                            DQMStore::IGetter& igetter,
                                            edm::LuminosityBlock const& lb,
                                            edm::EventSetup const& setup) {
  edm::LogVerbatim("rpceventsummary") << "[RPCEventSummary]: Begin run";

  if (!init_) {
    lumiCounter_ = prescaleFactor_;

    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

    int defaultValue = 1;
    isIn_ = true;

    if (auto runInfoRec = setup.tryToGet<RunInfoRcd>()) {
      defaultValue = -1;
      //get fed summary information
      auto sumFED = runInfoRec->get(runInfoToken_);
      const std::vector<int> FedsInIds = sumFED.m_fed_in;
      unsigned int f = 0;
      isIn_ = false;
      while (!isIn_ && f < FedsInIds.size()) {
        int fedID = FedsInIds[f];
        //make sure fed id is in allowed range
        if (fedID >= FEDRange_.first && fedID <= FEDRange_.second) {
          defaultValue = 1;
          isIn_ = true;
        }
        f++;
      }
    }

    ibooker.setCurrentFolder(eventInfoPath_);

    //a global summary float [0,1] providing a global summary of the status
    //and showing the goodness of the data taken by the the sub-system
    MonitorElement* meSummary = ibooker.bookFloat("reportSummary");
    meSummary->Fill(defaultValue);

    //TH2F ME providing a mapof values[0-1] to show if problems are localized or distributed
    MonitorElement* me = RPCSummaryMapHisto::book(ibooker, "reportSummaryMap", "RPC Report Summary Map");
    MonitorElement* me2 = RPCSummaryMapHisto::book(ibooker, "noisySummaryMap", "RPC Noisy Chamber Summary Map");
    RPCSummaryMapHisto::setBinsBarrel(me, defaultValue);
    RPCSummaryMapHisto::setBinsEndcap(me, defaultValue);
    RPCSummaryMapHisto::setBinsBarrel(me2, defaultValue);
    RPCSummaryMapHisto::setBinsEndcap(me2, defaultValue);

    //the reportSummaryContents folder containins a collection of ME floats [0-1] (order of 5-10)
    // which describe the behavior of the respective subsystem sub-components.
    ibooker.setCurrentFolder(eventInfoPath_ + "/reportSummaryContents");

    std::vector<std::string> segmentNames;
    for (int i = -2; i <= 2; i++) {
      const std::string segName = fmt::format("RPC_Wheel{}", i);
      segmentNames.push_back(segName);
    }

    for (int i = -numberDisk_; i <= numberDisk_; i++) {
      if (i == 0)
        continue;
      const std::string segName = fmt::format("RPC_Disk{}", i);
      segmentNames.push_back(segName);
    }

    for (const auto& segmentName : segmentNames) {
      MonitorElement* me3 = ibooker.bookFloat(segmentName);
      me3->Fill(defaultValue);
    }

    lumiCounter_ = prescaleFactor_;
    init_ = true;
  }

  if (isIn_ && !offlineDQM_ && lumiCounter_ % prescaleFactor_ == 0) {
    this->clientOperation(igetter);
  }

  ++lumiCounter_;
}

void RPCEventSummary::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (isIn_) {
    this->clientOperation(igetter);
  }
}

void RPCEventSummary::clientOperation(DQMStore::IGetter& igetter) {
  float rpcevents = minimumEvents_;
  MonitorElement* RPCEvents = igetter.get(prefixFolder_ + "/RPCEvents");

  if (RPCEvents) {
    rpcevents = RPCEvents->getBinContent(1);
  }

  if (rpcevents < minimumEvents_)
    return;

  MonitorElement* reportMe = igetter.get(eventInfoPath_ + "/reportSummaryMap");
  MonitorElement* reportMe2 = igetter.get(eventInfoPath_ + "/noisySummaryMap");

  //BARREL
  float barrelFactor = 0;

  for (int w = -2; w < 3; w++) {
    const std::string meName1 = fmt::format("{}/DeadChannelFraction_Roll_vs_Sector_Wheel{}", globalFolder_, w);
    const std::string meName2 = fmt::format("{}/RPCNoisyStrips_Roll_vs_Sector_Wheel{}", globalFolder_, w);
    MonitorElement* myMe1 = igetter.get(meName1);
    MonitorElement* myMe2 = igetter.get(meName2);

    if (myMe1 && myMe2) {
      float wheelFactor = 0;

      for (int s = 1; s <= myMe1->getNbinsX(); s++) {
        float sectorFactor = 0;
        float sectorFactor2 = 0;
        int rollInSector = 0;

        for (int r = 1; r <= myMe1->getNbinsY(); r++) {
          if ((s != 4 && r > 17) || ((s == 9 || s == 11) && r > 15))
            continue;
          rollInSector++;
          sectorFactor += 1 - myMe1->getBinContent(s, r);
          sectorFactor2 += myMe2->getBinContent(s, r);
        }

        if (rollInSector != 0) {
          sectorFactor = sectorFactor / rollInSector;
          sectorFactor2 = sectorFactor2 / rollInSector;
        }

        if (reportMe)
          reportMe->setBinContent(w + 8, s, sectorFactor);
        if (reportMe2)
          reportMe2->setBinContent(w + 8, s, sectorFactor2);
        wheelFactor += sectorFactor;

      }  //end loop on sectors

      wheelFactor = wheelFactor / myMe1->getNbinsX();

      const std::string globalMEName = fmt::format("{}/reportSummaryContents/RPC_Wheel{}", eventInfoPath_, w);
      MonitorElement* globalMe = igetter.get(globalMEName);
      if (globalMe)
        globalMe->Fill(wheelFactor);

      barrelFactor += wheelFactor;
    }  //
  }    //end loop on wheel

  barrelFactor = barrelFactor / 5;

  float endcapFactor = 0;

  if (doEndcapCertification_) {
    //Endcap
    for (int d = -numberDisk_; d <= numberDisk_; d++) {
      if (d == 0)
        continue;

      const std::string meName1 = fmt::format("{}/DeadChannelFraction_Ring_vs_Segment_Disk{}", globalFolder_, d);
      const std::string meName2 = fmt::format("{}/RPCNoisyStrips_Ring_vs_Segment_Disk{}", globalFolder_, d);
      MonitorElement* myMe1 = igetter.get(meName1);
      MonitorElement* myMe2 = igetter.get(meName2);

      if (myMe1 && myMe2) {
        float diskFactor = 0;

        float sectorFactor[6] = {0, 0, 0, 0, 0, 0};
        float sectorFactor2[6] = {0, 0, 0, 0, 0, 0};

        for (int i = 0; i < 6; i++) {
          int firstSeg = (i * 6) + 1;
          int lastSeg = firstSeg + 6;
          int rollInSector = 0;
          for (int seg = firstSeg; seg < lastSeg; seg++) {
            for (int y = 1; y <= myMe1->getNbinsY(); y++) {
              rollInSector++;
              sectorFactor[i] += 1 - myMe1->getBinContent(seg, y);
              sectorFactor2[i] += myMe2->getBinContent(seg, y);
            }
          }
          sectorFactor[i] = sectorFactor[i] / rollInSector;
          sectorFactor2[i] = sectorFactor2[i] / rollInSector;
        }  //end loop on Sectors

        for (int sec = 0; sec < 6; sec++) {
          diskFactor += sectorFactor[sec];
          if (reportMe) {
            if (d < 0)
              reportMe->setBinContent(d + 5, sec + 1, sectorFactor[sec]);
            else
              reportMe->setBinContent(d + 11, sec + 1, sectorFactor[sec]);
          }
          if (reportMe2) {
            if (d < 0)
              reportMe2->setBinContent(d + 5, sec + 1, sectorFactor2[sec]);
            else
              reportMe2->setBinContent(d + 11, sec + 1, sectorFactor2[sec]);
          }
        }

        diskFactor = diskFactor / 6;

        const std::string globalMEName = fmt::format("{}/reportSummaryContents/RPC_Disk{}", eventInfoPath_, d);
        MonitorElement* globalMe = igetter.get(globalMEName);
        if (globalMe)
          globalMe->Fill(diskFactor);

        endcapFactor += diskFactor;
      }  //end loop on disks
    }

    endcapFactor = endcapFactor / (numberDisk_ * 2);
  }

  //Fill repor summary
  float rpcFactor = barrelFactor;
  if (doEndcapCertification_) {
    rpcFactor = (barrelFactor + endcapFactor) / 2;
  }

  MonitorElement* globalMe = igetter.get(eventInfoPath_ + "/reportSummary");
  if (globalMe)
    globalMe->Fill(rpcFactor);
}
