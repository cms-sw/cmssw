/*
 * Original author: A. Raval / A. Meyer - DESY
 * Rewritten by:    B. van Besien - CERN
 * Improved by:     S. Di Guida - INFN and Marconi University
 */

#include "DQMProvInfo.h"
#include <TSystem.h>
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// The LHC beam info used to come from FED812, but since the new TCDS this
// info is in FED1024. We retrieve the BST record from the TCDS digis, and
// we get the LHC beam info using a dedicated data format.

const int DQMProvInfo::MAX_VBINS;
const int DQMProvInfo::MAX_LUMIS;

// Constructor
DQMProvInfo::DQMProvInfo(const edm::ParameterSet& ps) {
  // Initialization of DQM parameters
  subsystemname_ = ps.getUntrackedParameter<std::string>("subSystemFolder", "Info");
  provinfofolder_ = ps.getUntrackedParameter<std::string>("provInfoFolder", "ProvInfo");
  runType_ = ps.getUntrackedParameter<std::string>("runType", "No run type selected");

  // Initialization of the input
  // Used to get the DCS bits:
  dcsStatusCollection_ =
      consumes<DcsStatusCollection>(ps.getUntrackedParameter<std::string>("dcsStatusCollection", "scalersRawToDigi"));

  // Used to get the BST record from the TCDS information
  tcdsrecord_ = consumes<TCDSRecord>(
      ps.getUntrackedParameter<edm::InputTag>("tcdsData", edm::InputTag("tcdsDigis", "tcdsRecord")));

  // Used to get the DCS bits:
  dcsRecordToken_ = consumes<DCSRecord>(
      ps.getUntrackedParameter<edm::InputTag>("dcsRecord", edm::InputTag("onlineMetaDataRawToDigi")));

  // Initialization of the global tag
  globalTag_ = "MODULE::DEFAULT";  // default
  globalTagRetrieved_ = false;     // set as soon as retrieved from first event
}

// Destructor
DQMProvInfo::~DQMProvInfo() = default;

void DQMProvInfo::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iEventSetup) {
  // Here we do everything that needs to be done before the booking
  // Getting the HLT key from HLTConfigProvider:
  hltKey_ = "";
  HLTConfigProvider hltConfig;
  bool changed(true);
  if (!hltConfig.init(iRun, iEventSetup, "HLT", changed)) {
    edm::LogInfo("DQMProvInfo") << "errorHltConfigExtraction" << std::endl;
    hltKey_ = "error extraction";
  } else if (hltConfig.size() <= 0) {
    edm::LogInfo("DQMProvInfo") << "hltConfig" << std::endl;
    hltKey_ = "error key of length 0";
  } else {
    edm::LogInfo("DQMProvInfo") << "HLT key (run): " << hltConfig.tableName() << std::endl;
    hltKey_ = hltConfig.tableName();
  }
}

void DQMProvInfo::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& iRun, edm::EventSetup const& iEventSetup) {
  iBooker.cd();
  // This module will create elements in 3 different folders:
  // - Info/LhcInfo
  // - Info/EventInfo
  // - Info/ProvInfo
  // (string "Info" configurable through subsystemname_)
  // (string "Provinfo" configurable through provinfofolder_)
  iBooker.setCurrentFolder(subsystemname_ + "/LhcInfo/");
  bookHistogramsLhcInfo(iBooker);

  iBooker.setCurrentFolder(subsystemname_ + "/EventInfo/");
  bookHistogramsEventInfo(iBooker);

  iBooker.setCurrentFolder(subsystemname_ + "/" + provinfofolder_);
  bookHistogramsProvInfo(iBooker);
}

void DQMProvInfo::bookHistogramsLhcInfo(DQMStore::IBooker& iBooker) {
  // Element: beamMode
  // Beam parameters provided by BST are defined in:
  // https://edms.cern.ch/document/638899/2.0
  hBeamMode_ = iBooker.book1D("beamMode", "beamMode", MAX_LUMIS, 1., MAX_LUMIS + 1);
  hBeamMode_->getTH1F()->GetYaxis()->Set(21, 0.5, 21.5);
  hBeamMode_->getTH1F()->SetMaximum(21.5);
  hBeamMode_->setBinContent(0., 22.);  // Not clear, remove when testable

  hBeamMode_->setAxisTitle("Luminosity Section", 1);
  hBeamMode_->setBinLabel(1, "no mode", 2);
  hBeamMode_->setBinLabel(2, "setup", 2);
  hBeamMode_->setBinLabel(3, "inj pilot", 2);
  hBeamMode_->setBinLabel(4, "inj intr", 2);
  hBeamMode_->setBinLabel(5, "inj nomn", 2);
  hBeamMode_->setBinLabel(6, "pre ramp", 2);
  hBeamMode_->setBinLabel(7, "ramp", 2);
  hBeamMode_->setBinLabel(8, "flat top", 2);
  hBeamMode_->setBinLabel(9, "squeeze", 2);
  hBeamMode_->setBinLabel(10, "adjust", 2);
  hBeamMode_->setBinLabel(11, "stable", 2);
  hBeamMode_->setBinLabel(12, "unstable", 2);
  hBeamMode_->setBinLabel(13, "beam dump", 2);
  hBeamMode_->setBinLabel(14, "ramp down", 2);
  hBeamMode_->setBinLabel(15, "recovery", 2);
  hBeamMode_->setBinLabel(16, "inj dump", 2);
  hBeamMode_->setBinLabel(17, "circ dump", 2);
  hBeamMode_->setBinLabel(18, "abort", 2);
  hBeamMode_->setBinLabel(19, "cycling", 2);
  hBeamMode_->setBinLabel(20, "warn b-dump", 2);
  hBeamMode_->setBinLabel(21, "no beam", 2);

  // Element: intensity1
  hIntensity1_ = iBooker.book1D("intensity1", "Intensity Beam 1", MAX_LUMIS, 1., MAX_LUMIS + 1);
  hIntensity1_->setAxisTitle("Luminosity Section", 1);
  hIntensity1_->setAxisTitle("N [E10]", 2);

  // Element: intensity2
  hIntensity2_ = iBooker.book1D("intensity2", "Intensity Beam 2", MAX_LUMIS, 1., MAX_LUMIS + 1);
  hIntensity2_->setAxisTitle("Luminosity Section", 1);
  hIntensity2_->setAxisTitle("N [E10]", 2);

  // Element: lhcFill
  hLhcFill_ = iBooker.book1D("lhcFill", "LHC Fill Number", MAX_LUMIS, 1., MAX_LUMIS + 1);
  hLhcFill_->setAxisTitle("Luminosity Section", 1);

  // Element: momentum
  hMomentum_ = iBooker.book1D("momentum", "Beam Energy [GeV]", MAX_LUMIS, 1., MAX_LUMIS + 1);
  hMomentum_->setAxisTitle("Luminosity Section", 1);
}

void DQMProvInfo::bookHistogramsEventInfo(DQMStore::IBooker& iBooker) {
  // Element: reportSummary
  reportSummary_ = iBooker.bookFloat("reportSummary");

  // Element: reportSummaryMap   (this is the famous HV plot)
  reportSummaryMap_ = iBooker.book2D("reportSummaryMap",
                                     "DCS HV Status and Beam Status per Lumisection",
                                     MAX_LUMIS,
                                     0,
                                     MAX_LUMIS,
                                     MAX_VBINS,
                                     0.,
                                     MAX_VBINS);
  reportSummaryMap_->setAxisTitle("Luminosity Section");

  reportSummaryMap_->setBinLabel(VBIN_CSC_P, "CSC+", 2);
  reportSummaryMap_->setBinLabel(VBIN_CSC_M, "CSC-", 2);
  reportSummaryMap_->setBinLabel(VBIN_DT_0, "DT0", 2);
  reportSummaryMap_->setBinLabel(VBIN_DT_P, "DT+", 2);
  reportSummaryMap_->setBinLabel(VBIN_DT_M, "DT-", 2);
  reportSummaryMap_->setBinLabel(VBIN_EB_P, "EB+", 2);
  reportSummaryMap_->setBinLabel(VBIN_EB_M, "EB-", 2);
  reportSummaryMap_->setBinLabel(VBIN_EE_P, "EE+", 2);
  reportSummaryMap_->setBinLabel(VBIN_EE_M, "EE-", 2);
  reportSummaryMap_->setBinLabel(VBIN_ES_P, "ES+", 2);
  reportSummaryMap_->setBinLabel(VBIN_ES_M, "ES-", 2);
  reportSummaryMap_->setBinLabel(VBIN_HBHE_A, "HBHEa", 2);
  reportSummaryMap_->setBinLabel(VBIN_HBHE_B, "HBHEb", 2);
  reportSummaryMap_->setBinLabel(VBIN_HBHE_C, "HBHEc", 2);
  reportSummaryMap_->setBinLabel(VBIN_HF, "HF", 2);
  reportSummaryMap_->setBinLabel(VBIN_HO, "HO", 2);
  reportSummaryMap_->setBinLabel(VBIN_BPIX, "BPIX", 2);
  reportSummaryMap_->setBinLabel(VBIN_FPIX, "FPIX", 2);
  reportSummaryMap_->setBinLabel(VBIN_RPC, "RPC", 2);
  reportSummaryMap_->setBinLabel(VBIN_TIBTID, "TIBTID", 2);
  reportSummaryMap_->setBinLabel(VBIN_TOB, "TOB", 2);
  reportSummaryMap_->setBinLabel(VBIN_TEC_P, "TECp", 2);
  reportSummaryMap_->setBinLabel(VBIN_TE_M, "TECm", 2);
  reportSummaryMap_->setBinLabel(VBIN_CASTOR, "CASTOR", 2);
  reportSummaryMap_->setBinLabel(VBIN_ZDC, "ZDC", 2);
  reportSummaryMap_->setBinLabel(VBIN_PHYSICS_DECLARED, "PhysDecl", 2);
  reportSummaryMap_->setBinLabel(VBIN_MOMENTUM, "13 TeV", 2);
  reportSummaryMap_->setBinLabel(VBIN_STABLE_BEAM, "Stable B", 2);
  reportSummaryMap_->setBinLabel(VBIN_VALID, "Valid", 2);

  blankAllLumiSections();
}

void DQMProvInfo::bookHistogramsProvInfo(DQMStore::IBooker& iBooker) {
  // Note: Given that all these elements are only filled once per run, they
  //       are filled here right away. (except for isCollisionsRun)

  // Element: CMMSW
  versCMSSW_ = iBooker.bookString("CMSSW", edm::getReleaseVersion().c_str());

  // Element: Globaltag
  versGlobaltag_ = iBooker.bookString("Globaltag", globalTag_);

  // Element: RunType
  versRuntype_ = iBooker.bookString("Run Type", runType_);

  // Element: hltKey
  hHltKey_ = iBooker.bookString("hltKey", hltKey_);

  // Element: hostName
  hostName_ = iBooker.bookString("hostName", gSystem->HostName());

  // Element: isCollisionsRun (filled for real in EndLumi)
  hIsCollisionsRun_ = iBooker.bookInt("isCollisionsRun");
  hIsCollisionsRun_->Fill(0);

  // Element: processID
  processId_ = iBooker.bookInt("processID");
  processId_->Fill(gSystem->GetPid());

  // Element: workingDir
  workingDir_ = iBooker.bookString("workingDir", gSystem->pwd());
}

void DQMProvInfo::analyze(const edm::Event& event, const edm::EventSetup& c) {
  // This happens on an event by event base
  // We extract information from events, placing them in local variables
  // and then at the end of each lumisection, we fill them in the MonitorElement
  // (Except for the global tag, which we only extract from the first event we
  //  ever encounter and put in the MonitorElement right away)

  // We set the top value to "Valid" to 1 for each LS we encounter
  setupLumiSection(event.id().luminosityBlock());

  analyzeLhcInfo(event);
  analyzeEventInfo(event);
  analyzeProvInfo(event);
}

void DQMProvInfo::analyzeLhcInfo(const edm::Event& event) {
  unsigned int currentLSNumber = event.id().luminosityBlock();
  edm::Handle<TCDSRecord> tcdsData;
  event.getByToken(tcdsrecord_, tcdsData);
  // We unpack the TCDS record from TCDS
  if (tcdsData.isValid()) {
    //and we look at the BST information
    auto lhcFill = static_cast<int>(tcdsData->getBST().getLhcFill());
    beamMode_ = static_cast<int>(tcdsData->getBST().getBeamMode());
    auto momentum = static_cast<int>(tcdsData->getBST().getBeamMomentum());
    auto intensity1 = static_cast<int>(tcdsData->getBST().getIntensityBeam1());
    auto intensity2 = static_cast<int>(tcdsData->getBST().getIntensityBeam2());

    // Quite straightforward: Fill in the value for the LS in each plot:
    hLhcFill_->setBinContent(currentLSNumber, lhcFill);
    hBeamMode_->setBinContent(currentLSNumber, beamMode_);
    hMomentum_->setBinContent(currentLSNumber, momentum);
    hIntensity1_->setBinContent(currentLSNumber, intensity1);
    hIntensity2_->setBinContent(currentLSNumber, intensity2);

    // Part3: Using LHC status info, fill in VBIN_MOMENTUM and VBIN_STABLE_BEAM
    // Fill 13 TeV bit in y bin VBIN_MOMENTUM
    if (momentum >= MAX_MOMENTUM - MOMENTUM_OFFSET) {
      fillSummaryMapBin(currentLSNumber, VBIN_MOMENTUM, 1.);
    } else {
      fillSummaryMapBin(currentLSNumber, VBIN_MOMENTUM, 0.);
    }

    // Fill stable beams bit in y bin VBIN_STABLE_BEAM
    if (beamMode_ == 11) {
      hIsCollisionsRun_->Fill(1);
      reportSummary_->Fill(1.);
      fillSummaryMapBin(currentLSNumber, VBIN_STABLE_BEAM, 1.);
    } else {
      reportSummary_->Fill(0.);
      fillSummaryMapBin(currentLSNumber, VBIN_STABLE_BEAM, 0.);
    }
  } else {
    edm::LogWarning("DQMProvInfo") << "TCDS Data inaccessible.";
  }
}

void DQMProvInfo::analyzeEventInfo(const edm::Event& event) {
  unsigned int currentLSNumber = event.id().luminosityBlock();
  // Part 1:
  // If FED#735 is available use it to extract DcsStatusCollection.
  // If not, use softFED#1022 to extract DCSRecord.

  edm::Handle<DcsStatusCollection> dcsStatusCollection;
  event.getByToken(dcsStatusCollection_, dcsStatusCollection);
  edm::Handle<DCSRecord> dcsRecord;
  event.getByToken(dcsRecordToken_, dcsRecord);

  // Populate dcsBits array with received information.
  bool dcsBits[MAX_DCS_VBINS + 1] = {};

  if (dcsStatusCollection.isValid() && !dcsStatusCollection->empty()) {
    edm::LogInfo("DQMProvInfo") << "Using FED#735 for reading DCS bits" << std::endl;
    fillDcsBitsFromDcsStatusCollection(dcsStatusCollection, dcsBits);
  } else if (dcsRecord.isValid()) {
    edm::LogInfo("DQMProvInfo") << "Using softFED#1022 for reading DCS bits" << std::endl;
    fillDcsBitsFromDCSRecord(*dcsRecord, dcsBits);
  } else {
    edm::LogError("DQMProvInfo") << "No DCS information found!" << std::endl;
  }

  // Part 2: Compute the PhysicsDeclared bit from the event
  auto physicsDeclared = isPhysicsDeclared(dcsBits);

  // Some info-level logging
  edm::LogInfo("DQMProvInfo") << "Physics declared bit: " << physicsDeclared << std::endl;

  // Part 1: Physics declared bit in y bin VBIN_PHYSICS_DECLARED
  // This also is used as the global value of the summary.
  if (physicsDeclared) {
    fillSummaryMapBin(currentLSNumber, VBIN_PHYSICS_DECLARED, 1.);
  } else {
    fillSummaryMapBin(currentLSNumber, VBIN_PHYSICS_DECLARED, 0.);
  }

  // Part2: DCS bits in y bins 1 to MAX_DCS_VBINS
  for (int vbin = 1; vbin <= MAX_DCS_VBINS; vbin++) {
    if (dcsBits[vbin]) {
      fillSummaryMapBin(currentLSNumber, vbin, 1.);
    } else {
      fillSummaryMapBin(currentLSNumber, vbin, 0.);
    }
  }
}

void DQMProvInfo::analyzeProvInfo(const edm::Event& event) {
  // Only trying to retrieve the global tag for the first event we ever
  // encounter.
  if (!globalTagRetrieved_) {
    // Getting the real process name for the given event
    std::string processName = event.processHistory()[event.processHistory().size() - 1].processName();
    // Getting parameters for that process
    edm::ParameterSet ps;
    event.getProcessParameterSet(processName, ps);
    // Getting the global tag
    globalTag_ = ps.getParameterSet("PoolDBESSource@GlobalTag").getParameter<std::string>("globaltag");
    versGlobaltag_->Fill(globalTag_);
    // Finaly: Setting globalTagRetrieved_ to true, since we got it now
    globalTagRetrieved_ = true;
  }
}

void DQMProvInfo::fillDcsBitsFromDCSRecord(const DCSRecord& dcsRecord, bool* dcsBits) {
  dcsBits[VBIN_CSC_P] = dcsRecord.highVoltageReady(DCSRecord::Partition::CSCp);
  dcsBits[VBIN_CSC_M] = dcsRecord.highVoltageReady(DCSRecord::Partition::CSCm);
  dcsBits[VBIN_DT_0] = dcsRecord.highVoltageReady(DCSRecord::Partition::DT0);
  dcsBits[VBIN_DT_P] = dcsRecord.highVoltageReady(DCSRecord::Partition::DTp);
  dcsBits[VBIN_DT_M] = dcsRecord.highVoltageReady(DCSRecord::Partition::DTm);
  dcsBits[VBIN_EB_P] = dcsRecord.highVoltageReady(DCSRecord::Partition::EBp);
  dcsBits[VBIN_EB_M] = dcsRecord.highVoltageReady(DCSRecord::Partition::EBm);
  dcsBits[VBIN_EE_P] = dcsRecord.highVoltageReady(DCSRecord::Partition::EEp);
  dcsBits[VBIN_EE_M] = dcsRecord.highVoltageReady(DCSRecord::Partition::EEm);
  dcsBits[VBIN_ES_P] = dcsRecord.highVoltageReady(DCSRecord::Partition::ESp);
  dcsBits[VBIN_ES_M] = dcsRecord.highVoltageReady(DCSRecord::Partition::ESm);
  dcsBits[VBIN_HBHE_A] = dcsRecord.highVoltageReady(DCSRecord::Partition::HBHEa);
  dcsBits[VBIN_HBHE_B] = dcsRecord.highVoltageReady(DCSRecord::Partition::HBHEb);
  dcsBits[VBIN_HBHE_C] = dcsRecord.highVoltageReady(DCSRecord::Partition::HBHEc);
  dcsBits[VBIN_HF] = dcsRecord.highVoltageReady(DCSRecord::Partition::HF);
  dcsBits[VBIN_HO] = dcsRecord.highVoltageReady(DCSRecord::Partition::HO);
  dcsBits[VBIN_BPIX] = dcsRecord.highVoltageReady(DCSRecord::Partition::BPIX);
  dcsBits[VBIN_FPIX] = dcsRecord.highVoltageReady(DCSRecord::Partition::FPIX);
  dcsBits[VBIN_RPC] = dcsRecord.highVoltageReady(DCSRecord::Partition::RPC);
  dcsBits[VBIN_TIBTID] = dcsRecord.highVoltageReady(DCSRecord::Partition::TIBTID);
  dcsBits[VBIN_TOB] = dcsRecord.highVoltageReady(DCSRecord::Partition::TOB);
  dcsBits[VBIN_TEC_P] = dcsRecord.highVoltageReady(DCSRecord::Partition::TECp);
  dcsBits[VBIN_TE_M] = dcsRecord.highVoltageReady(DCSRecord::Partition::TECm);
  dcsBits[VBIN_CASTOR] = dcsRecord.highVoltageReady(DCSRecord::Partition::CASTOR);
  dcsBits[VBIN_ZDC] = dcsRecord.highVoltageReady(DCSRecord::Partition::ZDC);
}

void DQMProvInfo::fillDcsBitsFromDcsStatusCollection(const edm::Handle<DcsStatusCollection>& dcsStatusCollection,
                                                     bool* dcsBits) {
  // Loop over the DCSStatus entries in the DcsStatusCollection
  // (Typically there is only one)
  bool first = true;
  for (auto const& dcsStatusItr : *dcsStatusCollection) {
    // By default all the bits are false. We put all the bits on true only
    // for the first DCSStatus that we encounter:
    if (first) {
      for (int vbin = 1; vbin <= MAX_DCS_VBINS; vbin++) {
        dcsBits[vbin] = true;
      }
      first = false;
    }
    dcsBits[VBIN_CSC_P] &= dcsStatusItr.ready(DcsStatus::CSCp);
    dcsBits[VBIN_CSC_M] &= dcsStatusItr.ready(DcsStatus::CSCm);
    dcsBits[VBIN_DT_0] &= dcsStatusItr.ready(DcsStatus::DT0);
    dcsBits[VBIN_DT_P] &= dcsStatusItr.ready(DcsStatus::DTp);
    dcsBits[VBIN_DT_M] &= dcsStatusItr.ready(DcsStatus::DTm);
    dcsBits[VBIN_EB_P] &= dcsStatusItr.ready(DcsStatus::EBp);
    dcsBits[VBIN_EB_M] &= dcsStatusItr.ready(DcsStatus::EBm);
    dcsBits[VBIN_EE_P] &= dcsStatusItr.ready(DcsStatus::EEp);
    dcsBits[VBIN_EE_M] &= dcsStatusItr.ready(DcsStatus::EEm);
    dcsBits[VBIN_ES_P] &= dcsStatusItr.ready(DcsStatus::ESp);
    dcsBits[VBIN_ES_M] &= dcsStatusItr.ready(DcsStatus::ESm);
    dcsBits[VBIN_HBHE_A] &= dcsStatusItr.ready(DcsStatus::HBHEa);
    dcsBits[VBIN_HBHE_B] &= dcsStatusItr.ready(DcsStatus::HBHEb);
    dcsBits[VBIN_HBHE_C] &= dcsStatusItr.ready(DcsStatus::HBHEc);
    dcsBits[VBIN_HF] &= dcsStatusItr.ready(DcsStatus::HF);
    dcsBits[VBIN_HO] &= dcsStatusItr.ready(DcsStatus::HO);
    dcsBits[VBIN_BPIX] &= dcsStatusItr.ready(DcsStatus::BPIX);
    dcsBits[VBIN_FPIX] &= dcsStatusItr.ready(DcsStatus::FPIX);
    dcsBits[VBIN_RPC] &= dcsStatusItr.ready(DcsStatus::RPC);
    dcsBits[VBIN_TIBTID] &= dcsStatusItr.ready(DcsStatus::TIBTID);
    dcsBits[VBIN_TOB] &= dcsStatusItr.ready(DcsStatus::TOB);
    dcsBits[VBIN_TEC_P] &= dcsStatusItr.ready(DcsStatus::TECp);
    dcsBits[VBIN_TE_M] &= dcsStatusItr.ready(DcsStatus::TECm);
    dcsBits[VBIN_CASTOR] &= dcsStatusItr.ready(DcsStatus::CASTOR);
    dcsBits[VBIN_ZDC] &= dcsStatusItr.ready(DcsStatus::ZDC);

    // Some info-level logging
    edm::LogInfo("DQMProvInfo") << "DCS status: 0x" << std::hex << dcsStatusItr.ready() << std::dec << std::endl;
  }
}

bool DQMProvInfo::isPhysicsDeclared(bool* dcsBits) {
  // Compute the PhysicsDeclared bit from the event
  // The bit is set to to true if:
  // - the LHC is in stable beams
  // - all the pixel and strips partitions have DCSStatus ON
  // - at least one muon partition has DCSStatus ON
  // Basically: we do an AND of the physicsDeclared of ALL events.
  // As soon as one value is not "1", physicsDeclared_ becomes false.
  return (beamMode_ == 11) &&
         (dcsBits[VBIN_BPIX] && dcsBits[VBIN_FPIX] && dcsBits[VBIN_TIBTID] && dcsBits[VBIN_TOB] &&
          dcsBits[VBIN_TEC_P] && dcsBits[VBIN_TE_M]) &&
         (dcsBits[VBIN_CSC_P] || dcsBits[VBIN_CSC_M] || dcsBits[VBIN_DT_0] || dcsBits[VBIN_DT_P] ||
          dcsBits[VBIN_DT_M] || dcsBits[VBIN_RPC]);
}

void DQMProvInfo::blankAllLumiSections() {
  // Initially we want all lumisection to be blank (-1) and
  // white instead of red which is misleading.
  for (int ls = 0; ls < MAX_LUMIS; ls++) {
    // Color all the bins white (-1)
    for (int vBin = 1; vBin <= MAX_VBINS; vBin++) {
      reportSummaryMap_->setBinContent(ls, vBin, -1.);
    }
  }
}

void DQMProvInfo::fillSummaryMapBin(int ls, int bin, double value) {
  // All lumis are initialized as -1 (white).
  // We'll set them to red (0) whenever we see a 0 -- else, the value should be
  // green (1).
  // This need to be atomic, DQMOneEDAnalyzer for this reason.
  double current = reportSummaryMap_->getBinContent(ls, bin);
  if (current == -1) {
    reportSummaryMap_->setBinContent(ls, bin, value);
  } else if (value < current) {
    reportSummaryMap_->setBinContent(ls, bin, value);
  }  // else: ignore, keep min value.
}

void DQMProvInfo::setupLumiSection(int currentLSNumber) {
  if (reportSummaryMap_->getBinContent(currentLSNumber, VBIN_VALID) < 1.) {
    reportSummaryMap_->setBinContent(currentLSNumber, VBIN_VALID, 1.);

    // Mark all lower LS as invalid, if they are not set valid yet.
    // This is a hint for the render plugin to show the correct range.
    for (int ls = 1; ls < currentLSNumber; ls++) {
      if (reportSummaryMap_->getBinContent(ls, VBIN_VALID) == -1.) {
        reportSummaryMap_->setBinContent(ls, VBIN_VALID, 0.);
      }
    }
  }
}
