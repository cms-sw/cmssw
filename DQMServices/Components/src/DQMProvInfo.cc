/*
 * Original author: A. Raval / A. Meyer - DESY
 * Rewritten by:    B. van Besien - CERN
 */

#include "DQMProvInfo.h"
#include <TSystem.h>
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
////#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// In this module, the part to get the LHC beam info is currently broken.
// Previously this info came from FED812, but since the new TCDS this info is
// in FED1024. The general module to decode the info, does however not exist
// yet.
// We leave the old code here, because we expect a solution to come at some
// point and then we can update.
// Also note that at the moment we will still create the monitor elements and
// book the plots containing the LHC info, but we will simply not fill them.
// The 2 lines in the EventInfo (HV) plot which show beam status info will be
// hidden.

const int DQMProvInfo::MAX_VBINS;
const int DQMProvInfo::MAX_LUMIS;

// Constructor
DQMProvInfo::DQMProvInfo(const edm::ParameterSet& ps) {
  // Initialization of DQM parameters
  subsystemname_ =
      ps.getUntrackedParameter<std::string>("subSystemFolder", "Info");
  provinfofolder_ =
      ps.getUntrackedParameter<std::string>("provInfoFolder", "ProvInfo");
  runType_ =
      ps.getUntrackedParameter<std::string>("runType", "No run type selected");

  // Initialization of the input
  // Used to get the LHC beam status info, pre 2014-09:
  ////L1gtEvm_ = consumes<L1GlobalTriggerEvmReadoutRecord>(
  ////    ps.getUntrackedParameter<std::string>("L1gtEvm", "gtEvmDigis"));
  // Used to get the DCS bits:
  dcsStatusCollection_ =
      consumes<DcsStatusCollection>(ps.getUntrackedParameter<std::string>(
          "dcsStatusCollection", "scalersRawToDigi"));
  // Used to get the physics declared bit:
  L1gt_ = consumes<L1GlobalTriggerReadoutRecord>(
      ps.getUntrackedParameter<std::string>("L1gt", "gtDigis"));

  // Initialization of the global tag
  globalTag_ = "MODULE::DEFAULT";  // default
  globalTagRetrieved_ = false;     // set as soon as retrieved from first event

  // Initialization of run scope variables
  previousLSNumber_ = 0;  // Previous LS compared to current, initializing at 0
}

// Destructor
DQMProvInfo::~DQMProvInfo() {}

void DQMProvInfo::dqmBeginRun(const edm::Run& iRun,
                              const edm::EventSetup& iEventSetup) {
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
    edm::LogInfo("DQMProvInfo") << "HLT key (run): " << hltConfig.tableName()
                                << std::endl;
    hltKey_ = hltConfig.tableName();
  }
}

void DQMProvInfo::bookHistograms(DQMStore::IBooker& iBooker,
                                 edm::Run const& iRun,
                                 edm::EventSetup const& iEventSetup) {
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
  hBeamMode_ =
      iBooker.book1D("beamMode", "beamMode", MAX_LUMIS, 1., MAX_LUMIS + 1);
  hBeamMode_->getTH1F()->GetYaxis()->Set(21, 0.5, 21.5);
  hBeamMode_->getTH1F()->SetMaximum(21.5);
  hBeamMode_->getTH1F()->SetCanExtend(TH1::kAllAxes);
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
  hIntensity1_ = iBooker.book1D("intensity1", "Intensity Beam 1", MAX_LUMIS, 1.,
                                MAX_LUMIS + 1);
  hIntensity1_->setAxisTitle("Luminosity Section", 1);
  hIntensity1_->setAxisTitle("N [E10]", 2);
  hIntensity1_->getTH1F()->SetCanExtend(TH1::kAllAxes);

  // Element: intensity2
  hIntensity2_ = iBooker.book1D("intensity2", "Intensity Beam 2", MAX_LUMIS, 1.,
                                MAX_LUMIS + 1);
  hIntensity2_->setAxisTitle("Luminosity Section", 1);
  hIntensity2_->setAxisTitle("N [E10]", 2);
  hIntensity2_->getTH1F()->SetCanExtend(TH1::kAllAxes);

  // Element: lhcFill
  hLhcFill_ = iBooker.book1D("lhcFill", "LHC Fill Number", MAX_LUMIS, 1.,
                             MAX_LUMIS + 1);
  hLhcFill_->setAxisTitle("Luminosity Section", 1);
  hLhcFill_->getTH1F()->SetCanExtend(TH1::kAllAxes);

  // Element: momentum
  hMomentum_ = iBooker.book1D("momentum", "Beam Energy [GeV]", MAX_LUMIS, 1.,
                              MAX_LUMIS + 1);
  hMomentum_->setAxisTitle("Luminosity Section", 1);
  hMomentum_->getTH1F()->SetCanExtend(TH1::kAllAxes);
}

void DQMProvInfo::bookHistogramsEventInfo(DQMStore::IBooker& iBooker) {
  // Element: reportSummary
  reportSummary_ = iBooker.bookFloat("reportSummary");

  // Element: reportSummaryMap   (this is the famous HV plot)
  reportSummaryMap_ = iBooker.book2D(
      ////"reportSummaryMap", "DCS HV Status and Beam Status per Lumisection",
      "reportSummaryMap", "DCS High Voltage Status per Lumisection",
      MAX_LUMIS, 0, MAX_LUMIS, MAX_VBINS, 0., MAX_VBINS);
  reportSummaryMap_->setAxisTitle("Luminosity Section");
  reportSummaryMap_->getTH2F()->SetCanExtend(TH1::kAllAxes);

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
  // reportSummaryMap_->setBinLabel(VBIN_MOMENTUM, "13 TeV", 2);
  // reportSummaryMap_->setBinLabel(VBIN_STABLE_BEAM, "Stable B", 2);
  reportSummaryMap_->setBinLabel(VBIN_VALID, "Valid", 2);
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

void DQMProvInfo::beginLuminosityBlock(const edm::LuminosityBlock& l,
                                       const edm::EventSetup& c) {
  // By default we set the Physics Declared bit to false at the beginning of
  // every LS
  physicsDeclared_ = false;
  // Boolean that tells the analyse method that we encountered the first real
  // dcs info
  foundFirstPhysicsDeclared_ = false;
  // By default we set all the DCS bits to false at the beginning of every LS
  for (int vbin = 1; vbin <= MAX_DCS_VBINS; vbin++) {
    dcsBits_[vbin] = false;
  }
  // Boolean that tells the analyse method that we encountered the first real
  // dcs info
  foundFirstDcsBits_ = false;
}

void DQMProvInfo::analyze(const edm::Event& event, const edm::EventSetup& c) {
  // This happens on an event by event base
  // We extract information from events, placing them in local variables
  // and then at the end of each lumisection, we fill them in the MonitorElement
  // (Except for the global tag, which we only extract from the first event we
  //  ever encounter and put in the MonitorElement right away)
  analyzeLhcInfo(event);
  analyzeEventInfo(event);
  analyzeProvInfo(event);
}

void DQMProvInfo::analyzeLhcInfo(const edm::Event& event) {
  // Currently broken, see comments at the top of this file.
  ////edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvm_handle;
  ////event.getByToken(L1gtEvm_, gtEvm_handle);
  ////L1GlobalTriggerEvmReadoutRecord const* gtevm = gtEvm_handle.product();
  ////L1GtfeWord gtfeEvmWord;
  ////L1GtfeExtWord gtfeEvmExtWord;
  ////if (gtevm) {
  ////  gtfeEvmWord = gtevm->gtfeWord();
  ////  gtfeEvmExtWord = gtevm->gtfeWord();
  ////} else {
  ////  edm::LogWarning("DQMProvInfo") << " gtfeEvmWord inaccessible";
  ////}
  ////lhcFill_ = gtfeEvmExtWord.lhcFillNumber();
  ////beamMode_ = gtfeEvmExtWord.beamMode();
  ////momentum_ = gtfeEvmExtWord.beamMomentum();
  ////intensity1_ = gtfeEvmExtWord.totalIntensityBeam1();
  ////intensity2_ = gtfeEvmExtWord.totalIntensityBeam2();
}

void DQMProvInfo::analyzeEventInfo(const edm::Event& event) {
  // Part 1:
  // Extract the PhysicsDeclared bit from the event
  edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
  event.getByToken(L1gt_, gtrr_handle);
  L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();
  if (gtrr) {
    // By default Physics Declared is false. We put it on true only for the
    // first trigger record that we encounter:
    if (!foundFirstPhysicsDeclared_) {
      physicsDeclared_ = true;
      foundFirstPhysicsDeclared_ = true;
    }
    L1GtFdlWord fdlWord = gtrr->gtFdlWord();
    // Basically: we do an AND of the physicsDeclared of ALL events.
    // As soon as one value is not "1", physicsDeclared_ becomes false.
    physicsDeclared_ &= (fdlWord.physicsDeclared() == 1);
    // cout << "phys decl. bit =" << static_cast<int>(fdlWord.physicsDeclared())
    // << endl;
  } else {
    // If for some reason the record is not accessible, we also revert to false.
    edm::LogWarning("DQMProvInfo") << "Physics declared bit not accessible!";
    physicsDeclared_ = false;
  }

  // Part 2:
  // Extract the DcsStatusCollection from the event
  // and put it into the dcsBits_ array
  edm::Handle<DcsStatusCollection> dcsStatus;
  event.getByToken(dcsStatusCollection_, dcsStatus);
  // Loop over the DCSStatus entries in the DcsStatusCollection
  // (Typically there is only one)
  for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin();
       dcsStatusItr != dcsStatus->end(); ++dcsStatusItr) {
    // By default all the bits are false. We put all the bits on true only
    // for the first DCSStatus that we encounter:
    if (!foundFirstDcsBits_) {
      for (int vbin = 1; vbin <= MAX_DCS_VBINS; vbin++) {
        dcsBits_[vbin] = true;
      }
      foundFirstDcsBits_ = true;
    }
    // The DCS on lumi level is considered ON if the bit is set in EVERY event
    dcsBits_[VBIN_CSC_P] &= dcsStatusItr->ready(DcsStatus::CSCp);
    dcsBits_[VBIN_CSC_M] &= dcsStatusItr->ready(DcsStatus::CSCm);
    dcsBits_[VBIN_DT_0] &= dcsStatusItr->ready(DcsStatus::DT0);
    dcsBits_[VBIN_DT_P] &= dcsStatusItr->ready(DcsStatus::DTp);
    dcsBits_[VBIN_DT_M] &= dcsStatusItr->ready(DcsStatus::DTm);
    dcsBits_[VBIN_EB_P] &= dcsStatusItr->ready(DcsStatus::EBp);
    dcsBits_[VBIN_EB_M] &= dcsStatusItr->ready(DcsStatus::EBm);
    dcsBits_[VBIN_EE_P] &= dcsStatusItr->ready(DcsStatus::EEp);
    dcsBits_[VBIN_EE_M] &= dcsStatusItr->ready(DcsStatus::EEm);
    dcsBits_[VBIN_ES_P] &= dcsStatusItr->ready(DcsStatus::ESp);
    dcsBits_[VBIN_ES_M] &= dcsStatusItr->ready(DcsStatus::ESm);
    dcsBits_[VBIN_HBHE_A] &= dcsStatusItr->ready(DcsStatus::HBHEa);
    dcsBits_[VBIN_HBHE_B] &= dcsStatusItr->ready(DcsStatus::HBHEb);
    dcsBits_[VBIN_HBHE_C] &= dcsStatusItr->ready(DcsStatus::HBHEc);
    dcsBits_[VBIN_HF] &= dcsStatusItr->ready(DcsStatus::HF);
    dcsBits_[VBIN_HO] &= dcsStatusItr->ready(DcsStatus::HO);
    dcsBits_[VBIN_BPIX] &= dcsStatusItr->ready(DcsStatus::BPIX);
    dcsBits_[VBIN_FPIX] &= dcsStatusItr->ready(DcsStatus::FPIX);
    dcsBits_[VBIN_RPC] &= dcsStatusItr->ready(DcsStatus::RPC);
    dcsBits_[VBIN_TIBTID] &= dcsStatusItr->ready(DcsStatus::TIBTID);
    dcsBits_[VBIN_TOB] &= dcsStatusItr->ready(DcsStatus::TOB);
    dcsBits_[VBIN_TEC_P] &= dcsStatusItr->ready(DcsStatus::TECp);
    dcsBits_[VBIN_TE_M] &= dcsStatusItr->ready(DcsStatus::TECm);
    dcsBits_[VBIN_CASTOR] &= dcsStatusItr->ready(DcsStatus::CASTOR);
    dcsBits_[VBIN_ZDC] &= dcsStatusItr->ready(DcsStatus::ZDC);
    // Some info-level logging
    edm::LogInfo("DQMProvInfo") << "DCS status: 0x" << std::hex
                                << dcsStatusItr->ready() << std::dec
                                << std::endl;
  }
}

void DQMProvInfo::analyzeProvInfo(const edm::Event& event) {
  // Only trying to retrieve the global tag for the first event we ever
  // encounter.
  if (!globalTagRetrieved_) {
    // Getting the real process name for the given event
    std::string processName =
        event.processHistory()[event.processHistory().size() - 1].processName();
    // Getting parameters for that process
    edm::ParameterSet ps;
    event.getProcessParameterSet(processName, ps);
    // Getting the global tag
    globalTag_ = ps.getParameterSet("PoolDBESSource@GlobalTag")
                     .getParameter<std::string>("globaltag");
    versGlobaltag_->Fill(globalTag_);
    // Finaly: Setting globalTagRetrieved_ to true, since we got it now
    globalTagRetrieved_ = true;
  }
}

void DQMProvInfo::endLuminosityBlock(const edm::LuminosityBlock& iLumi,
                                     const edm::EventSetup& c) {
  int currentLSNumber = iLumi.id().luminosityBlock();

  // We assume that we encounter the LumiSections in chronological order
  // We only process a LS if it's greater than the previous one:
  if (currentLSNumber > previousLSNumber_) {
    endLuminosityBlockLhcInfo(currentLSNumber);
    endLuminosityBlockEventInfo(currentLSNumber);
  }

  // Set current LS number as previous number for the next cycle:
  previousLSNumber_ = currentLSNumber;
}

void DQMProvInfo::endLuminosityBlockLhcInfo(const int currentLSNumber) {
  // Quite straightforward: Fill in the value for the LS in each plot:
  hBeamMode_->setBinContent(currentLSNumber, beamMode_);
  hIntensity1_->setBinContent(currentLSNumber, intensity1_);
  hIntensity2_->setBinContent(currentLSNumber, intensity2_);
  hLhcFill_->setBinContent(currentLSNumber, lhcFill_);
  hMomentum_->setBinContent(currentLSNumber, momentum_);
}

void DQMProvInfo::endLuminosityBlockEventInfo(const int currentLSNumber) {
  // If we skipped LumiSections, we make them "white"
  blankPreviousLumiSections(currentLSNumber);

  // We set the top value to "Valid" to 1 for each LS we end
  reportSummaryMap_->setBinContent(currentLSNumber, VBIN_VALID, 1.);

  // Part 1: Physics declared bit in y bin VBIN_PHYSICS_DECLARED
  // This also is used as the global value of the summary.
  if (physicsDeclared_) {
    reportSummary_->Fill(1.);
    reportSummaryMap_->setBinContent(currentLSNumber, VBIN_PHYSICS_DECLARED,
                                     1.);
  } else {
    reportSummary_->Fill(0.);
    reportSummaryMap_->setBinContent(currentLSNumber, VBIN_PHYSICS_DECLARED,
                                     0.);
  }

  // Part2: DCS bits in y bins 1 to MAX_DCS_VBINS
  for (int vbin = 1; vbin <= MAX_DCS_VBINS; vbin++) {
    if (dcsBits_[vbin]) {
      reportSummaryMap_->setBinContent(currentLSNumber, vbin, 1.);
    } else {
      reportSummaryMap_->setBinContent(currentLSNumber, vbin, 0.);
    }
  }

  // Part3: Using LHC status info, fill in VBIN_MOMENTUM and VBIN_STABLE_BEAM

  // Fill ? TeV bit in y bin VBIN_MOMENTUM
  // When we get this working again, we have to understand which momentum we
  // actually receive.
  ////if (momentum_ == 3500 || momentum_ == 4000) {
  ////  reportSummary_->Fill(1.);
  ////  reportSummaryMap_->setBinContent(currentLSNumber, VBIN_MOMENTUM, 1.);
  ////} else {
  ////  reportSummary_->Fill(0.);
  ////  reportSummaryMap_->setBinContent(currentLSNumber, VBIN_MOMENTUM, 0.);
  ////}

  // Fill stable beams bit in y bin VBIN_STABLE_BEAM
  ////if (beamMode_ == 11) {
  ////  hIsCollisionsRun_->Fill(1);
  ////  reportSummary_->Fill(1.);
  ////  reportSummaryMap_->setBinContent(currentLSNumber, VBIN_STABLE_BEAM, 1.);
  ////} else {
  ////  reportSummary_->Fill(0.);
  ////  reportSummaryMap_->setBinContent(currentLSNumber, VBIN_STABLE_BEAM, 0.);
  ////}
}

void DQMProvInfo::blankPreviousLumiSections(const int currentLSNumber) {
  // In case we skipped lumisections, the current lumisection number will be
  // more than the previous lumisection number + 1.
  // We paint all the skipped lumisections completely white (-1), except for
  // the top flag (the "valid" flag), which we paint red (0).
  for (int ls = previousLSNumber_ + 1; ls < currentLSNumber; ls++) {
    // Color the "Valid" bin red (0)
    reportSummaryMap_->setBinContent(ls, VBIN_VALID, 0.);
    // Color all the other bins white (-1)
    for (int vBin = 1; vBin < VBIN_VALID; vBin++) {
      reportSummaryMap_->setBinContent(ls, vBin, -1.);
    }
  }
}
