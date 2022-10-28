/** \class TKStatus
 * *
 *  \author  Geng-yuan Jeng/UC Riverside
 *           Francisco Yumiceva/FNAL
 *
 */

// C++
#include <array>
#include <fstream>
#include <string>

// CMS
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class declaration
//
class TKStatus : public edm::one::EDAnalyzer<> {
public:
  TKStatus(const edm::ParameterSet&);

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  void dumpTkDcsStatus(std::string const&, edm::RunNumber_t, std::array<bool, 6> const&);

  std::string dcsTkFileName_;
  const edm::EDGetTokenT<DcsStatusCollection> dcsStatusToken_;
  const edm::EDGetTokenT<DCSRecord> dcsRecordToken_;

  int lastlumi_ = -1;
  // ----------member data ---------------------------
};

// ----------------------------------------------------------
TKStatus::TKStatus(const edm::ParameterSet& ps)
    : dcsStatusToken_(consumes<DcsStatusCollection>(edm::InputTag("scalersRawToDigi"))),
      dcsRecordToken_(consumes<DCSRecord>(edm::InputTag("onlineMetaDataDigis"))) {
  dcsTkFileName_ = ps.getParameter<edm::ParameterSet>("BeamFitter").getUntrackedParameter<std::string>("DIPFileName");
  {
    std::string tmpname = dcsTkFileName_;
    tmpname.insert(dcsTkFileName_.length() - 4, "_TkStatus");
    dcsTkFileName_ = std::move(tmpname);
  }
}

// ----------------------------------------------------------
void TKStatus::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  int nthlumi = iEvent.luminosityBlock();
  if (nthlumi > lastlumi_) {  // check every LS
    lastlumi_ = nthlumi;

    // Checking TK status
    const auto& dcsStatus = iEvent.getHandle(dcsStatusToken_);
    const auto& dcsRecord = iEvent.getHandle(dcsRecordToken_);

    std::array<bool, 6> dcsTk;
    for (auto& e : dcsTk) {
      e = true;
    }

    // Check that the DCS information is available in some form
    if (!dcsStatus.isValid() && !dcsRecord.isValid()) {
      edm::LogWarning("TkStatus") << "DcsStatusCollection product with InputTag \"scalersRawToDigi\" not in event \n"
                                  << "DCSRecord product with InputTag \"onlineMetaDataDigis\" not in event \n";
      dumpTkDcsStatus(dcsTkFileName_, iEvent.run(), dcsTk);
      return;
    }

    if (dcsStatus.isValid() && (*dcsStatus).empty()) {
      if (iEvent.eventAuxiliary().isRealData()) {
        // This is the Data case for >= Run3, DCSStatus is available (unpacked), but empty
        // because SCAL is not in data-taking. In this case we fall back to s/w FED 1022
        if (dcsRecord.isValid()) {
          edm::LogPrint("TkStatus") << "Using dcsRecord because dcsStatus is empty";
          dcsTk[0] = (*dcsRecord).highVoltageReady(DCSRecord::Partition::BPIX);
          dcsTk[1] = (*dcsRecord).highVoltageReady(DCSRecord::Partition::FPIX);
          dcsTk[2] = (*dcsRecord).highVoltageReady(DCSRecord::Partition::TIBTID);
          dcsTk[3] = (*dcsRecord).highVoltageReady(DCSRecord::Partition::TOB);
          dcsTk[4] = (*dcsRecord).highVoltageReady(DCSRecord::Partition::TECp);
          dcsTk[5] = (*dcsRecord).highVoltageReady(DCSRecord::Partition::TECm);
        } else {
          // DCS status is empty, and the DCS record is not available
          edm::LogWarning("TkStatus") << "DCSRecord product with InputTag \"onlineMetaDataDigis\" empty \n";
          dumpTkDcsStatus(dcsTkFileName_, iEvent.run(), dcsTk);
          return;
        }
      } else {
        // DCS status is empty, but it's not real data
        edm::LogPrint("TkStatus") << "Running on MC";
        dumpTkDcsStatus(dcsTkFileName_, iEvent.run(), dcsTk);
        return;
      }
    } else {
      // This is the case for <= Run2
      edm::LogPrint("TkStatus") << "Using dcsStatus (Run 1 and Run 2 SCAL)";
      for (auto const& status : *dcsStatus) {
        if (!status.ready(DcsStatus::BPIX))
          dcsTk[0] = false;
        if (!status.ready(DcsStatus::FPIX))
          dcsTk[1] = false;
        if (!status.ready(DcsStatus::TIBTID))
          dcsTk[2] = false;
        if (!status.ready(DcsStatus::TOB))
          dcsTk[3] = false;
        if (!status.ready(DcsStatus::TECp))
          dcsTk[4] = false;
        if (!status.ready(DcsStatus::TECm))
          dcsTk[5] = false;
      }
    }

    dumpTkDcsStatus(dcsTkFileName_, iEvent.run(), dcsTk);
  }
}

//--------------------------------------------------------
void TKStatus::dumpTkDcsStatus(std::string const& fileName, edm::RunNumber_t runnum, std::array<bool, 6> const& dcsTk) {
  std::ofstream outFile;

  outFile.open(fileName.c_str());
  outFile << "BPIX " << (dcsTk[0] ? "On" : "Off") << std::endl;
  outFile << "FPIX " << (dcsTk[1] ? "On" : "Off") << std::endl;
  outFile << "TIBTID " << (dcsTk[2] ? "On" : "Off") << std::endl;
  outFile << "TOB " << (dcsTk[3] ? "On" : "Off") << std::endl;
  outFile << "TECp " << (dcsTk[4] ? "On" : "Off") << std::endl;
  outFile << "TECm " << (dcsTk[5] ? "On" : "Off") << std::endl;
  bool AllTkOn = true;
  for (auto status : dcsTk) {
    if (!status) {
      AllTkOn = false;
      break;
    }
  }
  outFile << "WholeTrackerOn " << (AllTkOn ? "Yes" : "No") << std::endl;
  outFile << "Runnumber " << runnum << std::endl;

  outFile.close();
}

DEFINE_FWK_MODULE(TKStatus);
