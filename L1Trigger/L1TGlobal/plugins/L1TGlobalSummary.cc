// L1TGlobalSummary:  Use L1TGlobalUtils to print summary of L1TGlobal output
//
// author: Brian Winer Ohio State
//

#include <fstream>
#include <iomanip>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace l1t;

// class declaration
class L1TGlobalSummary : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit L1TGlobalSummary(const edm::ParameterSet&);
  ~L1TGlobalSummary() override{};
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginRun(Run const&, EventSetup const&) override;
  void endRun(Run const&, EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  InputTag algInputTag_;
  InputTag extInputTag_;
  EDGetToken algToken_;
  EDGetToken extToken_;
  bool dumpRecord_;
  bool dumpTriggerResults_;
  bool dumpTriggerSummary_;
  bool readPrescalesFromFile_;
  int minBx_;
  int maxBx_;
  L1TGlobalUtil* gtUtil_;

  std::vector<int> decisionCount_;
  std::vector<int> intermCount_;
  std::vector<int> finalCount_;
  int finalOrCount;
};

L1TGlobalSummary::L1TGlobalSummary(const edm::ParameterSet& iConfig) {
  algInputTag_ = iConfig.getParameter<InputTag>("AlgInputTag");
  extInputTag_ = iConfig.getParameter<InputTag>("ExtInputTag");
  algToken_ = consumes<BXVector<GlobalAlgBlk>>(algInputTag_);
  extToken_ = consumes<BXVector<GlobalExtBlk>>(extInputTag_);
  dumpRecord_ = iConfig.getParameter<bool>("DumpRecord");
  dumpTriggerResults_ = iConfig.getParameter<bool>("DumpTrigResults");
  dumpTriggerSummary_ = iConfig.getParameter<bool>("DumpTrigSummary");
  readPrescalesFromFile_ = iConfig.getParameter<bool>("ReadPrescalesFromFile");
  minBx_ = iConfig.getParameter<int>("MinBx");
  maxBx_ = iConfig.getParameter<int>("MaxBx");
  l1t::UseEventSetupIn useEventSetupIn = l1t::UseEventSetupIn::Run;
  if (dumpTriggerResults_ || dumpTriggerSummary_) {
    useEventSetupIn = l1t::UseEventSetupIn::RunAndEvent;
  }
  gtUtil_ = new L1TGlobalUtil(iConfig, consumesCollector(), *this, algInputTag_, extInputTag_, useEventSetupIn);
  finalOrCount = 0;

  if (readPrescalesFromFile_) {
    std::string preScaleFileName = iConfig.getParameter<std::string>("psFileName");
    unsigned int preScColumn = iConfig.getParameter<int>("psColumn");
    gtUtil_->OverridePrescalesAndMasks(preScaleFileName, preScColumn);
  }
}

void L1TGlobalSummary::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // These parameters are part of the L1T/HLT interface, avoid changing if possible::
  desc.add<edm::InputTag>("AlgInputTag", edm::InputTag(""))
      ->setComment("InputTag for uGT Algorithm Block (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("ExtInputTag", edm::InputTag(""))
      ->setComment("InputTag for uGT External Block (required parameter:  default value is invalid)");
  // These parameters have well defined  default values and are not currently
  // part of the L1T/HLT interface.  They can be cleaned up or updated at will:
  desc.add<int>("MinBx", 0);
  desc.add<int>("MaxBx", 0);
  desc.add<bool>("DumpTrigResults", false);
  desc.add<bool>("DumpRecord", false);
  desc.add<bool>("DumpTrigSummary", true);
  desc.add<bool>("ReadPrescalesFromFile", false);
  desc.add<std::string>("psFileName", "prescale_L1TGlobal.csv")
      ->setComment("File should be located in directory: L1Trigger/L1TGlobal/data/Luminosity/startup");
  desc.add<int>("psColumn", 0);
  descriptions.add("L1TGlobalSummary", desc);
}

void L1TGlobalSummary::beginRun(Run const&, EventSetup const& evSetup) {
  decisionCount_.clear();
  intermCount_.clear();
  finalCount_.clear();

  finalOrCount = 0;
  gtUtil_->retrieveL1Setup(evSetup);

  int size = gtUtil_->decisionsInitial().size();
  decisionCount_.resize(size);
  intermCount_.resize(size);
  finalCount_.resize(size);
  std::fill(decisionCount_.begin(), decisionCount_.end(), 0);
  std::fill(intermCount_.begin(), intermCount_.end(), 0);
  std::fill(finalCount_.begin(), finalCount_.end(), 0);
}

void L1TGlobalSummary::endRun(Run const&, EventSetup const&) {
  if (dumpTriggerSummary_) {
    LogVerbatim out("L1TGlobalSummary");
    if (gtUtil_->valid()) {
      out << "==================  L1 Trigger Report  "
             "=====================================================================\n";
      out << '\n';
      out << " L1T menu Name   : " << gtUtil_->gtTriggerMenuName() << '\n';
      out << " L1T menu Version: " << gtUtil_->gtTriggerMenuVersion() << '\n';
      out << " L1T menu Comment: " << gtUtil_->gtTriggerMenuComment() << '\n';
      out << '\n';
      out << "    Bit                  Algorithm Name                  Init    PScd  Final   PS Factor     Num Bx "
             "Masked\n";
      out << "========================================================================================================="
             "===\n";
      auto const& prescales = gtUtil_->prescales();
      auto const& masks = gtUtil_->masks();
      for (unsigned int i = 0; i < prescales.size(); i++) {
        // get the prescale and mask (needs some error checking here)
        int resultInit = decisionCount_[i];
        int resultPre = intermCount_[i];
        int resultFin = finalCount_[i];

        auto const& name = prescales.at(i).first;
        if (name != "NULL") {
          int prescale = prescales.at(i).second;
          auto const& mask = masks.at(i).second;
          out << std::dec << setfill(' ') << "   " << setw(5) << i << "   " << setw(40) << name << "   " << setw(7)
              << resultInit << setw(7) << resultPre << setw(7) << resultFin << setw(10) << prescale << setw(11)
              << mask.size() << '\n';
        }
      }
      out << "                                                      Final OR Count = " << finalOrCount << '\n';
      out << "========================================================================================================="
             "===\n";
    } else {
      out << "==================  No Level-1 Trigger menu  "
             "===============================================================\n";
    }
  }
}

// loop over events
void L1TGlobalSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  Handle<BXVector<GlobalAlgBlk>> alg;
  iEvent.getByToken(algToken_, alg);

  Handle<BXVector<GlobalExtBlk>> ext;
  iEvent.getByToken(extToken_, ext);

  LogDebug("l1t|Global") << "retrieved L1 GT data blocks" << endl;

  if (dumpTriggerResults_ || dumpTriggerSummary_) {
    //Fill the L1 result maps
    gtUtil_->retrieveL1(iEvent, evSetup, algToken_);

    LogDebug("l1t|Global") << "retrieved L1 data from GT Util" << endl;

    // grab the map for the final decisions
    const std::vector<std::pair<std::string, bool>> initialDecisions = gtUtil_->decisionsInitial();
    const std::vector<std::pair<std::string, bool>> intermDecisions = gtUtil_->decisionsInterm();
    const std::vector<std::pair<std::string, bool>> finalDecisions = gtUtil_->decisionsFinal();
    const std::vector<std::pair<std::string, int>> prescales = gtUtil_->prescales();
    const std::vector<std::pair<std::string, std::vector<int>>> masks = gtUtil_->masks();

    if ((decisionCount_.size() != gtUtil_->decisionsInitial().size()) ||
        (intermCount_.size() != gtUtil_->decisionsInterm().size()) ||
        (finalCount_.size() != gtUtil_->decisionsFinal().size())) {
      LogError("l1t|Global") << "gtUtil sizes inconsistent across run." << endl;
      return;
    }

    if (dumpTriggerResults_) {
      cout << "\n===================================== Trigger Results for BX=0 "
              "=============================================\n"
           << endl;
      cout << "    Bit                  Algorithm Name                  Init    aBXM  Final   PS Factor     Num Bx "
              "Masked"
           << endl;
      cout << "========================================================================================================"
              "===="
           << endl;
    }
    for (unsigned int i = 0; i < initialDecisions.size(); i++) {
      // get the name and trigger result
      std::string name = (initialDecisions.at(i)).first;
      if (name == "NULL")
        continue;

      bool resultInit = (initialDecisions.at(i)).second;

      // get prescaled and final results (need some error checking here)
      bool resultInterm = (intermDecisions.at(i)).second;
      bool resultFin = (finalDecisions.at(i)).second;

      // get the prescale and mask (needs some error checking here)
      int prescale = (prescales.at(i)).second;
      std::vector<int> mask = (masks.at(i)).second;

      if (resultInit)
        decisionCount_[i]++;
      if (resultInterm)
        intermCount_[i]++;
      if (resultFin)
        finalCount_[i]++;

      //cout << i << " " << decisionCount_[i] << "\n";

      if (dumpTriggerResults_) {
        cout << std::dec << setfill(' ') << "   " << setw(5) << i << "   " << setw(40) << name.c_str() << "   "
             << setw(7) << resultInit << setw(7) << resultInterm << setw(7) << resultFin << setw(10) << prescale
             << setw(11) << mask.size() << endl;
      }
    }
    bool finOR = gtUtil_->getFinalOR();
    if (finOR)
      finalOrCount++;
    if (dumpTriggerResults_) {
      cout << "                                                                FinalOR = " << finOR << endl;
      cout << "========================================================================================================"
              "==="
           << endl;
    }
  }

  if (dumpRecord_) {
    //int i = 0; // now now just printing BX=0...
    for (int i = minBx_; i <= maxBx_; i++) {
      // Dump the coutput record
      cout << " ------ Bx= " << i << " ext ----------" << endl;
      if (ext.isValid()) {
        if (i >= ext->getFirstBX() && i <= ext->getLastBX()) {
          for (std::vector<GlobalExtBlk>::const_iterator extBlk = ext->begin(i); extBlk != ext->end(i); ++extBlk) {
            extBlk->print(cout);
            cout << std::dec;
          }
        } else {
          cout << "No Ext Conditions stored for this bx " << i << endl;
        }
      } else {
        LogError("L1TGlobalSummary") << "No ext Data in this event " << endl;
      }

      // Dump the coutput record
      cout << " ------ Bx= " << i << " alg ----------" << endl;
      if (alg.isValid()) {
        if (i >= alg->getFirstBX() && i <= alg->getLastBX()) {
          for (std::vector<GlobalAlgBlk>::const_iterator algBlk = alg->begin(i); algBlk != alg->end(i); ++algBlk) {
            algBlk->print(cout);
            cout << std::dec;
          }
        } else {
          cout << "No Alg Decisions stored for this bx " << i << endl;
        }
      } else {
        LogError("L1TGlobalSummary") << "No alg Data in this event " << endl;
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TGlobalSummary);
