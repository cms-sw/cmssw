#ifndef L1TdeStage2CaloLayer1_H
#define L1TdeStage2CaloLayer1_H

// system include files
#include <memory>
#include <string>
#include <array>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class L1TdeStage2CaloLayer1 : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  L1TdeStage2CaloLayer1(const edm::ParameterSet &ps);
  ~L1TdeStage2CaloLayer1() override;

protected:
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &, const edm::EventSetup &) override;
  void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override;
  void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override;

private:
  void updateMismatch(const edm::Event &e, int mismatchType);
  // Input and config info
  edm::InputTag dataLabel_;
  edm::EDGetTokenT<l1t::CaloTowerBxCollection> dataSource_;
  edm::InputTag emulLabel_;
  edm::EDGetTokenT<l1t::CaloTowerBxCollection> emulSource_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTowers_;
  edm::EDGetTokenT<FEDRawDataCollection> fedRawData_;
  std::string histFolder_;
  int tpFillThreshold_;

  // Some types to ease the pain of compring the sets of towers
  class SimpleTower {
  public:
    int ieta_;
    int iphi_;
    int data_;
    bool compareLeft_;
    SimpleTower(int ieta, int iphi, int data, bool compareLeft = true)
        : ieta_(ieta), iphi_(iphi), data_(data), compareLeft_(compareLeft){};
    inline int et() const { return (data_ & 0x01ff) >> 0; };   // ET bits 0-8
    inline int er() const { return (data_ & 0x0e00) >> 9; };   // ET ratio bits 9-11
    inline int fb() const { return (data_ & 0xf000) >> 12; };  // Feature bits 12-15

    // Lexicographical strict weak ordering
    // allows use of std::set_intersection, etc.
    bool operator<(const SimpleTower &b) const {
      if (ieta_ < b.ieta_)
        return true;
      if (ieta_ == b.ieta_ && iphi_ < b.iphi_)
        return true;
      if (ieta_ == b.ieta_ && iphi_ == b.iphi_ && data_ < b.data_)
        return true;
      return false;
    };
  };
  typedef std::set<SimpleTower> SimpleTowerSet;

  enum SummaryColumn {
    EtMismatch,
    ErMismatch,
    FbMismatch,
    TowerCountMismatch,
    NSummaryColumns,
  };
  MonitorElement *dataEmulSummary_;
  std::array<double, NSummaryColumns> dataEmulNumerator_;
  double dataEmulDenominator_;
  MonitorElement *mismatchesPerBxMod9_;

  MonitorElement *dataOcc_;
  MonitorElement *emulOcc_;
  MonitorElement *matchOcc_;
  MonitorElement *failureOccEtMismatch_;
  MonitorElement *failureOccEtDataZero_;
  MonitorElement *failureOccEtEmulZero_;
  MonitorElement *failureOccErMismatch_;
  MonitorElement *failureOccFbMismatch_;

  MonitorElement *dataEtDistribution_;
  MonitorElement *dataEtDistributionFBMismatch_;
  MonitorElement *emulEtDistribution_;
  MonitorElement *etCorrelation_;
  MonitorElement *matchEtDistribution_;
  MonitorElement *etMismatchDiff_;
  MonitorElement *fbCorrelation_;
  MonitorElement *fbCorrelationHF_;

  MonitorElement *etMismatchByLumi_;
  MonitorElement *erMismatchByLumi_;
  MonitorElement *fbMismatchByLumi_;

  MonitorElement *etMismatchesPerBx_;
  MonitorElement *erMismatchesPerBx_;
  MonitorElement *fbMismatchesPerBx_;
  MonitorElement *towerCountMismatchesPerBx_;

  MonitorElement *last20Mismatches_;
  std::array<std::pair<std::string, int>, 20> last20MismatchArray_;
  size_t lastMismatchIndex_{0};
};

#endif
