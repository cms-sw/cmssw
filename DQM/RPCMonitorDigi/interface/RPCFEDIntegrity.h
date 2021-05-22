#ifndef RPCFEDIntegrity_H
#define RPCFEDIntegrity_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/RPCDigi/interface/RPCRawDataCounts.h"

#include <vector>
#include <string>

class RPCFEDIntegrity : public DQMEDAnalyzer {
public:
  /// Constructor
  RPCFEDIntegrity(const edm::ParameterSet &ps);

  /// Destructor
  ~RPCFEDIntegrity() override;

  /// Begin Lumi block

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  /// Analyze
  void analyze(const edm::Event &iEvent, const edm::EventSetup &c) override;

private:
  void labelBins(MonitorElement *myMe);
  edm::EDGetTokenT<RPCRawDataCounts> rawCountsLabel_;
  void bookFEDMe(DQMStore::IBooker &);

  std::string prefixDir_;

  bool merge_, init_;

  int FATAL_LIMIT;

  enum fedHisto { Entries, Fatal, NonFatal };

  MonitorElement *fedMe_[3];

  int numOfFED_, minFEDNum_, maxFEDNum_;
  std::vector<std::string> histoName_;
};

#endif
