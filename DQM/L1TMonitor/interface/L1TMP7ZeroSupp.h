#ifndef DQM_L1TMonitor_L1TMP7ZeroSupp_h
#define DQM_L1TMonitor_L1TMP7ZeroSupp_h

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
//#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "EventFilter/L1TRawToDigi/interface/AMC13Spec.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class L1TMP7ZeroSupp : public DQMEDAnalyzer {
public:
  L1TMP7ZeroSupp(const edm::ParameterSet& ps);
  ~L1TMP7ZeroSupp() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  void bookCapIdHistograms(DQMStore::IBooker& ibooker, const unsigned int& id);

  // Add additional bins only before NBINLABELS
  enum binlabels {
    EVTS = 0,
    EVTSGOOD,
    EVTSBAD,
    BLOCKS,
    ZSBLKSGOOD,
    ZSBLKSBAD,
    ZSBLKSBADFALSEPOS,
    ZSBLKSBADFALSENEG,
    BXBLOCKS,
    ZSBXBLKSGOOD,
    ZSBXBLKSBAD,
    ZSBXBLKSBADFALSEPOS,
    ZSBXBLKSBADFALSENEG,
    NBINLABELS
  };
  enum ratioBinlabels {
    REVTS = 0,
    RBLKS,
    RBLKSFALSEPOS,
    RBLKSFALSENEG,
    RBXBLKS,
    RBXBLKSFALSEPOS,
    RBXBLKSFALSENEG,
    RNBINLABELS
  };

  edm::EDGetTokenT<FEDRawDataCollection> fedDataToken_;
  bool zsEnabled_;
  std::vector<int> fedIds_;
  std::vector<std::vector<int>> masks_;

  // header and trailer sizes in chars
  int slinkHeaderSize_;
  int slinkTrailerSize_;
  int amc13HeaderSize_;
  int amc13TrailerSize_;
  int amcHeaderSize_;
  int amcTrailerSize_;
  int newZsFlagMask_;
  int zsFlagMask_;
  int dataInvFlagMask_;

  int maxFedReadoutSize_;

  bool checkOnlyCapIdsWithMasks_;

  std::string monitorDir_;
  bool verbose_;

  static const unsigned int maxMasks_;

  std::vector<unsigned int> definedMaskCapIds_;

  std::map<unsigned int, MonitorElement*> zeroSuppValMap_;
  std::map<unsigned int, MonitorElement*> errorSummaryNumMap_;
  std::map<unsigned int, MonitorElement*> errorSummaryDenMap_;
  std::map<unsigned int, MonitorElement*> readoutSizeNoZSMap_;
  std::map<unsigned int, MonitorElement*> readoutSizeZSMap_;
  std::map<unsigned int, MonitorElement*> readoutSizeZSExpectedMap_;
  MonitorElement* capIds_;
};

#endif
