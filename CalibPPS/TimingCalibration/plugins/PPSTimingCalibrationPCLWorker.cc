/****************************************************************************
 *
 * This is a part of PPS offline software.
 * Authors:
 *   Edoardo Bossini
 *   Piotr Maciej Cwiklicki
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"

//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLWorker : public DQMEDAnalyzer {
public:
  explicit PPSTimingCalibrationPCLWorker(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondDigi>> digiToken_;
  std::string dqmDir_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLWorker::PPSTimingCalibrationPCLWorker(const edm::ParameterSet& iConfig)
  :digiToken_(consumes<edm::DetSetVector<CTPPSDiamondDigi>>(iConfig.getParameter<edm::InputTag>("digiTag"))),
   dqmDir_(iConfig.getParameter<std::string>("dqmDir"))
{
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  iBooker.cd();
  iBooker.setCurrentFolder(dqmDir_);
  for (unsigned short arm = 0; arm < 2; ++arm) {
    for (unsigned short st = 0; st < 2; ++st) {
      for (unsigned short pl = 0; pl < 4; ++pl) {
        for (unsigned short ch = 0; ch < 12; ++ch) {
        } // loop over channels
      } // loop over arms
    } // loop over stations
  } // loop over arms
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<edm::DetSetVector<CTPPSDiamondDigi>> dsv_digis;
  iEvent.getByToken(digiToken_, dsv_digis);
  if (dsv_digis->empty()) {
    edm::LogWarning("PPSTimingCalibrationPCLWorker:analyze") << "No digis retrieved from the event content.";
    return;
  }
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("ctppsDiamondRawToDigi", "TimingDiamond"))
    ->setComment("input tag for the PPS diamond detectors digis");
  desc.add<std::string>("dqmDir", "AlCaReco/PPSTimingCalibrationPCL")
    ->setComment("output path for the various DQM plots");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLWorker);
