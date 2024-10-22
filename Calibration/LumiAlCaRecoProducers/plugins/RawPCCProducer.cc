/**_________________________________________________________________
class:   RawPCCProducer.cc

description: Creates a LumiInfo object that will contain the luminosity per bunch crossing,
along with the total luminosity and the statistical error.

authors:Sam Higginbotham (shigginb@cern.ch), Chris Palmer (capalmer@cern.ch), Jose Benitez (jose.benitez@cern.ch)

________________________________________________________________**/
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <cmath>
#include "DataFormats/Luminosity/interface/PixelClusterCounts.h"
#include "DataFormats/Luminosity/interface/LumiInfo.h"
#include "DataFormats/Luminosity/interface/LumiConstants.h"
#include "CondFormats/Luminosity/interface/LumiCorrections.h"
#include "CondFormats/DataRecord/interface/LumiCorrectionsRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

class RawPCCProducer : public edm::global::EDProducer<edm::EndLuminosityBlockProducer> {
public:
  explicit RawPCCProducer(const edm::ParameterSet&);
  ~RawPCCProducer() override;

private:
  void globalEndLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) const final;
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const final;

  //input object labels
  edm::EDGetTokenT<reco::PixelClusterCounts> pccToken_;
  //background corrections from DB
  const edm::ESGetToken<LumiCorrections, LumiCorrectionsRcd> lumiCorrectionsToken_;
  //The list of modules to skip in the lumi calc.
  const std::vector<int> modVeto_;
  //background corrections
  const bool applyCorr_;
  //Output average values
  const std::string takeAverageValue_;

  //output object labels
  const edm::EDPutTokenT<LumiInfo> putToken_;

  //produce csv lumi file
  const bool saveCSVFile_;
  const std::string csvOutLabel_;
  mutable std::mutex fileLock_;
};

//--------------------------------------------------------------------------------------------------
RawPCCProducer::RawPCCProducer(const edm::ParameterSet& iConfig)
    : pccToken_(consumes<reco::PixelClusterCounts, edm::InLumi>(edm::InputTag(
          iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::string>("inputPccLabel"),
          iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::string>("ProdInst")))),
      lumiCorrectionsToken_(esConsumes<edm::Transition::EndLuminosityBlock>()),
      modVeto_(
          iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::vector<int>>("modVeto")),
      applyCorr_(iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters")
                     .getUntrackedParameter<bool>("ApplyCorrections", false)),
      takeAverageValue_(iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters")
                            .getUntrackedParameter<std::string>("OutputValue", std::string("Average"))),
      putToken_(produces<LumiInfo, edm::Transition::EndLuminosityBlock>(
          iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters")
              .getUntrackedParameter<std::string>("outputProductName", "alcaLumi"))),
      saveCSVFile_(iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters")
                       .getUntrackedParameter<bool>("saveCSVFile", false)),
      csvOutLabel_(iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters")
                       .getUntrackedParameter<std::string>("label", std::string("rawPCC.csv"))) {}

//--------------------------------------------------------------------------------------------------
RawPCCProducer::~RawPCCProducer() {}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::globalEndLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg,
                                                     const edm::EventSetup& iSetup) const {
  //The total raw luminosity from the pixel clusters - not scaled
  float totalLumi = 0.0;
  //the statistical error on the lumi - large num ie sqrt(N)
  float statErrOnLumi = 0.0;

  //new vector containing clusters per bxid
  std::vector<int> clustersPerBXOutput(LumiConstants::numBX, 0);
  //new vector containing clusters per bxid with afterglow corrections
  std::vector<float> corrClustersPerBXOutput(LumiConstants::numBX, 0);

  //////////////////////////////////
  /// read input , clusters per module per bx
  /////////////////////////////////
  const edm::Handle<reco::PixelClusterCounts> pccHandle = lumiSeg.getHandle(pccToken_);
  const reco::PixelClusterCounts& inputPcc = *(pccHandle.product());
  //vector with Module IDs 1-1 map to bunch x-ing in clusers
  auto modID = inputPcc.readModID();
  //vector with total events at each bxid.
  auto events = inputPcc.readEvents();
  //cluster counts per module per bx
  auto clustersPerBXInput = inputPcc.readCounts();

  ////////////////////////////
  ///Apply the module veto
  ///////////////////////////
  std::vector<int> goodMods;
  for (unsigned int i = 0; i < modID.size(); i++) {
    if (std::find(modVeto_.begin(), modVeto_.end(), modID.at(i)) == modVeto_.end()) {
      goodMods.push_back(i);
    }
  }
  for (int bx = 0; bx < int(LumiConstants::numBX); bx++) {
    for (unsigned int i = 0; i < goodMods.size(); i++) {
      clustersPerBXOutput.at(bx) += clustersPerBXInput.at(goodMods.at(i) * int(LumiConstants::numBX) + bx);
    }
  }

  //////////////////////////////
  //// Apply afterglow corrections
  //////////////////////////////
  std::vector<float> correctionScaleFactors;
  if (applyCorr_) {
    const auto pccCorrections = &iSetup.getData(lumiCorrectionsToken_);
    correctionScaleFactors = pccCorrections->getCorrectionsBX();
  } else {
    correctionScaleFactors.resize(LumiConstants::numBX, 1.0);
  }

  for (unsigned int i = 0; i < clustersPerBXOutput.size(); i++) {
    if (events.at(i) != 0) {
      corrClustersPerBXOutput[i] = clustersPerBXOutput[i] * correctionScaleFactors[i];
    } else {
      corrClustersPerBXOutput[i] = 0.0;
    }
    totalLumi += corrClustersPerBXOutput[i];
    statErrOnLumi += float(events[i]);
  }

  std::vector<float> errorPerBX;  //Stat error (or number of events)
  errorPerBX.assign(events.begin(), events.end());

  //////////////////////////////////
  /// Compute average number of clusters per event
  ////////////////////////////////
  if (takeAverageValue_ == "Average") {
    unsigned int NActiveBX = 0;
    for (int bx = 0; bx < int(LumiConstants::numBX); bx++) {
      if (events[bx] > 0) {
        NActiveBX++;
        corrClustersPerBXOutput[bx] /= float(events[bx]);
        errorPerBX[bx] = 1 / sqrt(float(events[bx]));
      }
    }
    if (statErrOnLumi != 0) {
      totalLumi = totalLumi / statErrOnLumi * float(NActiveBX);
      statErrOnLumi = 1 / sqrt(statErrOnLumi) * totalLumi;
    }
  }

  ///////////////////////////////////////////////////////
  ///Lumi saved in the LuminosityBlocks
  LumiInfo outputLumiInfo;
  outputLumiInfo.setTotalInstLumi(totalLumi);
  outputLumiInfo.setTotalInstStatError(statErrOnLumi);
  outputLumiInfo.setErrorLumiAllBX(errorPerBX);
  outputLumiInfo.setInstLumiAllBX(corrClustersPerBXOutput);
  lumiSeg.emplace(putToken_, std::move(outputLumiInfo));

  //Lumi saved in the csv file
  if (saveCSVFile_) {
    std::lock_guard<std::mutex> lock(fileLock_);
    std::ofstream csfile(csvOutLabel_, std::ios_base::app);
    csfile << std::to_string(lumiSeg.run()) << ",";
    csfile << std::to_string(lumiSeg.luminosityBlock()) << ",";
    csfile << std::to_string(totalLumi);

    for (unsigned int bx = 0; bx < LumiConstants::numBX; bx++)
      csfile << "," << std::to_string(corrClustersPerBXOutput[bx]);
    csfile << std::endl;

    csfile.close();
  }
}

DEFINE_FWK_MODULE(RawPCCProducer);
