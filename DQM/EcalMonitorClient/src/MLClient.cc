#include "DQM/EcalMonitorClient/interface/MLClient.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "DQM/EcalCommon/interface/MESetNonObject.h"

#include <string>

using namespace cms::Ort;

namespace ecaldqm {
  MLClient::MLClient() : DQWorkerClient() { qualitySummaries_.insert("MLQualitySummary"); }

  void MLClient::setParams(edm::ParameterSet const& _params) {
    MLThreshold_ = _params.getUntrackedParameter<double>("MLThreshold");
    PUcorr_slope_ = _params.getUntrackedParameter<double>("PUcorr_slope");
    PUcorr_intercept_ = _params.getUntrackedParameter<double>("PUcorr_intercept");
    avgOcc_ = _params.getUntrackedParameter<std::vector<double>>("avgOcc");
    if (!onlineMode_) {
      MEs_.erase(std::string("MLQualitySummary"));
      MEs_.erase(std::string("EventsperMLImage"));
      sources_.erase(std::string("PU"));
      sources_.erase(std::string("NumEvents"));
      sources_.erase(std::string("DigiAllByLumi"));
      sources_.erase(std::string("AELoss"));
    }
  }

  void MLClient::producePlots(ProcessType) {
    if (!onlineMode_)
      return;
    using namespace std;
    MESet& meMLQualitySummary(MEs_.at("MLQualitySummary"));
    MESet& meEventsperMLImage(MEs_.at("EventsperMLImage"));

    MESetNonObject const& sPU(static_cast<MESetNonObject&>(sources_.at("PU")));
    MESetNonObject const& sNumEvents(static_cast<MESetNonObject&>(sources_.at("NumEvents")));

    //Get the no.of events and the PU per LS calculated in OccupancyTask
    int nEv = sNumEvents.getFloatValue();
    double pu = sPU.getFloatValue();
    //Do not compute ML quality if PU is non existent.
    if (pu < 0.) {
      return;
    }
    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR);

    //////////////// ML Data Preprocessing //////////////////////////////////
    //Inorder to feed the data into the ML model we apply some preprocessing.
    //We use the Digi Occupancy per Lumisection as the input source.
    //The model was trained on each occupancy plot having 500 events.
    //In apprehension of the low luminosity in the beginning of Run3, where in online DQM
    //the no.of events per LS could be lower than 500, we sum the occupancies over a fixed no.of lumisections as a running sum,
    //and require that the total no.of events on this summed occupancy to be atleast 200.
    //(This no.of LS and the no.of events are parameters which would require tuning later)
    //This summed occupancy is now the input image, which is then corrected for PileUp(PU) dependence and
    //change in no.of events, which are derived from training.
    //The input image is also padded by replicating the top and bottom rows so as to prevent the "edge effect"
    //wherein the ML model's learning degrades near the edge of the data set it sees.
    //This padding is then removed during inference on the model output.

    //Get the histogram of the input digi occupancy per lumisection.
    TH2F* hEbDigiMap((sources_.at("DigiAllByLumi")).getME(1)->getTH2F());

    size_t nTowers = nEtaTowers * nPhiTowers;  //Each occupancy map is of size 34x72 towers
    std::vector<float> ebOccMap1dCumulPad;     //Vector to feed into the ML network
    std::valarray<float> ebOccMap1d(nTowers);  //Array to store occupancy map of size 34x72
    //Store the values from the input histogram into the array
    //to do preprocessing
    for (int i = 0; i < hEbDigiMap->GetNbinsY(); i++) {  //NbinsY = 34, NbinsX = 72
      for (int j = 0; j < hEbDigiMap->GetNbinsX(); j++) {
        int bin = hEbDigiMap->GetBin(j + 1, i + 1);
        int k = (i * nPhiTowers) + j;
        ebOccMap1d[k] = hEbDigiMap->GetBinContent(bin);
      }
    }
    ebOccMap1dQ.push_back(ebOccMap1d);  //Queue which stores input occupancy maps for nLS lumis
    NEventQ.push_back(nEv);             //Queue which stores the no.of events per LS for nLS lumis

    if (NEventQ.size() < nLS) {
      return;  //Should have nLS lumis to add the occupancy over.
    }
    if (NEventQ.size() > nLS) {
      NEventQ.pop_front();  //Keep only nLS consecutive LS. Pop the first one if size greater than nLS
    }
    if (ebOccMap1dQ.size() > nLS) {
      ebOccMap1dQ.pop_front();  //Same conditon for the input occupancy maps.
    }

    int TNum = 0;
    for (size_t i = 0; i < nLS; i++) {
      TNum += NEventQ[i];  //Total no.of events over nLS lumis
    }
    if (TNum < 200) {
      return;  //The total no.of events should be atleast 200 over nLS for meaningful statistics
    }
    //Fill the ME to monitor the trend of the total no.of events in each input image to the ML model
    meEventsperMLImage.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), double(TNum));

    //Array to hold the sum of inputs, which make atleast 200 events.
    std::valarray<float> ebOccMap1dCumul(0., nTowers);

    for (size_t i = 0; i < ebOccMap1dQ.size(); i++) {
      ebOccMap1dCumul += ebOccMap1dQ[i];  //Sum the input arrays of N LS.
    }
    //Applying PU correction derived from training
    ebOccMap1dCumul = ebOccMap1dCumul / (PUcorr_slope_ * pu + PUcorr_intercept_);

    //Scaling up to match input dimensions. 36*72 used instead of 34*72 to accommodate the additional padding
    //of 2 rows to prevent the "edge effect" which is done below
    ebOccMap1dCumul = ebOccMap1dCumul * (nEtaTowersPad * nPhiTowers);

    //Correction for no.of events in each input image as originally model trained with 500 events per image
    ebOccMap1dCumul = ebOccMap1dCumul * (500. / TNum);

    //The pre-processed input is now fed into the input tensor vector which will go into the ML model
    ebOccMap1dCumulPad.assign(std::begin(ebOccMap1dCumul), std::end(ebOccMap1dCumul));

    //Replicate and pad with the first and last row to prevent the edge effect
    for (int k = 0; k < nPhiTowers; k++) {
      float val = ebOccMap1dCumulPad[nPhiTowers - 1];
      ebOccMap1dCumulPad.insert(ebOccMap1dCumulPad.begin(),
                                val);  //padding in the beginning with the first row elements
    }

    int size = ebOccMap1dCumulPad.size();
    for (int k = (size - nPhiTowers); k < size; k++) {
      float val = ebOccMap1dCumulPad[k];
      ebOccMap1dCumulPad.push_back(val);  //padding at the end with the last row elements
    }

    ///// Model Inference //////
    //An Autoencoder (AE) network with resnet architecture is used here which is trained on
    //certified good data (EB digi occupancy) from Run 2018 data.
    //On giving an input occupancy map, the encoder part of the AE compresses and reduces the input data, learning its features,
    //and the decoder reconstructs the data from the encoded form into a representation as close to the original input as possible.
    //We then compute the Mean squared error (MSE) between the input and output image, also called the Reconstruction Loss,
    //calculated at a tower by tower basis.
    //Thus, given an anomalous tower the loss should be significantly higher than the loss with respect to good towers, which the model
    //has already seen --> anomaly detection.
    //When calculating the loss we also apply a response correction by dividing each input and output image with the average occupancy from
    //all 2018 data (also to be tuned),to accommodate the difference in response of crystals in different regions of the Ecal Barrel
    //Further each loss map from each input image is then multiplied by the last N loss maps,
    ///so that real anomalies which persist with time are enhanced and fluctuations are suppressed.
    //A quality threshold is then applied on this time multiplied loss map, to mark them as GOOD or BAD,
    //after which it is stored as a quality summary ME.

    ///ONNX model running///
    std::string instanceName{"AE-DQM-inference"};
    std::string modelFilepath = edm::FileInPath("DQM/EcalMonitorClient/data/onnxModels/resnet.onnx").fullPath();

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    // Strings returned by session.GetInputNameAllocated are temporary, need to copy them before they are deallocated
    std::string inputName {session.GetInputNameAllocated(0, allocator).get()};

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    std::string outputName {session.GetOutputNameAllocated(0, allocator).get()};

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    size_t TensorSize = nEtaTowersPad * nPhiTowers;
    std::vector<float> ebRecoOccMap1dPad(TensorSize);  //To store the output reconstructed occupancy

    std::vector<const char*> inputNames{inputName.c_str()};
    std::vector<const char*> outputNames{outputName.c_str()};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, ebOccMap1dCumulPad.data(), TensorSize, inputDims.data(), inputDims.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, ebRecoOccMap1dPad.data(), TensorSize, outputDims.data(), outputDims.size()));

    session.Run(Ort::RunOptions{nullptr},
                inputNames.data(),
                inputTensors.data(),
                1,
                outputNames.data(),
                outputTensors.data(),
                1);

    ///Inference on the output from the model///
    //2D Loss map to store tower by tower loss between the output (reconstructed) and input occupancies,
    //Have same dimensions as the occupancy plot
    std::valarray<std::valarray<float>> lossMap2d(std::valarray<float>(nPhiTowers), nEtaTowers);

    //1D val arrays to store row wise information corresponding to the reconstructed, input and average occupancies, and loss.
    //and to do element wise (tower wise) operations on them to calculate the MSE loss between the reco and input occupancy.
    std::valarray<float> recoOcc1d(0., nPhiTowers);
    std::valarray<float> inputOcc1d(0., nPhiTowers);
    std::valarray<float> avgOcc1d(0., nPhiTowers);
    std::valarray<float> loss_;

    //Loss calculation
    //Ignore the top and bottom replicated padded rows when doing inference
    //by making index i run over (1,35) instead of (0,36)
    for (int i = 1; i < 35; i++) {
      for (int j = 0; j < nPhiTowers; j++) {
        int k = (i * nPhiTowers) + j;
        recoOcc1d[j] = ebRecoOccMap1dPad[k];
        inputOcc1d[j] = ebOccMap1dCumulPad[k];
        avgOcc1d[j] = avgOcc_[k];
      }
      //Calculate the MSE loss = (output-input)^2, with avg response correction
      loss_ = std::pow((recoOcc1d / avgOcc1d - inputOcc1d / avgOcc1d), 2);
      lossMap2d[i - 1] = (loss_);
    }

    lossMap2dQ.push_back(lossMap2d);  //Store each loss map from the output in the queue
    if (lossMap2dQ.size() > nLSloss) {
      lossMap2dQ.pop_front();  //Keep exactly nLSloss loss maps to multiply
    }
    if (lossMap2dQ.size() < nLSloss) {  //Exit if there are not nLSloss loss maps
      return;
    }
    //To hold the final multiplied loss
    std::valarray<std::valarray<float>> lossMap2dMult(std::valarray<float>(1., nPhiTowers), nEtaTowers);

    //Multiply together the last nLSloss loss maps
    //So that real anomalies which persist with time are enhanced and fluctuations are suppressed.
    for (size_t i = 0; i < lossMap2dQ.size(); i++) {
      lossMap2dMult *= lossMap2dQ[i];
    }

    //Fill the AELoss ME with the values of this time multiplied loss map
    MESet const& sAELoss(sources_.at("AELoss"));
    TH2F* hLossMap2dMult(sAELoss.getME(1)->getTH2F());
    for (int i = 0; i < hLossMap2dMult->GetNbinsY(); i++) {
      for (int j = 0; j < hLossMap2dMult->GetNbinsX(); j++) {
        int bin_ = hLossMap2dMult->GetBin(j + 1, i + 1);
        double content = lossMap2dMult[i][j];
        hLossMap2dMult->SetBinContent(bin_, content);
      }
    }
    ///////////////////// ML Quality Summary /////////////////////
    //Apply the quality threshold on the time multiplied loss map stored in the ME AELoss
    //If anomalous, the tower entry will have a large loss value. If good, the value will be close to zero.

    MESet::const_iterator dAEnd(sAELoss.end(GetElectronicsMap()));
    for (MESet::const_iterator dItr(sAELoss.beginChannel(GetElectronicsMap())); dItr != dAEnd;
         dItr.toNextChannel(GetElectronicsMap())) {
      DetId id(dItr->getId());

      bool doMaskML(meMLQualitySummary.maskMatches(id, mask, statusManager_, GetTrigTowerMap()));

      float entries(dItr->getBinContent());
      int quality(doMaskML ? kMGood : kGood);
      //If a trigger tower entry is greater than the ML threshold, set it to Bad quality, otherwise Good.
      if (entries > MLThreshold_) {
        quality = doMaskML ? kMBad : kBad;
      }
      //Fill the quality summary with the quality of the given tower id.
      meMLQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, double(quality));
    }  // ML Quality Summary
  }    // producePlots()

  DEFINE_ECALDQM_WORKER(MLClient);
}  // namespace ecaldqm
