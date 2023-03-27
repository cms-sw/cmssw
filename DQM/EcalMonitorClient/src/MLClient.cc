#include "DQM/EcalMonitorClient/interface/MLClient.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "DQM/EcalCommon/interface/MESetNonObject.h"

#include <fstream>

using namespace cms::Ort;

namespace ecaldqm {
  MLClient::MLClient() : DQWorkerClient() { qualitySummaries_.insert("MLQualitySummary"); }

  void MLClient::setParams(edm::ParameterSet const& _params) {
    EBThreshold_ = _params.getUntrackedParameter<double>("EBThreshold");
    EEpThreshold_ = _params.getUntrackedParameter<double>("EEpThreshold");
    EEmThreshold_ = _params.getUntrackedParameter<double>("EEmThreshold");
    EB_PUcorr_slope_ = _params.getUntrackedParameter<double>("EB_PUcorr_slope");
    EB_PUcorr_intercept_ = _params.getUntrackedParameter<double>("EB_PUcorr_intercept");
    EEp_PUcorr_slope_ = _params.getUntrackedParameter<double>("EEp_PUcorr_slope");
    EEp_PUcorr_intercept_ = _params.getUntrackedParameter<double>("EEp_PUcorr_intercept");
    EEm_PUcorr_slope_ = _params.getUntrackedParameter<double>("EEm_PUcorr_slope");
    EEm_PUcorr_intercept_ = _params.getUntrackedParameter<double>("EEm_PUcorr_intercept");

    if (!onlineMode_) {
      MEs_.erase(std::string("MLQualitySummary"));
      MEs_.erase(std::string("EventsperMLImage"));
      sources_.erase(std::string("PU"));
      sources_.erase(std::string("NumEvents"));
      sources_.erase(std::string("DigiAllByLumi"));
      sources_.erase(std::string("AELoss"));
      sources_.erase(std::string("BadTowerCount"));
      sources_.erase(std::string("BadTowerCountNorm"));
    }
  }

  void MLClient::producePlots(ProcessType) {
    if (!onlineMode_)
      return;
    nbadtowerEB = 0;
    nbadtowerEE = 0;

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
    TH2F* hEEmDigiMap((sources_.at("DigiAllByLumi")).getME(0)->getTH2F());
    TH2F* hEbDigiMap((sources_.at("DigiAllByLumi")).getME(1)->getTH2F());
    TH2F* hEEpDigiMap((sources_.at("DigiAllByLumi")).getME(2)->getTH2F());

    size_t nEBTowers = nEBEtaTowers * nEBPhiTowers;  //Each EB occupancy map is of size 34x72 towers
    size_t nEETowers = nEEEtaTowers * nEEPhiTowers;  //Each EE occupancy map is of size 20x20 towers

    //Vectors to feed into the ML network
    std::vector<float> ebOccMap1dCumulPad;
    std::vector<float> eemOccMap1dCumulPad;
    std::vector<float> eepOccMap1dCumulPad;

    //Array to store occupancy maps
    std::valarray<float> ebOccMap1d(nEBTowers);
    std::valarray<float> eemOccMap1d(nEETowers);
    std::valarray<float> eepOccMap1d(nEETowers);

    //Store the values from the input histogram into the array
    //to do preprocessing
    for (int i = 0; i < hEbDigiMap->GetNbinsY(); i++) {  //NbinsY = 34, NbinsX = 72
      for (int j = 0; j < hEbDigiMap->GetNbinsX(); j++) {
        int bin = hEbDigiMap->GetBin(j + 1, i + 1);
        int k = (i * nEBPhiTowers) + j;
        ebOccMap1d[k] = hEbDigiMap->GetBinContent(bin);
      }
    }
    ebOccMap1dQ.push_back(ebOccMap1d);  //Queue which stores input occupancy maps for nLS lumis

    for (int i = 0; i < hEEpDigiMap->GetNbinsY(); i++) {  //NbinsY = 20, NbinsX = 20
      for (int j = 0; j < hEEpDigiMap->GetNbinsX(); j++) {
        int bin = hEEpDigiMap->GetBin(j + 1, i + 1);
        int k = (i * nEEPhiTowers) + j;
        eemOccMap1d[k] = hEEmDigiMap->GetBinContent(bin);
        eepOccMap1d[k] = hEEpDigiMap->GetBinContent(bin);
      }
    }

    //Queue which stores input occupancy maps for nLS lumis
    eemOccMap1dQ.push_back(eemOccMap1d);
    eepOccMap1dQ.push_back(eepOccMap1d);

    NEventQ.push_back(nEv);  //Queue which stores the no.of events per LS for nLS lumis

    if (NEventQ.size() < nLS) {
      return;  //Should have nLS lumis to add the occupancy over.
    }
    if (NEventQ.size() > nLS) {
      NEventQ.pop_front();  //Keep only nLS consecutive LS. Pop the first one if size greater than nLS
    }
    if (ebOccMap1dQ.size() > nLS) {
      ebOccMap1dQ.pop_front();  //Same conditon for the input occupancy maps.
      eemOccMap1dQ.pop_front();
      eepOccMap1dQ.pop_front();
    }

    int TNum = 0;
    for (size_t i = 0; i < nLS; i++) {
      TNum += NEventQ[i];  //Total no.of events over nLS lumis
    }

    if (TNum < 400) {
      return;  //The total no.of events should be atleast 400 over nLS for meaningful statistics
    }
    //Fill the ME to monitor the trend of the total no.of events in each input image to the ML model
    meEventsperMLImage.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), double(TNum));

    //Array to hold the sum of inputs, which make atleast 400 events.
    std::valarray<float> ebOccMap1dCumul(0., nEBTowers);
    std::valarray<float> eemOccMap1dCumul(0., nEETowers);
    std::valarray<float> eepOccMap1dCumul(0., nEETowers);

    //Sum the input arrays of nLS.
    for (size_t i = 0; i < ebOccMap1dQ.size(); i++) {
      ebOccMap1dCumul += ebOccMap1dQ[i];
      eemOccMap1dCumul += eemOccMap1dQ[i];
      eepOccMap1dCumul += eepOccMap1dQ[i];
    }

    //Applying PU correction derived from training
    ebOccMap1dCumul = ebOccMap1dCumul / (EB_PUcorr_slope_ * pu + EB_PUcorr_intercept_);
    eemOccMap1dCumul = eemOccMap1dCumul / (EEm_PUcorr_slope_ * pu + EEm_PUcorr_intercept_);
    eepOccMap1dCumul = eepOccMap1dCumul / (EEp_PUcorr_slope_ * pu + EEp_PUcorr_intercept_);

    //Scaling up to match input dimensions.
    ebOccMap1dCumul = ebOccMap1dCumul * (nEBEtaTowers * nEBPhiTowers);
    eemOccMap1dCumul = eemOccMap1dCumul * nEEEtaTowers * nEEPhiTowers;  //(nEETowersPad * nEETowersPad);
    eepOccMap1dCumul = eepOccMap1dCumul * nEEEtaTowers * nEEPhiTowers;  //(nEETowersPad * nEETowersPad);

    //Correction for no.of events in each input image as originally model trained with 500 events per image
    ebOccMap1dCumul = ebOccMap1dCumul * (500. / TNum);
    eemOccMap1dCumul = eemOccMap1dCumul * (500. / TNum);
    eepOccMap1dCumul = eepOccMap1dCumul * (500. / TNum);

    std::vector<std::vector<float>> ebOccMap2dCumul(nEBEtaTowers, std::vector<float>(nEBPhiTowers, 0.));
    //Convert 1dCumul to 2d
    for (size_t i = 0; i < nEBEtaTowers; i++) {
      for (size_t j = 0; j < nEBPhiTowers; j++) {
        int k = (i * nEBPhiTowers) + j;
        ebOccMap2dCumul[i][j] = ebOccMap1dCumul[k];
      }
    }

    std::vector<float> pad_top;
    std::vector<float> pad_bottom;
    std::vector<float> pad_left;
    std::vector<float> pad_right;

    pad_top = ebOccMap2dCumul[0];
    pad_bottom = ebOccMap2dCumul[ebOccMap2dCumul.size() - 1];

    ebOccMap2dCumul.insert(ebOccMap2dCumul.begin(), pad_top);
    ebOccMap2dCumul.push_back(pad_bottom);

    //// Endcaps ///
    std::vector<std::vector<float>> eemOccMap2dCumul(nEEEtaTowers, std::vector<float>(nEEPhiTowers, 0.));
    std::vector<std::vector<float>> eepOccMap2dCumul(nEEEtaTowers, std::vector<float>(nEEPhiTowers, 0.));

    for (size_t i = 0; i < nEEEtaTowers; i++) {
      for (size_t j = 0; j < nEEPhiTowers; j++) {
        int k = (i * nEEPhiTowers) + j;
        eemOccMap2dCumul[i][j] = eemOccMap1dCumul[k];
        eepOccMap2dCumul[i][j] = eepOccMap1dCumul[k];
      }
    }

    // EE - //
    pad_top.clear();
    pad_bottom.clear();
    pad_left.clear();
    pad_right.clear();

    pad_top = eemOccMap2dCumul[0];
    pad_bottom = eemOccMap2dCumul[eemOccMap2dCumul.size() - 1];

    eemOccMap2dCumul.insert(eemOccMap2dCumul.begin(), pad_top);
    eemOccMap2dCumul.push_back(pad_bottom);

    for (auto& row : eemOccMap2dCumul) {
      pad_left.push_back(row[0]);
      pad_right.push_back(row[row.size() - 1]);
    }

    std::size_t Lindex = 0;
    std::size_t Rindex = 0;

    for (auto& row : eemOccMap2dCumul) {
      row.insert(row.begin(), pad_left[Lindex++]);
      row.insert(row.end(), pad_right[Rindex++]);
    }

    // EE + //
    pad_top.clear();
    pad_bottom.clear();

    pad_top = eepOccMap2dCumul[0];
    pad_bottom = eepOccMap2dCumul[eepOccMap2dCumul.size() - 1];

    eepOccMap2dCumul.insert(eepOccMap2dCumul.begin(), pad_top);
    eepOccMap2dCumul.push_back(pad_bottom);

    for (auto& row : eepOccMap2dCumul) {
      pad_left.push_back(row[0]);
      pad_right.push_back(row[row.size() - 1]);
    }

    Lindex = 0;
    Rindex = 0;

    for (auto& row : eepOccMap2dCumul) {
      row.insert(row.begin(), pad_left[Lindex++]);
      row.insert(row.end(), pad_right[Rindex++]);
    }

    //The pre-processed input is now fed into the 1D input tensor vector which will go into the ML model
    for (auto& row : ebOccMap2dCumul) {
      ebOccMap1dCumulPad.insert(ebOccMap1dCumulPad.end(), row.begin(), row.end());
    }

    for (auto& row : eemOccMap2dCumul) {
      eemOccMap1dCumulPad.insert(eemOccMap1dCumulPad.end(), row.begin(), row.end());
    }

    for (auto& row : eepOccMap2dCumul) {
      eepOccMap1dCumulPad.insert(eepOccMap1dCumulPad.end(), row.begin(), row.end());
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

    const char* inputName = session.GetInputName(0, allocator);

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    const char* outputName = session.GetOutputName(0, allocator);

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    size_t TensorSize = nEBEtaTowersPad * nEBPhiTowers;
    std::vector<float> ebRecoOccMap1dPad(TensorSize);  //To store the output reconstructed occupancy

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
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

    //Endcaps
    // EE- //

    inputDims.clear();
    outputDims.clear();
    inputNames.clear();
    outputNames.clear();
    inputTensors.clear();
    outputTensors.clear();

    modelFilepath = edm::FileInPath("DQM/EcalMonitorClient/data/onnxModels/EEm_resnet2018.onnx").fullPath();

    Ort::Session EEm_session(env, modelFilepath.c_str(), sessionOptions);

    inputName = EEm_session.GetInputName(0, allocator);

    inputTypeInfo = EEm_session.GetInputTypeInfo(0);
    auto EEm_inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    inputDims = EEm_inputTensorInfo.GetShape();

    outputName = EEm_session.GetOutputName(0, allocator);

    //Ort::TypeInfo
    outputTypeInfo = EEm_session.GetOutputTypeInfo(0);
    auto EEm_outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    outputDims = EEm_outputTensorInfo.GetShape();

    size_t EE_TensorSize = nEETowersPad * nEETowersPad;
    std::vector<float> eemRecoOccMap1dPad(EE_TensorSize);  //To store the output reconstructed occupancy

    inputNames.push_back(inputName);
    outputNames.push_back(outputName);

    //Ort::MemoryInfo
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, eemOccMap1dCumulPad.data(), EE_TensorSize, inputDims.data(), inputDims.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, eemRecoOccMap1dPad.data(), EE_TensorSize, outputDims.data(), outputDims.size()));

    EEm_session.Run(Ort::RunOptions{nullptr},
                    inputNames.data(),
                    inputTensors.data(),
                    1,
                    outputNames.data(),
                    outputTensors.data(),
                    1);

    // EE+ //
    inputDims.clear();
    outputDims.clear();
    inputNames.clear();
    outputNames.clear();
    inputTensors.clear();
    outputTensors.clear();

    modelFilepath = edm::FileInPath("DQM/EcalMonitorClient/data/onnxModels/EEp_resnet2018.onnx").fullPath();

    Ort::Session EEp_session(env, modelFilepath.c_str(), sessionOptions);

    inputName = EEp_session.GetInputName(0, allocator);

    inputTypeInfo = EEp_session.GetInputTypeInfo(0);
    auto EEp_inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    inputDims = EEp_inputTensorInfo.GetShape();

    outputName = EEp_session.GetOutputName(0, allocator);

    outputTypeInfo = EEp_session.GetOutputTypeInfo(0);

    auto EEp_outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    outputDims = EEp_outputTensorInfo.GetShape();

    std::vector<float> eepRecoOccMap1dPad(EE_TensorSize);  //To store the output reconstructed occupancy

    inputNames.push_back(inputName);
    outputNames.push_back(outputName);

    //Ort::MemoryInfo
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, eepOccMap1dCumulPad.data(), EE_TensorSize, inputDims.data(), inputDims.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, eepRecoOccMap1dPad.data(), EE_TensorSize, outputDims.data(), outputDims.size()));

    EEp_session.Run(Ort::RunOptions{nullptr},
                    inputNames.data(),
                    inputTensors.data(),
                    1,
                    outputNames.data(),
                    outputTensors.data(),
                    1);

    ///Inference on the output from the model///
    //2D Loss map to store tower by tower loss between the output (reconstructed) and input occupancies,
    //Have same dimensions as the occupancy plot
    std::valarray<std::valarray<float>> EBlossMap2d(std::valarray<float>(nEBPhiTowers), nEBEtaTowers);
    std::valarray<std::valarray<float>> EEmlossMap2d(std::valarray<float>(nEEPhiTowers), nEEEtaTowers);
    std::valarray<std::valarray<float>> EEplossMap2d(std::valarray<float>(nEEPhiTowers), nEEEtaTowers);

    //1D val arrays to store row wise information corresponding to the reconstructed, input and average occupancies, and loss.
    //and to do element wise (tower wise) operations on them to calculate the MSE loss between the reco and input occupancy.
    std::valarray<float> EBrecoOcc1d(0., nEBPhiTowers);
    std::valarray<float> EBinputOcc1d(0., nEBPhiTowers);
    std::valarray<float> EBavgOcc1d(0., nEBPhiTowers);
    std::valarray<float> EBloss_;

    std::valarray<float> EEmrecoOcc1d(0., nEEPhiTowers);
    std::valarray<float> EEminputOcc1d(0., nEEPhiTowers);
    std::valarray<float> EEmavgOcc1d(0., nEEPhiTowers);
    std::valarray<float> EEmloss_;

    std::valarray<float> EEprecoOcc1d(0., nEEPhiTowers);
    std::valarray<float> EEpinputOcc1d(0., nEEPhiTowers);
    std::valarray<float> EEpavgOcc1d(0., nEEPhiTowers);
    std::valarray<float> EEploss_;

    std::string EBOccpath =
        edm::FileInPath("DQM/EcalMonitorClient/data/MLAvgOccupancy/EB_avgocc_Run2022_500ev.dat").fullPath();
    std::ifstream inFile;
    double val;
    inFile.open((EBOccpath).c_str());
    while (inFile) {
      inFile >> val;
      if (inFile.eof())
        break;
      EBavgOcc.push_back(val);
    }
    inFile.close();

    std::string EEmOccpath =
        edm::FileInPath("DQM/EcalMonitorClient/data/MLAvgOccupancy/EEm_avgocc_Run2022_500ev.dat").fullPath();
    inFile.open((EEmOccpath).c_str());
    while (inFile) {
      inFile >> val;
      if (inFile.eof())
        break;
      EEmavgOcc.push_back(val);
    }
    inFile.close();

    std::string EEpOccpath =
        edm::FileInPath("DQM/EcalMonitorClient/data/MLAvgOccupancy/EEp_avgocc_Run2022_500ev.dat").fullPath();
    inFile.open((EEpOccpath).c_str());
    while (inFile) {
      inFile >> val;
      if (inFile.eof())
        break;
      EEpavgOcc.push_back(val);
    }
    inFile.close();

    //Loss calculation
    //Ignore the top and bottom replicated padded rows when doing inference
    //by making index i run over (1,35) instead of (0,36) for EB, and over (1,21) for EE

    MESet const& sAEReco(sources_.at("AEReco"));
    TH2F* hEBRecoMap2d(sAEReco.getME(1)->getTH2F());

    for (int i = 1; i < nEBEtaTowersPad - 1; i++) {
      for (int j = 0; j < nEBPhiTowers; j++) {
        int k = (i * nEBPhiTowers) + j;
        int bin_ = hEBRecoMap2d->GetBin(j + 1, i);
        EBrecoOcc1d[j] = ebRecoOccMap1dPad[k];
        EBinputOcc1d[j] = ebOccMap1dCumulPad[k];
        EBavgOcc1d[j] = EBavgOcc[k];
        double content = ebRecoOccMap1dPad[k];
        hEBRecoMap2d->SetBinContent(bin_, content);
      }
      //Calculate the MSE loss = (output-input)^2, with avg response correction
      EBloss_ = std::pow((EBrecoOcc1d / EBavgOcc1d - EBinputOcc1d / EBavgOcc1d), 2);
      EBlossMap2d[i - 1] = (EBloss_);
    }

    TH2F* hEEmRecoMap2d(sAEReco.getME(0)->getTH2F());
    TH2F* hEEpRecoMap2d(sAEReco.getME(2)->getTH2F());

    for (int i = 1; i < nEETowersPad - 1; i++) {
      for (int j = 0; j < nEEPhiTowers; j++) {
        int k = (i * nEETowersPad) + j + 1;
        int bin_ = hEEmRecoMap2d->GetBin(j + 1, i);

        EEmrecoOcc1d[j] = eemRecoOccMap1dPad[k];
        EEminputOcc1d[j] = eemOccMap1dCumulPad[k];
        EEmavgOcc1d[j] = EEmavgOcc[k];
        double EEmcontent = eemRecoOccMap1dPad[k];
        hEEmRecoMap2d->SetBinContent(bin_, EEmcontent);

        EEprecoOcc1d[j] = eepRecoOccMap1dPad[k];
        EEpinputOcc1d[j] = eepOccMap1dCumulPad[k];
        EEpavgOcc1d[j] = EEpavgOcc[k];
        double EEpcontent = eepRecoOccMap1dPad[k];
        hEEpRecoMap2d->SetBinContent(bin_, EEpcontent);
      }
      //Calculate the MSE loss = (output-input)^2, with avg response correction
      EEmloss_ = std::pow((EEmrecoOcc1d / EEmavgOcc1d - EEminputOcc1d / EEmavgOcc1d), 2);
      EEmlossMap2d[i - 1] = (EEmloss_);

      EEploss_ = std::pow((EEprecoOcc1d / EEpavgOcc1d - EEpinputOcc1d / EEpavgOcc1d), 2);
      EEplossMap2d[i - 1] = (EEploss_);
    }

    //Store each loss map from the output in the queue
    EBlossMap2dQ.push_back(EBlossMap2d);
    EEmlossMap2dQ.push_back(EEmlossMap2d);
    EEplossMap2dQ.push_back(EEplossMap2d);

    //Keep exactly nLSloss loss maps to multiply
    if (EBlossMap2dQ.size() > nLSloss) {
      EBlossMap2dQ.pop_front();
      EEmlossMap2dQ.pop_front();
      EEplossMap2dQ.pop_front();
    }
    if (EBlossMap2dQ.size() < nLSloss) {  //Exit if there are not nLSloss loss maps
      return;
    }

    //To hold the final multiplied loss
    std::valarray<std::valarray<float>> EBlossMap2dMult(std::valarray<float>(1., nEBPhiTowers), nEBEtaTowers);
    std::valarray<std::valarray<float>> EEmlossMap2dMult(std::valarray<float>(1., nEEPhiTowers), nEEEtaTowers);
    std::valarray<std::valarray<float>> EEplossMap2dMult(std::valarray<float>(1., nEEPhiTowers), nEEEtaTowers);

    //Multiply together the last nLSloss loss maps
    //So that real anomalies which persist with time are enhanced and fluctuations are suppressed.
    for (size_t i = 0; i < EBlossMap2dQ.size(); i++) {
      EBlossMap2dMult *= EBlossMap2dQ[i];
      EEmlossMap2dMult *= EEmlossMap2dQ[i];
      EEplossMap2dMult *= EEplossMap2dQ[i];
    }

    //Fill the AELoss ME with the values of this time multiplied loss map
    //MESet const& sAELoss(sources_.at("AELoss"));
    MESet& sAELoss(sources_.at("AELoss"));

    TH2F* hEBLossMap2dMult(sAELoss.getME(1)->getTH2F());

    for (int i = 0; i < hEBLossMap2dMult->GetNbinsY(); i++) {
      for (int j = 0; j < hEBLossMap2dMult->GetNbinsX(); j++) {
        int bin_ = hEBLossMap2dMult->GetBin(j + 1, i + 1);
        double content = EBlossMap2dMult[i][j];
        hEBLossMap2dMult->SetBinContent(bin_, content);
      }
    }

    TH2F* hEEmLossMap2dMult(sAELoss.getME(0)->getTH2F());
    TH2F* hEEpLossMap2dMult(sAELoss.getME(2)->getTH2F());

    for (int i = 0; i < hEEmLossMap2dMult->GetNbinsY(); i++) {
      for (int j = 0; j < hEEmLossMap2dMult->GetNbinsX(); j++) {
        int bin_ = hEEmLossMap2dMult->GetBin(j + 1, i + 1);

        double EEmcontent = EEmlossMap2dMult[i][j];
        hEEmLossMap2dMult->SetBinContent(bin_, EEmcontent);

        double EEpcontent = EEplossMap2dMult[i][j];
        hEEpLossMap2dMult->SetBinContent(bin_, EEpcontent);
      }
    }

    ///////////////////// ML Quality Summary /////////////////////
    //Apply the quality threshold on the time multiplied loss map stored in the ME AELoss
    //If anomalous, the tower entry will have a large loss value. If good, the value will be close to zero.

    MESet& meBadTowerCount(sources_.at("BadTowerCount"));
    MESet& meBadTowerCountNorm(sources_.at("BadTowerCountNorm"));
    MESet& meTrendMLBadTower(MEs_.at("TrendMLBadTower"));

    LScount++;

    MESet::const_iterator dAEnd(sAELoss.end(GetElectronicsMap()));
    for (MESet::const_iterator dItr(sAELoss.beginChannel(GetElectronicsMap())); dItr != dAEnd;
         dItr.toNextChannel(GetElectronicsMap())) {
      DetId id(dItr->getId());

      bool doMaskML(meMLQualitySummary.maskMatches(id, mask, statusManager_, GetTrigTowerMap()));

      float entries(dItr->getBinContent());

      int quality(doMaskML ? kMGood : kGood);
      float MLThreshold;

      if (id.subdetId() == EcalEndcap) {
        EEDetId eeid(id);
        if (eeid.zside() > 0)
          MLThreshold = EEpThreshold_;
        else
          MLThreshold = EEmThreshold_;
      } else {
        MLThreshold = EBThreshold_;
      }

      //If a trigger tower entry is greater than the ML threshold, set it to Bad quality, otherwise Good.
      if (entries > MLThreshold) {
        quality = doMaskML ? kMBad : kBad;
        meBadTowerCount.fill(getEcalDQMSetupObjects(), id);
        if (id.subdetId() == EcalEndcap)
          nbadtowerEE++;
        else
          nbadtowerEB++;
      }
      //Fill the quality summary with the quality of the given tower id.
      meMLQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, double(quality));

      double badtowcount(meBadTowerCount.getBinContent(getEcalDQMSetupObjects(), id));
      meBadTowerCountNorm.setBinContent(getEcalDQMSetupObjects(), id, double(badtowcount / LScount));
    }  // ML Quality Summary

    meTrendMLBadTower.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), double(nbadtowerEB));
    meTrendMLBadTower.fill(getEcalDQMSetupObjects(), EcalEndcap, double(timestamp_.iLumi), double(nbadtowerEE));

  }  // producePlots()

  DEFINE_ECALDQM_WORKER(MLClient);
}  // namespace ecaldqm
