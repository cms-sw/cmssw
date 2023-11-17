/*
 * OnlineDQMDigiAD_cmssw.cpp
 *
 * Created on: Jun 10, 2023
 * Author: Mulugeta W.Asres, UiA, Norway
 *
 * The implementation follows https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/ONNXRuntime
 */

// #include "FWCore/Utilities/interface/Exception.h"
// #include "FWCore/Utilities/interface/thread_safety_macros.h"
// #include "FWCore/Framework/interface/Event.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>

#include "DQM/HcalTasks/plugins/OnlineDQMDigiAD_cmssw.h"

// using namespace std;
using namespace cms::Ort;

// Constructor
OnlineDQMDigiAD::OnlineDQMDigiAD(const std::string model_system_name,
                                 const std::string &modelFilepath,
                                 Backend backend) {
  std::string instanceName{"DESMOD Digioccupancy Map AD inference"};

  /**************** Initailize Model Memory States ******************/
  InitializeState();  // initailize model memory states to zero

  /**************** Create ORT session ******************/
  // Set up options for session
  auto session_options = ONNXRuntime::defaultSessionOptions(backend);
  // Create session by loading the onnx model
  model_path = edm::FileInPath(modelFilepath).fullPath();
  auto uOrtSession = std::make_unique<ONNXRuntime>(model_path, &session_options);
  ort_mSession = std::move(uOrtSession);

  // check model availability
  hcal_subsystem_name = model_system_name;

  IsModelExist(hcal_subsystem_name);  // assert model integration for the given hcal system name

  if (hcal_subsystem_name == "he") {
    std::vector<std::vector<int64_t>> input_shapes_ = {
        {batch_size, 64, 72, 7, 1},
        {batch_size, 1},
        {1, 1},
        {batch_size, model_state_inner_dim, model_state_layer_dims[0][0]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[0][0]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[0][1]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[0][1]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[1][0]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[1][0]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[1][1]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[1][1]}};  // input dims
    input_shapes = input_shapes_;
  }

  else if (hcal_subsystem_name == "hb") {
    std::vector<std::vector<int64_t>> input_shapes_ = {
        {batch_size, 64, 72, 4, 1},
        {batch_size, 1},
        {1, 1},
        {batch_size, model_state_inner_dim, model_state_layer_dims[0][0]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[0][0]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[0][1]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[0][1]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[1][0]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[1][0]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[1][1]},
        {batch_size, model_state_inner_dim, model_state_layer_dims[1][1]}};  // input dims
    input_shapes = input_shapes_;
  }
}

void OnlineDQMDigiAD::IsModelExist(std::string hcal_subsystem_name) {
  if (std::find(hcal_modeled_systems.begin(), hcal_modeled_systems.end(), hcal_subsystem_name) ==
      hcal_modeled_systems.end()) {
    std::string err =
        "ML for OnlineDQM is not currently supported for the selected " + hcal_subsystem_name + " system!\n";
    throw std::invalid_argument(err);
  }
}

void OnlineDQMDigiAD::InitializeState() {
  // model memory states vectors init, only when the runs starts or for the first LS
  std::fill(input_model_state_memory_e_0_0.begin(),
            input_model_state_memory_e_0_0.end(),
            float(0.0));  // init model memory states-encoder_layer_0_state_0 to zero
  std::fill(input_model_state_memory_e_0_1.begin(),
            input_model_state_memory_e_0_1.end(),
            float(0.0));  // init model memory states-encoder_layer_0_state_1 to zero
  std::fill(input_model_state_memory_e_1_0.begin(),
            input_model_state_memory_e_1_0.end(),
            float(0.0));  // init model memory states-encoder_layer_1_state_0 to zero
  std::fill(input_model_state_memory_e_1_1.begin(),
            input_model_state_memory_e_1_1.end(),
            float(0.0));  // init model memory states-encoder_layer_1_state_1 to zero
  std::fill(input_model_state_memory_d_0_0.begin(),
            input_model_state_memory_d_0_0.end(),
            float(0.0));  // init model memory states-decoder_layer_0_state_0 to zero
  std::fill(input_model_state_memory_d_0_1.begin(),
            input_model_state_memory_d_0_1.end(),
            float(0.0));  // init model memory states-decoder_layer_0_state_1 to zero
  std::fill(input_model_state_memory_d_1_0.begin(),
            input_model_state_memory_d_1_0.end(),
            float(0.0));  // init model memory states-decoder_layer_1_state_0 to zero
  std::fill(input_model_state_memory_d_1_1.begin(),
            input_model_state_memory_d_1_1.end(),
            float(0.0));  // init model memory states-decoder_layer_1_state_1 to zero

  // model_state_refresh_counter = 15; // counter set due to onnx double datatype handling limitation that might cause precision error to propagate.
  model_state_refresh_counter =
      1;  // DQM multithread returns non-sequential LS. Hence, the model will not keep states (experimental)
}

std::vector<float> OnlineDQMDigiAD::Serialize2DVector(const std::vector<std::vector<float>> &input_2d_vec) {
  std::vector<float> output;
  for (const auto &row : input_2d_vec) {
    for (const auto &element : row) {
      output.push_back(element);
    }
  }
  return output;
}

std::vector<std::vector<float>> OnlineDQMDigiAD::Map1DTo2DVector(const std::vector<float> &input_1d_vec,
                                                                 const int numSplits) {
  if (numSplits <= 0)
    throw std::invalid_argument("numSplits must be greater than 0.");

  std::size_t const splitted_size = input_1d_vec.size() / numSplits;

  if (splitted_size * numSplits != input_1d_vec.size())
    throw std::invalid_argument("Conversion is not allowed! The input vector length " +
                                std::to_string(input_1d_vec.size()) + " must be divisible by the numSplits " +
                                std::to_string(numSplits) + ".");

  std::vector<std::vector<float>> output_2d_vec;

  for (int i = 0; i < numSplits; i++) {
    std::vector<float> chunch_vec(input_1d_vec.begin() + i * splitted_size,
                                  input_1d_vec.begin() + (i + 1) * splitted_size);
    output_2d_vec.push_back(chunch_vec);
  }
  return output_2d_vec;
}

std::vector<float> OnlineDQMDigiAD::PrepareONNXDQMMapVectors(
    std::vector<std::vector<std::vector<float>>> &digiHcal2DHist_depth_all) {
  std::vector<float> digi3DHistVector_serialized;

  for (const std::vector<std::vector<float>> &digiHcal2DHist_depth : digiHcal2DHist_depth_all) {
    std::vector<float> digiHcalDHist_serialized_depth = Serialize2DVector(digiHcal2DHist_depth);
    digi3DHistVector_serialized.insert(digi3DHistVector_serialized.end(),
                                       digiHcalDHist_serialized_depth.begin(),
                                       digiHcalDHist_serialized_depth.end());
  }

  return digi3DHistVector_serialized;
}

std::vector<std::vector<std::vector<float>>> OnlineDQMDigiAD::ONNXOutputToDQMHistMap(
    const std::vector<std::vector<float>> &ad_model_output_vectors,
    const int numDepth,
    const int numDIeta,
    const int selOutputIdx) {
  // each output_vector is a serialized 3d hist map

  const std::vector<float> &output_vector = ad_model_output_vectors[selOutputIdx];
  std::vector<std::vector<float>> output_2d_vec = Map1DTo2DVector(output_vector, numDepth);

  std::vector<std::vector<std::vector<float>>> digiHcal3DHist;
  for (const std::vector<float> &output_vector_depth : output_2d_vec) {
    std::vector<std::vector<float>> digiHcal2DHist_depth = Map1DTo2DVector(output_vector_depth, numDIeta);
    digiHcal3DHist.push_back(digiHcal2DHist_depth);
  }

  return digiHcal3DHist;
}

// Perform inference for a given dqm map
std::vector<std::vector<float>> OnlineDQMDigiAD::Inference(std::vector<float> &digiHcalMapTW,
                                                           std::vector<float> &numEvents,
                                                           std::vector<float> &adThr,
                                                           std::vector<float> &input_model_state_memory_e_0_0,
                                                           std::vector<float> &input_model_state_memory_e_0_1,
                                                           std::vector<float> &input_model_state_memory_e_1_0,
                                                           std::vector<float> &input_model_state_memory_e_1_1,
                                                           std::vector<float> &input_model_state_memory_d_0_0,
                                                           std::vector<float> &input_model_state_memory_d_0_1,
                                                           std::vector<float> &input_model_state_memory_d_1_0,
                                                           std::vector<float> &input_model_state_memory_d_1_1) {
  /**************** Preprocessing ******************/
  // Create input tensor (including size and value) from the loaded inputs
  // Compute the product of all input dimension
  // Assign memory for input tensor
  // inputTensors will be used by the Session Run for inference

  input_values.clear();
  input_values.emplace_back(digiHcalMapTW);
  input_values.emplace_back(numEvents);
  input_values.emplace_back(adThr);
  input_values.emplace_back(input_model_state_memory_e_0_0);
  input_values.emplace_back(input_model_state_memory_e_0_1);
  input_values.emplace_back(input_model_state_memory_e_1_0);
  input_values.emplace_back(input_model_state_memory_e_1_1);
  input_values.emplace_back(input_model_state_memory_d_0_0);
  input_values.emplace_back(input_model_state_memory_d_0_1);
  input_values.emplace_back(input_model_state_memory_d_1_0);
  input_values.emplace_back(input_model_state_memory_d_1_1);

  /**************** Inference ******************/

  output_values = ort_mSession->run(input_names, input_values, input_shapes, output_names, batch_size);

  return output_values;
}

// AD method to be called by the CMS system
std::vector<std::vector<float>> OnlineDQMDigiAD::Inference_CMSSW(
    const std::vector<std::vector<float>> &digiHcal2DHist_depth_1,
    const std::vector<std::vector<float>> &digiHcal2DHist_depth_2,
    const std::vector<std::vector<float>> &digiHcal2DHist_depth_3,
    const std::vector<std::vector<float>> &digiHcal2DHist_depth_4,
    const std::vector<std::vector<float>> &digiHcal2DHist_depth_5,
    const std::vector<std::vector<float>> &digiHcal2DHist_depth_6,
    const std::vector<std::vector<float>> &digiHcal2DHist_depth_7,
    const float LS_numEvents,
    const float flagDecisionThr)

{
  /**************** Prepare data ******************/
  // merging all 2d hist into one 3d depth[ieta[iphi]]
  std::vector<std::vector<std::vector<float>>> digiHcal2DHist_depth_all;

  if (hcal_subsystem_name == "he") {
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_1);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_2);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_3);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_4);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_5);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_6);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_7);
  }

  else if (hcal_subsystem_name == "hb") {
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_1);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_2);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_3);
    digiHcal2DHist_depth_all.push_back(digiHcal2DHist_depth_4);
  }

  // convert the 3d depth[ieta[iphi]] vector into 1d and commbined
  std::vector<float> digiHcalMapTW = PrepareONNXDQMMapVectors(digiHcal2DHist_depth_all);

  std::vector<float> adThr{flagDecisionThr};  // AD decision threshold, increase to reduce sensitivity
  std::vector<float> numEvents{LS_numEvents};

  // call model inference
  /**************** Inference ******************/
  std::vector<std::vector<float>> output_tensors = Inference(digiHcalMapTW,
                                                             numEvents,
                                                             adThr,
                                                             input_model_state_memory_e_0_0,
                                                             input_model_state_memory_e_0_1,
                                                             input_model_state_memory_e_1_0,
                                                             input_model_state_memory_e_1_1,
                                                             input_model_state_memory_d_0_0,
                                                             input_model_state_memory_d_0_1,
                                                             input_model_state_memory_d_1_0,
                                                             input_model_state_memory_d_1_1);

  // auto output_tensors = Inference(digiHcalMapTW, numEvents, adThr);
  //std::cout << "******* model inference is success *******" << std::endl;

  /**************** Output post processing ******************/
  //  split outputs into ad output vectors and state_memory vectors
  std::string state_output_name_tag = "rnn_hidden";
  std::vector<std::vector<float>> ad_model_output_vectors, ad_model_state_vectors;
  for (size_t i = 0; i < output_tensors.size(); i++) {
    std::string output_names_startstr = output_names[i].substr(
        2, state_output_name_tag.length());  // Extract the same number of characters as str2 from mOutputNames
    if (output_names_startstr == state_output_name_tag) {
      ad_model_state_vectors.emplace_back(output_tensors[i]);
    } else {
      ad_model_output_vectors.emplace_back(output_tensors[i]);
    }
  }

  if (ad_model_output_vectors.size() == num_state_vectors) {
    input_model_state_memory_e_0_0 = ad_model_state_vectors[0];
    input_model_state_memory_e_0_1 = ad_model_state_vectors[1];
    input_model_state_memory_e_1_0 = ad_model_state_vectors[2];
    input_model_state_memory_e_1_1 = ad_model_state_vectors[3];
    input_model_state_memory_d_0_0 = ad_model_state_vectors[4];
    input_model_state_memory_d_0_1 = ad_model_state_vectors[5];
    input_model_state_memory_d_1_0 = ad_model_state_vectors[6];
    input_model_state_memory_d_1_1 = ad_model_state_vectors[7];
  } else {
    std::cout << "Warning: the number of output state vectors does NOT equals to expected!. The states are set to  "
                 "default values."
              << std::endl;
    InitializeState();
  }

  // # if onnx is returning serialized 1d vectors instead of vector of 3d vectors
  // aml score and flag are at index 5 and 7 of the vector ad_model_output_vectors: anomaly score: ad_model_output_vectors[5], anomaly flags: ad_model_output_vectors[7]
  /*
      selOutputIdx: index to select of the onnx output. e.g. 5 is the anomaly score and 7 is the anomaly flag (1 is with anomaly, 0 is healthy)
      std::vector<std::vector<std::vector<float>>> digiHcal3DHist_ANOMALY_FLAG = ONNXOutputToDQMHistMap(ad_model_output_vectors, 7)
      std::vector<std::vector<std::vector<float>>> digiHcal3DHist_ANOMALY_SCORE = ONNXOutputToDQMHistMap(ad_model_output_vectors, 5)
      */

  // reduce counter for each ls call. due to onnx double datatype handling limitation that might cause precision error to propagate.
  if (--model_state_refresh_counter == 0)
    InitializeState();

  return ad_model_output_vectors;
}
