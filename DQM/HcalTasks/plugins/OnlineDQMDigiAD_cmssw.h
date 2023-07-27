/*
 * OnlineDQMDigiAD_cmssw.cpp
 *
 * Created on: Jun 10, 2023
 * Author: Mulugeta W.Asres, UiA, Norway
 *
 * The implementation follows https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/ONNXRuntime
 */
#ifndef OnlineDQMDigiAD_cmssw_H_
#define OnlineDQMDigiAD_cmssw_H_

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>

// Declare OnlineDQMDigiAD class
class OnlineDQMDigiAD {
public:
  /**
     * @brief Constructor
     * @param modelFilepath: path to the .onnx file
     * @param Backend: backend selection cpu or gpu
     */
  OnlineDQMDigiAD(const std::string model_system_name,
                  const std::string &modelFilepath,
                  cms::Ort::Backend backend = cms::Ort::Backend::cpu);

  /**
     * @brief check whether onnx model integration is added for the selected hcal system
     */
  void IsModelExist(std::string hcal_subsystem_name);

  /**
     * @brief Resets ml model memory states to default and function needs to be called when new collision run starts
     */
  void InitializeState();

  /**
     * @brief Perform inference on a single image
     * @param digiHcalMapTW: The input digipccupany maps in time window
     * @param numEvents: The input number of events for map renormalization in time window
     * @param adThr: The anomaly detection decision threshold
     * @param input_model_state_memory_: The model memory states
     * @param output_tensors: output arrays
     * @return the list of  multidimensional arrays
     */
  std::vector<std::vector<float>> Inference(std::vector<float> &digiHcalMapTW,
                                            std::vector<float> &numEvents,
                                            std::vector<float> &adThr,
                                            std::vector<float> &input_model_state_memory_e_0_0,
                                            std::vector<float> &input_model_state_memory_e_0_1,
                                            std::vector<float> &input_model_state_memory_e_1_0,
                                            std::vector<float> &input_model_state_memory_e_1_1,
                                            std::vector<float> &input_model_state_memory_d_0_0,
                                            std::vector<float> &input_model_state_memory_d_0_1,
                                            std::vector<float> &input_model_state_memory_d_1_0,
                                            std::vector<float> &input_model_state_memory_d_1_1);
  /**
     * @brief Perform inference on a single image
     * @param digiHcal2DHist_depth_1: 2D histogram digioccupancy of the 1st depth of the hcal-hehb
     * @param digiHcal2DHist_depth_2: 2D histogram digioccupancy of the 2nd depth of the hcal-hehb
     * @param digiHcal2DHist_depth_3: 2D histogram digioccupancy of the 3rd depth of the hcal-hehb
     * @param digiHcal2DHist_depth_4: 2D histogram digioccupancy of the 4th depth of the hcal-hehb
     * @param digiHcal2DHist_depth_5: 2D histogram digioccupancy of the 5th depth of the hcal-hehb
     * @param digiHcal2DHist_depth_5: 2D histogram digioccupancy of the 6th depth of the hcal-hehb
     * @param digiHcal2DHist_depth_7: 2D histogram digioccupancy of the 7th depth of the hcal-hehb
     * @param LS_numEvents: The input number of events for digioccupancy map renormalization
     * @param flagDecisionThr: The anomaly detection decision threshold, decrease to increase sensitivity
     * @return ad_model_output_vectors: the vectors of multidimensional arrays: output_data_0, output_data_1, ...
     */
  std::vector<std::vector<float>> Inference_CMSSW(const std::vector<std::vector<float>> &digiHcal2DHist_depth_1,
                                                  const std::vector<std::vector<float>> &digiHcal2DHist_depth_2,
                                                  const std::vector<std::vector<float>> &digiHcal2DHist_depth_3,
                                                  const std::vector<std::vector<float>> &digiHcal2DHist_depth_4,
                                                  const std::vector<std::vector<float>> &digiHcal2DHist_depth_5,
                                                  const std::vector<std::vector<float>> &digiHcal2DHist_depth_6,
                                                  const std::vector<std::vector<float>> &digiHcal2DHist_depth_7,
                                                  const float LS_numEvents,
                                                  const float flagDecisionThr = 20);

  /**
     @brief Converts 1D serialized vector output of the onnx into 3d hcal-hehp vector
    @param ad_model_output_vectors: vector of 3D histogram maps the hcal-hehb, each vector output from the onnx. e.g 3d map of anomaly score and 3d map of anomaly flag or label
    @param selOutputIdx: index to select of the onnx output. e.g. 5 is the anomaly score and 7 is the anomaly flag (1 is with anomaly, 0 is healthy)
    @return ad_model_output_vectors: the vectors of multidimensional arrays: output_data_0, output_data_1, ...
    */
  std::vector<std::vector<std::vector<float>>> ONNXOutputToDQMHistMap(
      const std::vector<std::vector<float>> &ad_model_output_vectors, const int selOutputIdx = 7);

private:
  // onnx session
  const std::vector<std::string> hcal_modeled_systems = {"he", "hb"};
  std::string hcal_subsystem_name;
  std::unique_ptr<cms::Ort::ONNXRuntime> ort_mSession = nullptr;
  std::string model_path;  // onnx model path

  // names of onnx model input vectors; do not change
  const std::vector<std::string> input_names = {
      "input_data",
      "input_data_exo",
      "anomaly_std_th",
      "e_rnn_hidden__layer_0_state_0",
      "e_rnn_hidden__layer_0_state_1",
      "e_rnn_hidden__layer_1_state_0",
      "e_rnn_hidden__layer_1_state_1",
      "d_rnn_hidden__layer_0_state_0",
      "d_rnn_hidden__layer_0_state_1",
      "d_rnn_hidden__layer_1_state_0",
      "d_rnn_hidden__layer_1_state_1",
  };

  // names of onnx model outputs vectors; do not change
  const std::vector<std::string> output_names = {
      "target_data",
      "pred_data",
      "pred_err_spatial",
      "pred_err_window_spatial",
      "pred_err_spatial_scaled",
      "pred_err_window_spatial_scaled",
      "pred_err_spatial_scaled_aml",
      "pred_err_window_spatial_scaled_aml",
      "e_rnn_hidden__layer_0_state_0_o",
      "e_rnn_hidden__layer_0_state_1_o",
      "e_rnn_hidden__layer_1_state_0_o",
      "e_rnn_hidden__layer_1_state_1_o",
      "d_rnn_hidden__layer_0_state_0_o",
      "d_rnn_hidden__layer_0_state_1_o",
      "d_rnn_hidden__layer_1_state_0_o",
      "d_rnn_hidden__layer_1_state_1_o",
  };

  // model state network config declaration : encoder and decoder have each two lstm layers(each hold two state vectors, h0, c0)
  const size_t num_state_vectors = 8;
  const unsigned int model_state_inner_dim = 2;  // do not change
  const std::vector<std::vector<unsigned int>> model_state_layer_dims = {
      {128, 32}, {128, 640}};  // do not change, encoder[layer_0, layer_1] and decoder [layer_0, layer_1]
  const std::vector<std::vector<unsigned int>> model_state_layer_serialized_dims = {
      {256, 64},
      {256,
       1280}};  // do not change, model_state_inner_dim*encoder[layer_0, layer_1] and model_state_inner_dim*decoder [layer_0, layer_1]
  // unsigned model_state_refresh_counter = 15;               // do not change for now. set due to onnx double datatype handling limitation that might cause precision error to propagate.
  unsigned model_state_refresh_counter =
      1;  // DQM multithread returns non-sequential LS. Hence, the model will not keep states (experimental)

  std::vector<float> input_model_state_memory_e_0_0{std::vector<float>(model_state_layer_serialized_dims[0][0])};
  std::vector<float> input_model_state_memory_e_0_1{std::vector<float>(model_state_layer_serialized_dims[0][0])};
  std::vector<float> input_model_state_memory_e_1_0{std::vector<float>(model_state_layer_serialized_dims[0][1])};
  std::vector<float> input_model_state_memory_e_1_1{std::vector<float>(model_state_layer_serialized_dims[0][1])};
  std::vector<float> input_model_state_memory_d_0_0{std::vector<float>(model_state_layer_serialized_dims[1][0])};
  std::vector<float> input_model_state_memory_d_0_1{std::vector<float>(model_state_layer_serialized_dims[1][0])};
  std::vector<float> input_model_state_memory_d_1_0{std::vector<float>(model_state_layer_serialized_dims[1][1])};
  std::vector<float> input_model_state_memory_d_1_1{std::vector<float>(model_state_layer_serialized_dims[1][1])};

  // input and outputs
  int64_t batch_size = 1;  // number maps to  be evaluated at once, a single time-window
  std::vector<std::vector<float>> input_values, output_values;
  std::vector<std::vector<int64_t>> input_shapes;

  /**
     * @brief Serializes 2d vectors into 1d
     */
  std::vector<float> Serialize2DVector(const std::vector<std::vector<float>> &input_2d_vec);

  /**
     * @brief Converts serialized 1d vectors into 2d
     */
  std::vector<std::vector<float>> Map1DTo2DVector(const std::vector<float> &input_1d_vec, const int numSplits);

  /**
     * @brief Prepares model input serialized dqm histogram from 2D histogram inputs from the cmssw
     *  @param digiHcal2DHist_depth_all: 3D vector (depth[ieta[iphi]]) of combined 2D histogram digioccupancy of the any depth of the hcal
     */
  std::vector<float> PrepareONNXDQMMapVectors(std::vector<std::vector<std::vector<float>>> &digiHcal2DHist_depth_all);
};

#endif  // OnlineDQMDigiAD_cmssw_H_
