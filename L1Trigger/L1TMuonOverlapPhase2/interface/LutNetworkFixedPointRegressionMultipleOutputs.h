/*
 * LutNetworkFixedPointRegressionMultipleOutputs.h
 *
 *  Created on: April 13, 2021
 *      Author: Karol Bunkowski, kbunkow@cern.ch
 */

#ifndef L1Trigger_L1TMuonOverlapPhase2_LutNetworkFixedPointRegressionMultipleOutputs_h
#define L1Trigger_L1TMuonOverlapPhase2_LutNetworkFixedPointRegressionMultipleOutputs_h

#include "L1Trigger/L1TMuonOverlapPhase2/interface/LutNeuronLayerFixedPoint.h"

#include "ap_int.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace lutNN {

  //_I - number of integer bits in the ap_ufixed, _F - number of fractional bits in the ap_ufixed
  //the network has two outputs, and since each output can have different range, the LUTs in the last layer have different I and F
  template <
      int input_I,
      int input_F,
      std::size_t inputSize,
      int layer1_lut_I,
      int layer1_lut_F,
      int layer1_neurons,
      int layer1_output_I,  //to the layer1 output the bias is added to make the layer2 input, therefore the layer1_output_I and layer2_lut_I are different
      int layer1_output_F,
      int layer2_input_I,
      int layer2_lut_I,
      int layer2_lut_F,
      int layer2_neurons,
      int layer3_input_I,
      int layer3_input_F,
      int layer3_0_inputCnt,
      int layer3_0_lut_I,
      int layer3_0_lut_F,
      int output0_I,
      int output0_F,
      int layer3_0_multiplicity,
      int layer3_1_inputCnt,
      int layer3_1_lut_I,
      int layer3_1_lut_F,
      int output1_I,
      int output1_F,
      int layer3_1_multiplicity>
  class LutNetworkFixedPointRegressionMultipleOutputs : public LutNetworkFixedPointRegressionBase {
  public:
    LutNetworkFixedPointRegressionMultipleOutputs() {
      static_assert(layer2_neurons ==
                    (layer3_0_inputCnt * layer3_0_multiplicity + layer3_1_inputCnt * layer3_1_multiplicity));

      //std::cout << "LutNetworkFixedPoint" << std::endl;
      lutLayer1.setName("lutLayer1");
      lutLayer2.setName("lutLayer2");
      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_0.size(); iSubLayer++) {
        lutLayer3_0[iSubLayer].setName("lutLayer3_0_" + std::to_string(iSubLayer));
      }
      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_1.size(); iSubLayer++) {
        lutLayer3_1[iSubLayer].setName("lutLayer3_1_" + std::to_string(iSubLayer));
      }
    };

    ~LutNetworkFixedPointRegressionMultipleOutputs() override {};

    typedef LutNeuronLayerFixedPoint<input_I,
                                     input_F,
                                     inputSize,
                                     layer1_lut_I,
                                     layer1_lut_F,
                                     layer1_neurons,
                                     layer1_output_I,
                                     layer1_output_F>
        LutLayer1;
    LutLayer1 lutLayer1;

    static constexpr unsigned int noHitCntShift = layer1_output_I;  //FIXME should be layer1_output_I ???

    static constexpr int layer2_input_F = layer1_output_F;

    typedef LutNeuronLayerFixedPoint<layer2_input_I,
                                     layer2_input_F,
                                     layer1_neurons,
                                     layer2_lut_I,
                                     layer2_lut_F,
                                     layer2_neurons,
                                     layer3_input_I,
                                     layer3_input_F>
        LutLayer2;
    LutLayer2 lutLayer2;

    typedef LutNeuronLayerFixedPoint<layer3_input_I,
                                     layer3_input_F,
                                     layer3_0_inputCnt,
                                     layer3_0_lut_I,
                                     layer3_0_lut_F,
                                     1,
                                     output0_I,
                                     output0_F>
        LutLayer3_0;
    std::array<LutLayer3_0, layer3_0_multiplicity> lutLayer3_0;

    typedef LutNeuronLayerFixedPoint<layer3_input_I,
                                     layer3_input_F,
                                     layer3_1_inputCnt,
                                     layer3_1_lut_I,
                                     layer3_1_lut_F,
                                     1,
                                     output1_I,
                                     output1_F>
        LutLayer3_1;
    std::array<LutLayer3_1, layer3_1_multiplicity> lutLayer3_1;

    void runWithInterpolation() {
      lutLayer1.runWithInterpolation(inputArray);
      auto& layer1Out = lutLayer1.getOutWithOffset();

      std::array<ap_ufixed<layer2_input_I + layer2_input_F, layer2_input_I, AP_TRN, AP_SAT>, layer1_neurons>
          layer1OutWithBias;
      for (unsigned int i = 0; i < layer1Out.size(); i++) {
        layer1OutWithBias[i] = layer1Out[i] + layer1Bias;
      }

      lutLayer2.runWithInterpolation(layer1OutWithBias);
      auto& layer2Out = lutLayer2.getOutWithOffset();

      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_0.size(); iSubLayer++) {
        typename LutLayer3_0::inputArrayType lutLayer3_0_input;
        std::copy(layer2Out.begin() + lutLayer3_0_input.size() * iSubLayer,  //from
                  layer2Out.begin() + lutLayer3_0_input.size() * (iSubLayer + 1),
                  lutLayer3_0_input.begin());

        lutLayer3_0[iSubLayer].runWithInterpolation(lutLayer3_0_input);
      }

      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_1.size(); iSubLayer++) {
        typename LutLayer3_1::inputArrayType lutLayer3_1_input;
        std::copy(
            layer2Out.begin() + lutLayer3_1_input.size() * iSubLayer +
                layer3_0_inputCnt * layer3_0_multiplicity,  //from
            layer2Out.begin() + lutLayer3_1_input.size() * (iSubLayer + 1) + layer3_0_inputCnt * layer3_0_multiplicity,
            lutLayer3_1_input.begin());

        lutLayer3_1[iSubLayer].runWithInterpolation(lutLayer3_1_input);
      }
    }

    void run(std::vector<float>& inputs, float noHitVal, std::vector<double>& nnResult) override {
      unsigned int noHitsCnt = 0;
      for (unsigned int iInput = 0; iInput < inputs.size(); iInput++) {
        inputArray[iInput] = inputs[iInput];
        if (inputs[iInput] == noHitVal)
          noHitsCnt++;
      }

      //the minimum required number of hits for a good candidate is 3,
      //and the total number of possible hits is 18 (number of OMTF layers)
      //so the maximum noHitsCnt is 15. It must be constrained here, otherwise address for the the next layer would be out of LUT range
      if (noHitsCnt > 15)
        noHitsCnt = 15;

      unsigned int bias = (noHitsCnt << noHitCntShift);

      //layer1Bias switches the input of the layer2 (i.e. output of the layer1) do different regions in the LUTs
      //depending on the  number of layers without hits
      layer1Bias = bias;

      runWithInterpolation();

      //output0_I goes to the declaration of the lutLayer3_0, but it does not matter, as it is used only for the outputArray
      //auto layer3_0_out = ap_ufixed<output0_I+output0_F, output0_I, AP_RND_CONV, AP_SAT>(lutLayer3_0.getLutOutSum()[0]); //TODO should be AP_RND_CONV rather, but it affect the rate
      //auto layer3_1_out = ap_fixed <output1_I+output1_F, output1_I, AP_RND_CONV, AP_SAT>(lutLayer3_1.getLutOutSum()[0]); //here layer3_0_out has size 1

      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_0.size(); iSubLayer++) {
        nnResult[iSubLayer] = lutLayer3_0[iSubLayer].getLutOutSum()[0].to_float();  //here layer3_0_out has size 1
        //std::cout<<"nnResult["<<iSubLayer<<"] "<<nnResult[iSubLayer] <<std::endl;
      }
      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_1.size(); iSubLayer++) {
        nnResult[iSubLayer + layer3_0_multiplicity] = lutLayer3_1[iSubLayer].getLutOutSum()[0].to_float();
        //std::cout<<"nnResult["<<iSubLayer<<"] "<<nnResult[iSubLayer + layer3_0_multiplicity] <<std::endl;
      }
    }

    //pt in the hardware scale, ptGeV = (ptHw -1) / 2
    int getCalibratedHwPt() override {
      //auto lutAddr = ap_ufixed<output0_I+output0_F+output0_F, output0_I+output0_F, AP_RND_CONV, AP_SAT>(lutLayer3_0.getLutOutSum()[0]);
      //lutAddr = lutAddr<<output0_F;
      //std::cout<<"lutLayer3_0.getLutOutSum()[0] "<<lutLayer3_0.getLutOutSum()[0]<<" lutAddr.to_uint() "<<lutAddr.to_uint()<<" ptCalibrationArray[lutAddr] "<<ptCalibrationArray[lutAddr.to_uint()]<<std::endl;
      //return ptCalibrationArray[lutAddr.to_uint()].to_uint();

      return 0.216286 + 1.09483 * lutLayer3_0[0].getLutOutSum()[0].to_float();
      //TODO the same can be obtained by lut[x] -> scale * lut[x] + offset for the last layer
    }

    void save(const std::string& filename) override {
      // Create an empty property tree object.
      boost::property_tree::ptree tree;

      PUT_VAR(tree, name, output0_I)
      PUT_VAR(tree, name, output0_F)
      PUT_VAR(tree, name, output1_I)
      PUT_VAR(tree, name, output1_F)

      lutLayer1.save(tree, name);
      lutLayer2.save(tree, name);
      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_0.size(); iSubLayer++) {
        lutLayer3_0[iSubLayer].save(tree, name);
      }
      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_1.size(); iSubLayer++) {
        lutLayer3_1[iSubLayer].save(tree, name);
      }

      int size = ptCalibrationArray.size();
      std::string key = "LutNetworkFixedPointRegressionMultipleOutputs.ptCalibrationArray";
      PUT_VAR(tree, key, size)
      std::ostringstream ostr;
      for (auto& a : ptCalibrationArray) {
        ostr << a.to_uint() << ", ";
      }
      tree.put(key + ".values", ostr.str());

      boost::property_tree::write_xml(filename,
                                      tree,
                                      std::locale(),
                                      boost::property_tree::xml_parser::xml_writer_make_settings<std::string>(' ', 2));
    }

    void load(const std::string& filename) override {
      // Create an empty property tree object.
      boost::property_tree::ptree tree;

      boost::property_tree::read_xml(filename, tree);

      CHECK_VAR(tree, name, output0_I)
      CHECK_VAR(tree, name, output0_F)
      CHECK_VAR(tree, name, output1_I)
      CHECK_VAR(tree, name, output1_F)

      lutLayer1.load(tree, name);
      lutLayer2.load(tree, name);
      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_0.size(); iSubLayer++) {
        lutLayer3_0[iSubLayer].load(tree, name);
      }
      for (unsigned int iSubLayer = 0; iSubLayer < lutLayer3_1.size(); iSubLayer++) {
        lutLayer3_1[iSubLayer].load(tree, name);
      }

      std::string key = "LutNetworkFixedPointRegressionMultipleOutputs.ptCalibrationArray";
      int size = ptCalibrationArray.size();
      CHECK_VAR(tree, key, size)

      auto str = tree.get<std::string>(key + ".values");

      std::stringstream ss(str);
      std::string item;

      for (auto& a : ptCalibrationArray) {
        if (std::getline(ss, item, ',')) {
          a = std::stoul(item, nullptr, 10);
        } else {
          throw std::runtime_error(
              "LutNetworkFixedPointRegressionMultipleOutputs::read: number of items get from file is smaller than lut "
              "size");
        }
      }
    }

    auto& getPtCalibrationArray() { return ptCalibrationArray; }

  private:
    std::array<ap_ufixed<LutLayer1::input_W, input_I, AP_TRN, AP_SAT>, inputSize> inputArray;
    ap_uint<layer2_input_I> layer1Bias;

    //ptCalibrationArray size should be 1024, the LSB of the input 0.25 GeV,
    //the output is int, with range 0...511, the LSB of output 0.5 GeV
    std::array<ap_uint<9>, 1 << (output0_I + output0_F)> ptCalibrationArray;

    std::string name = "LutNetworkFixedPointRegressionMultipleOutputs";
  };

} /* namespace lutNN */

#endif /* L1Trigger_L1TMuonOverlapPhase2_LutNetworkFixedPointRegressionMultipleOutputs_h */
