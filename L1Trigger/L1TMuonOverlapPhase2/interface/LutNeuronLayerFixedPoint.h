//============================================================================
// Name        : LutNeuronLayerFixedPoint.h
// Author      : Karol Bunkowski
// Created on: Mar 12, 2021
// Version     :
// Copyright   : All right reserved
// Description : Fixed point LUT layer
//============================================================================

#ifndef L1Trigger_L1TMuonOverlapPhase2_LutNeuronlayerFixedPoint_h
#define L1Trigger_L1TMuonOverlapPhase2_LutNeuronlayerFixedPoint_h

#include <ap_fixed.h>
#include <ap_int.h>
#include <array>
#include <limits>
#include <iomanip>
#include <cassert>

#include <boost/property_tree/ptree.hpp>

#include "L1Trigger/L1TMuonOverlapPhase2/interface/LutNetworkFixedPointCommon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace lutNN {
  // constexpr for ceil(log2) from stackoverflow
  constexpr size_t floorlog2(size_t i) { return i == 1 ? 0 : 1 + floorlog2(i >> 1); }
  constexpr size_t ceillog2(size_t i) { return i == 1 ? 0 : floorlog2(i - 1) + 1; }

  template <int input_I, int input_F, std::size_t inputSize, int lut_I, int lut_F, int neurons, int output_I, int output_F>
  class LutNeuronLayerFixedPoint {
  public:
    static constexpr int input_W = input_I + input_F;
    static constexpr int lut_W = lut_I + lut_F;

    //the lut out values sum
    //static const int lutOutSum_I = lut_I + ceil(log2(inputSize)); //ceil(log2(inputSize)) is not constexpr which makes issue for code-checks
    static_assert(inputSize > 0);
    static constexpr int lutOutSum_I = lut_I + ceillog2(inputSize);
    static constexpr int lutOutSum_W = lutOutSum_I + output_F;

    static constexpr int output_W = output_I + output_F;

    //static_assert( (1<<input_I) <= lutSize);
    static constexpr std::size_t lutSize = 1 << input_I;

    typedef std::array<ap_ufixed<input_W, input_I, AP_TRN, AP_SAT>, inputSize> inputArrayType;

    //the lutSumArrayType lutOutSum_I is such that no overflow in summation is possible
    typedef std::array<ap_fixed<lutOutSum_W, lutOutSum_I>, neurons> lutSumArrayType;

    LutNeuronLayerFixedPoint() {  //FIXME initialise name(name)
      //static_assert(lut_I <= (output_I - ceil(log2(inputSize)) ), "not correct lut_I, output_I  and inputSize"); //TODO

      LogTrace("l1tOmtfEventPrint") << name << "\n     input_I " << std::setw(2) << input_I << "     input_F "
                                    << std::setw(2) << input_F << "     input_W " << std::setw(2) << input_W
                                    << " inputSize " << std::setw(2) << inputSize << "\n       lut_I " << std::setw(2)
                                    << lut_I << "       lut_F " << std::setw(2) << lut_F << "       lut_W "
                                    << std::setw(2) << lut_W << "   lutSize " << std::setw(2) << lutSize
                                    << "\n lutOutSum_I " << std::setw(2) << lutOutSum_I << " lutOutSum_F "
                                    << std::setw(2) << lut_F << " lutOutSum_W " << std::setw(2) << lutOutSum_W
                                    << "\n    output_I " << std::setw(2) << output_I << "    output_F " << std::setw(2)
                                    << output_F << "    output_W " << std::setw(2) << output_W << "\n neurons "
                                    << std::setw(2) << neurons << "\n outOffset " << outOffset << " = " << std::hex
                                    << outOffset << " width " << outOffset.width << std::dec << std::endl;
    }

    virtual ~LutNeuronLayerFixedPoint() {}

    void setName(std::string name) { this->name = name; }

    void save(boost::property_tree::ptree& tree, std::string keyPath) {
      PUT_VAR(tree, keyPath + "." + name, input_I)
      PUT_VAR(tree, keyPath + "." + name, input_F)
      PUT_VAR(tree, keyPath + "." + name, inputSize)
      PUT_VAR(tree, keyPath + "." + name, lut_I)
      PUT_VAR(tree, keyPath + "." + name, lut_F)
      PUT_VAR(tree, keyPath + "." + name, neurons)
      PUT_VAR(tree, keyPath + "." + name, output_I)
      PUT_VAR(tree, keyPath + "." + name, output_F)

      for (unsigned int iInput = 0; iInput < lutArray.size(); iInput++) {
        for (unsigned int iNeuron = 0; iNeuron < lutArray[iInput].size(); iNeuron++) {
          auto& lut = lutArray.at(iInput).at(iNeuron);
          std::ostringstream ostr;
          for (auto& a : lut) {
            ostr << std::fixed << std::setprecision(19) << a.to_float() << ", ";
          }
          tree.put(keyPath + "." + name + ".lutArray." + std::to_string(iInput) + "." + std::to_string(iNeuron),
                   ostr.str());
        }
      }
    }

    void load(boost::property_tree::ptree& tree, std::string keyPath) {
      CHECK_VAR(tree, keyPath + "." + name, input_I)
      CHECK_VAR(tree, keyPath + "." + name, input_F)
      CHECK_VAR(tree, keyPath + "." + name, inputSize)
      CHECK_VAR(tree, keyPath + "." + name, lut_I)
      CHECK_VAR(tree, keyPath + "." + name, lut_F)
      CHECK_VAR(tree, keyPath + "." + name, neurons)
      CHECK_VAR(tree, keyPath + "." + name, output_I)
      CHECK_VAR(tree, keyPath + "." + name, output_F)

      for (unsigned int iInput = 0; iInput < lutArray.size(); iInput++) {
        for (unsigned int iNeuron = 0; iNeuron < lutArray[iInput].size(); iNeuron++) {
          auto& lut = lutArray.at(iInput).at(iNeuron);
          auto str = tree.get<std::string>(keyPath + "." + name + ".lutArray." + std::to_string(iInput) + "." +
                                           std::to_string(iNeuron));

          std::stringstream ss(str);
          std::string item;

          for (auto& a : lut) {
            if (std::getline(ss, item, ',')) {
              a = std::stof(item, nullptr);
            } else {
              throw std::runtime_error(
                  "LutNeuronLayerFixedPoint::read: number of items get from file is smaller than lut size");
            }
          }
        }
      }
    }

    lutSumArrayType& runWithInterpolation(const inputArrayType& inputArray) {
      for (unsigned int iNeuron = 0; iNeuron < lutOutSumArray.size(); iNeuron++) {
        auto& lutOutSum = lutOutSumArray.at(iNeuron);
        lutOutSum = 0;
        for (unsigned int iInput = 0; iInput < inputArray.size(); iInput++) {
          auto address = inputArray.at(iInput).to_uint();  //address in principle is unsigned
          auto& lut = lutArray.at(iInput).at(iNeuron);

          auto addresPlus1 = address;
          //there is a protection is in getOutWithOffset()
          //but it does not include the bias node, and the noHit value, so the address == lut.size()-1 is still possible
          //so this protection is needed here
          if (addresPlus1 < (lut.size() - 1)) {
            addresPlus1 = address + 1;
          }

          auto derivative = lut.at(addresPlus1) - lut.at(address);  // must be signed

          //N.B. the address and fractionalPart is the same for all neurons, what matters for the firmware
          ap_ufixed<input_W - input_I, 0> fractionalPart = inputArray.at(iInput);

          //the W of derivative is (lut_W + 1)
          //the W of (fractionalPart * derivative) is (input_W - input_I) + (lut_W + 1), the F is input_W-input_I + lut_F
          //so the F of result is the F of the multiplication, so input_W-input_I + lut_F
          //the F of lutOutSum is lut_F,so it is implicitly truncated
          auto result = lut.at(address) + fractionalPart * derivative;
          lutOutSum += result;
        }
      }

      return lutOutSumArray;
    }

    //Output without offset
    lutSumArrayType& getLutOutSum() { return lutOutSumArray; }

    //converts the output values from signed to unsigned by adding the offset = 1 << (output_I-1)
    //these values can be then directly used as inputs of the next LUT layer
    auto& getOutWithOffset() {
      for (unsigned int iOut = 0; iOut < lutOutSumArray.size(); iOut++) {
        //the integer part of the outputArray is usually smaller than inte case of the lutOutSumArray
        //therefore the outputArray has AP_SAT to avoid overflow
        outputArray[iOut] = lutOutSumArray[iOut] + outOffset;
        //in the next layer, in runWithInterpolation the addresPlus1 is calculated, so outputArray[iOut] must be smaller the max_ap_fixed<output_W, output_I>() -1
        //std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<name<<" "<<"iOut "<<iOut<<" lutOutSumArray[i] "<<lutOutSumArray[iOut]<<" outputArray[i] "<<outputArray[iOut]<<std::endl;
        if (outputArray[iOut] > (max_ap_ufixed<output_W, output_I>() - 1)) {
          edm::LogVerbatim("l1tOmtfEventPrint")
              << __FUNCTION__ << ":" << __LINE__ << " " << name << " "
              << "iOut " << iOut << " lutOutSumArray[i] " << lutOutSumArray[iOut] << " outputArray[i] "
              << outputArray[iOut] << " max_ap_ufixed " << max_ap_ufixed<output_W, output_I>() << " <<<<<<<<<<<<<<<"
              << std::endl;
          outputArray[iOut] = max_ap_ufixed<output_W, output_I>() - 1;
        }
      }

      return outputArray;
    }

    auto getName() { return name; }

  private:
    lutSumArrayType lutOutSumArray;
    std::array<ap_ufixed<output_W, output_I, AP_TRN, AP_SAT>, neurons> outputArray;

    ap_uint<output_I> outOffset = 1 << (output_I - 1);

    //[inputNum][outputNum =  neuronNum][address]
    std::array<std::array<std::array<ap_fixed<lut_W, lut_I>, lutSize>, neurons>, inputSize> lutArray;

    std::string name;
  };

} /* namespace lutNN */

#endif /* L1Trigger_L1TMuonOverlapPhase2_LutNeuronlayerFixedPoint_h */
