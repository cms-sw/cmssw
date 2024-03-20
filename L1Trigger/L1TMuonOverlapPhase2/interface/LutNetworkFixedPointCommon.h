/*
 * LutNetworkCommon.h
 *
 *  Created on: Jan 13, 2023
 *      Author: kbunkow
 */

#ifndef INTERFACE_LUTNETWORKFIXEDPOINTCOMMON_H_
#define INTERFACE_LUTNETWORKFIXEDPOINTCOMMON_H_

#include <ap_fixed.h>
#include <ap_int.h>
#include <array>
#include <vector>
#include <limits>

#include <boost/property_tree/ptree.hpp>

namespace lutNN {

  template <int W, int I>
  const ap_ufixed<W, I> max_ap_ufixed() {
    static_assert(I < 64, "this max_ap_ufixed works only for I < 64");
    return ap_ufixed<W, I, AP_RND, AP_SAT>(std::numeric_limits<uint64_t>::max());
    //AP_SAT Saturate the value to the maximum value in case of overflow
  }

  template <int W, int I>
  const ap_fixed<W, I> max_ap_fixed() {
    static_assert(I < 64, "this max_ap_ufixed works only for I < 64");
    return ap_fixed<W, I, AP_RND, AP_SAT>(std::numeric_limits<uint64_t>::max());
    //AP_SAT Saturate the value to the maximum value in case of overflow
  }

#define PUT_VAR(tree, keyPath, var) tree.put((keyPath) + "." + #var, (var));

#define CHECK_VAR(tree, keyPath, var)                 \
  if ((var) != tree.get<int>((keyPath) + "." + #var)) \
    throw std::runtime_error((keyPath) + "." + #var + " has different value in the file then given");

  /*template<typename arrayType>
void saveLut(boost::property_tree::ptree& tree, std::string keyPath, const arrayType& lut) {
    std::ostringstream ostr;
    for(auto& a : lut) {
        ostr<<std::hex<<a.bits_to_uint64()<<", ";
    }

    tree.put(keyPath + ".lutArray." + , ostr.str());
}*/

  class LutNetworkFixedPointRegressionBase {
  public:
    virtual ~LutNetworkFixedPointRegressionBase(){};

    virtual void save(const std::string& filename) = 0;
    virtual void load(const std::string& filename) = 0;

    /*    template <typename InputType>
    void run(std::vector<InputType>& inputs, InputType noHitVal, std::vector<double>& nnResult) = 0;*/

    virtual void run(std::vector<float>& inputs, float noHitVal, std::vector<double>& nnResult) = 0;

    //pt in the hardware scale, ptGeV = (ptHw -1) / 2
    virtual int getCalibratedHwPt() = 0;
  };

}  // namespace lutNN

#endif /* INTERFACE_LUTNETWORKFIXEDPOINTCOMMON_H_ */
