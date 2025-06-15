#ifndef PHYSICS_TOOLS__PYTORCH__INTERFACE__CONFIG_H_
#define PHYSICS_TOOLS__PYTORCH__INTERFACE__CONFIG_H_

#include <torch/script.h>
#include <torch/torch.h>

#include "FWCore/Utilities/interface/Exception.h"

namespace cms::torch {

  /** 
   * The following `constexpr` constants are aliases for various PyTorch data types.
   * 
   * Primarily used for specifying tensor types when working with PyTorch 
   * tensors in the CMS environment.
   */
  constexpr auto Byte = ::torch::kByte;     /**< unsigned 8-bit integer type. */
  constexpr auto Char = ::torch::kChar;     /**< signed 8-bit integer type. */
  constexpr auto Short = ::torch::kShort;   /**< signed 16-bit integer type. */
  constexpr auto Int = ::torch::kInt;       /**< signed 32-bit integer type. */
  constexpr auto Long = ::torch::kLong;     /**< signed 64-bit integer type. */
  constexpr auto UInt16 = ::torch::kUInt16; /**< unsigned 16-bit integer type. */
  constexpr auto UInt32 = ::torch::kUInt32; /**< unsigned 32-bit integer type. */
  constexpr auto UInt64 = ::torch::kUInt64; /**< unsigned 64-bit integer type. */
  constexpr auto Half = ::torch::kHalf;     /**< 16-bit floating point type. */
  constexpr auto Float = ::torch::kFloat;   /**< 32-bit floating point type. */
  constexpr auto Double = ::torch::kDouble; /**< 64-bit floating point type. */

  /**
   * @brief Loads a TorchScript model.
   *
   * This function wraps `torch::jit::load` to load a TorchScript model from a specified path.
   * In case of failure, it throws a CMS-specific `cms::Exception` with detailed context and error information.
   *
   * @param model_path The file path to the TorchScript model (.pt file).
   * @return A loaded `torch::jit::script::Module` ready for inference or further manipulation.
   *
   * @throws cms::Exception If loading fails due to file issues, format errors, or internal TorchScript problems.
   *
   * @note This function is intended for model loading in CMSSW environments, providing
   *       integration with the framework's exception handling and logging facilities.
   */
  inline ::torch::jit::script::Module load(const std::string &model_path) {
    try {
      return ::torch::jit::load(model_path);
    } catch (const c10::Error &e) {
      cms::Exception ex("ModelLoadingError");
      ex.addContext("Calling cms::torch::load(const std::string&)");
      ex.addAdditionalInfo("Error loading the model: " + std::string(e.what()));
      throw ex;
    }
  }

}  // namespace cms::torch

#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__CONFIG_H_
