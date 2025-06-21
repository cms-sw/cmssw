#ifndef PHYSICS_TOOLS__PYTORCH__INTERFACE__MODEL_H_
#define PHYSICS_TOOLS__PYTORCH__INTERFACE__MODEL_H_

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Converter.h"

namespace cms::torch::alpaka {

  /**
   * @class CompilationType
   * @brief Specifies the type of compilation used for the model.
   */
  enum class CompilationType {
    kJustInTime, /**< JIT compilation, load and compile at runtime from TorchScript */
    kAheadOfTime /**< AOT compilation, load precompiled shared library at runtime */
  };

  /**
   * @class Model
   * @brief Model base class.
   */
  template <CompilationType>
  class Model;

  /**
   * @class Model
   * @brief AOT Model specific implementation.
   *
   * Interface for loading and running models with AOT compilation models.
   */
  template <>
  class Model<CompilationType::kAheadOfTime> {
  public:
    explicit Model(const std::string &model_path) : loader_(model_path), runner_(loader_.get_runner()) {}

    /**
     * @brief Torch fallback for testing purposes.
     * @param inputs input tensors
     * @return output tensors
     */
    auto forward(std::vector<::torch::Tensor> &inputs) const { return runner_->run(inputs); }

    /**
     * @brief Torch portable inference with SoA buffers without explicit copies.
     * @param metadata Metadata specyfies how memory blob is organized and can be accessed.
     */
    template <typename InMemLayout, typename OutMemLayout>
    void forward(const ModelMetadata<InMemLayout, OutMemLayout> &metadata) const {
      std::vector<at::Tensor> inputs = Converter::convert_input_tensor(metadata, device_);

      if (metadata.multi_output) {
        auto out = runner_->run(inputs);
        Converter::convert_output(out, metadata, device_);
      } else {
        Converter::convert_output(metadata, device_) = runner_->run(inputs)[0];
      }
    }

    /**
     * @brief Change model metadata `device_` to a specified device.
     *
     * Utility function updates model metadata `device_` using Alpaka-aware 
     * device or queue objects. It is equivalent in purpose to `torch::to(device)`, 
     * but supports Alpaka's queue or device objects directly.
     * 
     * @note This function does not change the device of the model itself!
     *       It only updates the internal device metadata used for inference.
     *       Since AOT models are compiled for a specific arch and should loaded from a .so 
     *
     * @tparam T Type of the input argument. Can be an Alpaka device, Alpaka queue, or a `torch::Device`.
     * @param obj The object specifying the target device. Supported types:
     * - `alpaka::Device`: Direct specification of the Alpaka device.
     * - `alpaka::Queue`: Infers device from the queue.
     * - `torch::Device`: Standard PyTorch device.
     *
     * @note If the device is already set to the specified device, the function does nothing.
     * @throws A static assertion failure at compile-time if an unsupported type is passed.
     */
    template <typename T>
    void to(const T &obj) const {
      auto device = ::torch::Device(::torch::kCPU, 0);
      if constexpr (::alpaka::isDevice<T> || ::alpaka::isQueue<T>) {
        device = cms::torch::alpaka::device(obj);
      } else if constexpr (std::is_same_v<T, ::torch::Device>) {
        device = obj;
      } else {
        static_assert(false_value<T>, "Unsupported type passed -> to(const T&)");
      }

      if (device == device_)
        return;
      device_ = device;
    }

    /**
     * @brief Torch portable inference with SoA buffers without explicit copies.
     * @return Current device binded to the model.
     */
    ::torch::Device device() const { return device_; }

  private:
    mutable ::torch::Device device_ = ::torch::Device(::torch::kCPU, 0);    /**< Device metadata of the model */
    mutable ::torch::inductor::AOTIModelPackageLoader loader_;              /**< AOT model package loader */
    mutable ::torch::inductor::AOTIModelContainerRunner *runner_ = nullptr; /**< AOT model container runner */
  };

  /**
   * @class Model
   * @brief JIT Model specific implementation.
   *
   * Interface for loading and running models with JIT compilation models.
   */
  template <>
  class Model<CompilationType::kJustInTime> {
  public:
    Model(const std::string &model_path) : model_(cms::torch::load(model_path)) {}

    /**
     * @brief Moves the model to a specified device.
     *
     * Utility function updates the internal device of the model, using Alpaka-aware 
     * device or queue objects. It is equivalent in purpose to `torch::to(device)`, 
     * but supports Alpaka's queue or device objects directly.
     *
     * @tparam T Type of the input argument. Can be an Alpaka device, Alpaka queue, or a `torch::Device`.
     * @param obj The object specifying the target device. Supported types:
     * - `alpaka::Device`: Direct specification of the Alpaka device.
     * - `alpaka::Queue`: Infers device from the queue.
     * - `torch::Device`: Standard PyTorch device.
     *
     * @note If the device is already set to the specified device, the function does nothing.
     * @throws A static assertion failure at compile-time if an unsupported type is passed.
     */
    template <typename T>
    void to(const T &obj) const {
      auto device = ::torch::Device(::torch::kCPU, 0);
      if constexpr (::alpaka::isDevice<T> || ::alpaka::isQueue<T>) {
        device = cms::torch::alpaka::device(obj);
      } else if constexpr (std::is_same_v<T, ::torch::Device>) {
        device = obj;
      } else {
        static_assert(false_value<T>, "Unsupported type passed -> to(const T&)");
      }

      if (device == device_)
        return;

      device_ = device;
      model_.to(device_, true);
    }

    /**
     * @brief Torch fallback for testing purposes.
     * @param inputs input tensors
     * @return output tensors
     */
    auto forward(std::vector<::torch::IValue> &inputs) const { return model_.forward(inputs); }

    /**
     * @brief Torch portable inference with SoA buffers without explicit copies.
     * @param metadata Metadata specyfies how memory blob is organized and can be accessed.
     */
    template <typename InMemLayout, typename OutMemLayout>
    void forward(const ModelMetadata<InMemLayout, OutMemLayout> &metadata) const {
      auto input_tensor = Converter::convert_input(metadata, device_);
      // TODO: think about support for multi-output models (without temporary mem copy)
      Converter::convert_output(metadata, device_) = model_.forward(input_tensor).toTensor();
    };

    /**
     * @brief Torch portable inference with SoA buffers without explicit copies.
     * @return Current device binded to the model.
     */
    ::torch::Device device() const { return device_; }

  private:
    mutable ::torch::jit::script::Module model_;                         /**< JIT model */
    mutable ::torch::Device device_ = ::torch::Device(::torch::kCPU, 0); /**< Device binded to the model */
  };

}  // namespace cms::torch::alpaka

#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__MODEL_H_
