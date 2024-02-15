#ifndef PHYSICSTOOLS_TENSORFLOWAOT_MODEL_H
#define PHYSICSTOOLS_TENSORFLOWAOT_MODEL_H

/*
 * AOT model interface.
 *
 * Author: Marcel Rieger, Bogdan Wiederspan
 */

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/TensorFlowAOT/interface/Util.h"
#include "PhysicsTools/TensorFlowAOT/interface/Batching.h"

namespace tfaot {

  // model interface receiving the AOT wrapper type as a template argument
  template <class W>
  class Model {
  public:
    // constructor
    explicit Model() : wrapper_(std::make_unique<W>()) {}

    // destructor
    ~Model() { wrapper_.reset(); };

    // getter for the name
    const std::string& name() const { return wrapper_->name(); }

    // setter for the batch strategy
    void setBatchStrategy(const BatchStrategy& strategy) { batchStrategy_ = strategy; }

    // getter for the batch strategy
    const BatchStrategy& getBatchStrategy() const { return batchStrategy_; }

    // adds a new batch rule to the strategy
    void setBatchRule(size_t batchSize, const std::vector<size_t>& sizes, size_t lastPadding = 0) {
      batchStrategy_.setRule(BatchRule(batchSize, sizes, lastPadding));
    }

    // evaluates the model for multiple inputs and outputs of different types
    template <typename... Outputs, typename... Inputs>
    std::tuple<Outputs...> run(size_t batchSize, Inputs&&... inputs);

  private:
    std::unique_ptr<W> wrapper_;
    BatchStrategy batchStrategy_;

    // ensures that a batch rule exists for a certain batch size, and if not, registers a new one
    // based on the default algorithm
    const BatchRule& ensureRule(size_t batchSize);

    // reserves memory in a nested (batched) vector to accomodate the result output at an index
    template <typename T>
    void reserveOutput(size_t batchSize, size_t resultIndex, std::vector<std::vector<T>>& data) const;

    // injects data of a specific batch element into the argument data at an index
    template <typename T>
    void injectBatchInput(size_t batchSize, size_t batchIndex, size_t argIndex, const std::vector<T>& batchData);

    // extracts result data at an index into a specific batch
    template <typename T>
    void extractBatchOutput(size_t batchSize, size_t batchIndex, size_t resultIndex, std::vector<T>& batchData) const;
  };

  template <class W>
  const BatchRule& Model<W>::ensureRule(size_t batchSize) {
    // register a default rule if there is none yet for that batch size
    if (!batchStrategy_.hasRule(batchSize)) {
      batchStrategy_.setDefaultRule(batchSize, wrapper_->batchSizes());
    }
    return batchStrategy_.getRule(batchSize);
  }

  template <class W>
  template <typename T>
  void Model<W>::reserveOutput(size_t batchSize, size_t resultIndex, std::vector<std::vector<T>>& data) const {
    data.resize(batchSize, std::vector<T>(wrapper_->resultCountNoBatch(resultIndex)));
  }

  template <class W>
  template <typename T>
  void Model<W>::injectBatchInput(size_t batchSize,
                                  size_t batchIndex,
                                  size_t argIndex,
                                  const std::vector<T>& batchData) {
    size_t count = wrapper_->argCountNoBatch(argIndex);
    if (batchData.size() != count) {
      throw cms::Exception("InputMismatch")
          << "model '" << name() << "' received " << batchData.size() << " elements for argument " << argIndex
          << ", but " << count << " are expected";
    }
    T* argPtr = wrapper_->template argData<T>(batchSize, argIndex) + batchIndex * count;
    auto beg = batchData.cbegin();
    std::copy(beg, beg + count, argPtr);
  }

  template <class W>
  template <typename T>
  void Model<W>::extractBatchOutput(size_t batchSize,
                                    size_t batchIndex,
                                    size_t resultIndex,
                                    std::vector<T>& batchData) const {
    size_t count = wrapper_->resultCountNoBatch(resultIndex);
    const T* resPtr = wrapper_->template resultData<T>(batchSize, resultIndex) + batchIndex * count;
    batchData.assign(resPtr, resPtr + count);
  }

  template <class W>
  template <typename... Outputs, typename... Inputs>
  std::tuple<Outputs...> Model<W>::run(size_t batchSize, Inputs&&... inputs) {
    // check number of inputs
    size_t nInputs = sizeof...(Inputs);
    if (nInputs != wrapper_->nArgs()) {
      throw cms::Exception("InputMismatch")
          << "model '" << name() << "' received " << nInputs << " inputs, but " << wrapper_->nArgs() << " are expected";
    }

    // check number of outputs
    size_t nOutputs = sizeof...(Outputs);
    if (nOutputs != wrapper_->nResults()) {
      throw cms::Exception("OutputMismatch") << "requested " << nOutputs << " from model '" << name() << "', but "
                                             << wrapper_->nResults() << " are provided";
    }

    // get the corresponding batch rule
    const BatchRule& rule = ensureRule(batchSize);

    // create a callback that invokes lambdas over all outputs with normal indices
    auto forEachOutput = createIndexLooper<sizeof...(Outputs)>();

    // reserve output arrays
    std::tuple<Outputs...> outputs;
    forEachOutput([&](auto resultIndex) { reserveOutput(batchSize, resultIndex, std::get<resultIndex>(outputs)); });

    // loop over particular batch sizes, copy input, evaluate and compose the output
    size_t batchOffset = 0;
    size_t nSizes = rule.nSizes();
    for (size_t i = 0; i < nSizes; i++) {
      // get actual model batch size and optional padding
      size_t bs = rule.getSize(i);
      size_t padding = (i == nSizes - 1) ? rule.getLastPadding() : 0;

      // fill inputs separately per batch element
      for (size_t batchIndex = 0; batchIndex < bs - padding; batchIndex++) {
        size_t argIndex = 0;
        ([&] { injectBatchInput(bs, batchIndex, argIndex++, inputs[batchOffset + batchIndex]); }(), ...);
      }

      // model evaluation
      wrapper_->run(bs);

      // fill outputs separately per batch element
      for (size_t batchIndex = 0; batchIndex < bs - padding; batchIndex++) {
        forEachOutput([&](auto resultIndex) {
          extractBatchOutput(bs, batchIndex, resultIndex, std::get<resultIndex>(outputs)[batchOffset + batchIndex]);
        });
      }

      batchOffset += bs;
    }

    return outputs;
  }

}  // namespace tfaot

#endif  // PHYSICSTOOLS_TENSORFLOWAOT_MODEL_H
