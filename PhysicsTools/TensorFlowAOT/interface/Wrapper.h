#ifndef PHYSICSTOOLS_TENSORFLOWAOT_WRAPPER_H
#define PHYSICSTOOLS_TENSORFLOWAOT_WRAPPER_H

/*
 * AOT wrapper interface for interacting with xla functions compiled for different batch sizes.
 *
 * Author: Marcel Rieger, Bogdan Wiederspan
 */

#include <map>
#include <algorithm>

#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/TensorFlowAOT/interface/Util.h"

namespace tfaot {

  // object that wraps multiple variants of the same xla function, but each compiled for a different
  // batch size, and providing access to arguments (inputs) and results (outputs) by index
  class Wrapper {
  public:
    // constructor
    explicit Wrapper(const std::string& name)
        : name_(name), allocMode_(AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS) {}

    // disable copy constructor
    Wrapper(const Wrapper&) = delete;

    // disable assignment operator
    Wrapper& operator=(const Wrapper&) = delete;

    // disable move operator
    Wrapper& operator=(Wrapper&&) = delete;

    // destructor
    virtual ~Wrapper() = default;

    // getter for the name
    const std::string& name() const { return name_; }

    // getter for the allocation mode
    AllocMode allocMode() const { return allocMode_; }

    // getter for the compiled batch sizes
    virtual const std::vector<size_t>& batchSizes() const = 0;

    // returns the number of compiled batch sizes
    size_t nBatchSizes() const { return batchSizes().size(); }

    // returns whether a compiled xla function exists for a certain batch size
    // (batchSizes is sorted by default)
    bool hasBatchSize(size_t batchSize) const {
      const auto& bs = batchSizes();
      return std::binary_search(bs.begin(), bs.end(), batchSize);
    }

    // getter for the number of arguments (inputs)
    virtual size_t nArgs() const = 0;

    // number of elements in arguments per batch size
    virtual const std::map<size_t, std::vector<size_t>>& argCounts() const = 0;

    // number of elements in arguments, divided by batch size
    virtual const std::vector<size_t>& argCountsNoBatch() const = 0;

    // returns a pointer to the argument data at a certain index for the xla function at some batch
    // size
    template <typename T>
    T* argData(size_t batchSize, size_t argIndex);

    // returns a const pointer to the argument data at a certain index for the xla function at some
    // batch size
    template <typename T>
    const T* argData(size_t batchSize, size_t argIndex) const;

    // returns the total number of values in the argument data at a certain index for the xla
    // function at some batch size
    int argCount(size_t batchSize, size_t argIndex) const;

    // returns the number of values excluding the leading batch axis in the argument data at a
    // certain index for the xla function at some batch size
    int argCountNoBatch(size_t argIndex) const;

    // getter for the number of results (outputs)
    virtual size_t nResults() const = 0;

    // number of elements in results per batch size
    virtual const std::map<size_t, std::vector<size_t>>& resultCounts() const = 0;

    // number of elements in results, divided by batch size
    virtual const std::vector<size_t>& resultCountsNoBatch() const = 0;

    // returns a pointer to the result data at a certain index for the xla function at some batch
    // size
    template <typename T>
    T* resultData(size_t batchSize, size_t resultIndex);

    // returns a const pointer to the result data at a certain index for the xla function at some
    // batch size
    template <typename T>
    const T* resultData(size_t batchSize, size_t resultIndex) const;

    // returns the total number of values in the result data at a certain index for the xla function
    // at some batch size
    int resultCount(size_t batchSize, size_t resultIndex) const;

    // returns the number of values excluding the leading batch axis in the result data at a
    // certain index for the xla function at some batch size
    int resultCountNoBatch(size_t resultIndex) const;

    // evaluates the xla function at some batch size and returns whether the call succeeded
    virtual bool runSilent(size_t batchSize) = 0;

    // evaluates the xla function at some batch size and throws an exception in case of an error
    void run(size_t batchSize);

  protected:
    // throws an exception for the case where an unknown batch size was requested
    void unknownBatchSize(size_t batchSize, const std::string& method) const {
      throw cms::Exception("UnknownBatchSize")
          << "batch size " << batchSize << " not known to model '" << name_ << "' in '" << method << "'";
    }

    // throws an exception for the case where an unknown argument index was requested
    void unknownArgument(size_t argIndex, const std::string& method) const {
      throw cms::Exception("UnknownArgument")
          << "argument " << argIndex << " not known to model '" << name_ << "' in '" << method << "'";
    }

    // throws an exception for the case where an unknown result index was requested
    void unknownResult(size_t resultIndex, const std::string& method) const {
      throw cms::Exception("UnknownResult")
          << "result " << resultIndex << " not known to model '" << name_ << "' in '" << method << "'";
    }

  private:
    std::string name_;
    AllocMode allocMode_;
  };

}  // namespace tfaot

#endif  // PHYSICSTOOLS_TENSORFLOWAOT_WRAPPER_H
