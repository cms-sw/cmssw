#ifndef HeterogeneousCore_SonicCore_SonicClientTypes
#define HeterogeneousCore_SonicCore_SonicClientTypes

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//this base class exists to limit the impact of dependent scope in derived classes
template <typename InputT, typename OutputT = InputT>
class SonicClientTypes {
public:
  //typedefs for outside accessibility
  typedef InputT Input;
  typedef OutputT Output;
  //destructor
  virtual ~SonicClientTypes() = default;

  //accessors
  Input& input() { return input_; }
  const Input& input() const { return input_; }
  void setInput(const Input& inp) { input_ = inp; }
  const Output& output() const { return output_; }

protected:
  Input input_;
  Output output_;
};

#endif
