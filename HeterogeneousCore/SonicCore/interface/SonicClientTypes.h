#ifndef HeterogeneousCore_SonicCore_SonicClientTypes
#define HeterogeneousCore_SonicCore_SonicClientTypes

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
  const Output& output() const { return output_; }

protected:
  Input input_;
  Output output_;
};

#endif
