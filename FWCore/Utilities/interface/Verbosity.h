#ifndef Utilities_Verbosity_h
#define Utilities_Verbosity_h
// A first attempt to define a descriptive enumenration for verbosity.
namespace edm {
  enum Verbosity {
    Silent=0,
    Concise=2,
    Normal=5,
    Detailed=10
  };
}
#endif
