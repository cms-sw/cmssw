#ifndef Framework_LuminosityBlockPrincipalfwd_h
#define Framework_LuminosityBlockPrincipalfwd_h

namespace edm {
  class LuminosityBlockAux;
  template <typename T> class DataBlock;
  typedef DataBlock<LuminosityBlockAux> LuminosityBlockPrincipal;
}
#endif
