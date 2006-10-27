#ifndef Framework_RunPrincipalfwd_h
#define Framework_RunPrincipalfwd_h

namespace edm {
  class RunAux;
  template <typename T> class DataBlock;
  typedef DataBlock<RunAux> RunPrincipal;
}
#endif
