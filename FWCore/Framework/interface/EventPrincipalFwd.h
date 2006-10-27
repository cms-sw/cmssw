#ifndef Framework_EventPrincipalFwd_h
#define Framework_EventPrincipalFwd_h

namespace edm {
  class EventAux;
  template <typename T> class DataBlock;
  typedef DataBlock<EventAux> EventPrincipal;
}
#endif
