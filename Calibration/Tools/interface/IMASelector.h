#ifndef UtilAlgos_IMASelector_h
#define UtilAlgos_IMASelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"


struct IMASelector {
  IMASelector( double ESCOPinMin, double ESeedOPoutMin, double PinMPoutOPinMin,
               double ESCOPinMax, double ESeedOPoutMax, double PinMPoutOPinMax  ) : 
    ESCOPinMin_ (ESCOPinMin),
    ESeedOPoutMin_ (ESeedOPoutMin),
    PinMPoutOPinMin_ (PinMPoutOPinMin),
    ESCOPinMax_ (ESCOPinMax),
    ESeedOPoutMax_ (ESeedOPoutMax),
    PinMPoutOPinMax_ (PinMPoutOPinMax) {}
  template<typename T>
  bool operator()( const T & t ) const { 
    double pin = t.trackMomentumAtVtx ().R () ;
    double poMpiOpi = (pin - t.trackMomentumOut ().R ()) / pin ;
//    double ESC = t.energy () ;     
    double EseedOPout = t.eSeedClusterOverPout () ;
    double EoPin = t.eSuperClusterOverP () ;
    return (poMpiOpi > PinMPoutOPinMin_ && poMpiOpi < PinMPoutOPinMax_ &
            EseedOPout > ESeedOPoutMin_ && EseedOPout < ESeedOPoutMax_ &
            EoPin > ESCOPinMin_ && EoPin < ESCOPinMax_) ;
  }
private:
  double ESCOPinMin_, ESeedOPoutMin_, PinMPoutOPinMin_ ;
  double ESCOPinMax_, ESeedOPoutMax_, PinMPoutOPinMax_ ;
};


namespace reco { 
  namespace modules { 
    template<> 
    struct ParameterAdapter<IMASelector> { 
      static IMASelector make(const edm::ParameterSet & cfg) { 
        return IMASelector(cfg.getParameter<double>("ESCOPinMin"), 
                           cfg.getParameter<double>("ESeedOPoutMin"), 
                           cfg.getParameter<double>("PinMPoutOPinMin"), 
                           cfg.getParameter<double>("ESCOPinMax"), 
                           cfg.getParameter<double>("ESeedOPoutMax"), 
                           cfg.getParameter<double>("PinMPoutOPinMax") ); 
      } 
    }; 
  } 
} 

#endif
