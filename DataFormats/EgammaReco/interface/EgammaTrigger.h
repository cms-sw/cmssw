#ifndef EgammaReco_EgammaTrigger_h
#define EgammaReco_EgammaTrigger_h
/** \class reco::EgammaTrigger EgammaTrigger.h DataFormats/EgammaReco/interface/EgammaTrigger.h
 *  
 * Egamma trigger bit set
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: EgammaTrigger.h,v 1.2 2006/04/20 10:13:53 llista Exp $
 *
 */
#include "DataFormats/EgammaReco/interface/EgammaTriggerFwd.h"

namespace reco {

  class SuperCluster;

  namespace egamma {
    enum { L1Single = 0, L1Double, L1RelaxedDouble, IsolatedL1 };
    template<unsigned char L1>
    struct mask { enum { value = mask< L1 - 1 >::value << 1 }; };
    template<>
    struct mask<0> { enum { value = 1 }; };
  }

  class EgammaTrigger {
  public:
    /// default constructor
    EgammaTrigger() : l1word_( 0 ) { }

    /// constructor from boolean values
    EgammaTrigger( bool, bool, bool, bool ); 

    /// returns the trigger bit at positon L1 (L1 = 0, ..., 3)
    /// L1 couls be one of the enumerator defined in egamma namespace
    template<unsigned char L1>
    bool match() {
      return l1word_ & egamma::mask<L1>::value;
    }

    /// sets to 1 the trigger bit at positon L1 (L1 = 0, ..., 3)
    /// L1 couls be one of the enumerator defined in egamma namespace
    template<unsigned char L1>
    void set() {
      l1word_ |= egamma::mask<L1>::value;
    }

    /// return the trigger work
    unsigned char l1word() const { return l1word_; }

  private:
    /// trigger work (packed). Only 4 bits are used.
    unsigned char l1word_;
  };
}

#endif
