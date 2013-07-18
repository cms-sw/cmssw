// $Id: EgammaTrigger.cc,v 1.2 2005/12/07 09:26:49 llista Exp $
#include "DataFormats/EgammaReco/interface/EgammaTrigger.h"
using namespace reco;

EgammaTrigger::EgammaTrigger( bool matchL1Single, bool matchL1Double, 
                              bool matchL1RelaxedDouble, bool matchIsolatedL1 ) :
  l1word_( 0 ) {
  if ( matchL1Single        ) l1word_ |= 1 << egamma::L1Single;
  if ( matchL1Double        ) l1word_ |= 1 << egamma::L1Double;
  if ( matchL1RelaxedDouble ) l1word_ |= 1 << egamma::L1RelaxedDouble;
  if ( matchIsolatedL1      ) l1word_ |= 1 << egamma::IsolatedL1;
}
