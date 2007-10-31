#ifndef Parser_Combiner_h
#define Parser_Combiner_h
/* \class reco::parser::Combiner
 *
 * Combiner enumerator
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
namespace reco {
  namespace parser {    
    enum Combiner { kAnd, kOr, kNot };
  }
}

#endif
