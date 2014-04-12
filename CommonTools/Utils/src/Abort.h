#ifndef Parser_Abort_h
#define Parser_Abort_h
/* \class reco::parser::Abort
 *
 * Numerical expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */

namespace reco {
  namespace parser {
    struct Abort {
      void operator()( const char *, const char * ) const;
    };
  }
}

#endif
