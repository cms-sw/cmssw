#ifndef VertexReco_print_h
#define VertexReco_print_h
/* Vertex print utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: print.h,v 1.1 2006/05/08 07:59:08 llista Exp $
 * 
 */
#include "FWCore/Utilities/interface/Verbosity.h"
#include <string>

namespace reco {
   class Vertex;
   /// Vertex print utility
   std::string print( const Vertex &, edm::Verbosity = edm::Concise );
}

#endif
