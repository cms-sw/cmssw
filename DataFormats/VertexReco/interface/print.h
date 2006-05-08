#ifndef VertexReco_print_h
#define VertexReco_print_h
/* Vertex print utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: VertexPrint.h,v 1.1 2006/04/28 16:55:09 vanlaer Exp $
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
