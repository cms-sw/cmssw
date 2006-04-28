#ifndef VertexReco_print_h
#define VertexReco_print_h
/* Vertex print utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: print.h,v 1.2 2006/04/21 07:17:24 llista Exp $
 * 
 */
#include <string>

namespace reco {

   class Vertex;
   /// Vertex print utility

   //   std::string print( const Vertex &, verbosity = concise );
   std::string print( const Vertex & );
}

#endif
