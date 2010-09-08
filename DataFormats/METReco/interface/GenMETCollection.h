#ifndef METRECO_GenMETCOLLECTION_H
#define METRECO_GenMETCOLLECTION_H

/** \class GenMETCollection
 *
 * \short Collection of Gen MET
 *
 * GenMETCollection is a collection of GenMET objects
 *
 * \author Mike Schmitt, UFlorida
 *
 * \version   1st Version August 4, 2005.
 *
 ************************************************************/

#include <vector>
#include "DataFormats/METReco/interface/GenMETFwd.h" 

namespace reco
{
  typedef std::vector<reco::GenMET> GenMETCollection;
}  
#endif // METRECO_GenMETCOLLECTION_H
