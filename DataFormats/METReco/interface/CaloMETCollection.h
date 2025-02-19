#ifndef METRECO_CaloMETCOLLECTION_H
#define METRECO_CaloMETCOLLECTION_H

/** \class CaloMETCollection
 *
 * \short Collection of Calo MET
 *
 * CaloMETCollection is a collection of CaloMET objects
 *
 * \author Mike Schmitt, UFlorida
 *
 * \version   1st Version August 4, 2005.
 *
 ************************************************************/

#include <vector>
#include "DataFormats/METReco/interface/CaloMETFwd.h" 

namespace reco
{
  typedef std::vector<reco::CaloMET> CaloMETCollection;
}  
#endif // METRECO_CaloMETCOLLECTION_H
