#ifndef METRECO_METCOLLECTION_H
#define METRECO_METCOLLECTION_H

/** \class METCollection
 *
 * \short Collection of MET
 *
 * METCollection is a collection of MET objects
 *
 * \author Mike Schmitt, UFlorida
 *
 * \version   1st Version August 4, 2005.
 *
 ************************************************************/

#include <vector>
#include "DataFormats/METReco/interface/METFwd.h" 

//#warning "This header file is obsolete.  Please use METFwd.h instead"

namespace reco
{
  typedef std::vector<reco::MET> METCollection;
}  
#endif // METRECO_METCOLLECTION_H
