#ifndef METRECO_PFMETCOLLECTION_H
#define METRECO_PFMETCOLLECTION_H

/** \class PFMETCollection
 *
 * \short Collection of PF MET
 *
 * PFMETCollection is a collection of PFMET objects
 *
 * \author R.Remington, UFlorida
 *
 * \version   1st Version Oct, 2008.
 *
 ************************************************************/

#include <vector>
#include "DataFormats/METReco/interface/PFMETFwd.h" 

using namespace reco;
using namespace std;

namespace reco
{
  typedef std::vector<reco::PFMET> PFMETCollection;
}  
#endif // METRECO_PFMETCOLLECTION_H
