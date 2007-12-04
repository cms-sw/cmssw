#ifndef HLTReco_TriggerTypeDefs_h
#define HLTReco_TriggerTypeDefs_h

/** \class trigger::TriggerTypeDefs
 *
 *  Misc. typedefs
 *
 *  $Date: 2007/12/04 08:35:53 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Provenance/interface/ProductID.h"
#include<vector>

namespace trigger
{

  typedef uint16_t size_type;
  typedef std::vector<size_type> Keys;
  typedef std::pair<edm::ProductID,size_type> XRef;

}

#endif
