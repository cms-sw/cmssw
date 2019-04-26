/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef DataFormats_ProtonReco_ForwardProtonFwd_h
#define DataFormats_ProtonReco_ForwardProtonFwd_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

#include <vector>

namespace reco
{
  class ForwardProton;
  /// Collection of ForwardProton objects
  typedef std::vector<ForwardProton> ForwardProtonCollection;
  /// Persistent reference to a ForwardProton
  typedef edm::Ref<ForwardProtonCollection> ForwardProtonRef;
  /// Reference to a ForwardProton collection
  typedef edm::RefProd<ForwardProtonCollection> ForwardProtonRefProd;
  /// Vector of references to ForwardProton in the same collection
  typedef edm::RefVector<ForwardProtonCollection> ForwardProtonRefVector;
}

#endif

