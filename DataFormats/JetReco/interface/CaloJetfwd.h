// F.R.
// $Id: CaloJetfwd.h,v 1.5 2006/06/07 12:28:44 llista Exp $
#ifndef JetReco_CaloJetfwd_h
#define JetReco_CaloJetfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class CaloJet;
  /// collection of CaloJet objects 
  typedef std::vector<CaloJet> CaloJetCollection;
  /// edm references
  typedef edm::Ref<CaloJetCollection> CaloJetRef;
  typedef edm::RefVector<CaloJetCollection> CaloJetRefVector;
  typedef edm::RefProd<CaloJetCollection> CaloJetRefProd;
}
#endif
