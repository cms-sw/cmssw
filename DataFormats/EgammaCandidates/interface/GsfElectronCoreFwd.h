
#ifndef EgammaReco_GsfElectronCoreFwd_h
#define EgammaReco_GsfElectronCoreFwd_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

namespace reco {

  class GsfElectronCore ;
  typedef std::vector<GsfElectronCore> GsfElectronCoreCollection ;
  typedef edm::Ref<GsfElectronCoreCollection> GsfElectronCoreRef ;
  typedef edm::RefProd<GsfElectronCoreCollection> GsfElectronCoreRefProd ;
  typedef edm::RefVector<GsfElectronCoreCollection> GsfElectronCoreRefVector ;
  typedef GsfElectronCoreRefVector::iterator GsfElectronCore_iterator ;

}

#endif
