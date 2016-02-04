#ifndef DataFormats_TauReco_CaloTauDiscriminatorAgainstElectron_h
#define DataFormats_TauReco_CaloTauDiscriminatorAgainstElectron_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<CaloTauRefProd,std::vector<int> > CaloTauDiscriminatorAgainstElectronBase;
  
  class CaloTauDiscriminatorAgainstElectron : public CaloTauDiscriminatorAgainstElectronBase {
  public:
    CaloTauDiscriminatorAgainstElectron() :
      CaloTauDiscriminatorAgainstElectronBase()
      { }
    
    CaloTauDiscriminatorAgainstElectron(const reco::CaloTauRefProd & ref) :
      CaloTauDiscriminatorAgainstElectronBase(ref)
      { }
    
    CaloTauDiscriminatorAgainstElectron(const CaloTauDiscriminatorAgainstElectronBase &v) :
      CaloTauDiscriminatorAgainstElectronBase(v)
      { }
  };
  
  typedef CaloTauDiscriminatorAgainstElectron::value_type CaloTauDiscriminatorAgainstElectronVT;  
  typedef edm::Ref<CaloTauDiscriminatorAgainstElectron> CaloTauDiscriminatorAgainstElectronRef;  
  typedef edm::RefProd<CaloTauDiscriminatorAgainstElectron> CaloTauDiscriminatorAgainstElectronRefProd;  
  typedef edm::RefVector<CaloTauDiscriminatorAgainstElectron> CaloTauDiscriminatorAgainstElectronRefVector; 
}
#endif
