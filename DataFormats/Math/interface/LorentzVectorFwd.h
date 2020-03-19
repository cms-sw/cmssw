#ifndef Math_LorentzVectorFwd_h
#define Math_LorentzVectorFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace math {
  typedef std::vector<PtEtaPhiELorentzVectorD> PtEtaPhiELorentzVectorDCollection;
  typedef edm::Ref<PtEtaPhiELorentzVectorDCollection> PtEtaPhiELorentzVectorDRef;
  typedef edm::RefProd<PtEtaPhiELorentzVectorDCollection> PtEtaPhiELorentzVectorDRefProd;
  typedef edm::RefVector<PtEtaPhiELorentzVectorDCollection> PtEtaPhiELorentzVectorDRefVector;

  typedef std::vector<XYZTLorentzVectorD> XYZTLorentzVectorDCollection;
  typedef edm::Ref<XYZTLorentzVectorDCollection> XYZTLorentzVectorDRef;
  typedef edm::RefProd<XYZTLorentzVectorDCollection> XYZTLorentzVectorDRefProd;
  typedef edm::RefVector<XYZTLorentzVectorDCollection> XYZTLorentzVectorDRefVector;

  typedef std::vector<PtEtaPhiELorentzVectorF> PtEtaPhiELorentzVectorFCollection;
  typedef edm::Ref<PtEtaPhiELorentzVectorFCollection> PtEtaPhiELorentzVectorFRef;
  typedef edm::RefProd<PtEtaPhiELorentzVectorFCollection> PtEtaPhiELorentzVectorFRefProd;
  typedef edm::RefVector<PtEtaPhiELorentzVectorFCollection> PtEtaPhiELorentzVectorFRefVector;

  typedef std::vector<XYZTLorentzVectorF> XYZTLorentzVectorFCollection;
  typedef edm::Ref<XYZTLorentzVectorFCollection> XYZTLorentzVectorFRef;
  typedef edm::RefProd<XYZTLorentzVectorFCollection> XYZTLorentzVectorFRefProd;
  typedef edm::RefVector<XYZTLorentzVectorFCollection> XYZTLorentzVectorFRefVector;

  typedef std::vector<PtEtaPhiELorentzVector> PtEtaPhiELorentzVectorCollection;
  typedef edm::Ref<PtEtaPhiELorentzVectorCollection> PtEtaPhiELorentzVectorRef;
  typedef edm::RefProd<PtEtaPhiELorentzVectorCollection> PtEtaPhiELorentzVectorRefProd;
  typedef edm::RefVector<PtEtaPhiELorentzVectorCollection> PtEtaPhiELorentzVectorRefVector;

  typedef std::vector<XYZTLorentzVector> XYZTLorentzVectorCollection;
  typedef edm::Ref<XYZTLorentzVectorCollection> XYZTLorentzVectorRef;
  typedef edm::RefProd<XYZTLorentzVectorCollection> XYZTLorentzVectorRefProd;
  typedef edm::RefVector<XYZTLorentzVectorCollection> XYZTLorentzVectorRefVector;

}  // namespace math

#endif
