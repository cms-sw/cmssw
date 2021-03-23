#ifndef DataFormats_JetMatching_JetFlavourInfoMatching_h
#define DataFormats_JetMatching_JetFlavourInfoMatching_h

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfo.h"
#include <vector>

namespace reco {

  typedef edm::AssociationVector<edm::RefToBaseProd<reco::Jet>, std::vector<reco::JetFlavourInfo> >
      JetFlavourInfoMatchingCollectionBase;

  class JetFlavourInfoMatchingCollection : public JetFlavourInfoMatchingCollectionBase {
  public:
    JetFlavourInfoMatchingCollection() : JetFlavourInfoMatchingCollectionBase() {}

    JetFlavourInfoMatchingCollection(const reco::CaloJetRefProd &ref)
        : JetFlavourInfoMatchingCollectionBase(edm::RefToBaseProd<reco::Jet>(ref)) {}

    JetFlavourInfoMatchingCollection(const JetFlavourInfoMatchingCollectionBase &v)
        : JetFlavourInfoMatchingCollectionBase(v) {}
  };

  typedef JetFlavourInfoMatchingCollection::value_type JetFlavourInfoMatching;

  typedef edm::Ref<JetFlavourInfoMatchingCollection> JetFlavourInfoMatchingRef;

  typedef edm::RefProd<JetFlavourInfoMatchingCollection> JetFlavourInfoMatchingRefProd;

  typedef edm::RefVector<JetFlavourInfoMatchingCollection> JetFlavourInfoMatchingRefVector;

}  // namespace reco

#endif  // DataFormats_JetMatching_JetFlavourInfoMatching_h
