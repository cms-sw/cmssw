#ifndef DataFormats_JetMatching_JetFlavourMatching_h
#define DataFormats_JetMatching_JetFlavourMatching_h

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/JetMatching/interface/JetFlavour.h"
#include <vector>

namespace reco {

  typedef edm::AssociationVector<edm::RefToBaseProd<reco::Jet>, std::vector<reco::JetFlavour> >
      JetFlavourMatchingCollectionBase;

  class JetFlavourMatchingCollection : public JetFlavourMatchingCollectionBase {
  public:
    JetFlavourMatchingCollection() : JetFlavourMatchingCollectionBase() {}

    JetFlavourMatchingCollection(const reco::CaloJetRefProd &ref)
        : JetFlavourMatchingCollectionBase(edm::RefToBaseProd<reco::Jet>(ref)) {}

    JetFlavourMatchingCollection(const JetFlavourMatchingCollectionBase &v) : JetFlavourMatchingCollectionBase(v) {}
  };

  typedef JetFlavourMatchingCollection::value_type JetFlavourMatching;

  typedef edm::Ref<JetFlavourMatchingCollection> JetFlavourMatchingRef;

  typedef edm::RefProd<JetFlavourMatchingCollection> JetFlavourMatchingRefProd;

  typedef edm::RefVector<JetFlavourMatchingCollection> JetFlavourMatchingRefVector;

}  // namespace reco

#endif  // DataFormats_JetMatching_JetFlavourMatching_h
