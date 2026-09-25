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

  namespace io_v1 {

    class JetFlavourInfoMatchingCollection : public reco::JetFlavourInfoMatchingCollectionBase {
    public:
      JetFlavourInfoMatchingCollection() : reco::JetFlavourInfoMatchingCollectionBase() {}

      JetFlavourInfoMatchingCollection(const reco::CaloJetRefProd &ref)
          : reco::JetFlavourInfoMatchingCollectionBase(edm::RefToBaseProd<reco::Jet>(ref)) {}

      JetFlavourInfoMatchingCollection(const reco::JetFlavourInfoMatchingCollectionBase &v)
          : reco::JetFlavourInfoMatchingCollectionBase(v) {}
    };

  }  // namespace io_v1
  using JetFlavourInfoMatchingCollection = io_v1::JetFlavourInfoMatchingCollection;

  typedef JetFlavourInfoMatchingCollection::value_type JetFlavourInfoMatching;

  typedef edm::Ref<JetFlavourInfoMatchingCollection> JetFlavourInfoMatchingRef;

  typedef edm::RefProd<JetFlavourInfoMatchingCollection> JetFlavourInfoMatchingRefProd;

  typedef edm::RefVector<JetFlavourInfoMatchingCollection> JetFlavourInfoMatchingRefVector;

}  // namespace reco

#endif  // DataFormats_JetMatching_JetFlavourInfoMatching_h
