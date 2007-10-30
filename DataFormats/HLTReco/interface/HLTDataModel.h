#ifndef HLTReco_HLTDataModel_h
#define HLTReco_HLTDataModel_h

/** \class reco::HLTFilterObject
 *
 *
 *
 *  $Date: 2007/10/30 07:47:45 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <cassert>
#include <vector>

namespace reco
{

  typedef math::PtEtaPhiMLorentzVectorF TrigFourMomentum;

  /// individual physics object (e.g., electron or muon or jet)
  class TrigObject {

  private:
    /// 4-momentum of physics object
    TrigFourMomentum objectP4_;
    /// id or type - similar to pdgId
    int objectId_;

  public:
    /// constructors
    TrigObject(): objectP4_(), objectId_() { }
    TrigObject(const TrigFourMomentum& P4, int Id=0): objectP4_(P4), objectId_(Id) { }

    /// setters
    void setP4 (const TrigFourMomentum& P4) {objectP4_ = P4;}
    void setId (int Id=0) {objectId_ = Id;}

    /// getters
    const TrigFourMomentum& getP4() const {return objectP4_;}
    const int & getId() const {return objectId_;}

  };

  /// collection of physics objects (e.g., all electrons or all muons)
  class TrigCollection {

  private:
    /// objects making the collection
    std::vector<TrigObject> trigObjects_;
    /// id or type - of collection
    int collectionId_;

  public:
    /// constructors
    TrigCollection(): trigObjects_(), collectionId_() { }
    TrigCollection(const std::vector<TrigObject>& trigObjects, int Id=0):
      trigObjects_(trigObjects), collectionId_(Id) { }

    /// setters
    void setCollection(const std::vector<TrigObject>& trigObjects) {trigObjects_=trigObjects;}
    void setId(int Id) {collectionId_=Id;}

    /// getters
    const std::vector<TrigObject>& getCollection() {return trigObjects_;}
    int getId() const {return collectionId_;}

  };

  /// collection of physics object collections (all electrons and muons and...)
  class TrigGlobalCollection {

  private:
    ///
    std::vector<TrigCollection> trigCollections_;

  public:
    /// constructors
    TrigGlobalCollection(): trigCollections_() { }
    TrigGlobalCollection(const std::vector<TrigCollection>& trigCollections):
      trigCollections_(trigCollections) { }

  };



  /// Pointer to a physics object within a collection
  class TrigPointer { /// should be edm::Ptr ??

  private:
    edm::ProductID productID_;
    int            index_;

  public:
    /// constructors
    TrigPointer(): productID_(), index_() { }
    TrigPointer(const edm::ProductID& productID, int index=-1):
      productID_(productID), index_(index) { }

    /// setters
    void setProductID (const edm::ProductID& productID) {productID_=productID;}
    void setIndex (int index=-1) {index_=index;}

    /// getters
    const edm::ProductID& getProductID() const {return productID_;}
    int getIndex() const {return index_;}

  };

  /// Collection of pointers to physics objects belonging to a trigger path
  class TrigPathCollection {

  private:
    /// non-owning pointrers into collections
    std::vector<TrigPointer> trigObjects_;
    /// id or type - of collection
    int collectionId_;
    
  public:
    /// constructors
    TrigPathCollection(): trigObjects_(), collectionId_() { }
    TrigPathCollection(const std::vector<TrigPointer>& trigObjects, int Id=0):
      trigObjects_(trigObjects), collectionId_(Id) { }

    /// setters
    void setCollection(const std::vector<TrigPointer>& trigObjects) {trigObjects_=trigObjects;}
    void setId(int Id) {collectionId_=Id;}

    /// getters
    const std::vector<TrigPointer>& getCollection() {return trigObjects_;}
    int getId() const {return collectionId_;}

  };

  /// Collection of pointer collections describing the trigger table
  class TrigTableCollection {

  private:
    std::vector<TrigPathCollection> trigPathCollections_;

  public:
    void addPathCollection(const TrigPathCollection& trigPathCollection) {
      trigPathCollections_.push_back(trigPathCollection);}
    const TrigPathCollection& getPathCollection(int i) const {
      return trigPathCollections_.at(i);
    }
  };

}

#endif
