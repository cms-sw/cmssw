#ifndef HLTReco_HLTDataModel_h
#define HLTReco_HLTDataModel_h

/** \class reco::HLTDataModel
 *
 *  Classes for new HLT data model (to be split into separate header files)
 *
 *  $Date: 2007/10/30 15:33:28 $
 *  $Revision: 1.2 $
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

  typedef math::PtEtaPhiMLorentzVectorF TriggerFourMomentum;


  /// individual physics object (e.g., electron or muon or jet)
  class TriggerObject {

  private:
    /// 4-momentum of physics object
    TriggerFourMomentum objectP4_;
    /// id or type - similar to pdgId
    int objectId_;

  public:
    /// constructors
    TriggerObject(): objectP4_(), objectId_() { }
    TriggerObject(const TriggerFourMomentum& P4, int Id=0): objectP4_(P4), objectId_(Id) { }

    /// setters
    void setP4 (const TriggerFourMomentum& P4) {objectP4_ = P4;}
    void setId (int Id=0) {objectId_ = Id;}

    /// getters
    const TriggerFourMomentum& getP4() const {return objectP4_;}
    const int & getId() const {return objectId_;}

  };


  /// collection of physics objects (e.g., all electrons or all muons)
  class TriggerCollection {

  private:
    /// objects making the collection
    std::vector<TriggerObject> triggerObjects_;
    /// id or type - of collection
    int collectionId_;

  public:
    /// constructors
    TriggerCollection(): triggerObjects_(), collectionId_() { }
    TriggerCollection(const std::vector<TriggerObject>& triggerObjects, int Id=0):
      triggerObjects_(triggerObjects), collectionId_(Id) { }

    /// setters
    void setCollection(const std::vector<TriggerObject>& triggerObjects) {triggerObjects_=triggerObjects;}
    void setId(int Id) {collectionId_=Id;}

    /// getters
    const std::vector<TriggerObject>& getCollection() const {return triggerObjects_;}
    int getId() const {return collectionId_;}

  };


  /// collection of physics object collections (all electrons and muons and...)
  /// only needed if we want _one_ EDProduct containing _all_ collections!
  class TriggerGlobalCollection {

  private:
    ///
    std::vector<TriggerCollection> triggerCollections_;

  public:
    /// constructors
    TriggerGlobalCollection(): triggerCollections_() { }
    TriggerGlobalCollection(const std::vector<TriggerCollection>& triggerCollections):
      triggerCollections_(triggerCollections) { }
    ///
    void addCollection(const TriggerCollection& triggerCollection) {
      triggerCollections_.push_back(triggerCollection);}
    const TriggerCollection& getCollection(int i) const {
      return triggerCollections_.at(i);
    }

  };


  /// Pointer to a physics object within a collection
  class TriggerPointer { /// should be edm::Ptr<T> ??

  private:
    edm::ProductID productID_;
    int            index_;

  public:
    /// constructors
    TriggerPointer(): productID_(), index_() { }
    TriggerPointer(const edm::ProductID& productID, int index=-1):
      productID_(productID), index_(index) { }

    /// setters
    void setProductID (const edm::ProductID& productID) {productID_=productID;}
    void setIndex (int index=-1) {index_=index;}

    /// getters
    const edm::ProductID& getProductID() const {return productID_;}
    int getIndex() const {return index_;}

  };


  /// Collection of pointers to physics objects belonging to a trigger path
  class TriggerPathCollection {

  private:
    /// non-owning pointrers into collections
    std::vector<TriggerPointer> triggerObjects_;
    /// id or type - of collection
    int collectionId_;
    
  public:
    /// constructors
    TriggerPathCollection(): triggerObjects_(), collectionId_() { }
    TriggerPathCollection(const std::vector<TriggerPointer>& triggerObjects, int Id=0):
      triggerObjects_(triggerObjects), collectionId_(Id) { }

    /// setters
    void setCollection(const std::vector<TriggerPointer>& triggerObjects) {triggerObjects_=triggerObjects;}
    void setId(int Id) {collectionId_=Id;}

    /// getters
    const std::vector<TriggerPointer>& getCollection() const {return triggerObjects_;}
    int getId() const {return collectionId_;}

  };


  /// Collection of pointer collections describing the trigger table
  /// only needed if we want _one_ EDProduct containing _all_ collections!
  class TriggerTableCollection {

  private:
    std::vector<TriggerPathCollection> triggerPathCollections_;

  public:
    /// constructors
    TriggerTableCollection(): triggerPathCollections_() { }
    TriggerTableCollection(const std::vector<TriggerPathCollection>& triggerPathCollections):
      triggerPathCollections_(triggerPathCollections) { }
    ///
    void addPathCollection(const TriggerPathCollection& triggerPathCollection) {
      triggerPathCollections_.push_back(triggerPathCollection);}
    const TriggerPathCollection& getPathCollection(int i) const {
      return triggerPathCollections_.at(i);
    }

  };

}

#endif
