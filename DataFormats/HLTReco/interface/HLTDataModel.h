#ifndef HLTReco_HLTDataModel_h
#define HLTReco_HLTDataModel_h

/** \class reco::HLTDataModel
 *
 *  Classes for new HLT data model (to be split into separate header files)
 *
 *  $Date: 2007/11/02 07:06:39 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

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


  /// Non-templated pointer class, pointing to an object within a collection
  class TriggerPointer { // should be edm::Ptr<T> ??

  private:
    /// id of product pointed to
    edm::ProductID productID_;
    /// key to get item within collection pointed to (-1 = none)
    long int       key_;

  public:
    /// constructors
    TriggerPointer(): productID_(), key_() { key_=-1; }
    TriggerPointer(const edm::ProductID& productID, int key=-1):
      productID_(productID), key_(key) { }

    template <typename C> 
    TriggerPointer(const edm::Handle<C>& handle, int key=-1):
      productID_(handle.id()), key_(key) { }
    template <typename C> 
    TriggerPointer(const edm::RefProd<C>& refprod, int key=-1):
      productID_(refprod.id()), key_(key) { }
    template <typename C> 
    TriggerPointer(const edm::RefToBaseProd<C>& reftobaseprod, int key=-1):
      productID_(reftobaseprod.id()), key_(key) { }

    template <typename C> 
    TriggerPointer(const edm::Ref<C>& ref):
      productID_(ref.id()), key_(ref.key()) { }
    template <typename C> 
    TriggerPointer(const edm::RefToBase<C>& reftobase):
      productID_(reftobase.id()), key_(reftobase.key()) { }
    template <typename T> 
    TriggerPointer(const edm::Ptr<T> ptr):
      productID_(ptr.id()), key_(ptr.key()) { }

    /// setters
    void setProductID (const edm::ProductID& productID) {productID_=productID;}
    void setKey (int key=-1) {key_=key;}

    /// getters
    const edm::ProductID& id() const {return productID_;}
    const long int& key() const {return key_;}

    /// the user needs to provide a Handle to the EDProduct
    template <typename C>
    const edm::RefProd<C> getRefProd(const edm::Handle<C>& handle) const {
      assert(handle.isValid());
      assert(handle.id()==id()); 
      edm::RefProd<C> refprod(handle);
      return refprod;
    }
    template <typename C>
    const edm::Ref<C> getRef(const edm::Handle<C>& handle) const {
      edm::RefProd<C> refprod(getRefProd<C>(handle));
      assert (key_>=0);
      edm::Ref<C> ref(refprod,static_cast<unsigned long int>(key_));
      return ref;
    }
    template <typename C>
    const edm::Ref<C> getRef(const edm::RefProd<C>& refprod) const {
      assert (key_>=0);
      edm::Ref<C> ref(refprod,static_cast<unsigned int>(key_));
      return ref;
    }
    template <typename T>
    const edm::Ptr<T> getPtr() const {
      assert (key_>=0);
      edm::Ptr<T> ptr(id(),key());
      return ptr;
    }

  };


  /// Collection of pointers to physics objects belonging to a trigger path
  class TriggerPathCollection {

  private:
    /// non-owning pointers into collections
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
