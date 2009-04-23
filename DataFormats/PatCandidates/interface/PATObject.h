//
// $Id: PATObject.h,v 1.21 2008/10/09 17:48:23 lowette Exp $
//

#ifndef DataFormats_PatCandidates_PATObject_h
#define DataFormats_PatCandidates_PATObject_h

/**
  \class    pat::PATObject PATObject.h "DataFormats/PatCandidates/interface/PATObject.h"
  \brief    Templated PAT object container

   PATObject is the templated base PAT object that wraps around reco objects.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author   Steven Lowette, Giovanni Petrucciani, Frederic Ronga, Volker Adler, Sal Rappoccio
  \version  $Id: PATObject.h,v 1.21 2008/10/09 17:48:23 lowette Exp $
*/


#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>
#include <string>

#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"
#include "DataFormats/PatCandidates/interface/LookupTableRecord.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "DataFormats/Common/interface/OwnVector.h"


namespace pat {


  template <class ObjectType>
  class PATObject : public ObjectType {
    public:

      typedef  ObjectType             base_type;

      /// default constructor
      PATObject();
      /// constructor from a base object (leaves invalid reference to original object!)
      PATObject(const ObjectType & obj);
      /// constructor from reference
      PATObject(const edm::RefToBase<ObjectType> & ref);
      /// constructor from reference
      PATObject(const edm::Ptr<ObjectType> & ref);
      /// destructor
      virtual ~PATObject() {}    
    // returns a clone                                  // NO: ObjectType can be an abstract type like reco::Candidate
    //  virtual PATObject<ObjectType> * clone() const ; //     for which the clone() can't be defined

      /// access to the original object; returns zero for null Ref and throws for unavailable collection
      const reco::Candidate * originalObject() const;
      /// reference to original object. Returns a null reference if not available
      const edm::Ptr<reco::Candidate> & originalObjectRef() const;

      /// trigger matches
      const std::vector<TriggerPrimitive> & triggerMatches() const;
      const std::vector<TriggerPrimitive> triggerMatchesByFilter(const std::string & aFilt) const;
      /// add a trigger match
      void addTriggerMatch(const pat::TriggerPrimitive & aTrigPrim);

      /// Returns an efficiency given its name
      const pat::LookupTableRecord       & efficiency(const std::string &name) const ;
      /// Returns the efficiencies as <name,value> pairs (by value)
      std::vector<std::pair<std::string,pat::LookupTableRecord> > efficiencies() const ;
      /// Returns the list of the names of the stored efficiencies 
      const std::vector<std::string> & efficiencyNames() const { return efficiencyNames_; }
      /// Returns the list of the values of the stored efficiencies (the ordering is the same as in efficiencyNames())
      const std::vector<pat::LookupTableRecord> & efficiencyValues() const { return efficiencyValues_; }
      /// Store one efficiency in this item, in addition to the existing ones
      /// If an efficiency with the same name exists, the old value is replaced by this one
      /// Calling this method many times with names not sorted alphabetically will be slow
      void setEfficiency(const std::string &name, const pat::LookupTableRecord & value) ;
     
      /// Get generator level particle reference (might be a transient ref if the genParticle was embedded)
      /// If you stored multiple GenParticles, you can specify which one you want.
      reco::GenParticleRef      genParticleRef(size_t idx=0) const { 
            if (idx >= genParticlesSize()) return reco::GenParticleRef();
            return genParticleEmbedded_.empty() ? genParticleRef_[idx] : reco::GenParticleRef(&genParticleEmbedded_, idx); 
      }
      /// Get a generator level particle reference with a given pdg id and status
      /// If there is no MC match with that pdgId and status, it will return a null ref
      /// Note: this might be a transient ref if the genParticle was embedded
      reco::GenParticleRef      genParticleById(int pdgId, int status) const ;

      /// Get generator level particle, as C++ pointer (might be 0 if the ref was null)
      /// If you stored multiple GenParticles, you can specify which one you want.
      const reco::GenParticle * genParticle(size_t idx=0)    const { 
            reco::GenParticleRef ref = genParticleRef(idx); 
            return ref.isNonnull() ? ref.get() : 0; 
      }
      /// Number of generator level particles stored as ref or embedded
      size_t genParticlesSize() const { 
            return genParticleEmbedded_.empty() ? genParticleRef_.size() : genParticleEmbedded_.size(); 
      }
      /// Return the list of generator level particles.
      /// Note that the refs can be transient refs to embedded GenParticles
      std::vector<reco::GenParticleRef> genParticleRefs() const ;

      /// Set the generator level particle reference
      void setGenParticleRef(const reco::GenParticleRef &ref, bool embed=false) ; 
      /// Add a generator level particle reference
      /// If there is already an embedded particle, this ref will be embedded too
      void addGenParticleRef(const reco::GenParticleRef &ref) ; 
      /// Set the generator level particle from a particle not in the Event (embedding it, of course)
      void setGenParticle( const reco::GenParticle &particle ) ; 
      /// Embed the generator level particle(s) in this PATObject
      /// Note that generator level particles can only be all embedded or all not embedded.
      void embedGenParticle() ;

      /// Returns user-defined data. Returns NULL if the data is not present, or not of type T.
      template<typename T> const T * userData(const std::string &key) const {
          const pat::UserData * data = userDataObject_(key);
          return (data != 0 ? data->template get<T>() : 0);
          
      }
      /// Check if user data with a specific type is present
      bool hasUserData(const std::string &key) const {
          return (userDataObject_(key) != 0);
      }
      /// Get human-readable type of user data object, for debugging
      const std::string & userDataObjectType(const std::string &key) const { 
          static const std::string EMPTY("");
          const pat::UserData * data = userDataObject_(key);
          return (data != 0 ? data->typeName() : EMPTY);
      }; 
      /// Get list of user data object names
      const std::vector<std::string> & userDataNames() const  { return userDataLabels_; }

      /// Get the data as a void *, for CINT usage.
      /// COMPLETELY UNSUPPORTED, USE ONLY FOR DEBUGGING
      const void * userDataBare(const std::string &key) const {
          const pat::UserData * data = userDataObject_(key);
          return (data != 0 ? data->bareData() : 0);
      }
    
      /// Set user-defined data
      /// Needs dictionaries for T and for pat::UserHolder<T>, 
      /// and it will throw exception if they're missing,
      /// unless transientOnly is set to true 
      template<typename T>
      void addUserData( const std::string & label, const T & data, bool transientOnly=false ) {
          userDataLabels_.push_back(label);
          userDataObjects_.push_back(pat::UserData::make<T>(data, transientOnly));
      }

      /// Set user-defined data. To be used only to fill from ValueMap<Ptr<UserData>>
      /// Do not use unless you know what you are doing.
      void addUserDataFromPtr( const std::string & label, const edm::Ptr<pat::UserData> & data ) {
          userDataLabels_.push_back(label);
          userDataObjects_.push_back(data->clone());
      }
      
      /// Get user-defined float
      /// Note: it will return 0.0 if the key is not found; you can check if the key exists with 'hasUserFloat' method.
      float userFloat( const std::string & key ) const;
      /// Set user-defined float
      void addUserFloat( const  std::string & label, float data );
      /// Get list of user-defined float names
      const std::vector<std::string> & userFloatNames() const  { return userFloatLabels_; }
      /// Return true if there is a user-defined float with a given name
      bool hasUserFloat( const std::string & key ) const {
        return std::find(userFloatLabels_.begin(), userFloatLabels_.end(), key) != userFloatLabels_.end();
      }
      /// Get user-defined int
      /// Note: it will return 0 if the key is not found; you can check if the key exists with 'hasUserInt' method.
      int32_t userInt( const std::string & key ) const;
      /// Set user-defined int
      void addUserInt( const std::string & label,  int32_t data );
      /// Get list of user-defined int names
      const std::vector<std::string> & userIntNames() const  { return userIntLabels_; }
      /// Return true if there is a user-defined int with a given name
      bool hasUserInt( const std::string & key ) const {
        return std::find(userIntLabels_.begin(), userIntLabels_.end(), key) != userIntLabels_.end();
      }
 
    protected:
      // reference back to the original object
      edm::Ptr<reco::Candidate> refToOrig_;

      /// vector of trigger matches
      std::vector<pat::TriggerPrimitive> triggerMatches_;

      /// vector of the efficiencies (values)
      std::vector<pat::LookupTableRecord> efficiencyValues_;
      /// vector of the efficiencies (names)
      std::vector<std::string> efficiencyNames_;

      /// Reference to a generator level particle
      std::vector<reco::GenParticleRef> genParticleRef_;
      /// vector to hold an embedded generator level particle
      std::vector<reco::GenParticle>    genParticleEmbedded_; 

      /// User data object
      std::vector<std::string>      userDataLabels_;
      pat::UserDataCollection       userDataObjects_;
      // User float values
      std::vector<std::string>      userFloatLabels_;
      std::vector<float>            userFloats_;
      // User int values
      std::vector<std::string>      userIntLabels_;
      std::vector<int32_t>          userInts_;

    private:
      const pat::UserData *  userDataObject_(const std::string &key) const ;
  };


  template <class ObjectType> PATObject<ObjectType>::PATObject() {
  }

  template <class ObjectType> PATObject<ObjectType>::PATObject(const ObjectType & obj) :
    ObjectType(obj),
    refToOrig_() {
  }

  template <class ObjectType> PATObject<ObjectType>::PATObject(const edm::RefToBase<ObjectType> & ref) :
    ObjectType(*ref),
    refToOrig_(ref.id(), ref.get(), ref.key()) // correct way to convert RefToBase=>Ptr, if ref is guaranteed to be available
                                               // which happens to be true, otherwise the line before this throws ex. already
      {
      }

  template <class ObjectType> PATObject<ObjectType>::PATObject(const edm::Ptr<ObjectType> & ref) :
    ObjectType(*ref),
    refToOrig_(ref) {
  }


  template <class ObjectType> const reco::Candidate * PATObject<ObjectType>::originalObject() const {
    if (refToOrig_.isNull()) {
      // this object was not produced from a reference, so no link to the
      // original object exists -> return a 0-pointer
      return 0;
    } else if (!refToOrig_.isAvailable()) {
      throw edm::Exception(edm::errors::ProductNotFound) << "The original collection from which this PAT object was made is not present any more in the event, hence you cannot access the originating object anymore.";
    } else {
      return refToOrig_.get();
    }
  }

  template <class ObjectType> 
  const edm::Ptr<reco::Candidate> & PATObject<ObjectType>::originalObjectRef() const { return refToOrig_; }

  template <class ObjectType>
  const std::vector<TriggerPrimitive> & PATObject<ObjectType>::triggerMatches() const { return triggerMatches_; }
  
  template <class ObjectType>
  const std::vector<TriggerPrimitive> PATObject<ObjectType>::triggerMatchesByFilter(const std::string & aFilt) const {
    std::vector<TriggerPrimitive> selectedMatches;
    for ( size_t i = 0; i < triggerMatches_.size(); i++ ) {
      if ( triggerMatches_.at(i).filterName() == aFilt ) selectedMatches.push_back(triggerMatches_.at(i));
    }
    return selectedMatches;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::addTriggerMatch(const pat::TriggerPrimitive & aTrigPrim) {
    triggerMatches_.push_back(aTrigPrim);
  }

  template <class ObjectType>
  const pat::LookupTableRecord &  
  PATObject<ObjectType>::efficiency(const std::string &name) const {
    // find the name in the (sorted) list of names
    std::vector<std::string>::const_iterator it = std::lower_bound(efficiencyNames_.begin(), efficiencyNames_.end(), name);
    if ((it == efficiencyNames_.end()) || (*it != name)) {
        throw cms::Exception("Invalid Label") << "There is no efficiency with name '" << name << "' in this PAT Object\n";
    }
    return efficiencyValues_[it - efficiencyNames_.begin()];
  }

  template <class ObjectType>
  std::vector<std::pair<std::string,pat::LookupTableRecord> > 
  PATObject<ObjectType>::efficiencies() const {
    std::vector<std::pair<std::string,pat::LookupTableRecord> > ret;
    std::vector<std::string>::const_iterator itn = efficiencyNames_.begin(), edn = efficiencyNames_.end();
    std::vector<pat::LookupTableRecord>::const_iterator itv = efficiencyValues_.begin();
    for ( ; itn != edn; ++itn, ++itv) {
        ret.push_back( std::pair<std::string,pat::LookupTableRecord>(*itn, *itv) );
    }
    return ret;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::setEfficiency(const std::string &name, const pat::LookupTableRecord & value) {
    // look for the name, or to the place where we can insert it without violating the alphabetic order
    std::vector<std::string>::iterator it = std::lower_bound(efficiencyNames_.begin(), efficiencyNames_.end(), name);
    if (it == efficiencyNames_.end()) { // insert at the end
        efficiencyNames_.push_back(name);
        efficiencyValues_.push_back(value);
    } else if (*it == name) {           // replace existing
        efficiencyValues_[it - efficiencyNames_.begin()] = value;
    } else {                            // insert in the middle :-(
        efficiencyNames_. insert(it, name);
        efficiencyValues_.insert( efficiencyValues_.begin() + (it - efficiencyNames_.begin()), value );
    }
  }

  template <class ObjectType>
  void PATObject<ObjectType>::setGenParticleRef(const reco::GenParticleRef &ref, bool embed) {
          genParticleRef_ = std::vector<reco::GenParticleRef>(1,ref);
          genParticleEmbedded_.clear(); 
          if (embed) embedGenParticle();
  }

  template <class ObjectType>
  void PATObject<ObjectType>::addGenParticleRef(const reco::GenParticleRef &ref) {
      if (!genParticleEmbedded_.empty()) { // we're embedding
          if (ref.isNonnull()) genParticleEmbedded_.push_back(*ref);
      } else {
          genParticleRef_.push_back(ref);
      }
  }
  
  template <class ObjectType>
  void PATObject<ObjectType>::setGenParticle( const reco::GenParticle &particle ) {
      genParticleEmbedded_.clear(); 
      genParticleEmbedded_.push_back(particle);
      genParticleRef_.clear();
  }

  template <class ObjectType>
  void PATObject<ObjectType>::embedGenParticle() {
      genParticleEmbedded_.clear(); 
      for (std::vector<reco::GenParticleRef>::const_iterator it = genParticleRef_.begin(); it != genParticleRef_.end(); ++it) {
          if (it->isNonnull()) genParticleEmbedded_.push_back(**it); 
      }
      genParticleRef_.clear();
  }

  template <class ObjectType>
  std::vector<reco::GenParticleRef> PATObject<ObjectType>::genParticleRefs() const {
        if (genParticleEmbedded_.empty()) return genParticleRef_;
        std::vector<reco::GenParticleRef> ret(genParticleEmbedded_.size());
        for (size_t i = 0, n = ret.size(); i < n; ++i) {
            ret[i] = reco::GenParticleRef(&genParticleEmbedded_, i);
        }
        return ret;
  }

  template <class ObjectType>
  reco::GenParticleRef PATObject<ObjectType>::genParticleById(int pdgId, int status) const {
        // get a vector, avoiding an unneeded copy if there is no embedding
        const std::vector<reco::GenParticleRef> & vec = (genParticleEmbedded_.empty() ? genParticleRef_ : genParticleRefs());
        for (std::vector<reco::GenParticleRef>::const_iterator ref = vec.begin(), end = vec.end(); ref != end; ++ref) {
            if (ref->isNonnull() && ((*ref)->pdgId() == pdgId) && ((*ref)->status() == status)) return *ref;
        }
        return reco::GenParticleRef();
  }


  template <class ObjectType>
  const pat::UserData * PATObject<ObjectType>::userDataObject_( const std::string & key ) const
  {
    std::vector<std::string>::const_iterator it = std::find(userDataLabels_.begin(), userDataLabels_.end(), key);
    if (it != userDataLabels_.end()) {
        return & userDataObjects_[it - userDataLabels_.begin()];
    }
    return 0;
  }

  template <class ObjectType>
  float PATObject<ObjectType>::userFloat( const std::string &key ) const
  {
    std::vector<std::string>::const_iterator it = std::find(userFloatLabels_.begin(), userFloatLabels_.end(), key);
    if (it != userFloatLabels_.end()) {
        return userFloats_[it - userFloatLabels_.begin()];
    }
    return 0.0;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::addUserFloat( const std::string & label,
					    float data )
  {
    userFloatLabels_.push_back(label);
    userFloats_.push_back( data );
  }


  template <class ObjectType>
  int PATObject<ObjectType>::userInt( const std::string & key ) const
  {
    std::vector<std::string>::const_iterator it = std::find(userIntLabels_.begin(), userIntLabels_.end(), key);
    if (it != userIntLabels_.end()) {
        return userInts_[it - userIntLabels_.begin()];
    }
    return 0;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::addUserInt( const std::string &label,
					   int data )
  {
    userIntLabels_.push_back(label);
    userInts_.push_back( data );
  }


}

#endif
