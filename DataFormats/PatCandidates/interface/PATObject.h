#ifndef DataFormats_PatCandidates_PATObject_h
#define DataFormats_PatCandidates_PATObject_h

/** \class    pat::PATObject PATObject.h "DataFormats/PatCandidates/interface/PATObject.h"
 *
 *  \brief    Templated PAT object container
 *
 *  PATObject is the templated base PAT object that wraps around reco objects.
 *
 *  \author   Steven Lowette
 *
 *  \version  $Id: PATObject.h,v 1.11 2008/06/24 22:33:12 gpetrucc Exp $
 *
 */

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1DFloat.h"

#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace pat {

  template <class ObjectType>
  class PATObject : public ObjectType {
    public:

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
      /// standard deviation on A (see CMS Note 2006/023)
      float resolutionA() const;
      /// standard deviation on B (see CMS Note 2006/023)
      float resolutionB() const;
      /// standard deviation on C (see CMS Note 2006/023)
      float resolutionC() const;
      /// standard deviation on D (see CMS Note 2006/023)
      float resolutionD() const;
      /// standard deviation on transverse energy
      float resolutionEt() const;
      /// standard deviation on pseudorapidity
      float resolutionEta() const;
      /// standard deviation on azimuthal angle
      float resolutionPhi() const;
      /// standard deviation on polar angle
      float resolutionTheta() const;
      /// covariance matrix elements
      const std::vector<float> & covMatrix() const;
      /// trigger matches
      const std::vector<TriggerPrimitive> & triggerMatches() const;
      const std::vector<TriggerPrimitive> triggerMatchesByFilter(const std::string & aFilt) const;
      /// set standard deviation on A (see CMS Note 2006/023)
      void setResolutionA(float a);
      /// set standard deviation on B (see CMS Note 2006/023)
      void setResolutionB(float b);
      /// set standard deviation on C (see CMS Note 2006/023)
      void setResolutionC(float c);
      /// set standard deviation on D (see CMS Note 2006/023)
      void setResolutionD(float d);
      /// set standard deviation on transverse energy
      void setResolutionEt(float et);
      /// set standard deviation on pseudorapidity
      void setResolutionEta(float eta);
      /// set standard deviation on azimuthal angle
      void setResolutionPhi(float phi);
      /// set standard deviation on polar angle
      void setResolutionTheta(float theta);
      /// set covariance matrix elements
      void setCovMatrix(const std::vector<float> & c);
      /// add a trigger match
      void addTriggerMatch(const pat::TriggerPrimitive & aTrigPrim);

      /// Returns an efficiency given its name
      const Measurement1DFloat       & efficiency(const std::string &name) const ;
      /// Returns the efficiencies as <name,value> pairs (by value)
      std::vector<std::pair<std::string,Measurement1DFloat> > efficiencies() const ;
      /// Returns the list of the names of the stored efficiencies 
      const std::vector<std::string> & efficiencyNames() const { return efficiencyNames_; }
      /// Returns the list of the values of the stored efficiencies (the ordering is the same as in efficiencyNames())
      const std::vector<Measurement1DFloat> & efficiencyValues() const { return efficiencyValues_; }
      /// Store one efficiency in this item, in addition to the existing ones
      /// If an efficiency with the same name exists, the old value is replaced by this one
      /// Calling this method many times with names not sorted alphabetically will be slow
      void setEfficiency(const std::string &name, const Measurement1DFloat & value) ;
     
      /// Get generator level particle reference (might be a transient ref if the genParticle was embedded)
      reco::GenParticleRef      genParticleRef() const { return genParticleEmbedded_.empty() ? genParticleRef_ : reco::GenParticleRef(&genParticleEmbedded_, 0); }
      /// Get generator level particle, as C++ pointer (might be 0 if the ref was null)
      const reco::GenParticle * genParticle()    const { reco::GenParticleRef ref = genParticleRef(); return ref.isNonnull() ? ref.get() : 0; }
      /// Set the generator level particle reference
      void setGenParticleRef(const reco::GenParticleRef &ref, bool embed=false) ; 
      /// Set the generator level particle from a particle not in the Event (embedding it, of course)
      void setGenParticle( const reco::GenParticle &particle ) ; 
      /// Embed the generator level particle in this PATObject
      void embedGenParticle() ;
 
    protected:
      // reference back to the original object
      edm::Ptr<reco::Candidate> refToOrig_;
      /// standard deviation on transverse energy
      float resEt_;
      /// standard deviation on pseudorapidity
      float resEta_;
      /// standard deviation on azimuthal angle
      float resPhi_;
      /// standard deviation on A (see CMS Note 2006/023)
      float resA_;
      /// standard deviation on B (see CMS Note 2006/023)
      float resB_;
      /// standard deviation on C (see CMS Note 2006/023)
      float resC_;
      /// standard deviation on D (see CMS Note 2006/023)
      float resD_;
      /// standard deviation on polar angle
      float resTheta_;
      // covariance matrix elements
      std::vector<float> covM_;
      /// vector of trigger matches
      std::vector<pat::TriggerPrimitive> triggerMatches_;

      /// vector of the efficiencies (values)
      std::vector<Measurement1DFloat> efficiencyValues_;
      /// vector of the efficiencies (names)
      std::vector<std::string> efficiencyNames_;

      /// Reference to a generator level particle
      reco::GenParticleRef genParticleRef_;
      /// vector to hold an embedded generator level particle
      std::vector<reco::GenParticle> genParticleEmbedded_; 
   
  };


  template <class ObjectType> PATObject<ObjectType>::PATObject() :
    resEt_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0), resTheta_(0) {
  }

  template <class ObjectType> PATObject<ObjectType>::PATObject(const ObjectType & obj) :
    ObjectType(obj),
    refToOrig_(),
    resEt_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
  }

  template <class ObjectType> PATObject<ObjectType>::PATObject(const edm::RefToBase<ObjectType> & ref) :
    ObjectType(*ref),
    refToOrig_(ref.id(), ref.get(), ref.key()), // correct way to convert RefToBase=>Ptr, if ref is guaranteed to be available
                                                // which happens to be true, otherwise the line before this throws ex. already
    resEt_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
  }

  template <class ObjectType> PATObject<ObjectType>::PATObject(const edm::Ptr<ObjectType> & ref) :
    ObjectType(*ref),
    refToOrig_(ref),
    resEt_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
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
  float PATObject<ObjectType>::resolutionEt() const { return resEt_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionEta() const { return resEta_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionPhi() const { return resPhi_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionA() const { return resA_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionB() const { return resB_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionC() const { return resC_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionD() const { return resD_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionTheta() const { return resTheta_; }

  template <class ObjectType> 
  const std::vector<float> & PATObject<ObjectType>::covMatrix() const { return covM_; }
  
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
  void PATObject<ObjectType>::setResolutionEt(float et) { resEt_ = et; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionEta(float eta) { resEta_ = eta; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionPhi(float phi) { resPhi_ = phi; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionA(float a) { resA_ = a; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionB(float b) { resB_ = b; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionC(float c) { resC_ = c; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionD(float d) { resD_ = d; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionTheta(float theta) { resTheta_ = theta; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setCovMatrix(const std::vector<float> & c) {
    //    covM_.clear();
    //    for (size_t i = 0; i < c.size(); i++) covM_.push_back(c[i]); 
    covM_ = c;
  }
  
  template <class ObjectType>
  void PATObject<ObjectType>::addTriggerMatch(const pat::TriggerPrimitive & aTrigPrim) {
    triggerMatches_.push_back(aTrigPrim);
  }

  template <class ObjectType>
  const Measurement1DFloat &  
  PATObject<ObjectType>::efficiency(const std::string &name) const {
    // find the name in the (sorted) list of names
    std::vector<std::string>::const_iterator it = std::lower_bound(efficiencyNames_.begin(), efficiencyNames_.end(), name);
    if ((it == efficiencyNames_.end()) || (*it != name)) {
        throw cms::Exception("Invalid Label") << "There is no efficiency with name '" << name << "' in this PAT Object\n";
    }
    return efficiencyValues_[it - efficiencyNames_.begin()];
  }

  template <class ObjectType>
  std::vector<std::pair<std::string,Measurement1DFloat> > 
  PATObject<ObjectType>::efficiencies() const {
    std::vector<std::pair<std::string,Measurement1DFloat> > ret;
    std::vector<std::string>::const_iterator itn = efficiencyNames_.begin(), edn = efficiencyNames_.end();
    std::vector<Measurement1DFloat>::const_iterator itv = efficiencyValues_.begin();
    for ( ; itn != edn; ++itn, ++itv) {
        ret.push_back( std::pair<std::string,Measurement1DFloat>(*itn, *itv) );
    }
    return ret;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::setEfficiency(const std::string &name, const Measurement1DFloat & value) {
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
          genParticleRef_ = ref;
          genParticleEmbedded_.clear(); 
          if (embed) embedGenParticle();
  }
  
  template <class ObjectType>
  void PATObject<ObjectType>::setGenParticle( const reco::GenParticle &particle ) {
        genParticleEmbedded_.clear(); 
        genParticleEmbedded_.push_back(particle);
        genParticleRef_ = reco::GenParticleRef();
  }

  template <class ObjectType>
  void PATObject<ObjectType>::embedGenParticle() {
      if (genParticleRef_.isNonnull()) {
            genParticleEmbedded_.clear(); 
            genParticleEmbedded_.push_back(*genParticleRef_); 
      }
      genParticleRef_ = reco::GenParticleRef();
  }

}

#endif
