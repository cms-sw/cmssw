//
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
*/


#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>
#include <string>
#include <iosfwd>

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/LookupTableRecord.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "DataFormats/PatCandidates/interface/CandKinResolution.h"

#include "DataFormats/PatCandidates/interface/throwMissingLabel.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace pat {

  namespace pat_statics {
    static const reco::CandidatePtrVector EMPTY_CPV;
    static const std::string EMPTY_STR("");
  }

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
      ~PATObject() override {}
      // returns a clone                                  // NO: ObjectType can be an abstract type like reco::Candidate
      // virtual PATObject<ObjectType> * clone() const ;  //     for which the clone() can't be defined

      /// access to the original object; returns zero for null Ref and throws for unavailable collection
      const reco::Candidate * originalObject() const;
      /// reference to original object. Returns a null reference if not available
      const edm::Ptr<reco::Candidate> & originalObjectRef() const;

      /// access to embedded trigger matches:
      /// duplicated functions using 'char*' instead of 'std::string' are needed in order to work properly in CINT command lines;
      /// duplicated functions using 'unsigned' instead of 'bool' are needed in order to work properly in the cut string parser;

      /// get all matched trigger objects
      const TriggerObjectStandAloneCollection & triggerObjectMatches() const { return triggerObjectMatchesEmbedded_; };
      /// get one matched trigger object by index
      const TriggerObjectStandAlone * triggerObjectMatch( const size_t idx = 0 ) const;
      /// get all matched trigger objects of a certain type;
      /// trigger object types are defined in 'enum trigger::TriggerObjectType' (DataFormats/HLTReco/interface/TriggerTypeDefs.h)
      const TriggerObjectStandAloneCollection triggerObjectMatchesByType( const trigger::TriggerObjectType triggerObjectType ) const;
      const TriggerObjectStandAloneCollection triggerObjectMatchesByType( const unsigned triggerObjectType ) const {
        return triggerObjectMatchesByType( trigger::TriggerObjectType( triggerObjectType ) );
      };
      // for backward compatibility
      const TriggerObjectStandAloneCollection triggerObjectMatchesByFilterID( const unsigned triggerObjectType ) const {
        return triggerObjectMatchesByType( trigger::TriggerObjectType( triggerObjectType ) );
      };
      /// get one matched trigger object of a certain type by index
      const TriggerObjectStandAlone * triggerObjectMatchByType( const trigger::TriggerObjectType triggerObjectType, const size_t idx = 0 ) const;
      const TriggerObjectStandAlone * triggerObjectMatchByType( const unsigned triggerObjectType, const size_t idx = 0 ) const {
        return triggerObjectMatchByType( trigger::TriggerObjectType( triggerObjectType ), idx );
      };
      // for backward compatibility
      const TriggerObjectStandAlone * triggerObjectMatchByFilterID( const unsigned triggerObjectType, const size_t idx = 0 ) const {
        return triggerObjectMatchByType( trigger::TriggerObjectType( triggerObjectType ), idx );
      };
      /// get all matched trigger objects from a certain collection
      const TriggerObjectStandAloneCollection triggerObjectMatchesByCollection( const std::string & coll ) const;
      // for RooT command line
      const TriggerObjectStandAloneCollection triggerObjectMatchesByCollection( const char * coll ) const {
        return triggerObjectMatchesByCollection( std::string( coll ) );
      };
      /// get one matched trigger object from a certain collection by index
      const TriggerObjectStandAlone * triggerObjectMatchByCollection( const std::string & coll, const size_t idx = 0 ) const;
      // for RooT command line
      const TriggerObjectStandAlone * triggerObjectMatchByCollection( const char * coll, const size_t idx = 0 ) const {
        return triggerObjectMatchByCollection( std::string( coll ), idx );
      };
      /// get all matched L1 objects used in a succeeding object combination of a certain L1 condition
      const TriggerObjectStandAloneCollection triggerObjectMatchesByCondition( const std::string & nameCondition ) const;
      // for RooT command line
      const TriggerObjectStandAloneCollection triggerObjectMatchesByCondition( const char * nameCondition ) const {
        return triggerObjectMatchesByCondition( std::string( nameCondition ) );
      };
      /// get one matched L1 object used in a succeeding object combination of a certain L1 condition by index
      const TriggerObjectStandAlone * triggerObjectMatchByCondition( const std::string & nameCondition, const size_t idx = 0 ) const;
      // for RooT command line
      const TriggerObjectStandAlone * triggerObjectMatchByCondition( const char * nameCondition, const size_t idx = 0 ) const {
        return triggerObjectMatchByCondition( std::string( nameCondition ), idx );
      };
      /// get all matched L1 objects used in a succeeding object combination of a condition in a certain L1 (physics) algorithm;
      /// if 'algoCondAccepted' is set to 'true' (default), only objects used in succeeding conditions of succeeding algorithms are considered
      /// ("firing" objects)
      const TriggerObjectStandAloneCollection triggerObjectMatchesByAlgorithm( const std::string & nameAlgorithm, const bool algoCondAccepted = true ) const;
      // for RooT command line
      const TriggerObjectStandAloneCollection triggerObjectMatchesByAlgorithm( const char * nameAlgorithm, const bool algoCondAccepted = true ) const {
        return triggerObjectMatchesByAlgorithm( std::string( nameAlgorithm ), algoCondAccepted );
      };
      // for the cut string parser
      const TriggerObjectStandAloneCollection triggerObjectMatchesByAlgorithm( const std::string & nameAlgorithm, const unsigned algoCondAccepted ) const {
        return triggerObjectMatchesByAlgorithm( nameAlgorithm, bool( algoCondAccepted ) );
      };
      // for RooT command line and the cut string parser
      const TriggerObjectStandAloneCollection triggerObjectMatchesByAlgorithm( const char * nameAlgorithm, const unsigned algoCondAccepted ) const {
        return triggerObjectMatchesByAlgorithm( std::string( nameAlgorithm ), bool( algoCondAccepted ) );
      };
      /// get one matched L1 object used in a succeeding object combination of a condition in a certain L1 (physics) algorithm by index;
      /// if 'algoCondAccepted' is set to 'true' (default), only objects used in succeeding conditions of succeeding algorithms are considered
      /// ("firing" objects)
      const TriggerObjectStandAlone * triggerObjectMatchByAlgorithm( const std::string & nameAlgorithm, const bool algoCondAccepted = true, const size_t idx = 0 ) const;
      // for RooT command line
      const TriggerObjectStandAlone * triggerObjectMatchByAlgorithm( const char * nameAlgorithm, const bool algoCondAccepted = true, const size_t idx = 0 ) const {
        return triggerObjectMatchByAlgorithm( std::string( nameAlgorithm ), algoCondAccepted, idx );
      };
      // for the cut string parser
      const TriggerObjectStandAlone * triggerObjectMatchByAlgorithm( const std::string & nameAlgorithm, const unsigned algoCondAccepted, const size_t idx = 0 ) const {
        return triggerObjectMatchByAlgorithm( nameAlgorithm, bool( algoCondAccepted ), idx );
      };
      // for RooT command line and the cut string parser
      const TriggerObjectStandAlone * triggerObjectMatchByAlgorithm( const char * nameAlgorithm, const unsigned algoCondAccepted, const size_t idx = 0 ) const {
        return triggerObjectMatchByAlgorithm( std::string( nameAlgorithm ), bool( algoCondAccepted ), idx );
      };
      /// get all matched HLT objects used in a certain HLT filter
      const TriggerObjectStandAloneCollection triggerObjectMatchesByFilter( const std::string & labelFilter ) const;
      // for RooT command line
      const TriggerObjectStandAloneCollection triggerObjectMatchesByFilter( const char * labelFilter ) const {
        return triggerObjectMatchesByFilter( std::string( labelFilter ) );
      };
      /// get one matched HLT object used in a certain HLT filter by index
      const TriggerObjectStandAlone * triggerObjectMatchByFilter( const std::string & labelFilter, const size_t idx = 0 ) const;
      // for RooT command line
      const TriggerObjectStandAlone * triggerObjectMatchByFilter( const char * labelFilter, const size_t idx = 0 ) const {
        return triggerObjectMatchByFilter( std::string( labelFilter ), idx );
      };
      /// get all matched HLT objects used in a certain HLT path;
      /// if 'pathLastFilterAccepted' is set to 'true' (default), only objects used in the final filter of a succeeding path are considered
      /// ("firing" objects old style only valid for single object triggers);
      /// if 'pathL3FilterAccepted' is set to 'true' (default), only objects used in L3 filters (identified by the "saveTags" parameter being 'true')
      /// of a succeeding path are considered ("firing" objects old style only valid for single object triggers)
      const TriggerObjectStandAloneCollection triggerObjectMatchesByPath( const std::string & namePath, const bool pathLastFilterAccepted = false, const bool pathL3FilterAccepted = true ) const;
      // for RooT command line
      const TriggerObjectStandAloneCollection triggerObjectMatchesByPath( const char * namePath, const bool pathLastFilterAccepted = false, const bool pathL3FilterAccepted = true ) const {
        return triggerObjectMatchesByPath( std::string( namePath ), pathLastFilterAccepted, pathL3FilterAccepted );
      };
      // for the cut string parser
      const TriggerObjectStandAloneCollection triggerObjectMatchesByPath( const std::string & namePath, const unsigned pathLastFilterAccepted, const unsigned pathL3FilterAccepted = 1 ) const {
        return triggerObjectMatchesByPath( namePath, bool( pathLastFilterAccepted ), bool( pathL3FilterAccepted ) );
      };
      // for RooT command line and the cut string parser
      const TriggerObjectStandAloneCollection triggerObjectMatchesByPath( const char * namePath, const unsigned pathLastFilterAccepted, const unsigned pathL3FilterAccepted = 1 ) const {
        return triggerObjectMatchesByPath( std::string( namePath ), bool( pathLastFilterAccepted ), bool( pathL3FilterAccepted ) );
      };
      /// get one matched HLT object used in a certain HLT path by index;
      /// if 'pathLastFilterAccepted' is set to 'true' (default), only objects used in the final filter of a succeeding path are considered
      /// ("firing" objects, old style only valid for single object triggers);
      /// if 'pathL3FilterAccepted' is set to 'true' (default), only objects used in L3 filters (identified by the "saveTags" parameter being 'true')
      /// of a succeeding path are considered ("firing" objects also valid for x-triggers)
      const TriggerObjectStandAlone * triggerObjectMatchByPath( const std::string & namePath, const bool pathLastFilterAccepted = false, const bool pathL3FilterAccepted = true, const size_t idx = 0 ) const;
      // for RooT command line
      const TriggerObjectStandAlone * triggerObjectMatchByPath( const char * namePath, const bool pathLastFilterAccepted = false, const bool pathL3FilterAccepted = true, const size_t idx = 0 ) const {
        return triggerObjectMatchByPath( std::string( namePath ), pathLastFilterAccepted, pathL3FilterAccepted, idx );
      };
      // for the cut string parser
      const TriggerObjectStandAlone * triggerObjectMatchByPath( const std::string & namePath, const unsigned pathLastFilterAccepted, const unsigned pathL3FilterAccepted = 1, const size_t idx = 0 ) const {
        return triggerObjectMatchByPath( namePath, bool( pathLastFilterAccepted ), bool( pathL3FilterAccepted ), idx );
      };
      // for RooT command line and the cut string parser
      const TriggerObjectStandAlone * triggerObjectMatchByPath( const char * namePath, const unsigned pathLastFilterAccepted, const unsigned pathL3FilterAccepted = 1, const size_t idx = 0 ) const {
        return triggerObjectMatchByPath( std::string( namePath ), bool( pathLastFilterAccepted ), bool( pathL3FilterAccepted ), idx );
      };
      /// add a trigger match
      void addTriggerObjectMatch( const TriggerObjectStandAlone & trigObj ) { triggerObjectMatchesEmbedded_.push_back( trigObj ); };
      /// unpack path names of matched trigger objects (if they were packed before embedding, which is not normally the case)
      void unpackTriggerObjectPathNames( const edm::TriggerNames &names ) {
        for (std::vector<TriggerObjectStandAlone>::iterator it = triggerObjectMatchesEmbedded_.begin(), ed = triggerObjectMatchesEmbedded_.end(); it != ed; ++it) it->unpackPathNames(names);
      }

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
      /// If status == 0, only the pdgId will be checked; likewise, if pdgId == 0, only the status will be checked.
      /// When autoCharge is set to true, and a charged reco particle is matched to a charged gen particle,
      /// positive pdgId means 'same charge', negative pdgId means 'opposite charge';
      /// for example, electron.genParticleById(11,0,true) will get an e^+ matched to e^+ or e^- matched to e^-,
      /// while genParticleById(-15,0,true) will get e^+ matched to e^- or vice versa.
      /// If a neutral reco particle is matched to a charged gen particle, the sign of the pdgId passed to getParticleById must match that of the gen particle;
      /// for example photon.getParticleById(11) will match gamma to e^-, while genParticleById(-11) will match gamma to e^+ (pdgId=-11)
      // implementation note: uint8_t instead of bool, because the string parser doesn't allow bool currently
      reco::GenParticleRef      genParticleById(int pdgId, int status, uint8_t autoCharge=0) const ;

      /// Get generator level particle, as C++ pointer (might be 0 if the ref was null)
      /// If you stored multiple GenParticles, you can specify which one you want.
      const reco::GenParticle * genParticle(size_t idx=0)    const {
            reco::GenParticleRef ref = genParticleRef(idx);
            return ref.isNonnull() ? ref.get() : nullptr;
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

      /// Returns true if there was at least one overlap for this test label
      bool hasOverlaps(const std::string &label) const ;
      /// Return the list of overlaps for one label (can be empty)
      /// The original ordering of items is kept (usually it's by increasing deltaR from this item)
      const reco::CandidatePtrVector & overlaps(const std::string &label) const ;
      /// Returns the labels of the overlap tests that found at least one overlap
      const std::vector<std::string> & overlapLabels() const { return overlapLabels_; }
      /// Sets the list of overlapping items for one label
      /// Note that adding an empty PtrVector has no effect at all
      /// Items within the list should already be sorted appropriately (this method won't sort them)
      void setOverlaps(const std::string &label, const reco::CandidatePtrVector & overlaps) ;

      /// Returns user-defined data. Returns NULL if the data is not present, or not of type T.
      template<typename T> const T * userData(const std::string &key) const {
          const pat::UserData * data = userDataObject_(key);
          return (data != nullptr ? data->template get<T>() : nullptr);

      }
      /// Check if user data with a specific type is present
      bool hasUserData(const std::string &key) const {
          return (userDataObject_(key) != nullptr);
      }
      /// Get human-readable type of user data object, for debugging
      const std::string & userDataObjectType(const std::string &key) const {
          const pat::UserData * data = userDataObject_(key);
          return (data != nullptr ? data->typeName() : pat_statics::EMPTY_STR);
      };
      /// Get list of user data object names
      const std::vector<std::string> & userDataNames() const  { return userDataLabels_; }

      /// Get the data as a void *, for CINT usage.
      /// COMPLETELY UNSUPPORTED, USE ONLY FOR DEBUGGING
      const void * userDataBare(const std::string &key) const {
          const pat::UserData * data = userDataObject_(key);
          return (data != nullptr ? data->bareData() : nullptr);
      }

      /// Set user-defined data
      /// Needs dictionaries for T and for pat::UserHolder<T>,
      /// and it will throw exception if they're missing,
      /// unless transientOnly is set to true
      template<typename T>
      void addUserData( const std::string & label, const T & data, bool transientOnly=false, bool overwrite=false ) {
        std::unique_ptr<pat::UserData> made(pat::UserData::make<T>(data, transientOnly));
        addUserDataObject_( label, std::move(made), overwrite );
      }

      /// Set user-defined data. To be used only to fill from ValueMap<Ptr<UserData>>
      /// Do not use unless you know what you are doing.
      void addUserDataFromPtr( const std::string & label, const edm::Ptr<pat::UserData> & data, bool overwrite=false ) {
        std::unique_ptr<pat::UserData> cloned(data->clone());
        addUserDataObject_( label, std::move(cloned), overwrite );
      }

      /// Get user-defined float
      /// Note: throws if the key is not found; you can check if the key exists with 'hasUserFloat' method.
      float userFloat( const std::string & key ) const;
      /// return a range of values corresponding to key
      std::vector<float> userFloatRange( const std::string& key ) const;
      /// a CINT-friendly interface
      float userFloat( const char* key ) const { return userFloat( std::string(key) ); }
      
      /// Set user-defined float
      void addUserFloat( const  std::string & label, float data, const bool overwrite = false );
      /// Get list of user-defined float names
      const std::vector<std::string> & userFloatNames() const  { return userFloatLabels_; }
      /// Return true if there is a user-defined float with a given name
      bool hasUserFloat( const std::string & key ) const {
        auto it = std::lower_bound(userFloatLabels_.cbegin(), userFloatLabels_.cend(), key); 
        return ( it != userFloatLabels_.cend() && *it == key );
      }
      /// a CINT-friendly interface
      bool hasUserFloat( const char* key ) const {return hasUserFloat( std::string(key) );}

      /// Get user-defined int
      /// Note: throws if the key is not found; you can check if the key exists with 'hasUserInt' method.
      int32_t userInt( const std::string & key ) const;
      /// returns a range of values corresponding to key
      std::vector<int> userIntRange( const std::string& key ) const;
      /// Set user-defined int
      void addUserInt( const std::string & label,  int32_t data, const bool overwrite = false );
      /// Get list of user-defined int names
      const std::vector<std::string> & userIntNames() const  { return userIntLabels_; }
      /// Return true if there is a user-defined int with a given name
      bool hasUserInt( const std::string & key ) const {
        auto it = std::lower_bound(userIntLabels_.cbegin(), userIntLabels_.cend(), key);
        return ( it != userIntLabels_.cend() && *it == key );
      }

      /// Get user-defined candidate ptr
      /// Note: it will a null pointer if the key is not found; you can check if the key exists with 'hasUserInt' method.
      reco::CandidatePtr userCand( const std::string & key ) const;
      /// Set user-defined int
      void addUserCand( const std::string & label,  const reco::CandidatePtr & data, const bool overwrite = false );
      /// Get list of user-defined cand names
      const std::vector<std::string> & userCandNames() const  { return userCandLabels_; }
      /// Return true if there is a user-defined int with a given name
      bool hasUserCand( const std::string & key ) const {
        auto it = std::lower_bound(userCandLabels_.cbegin(), userCandLabels_.cend(), key);
        return ( it != userCandLabels_.cend() && *it == key );
      }

      // === New Kinematic Resolutions
      /// Return the kinematic resolutions associated to this object, possibly specifying a label for it.
      /// If not present, it will throw an exception.
      const pat::CandKinResolution & getKinResolution(const std::string &label="") const ;

      /// Check if the kinematic resolutions are stored into this object (possibly specifying a label for them)
      bool hasKinResolution(const std::string &label="") const ;

      /// Add a kinematic resolution to this object (possibly with a label)
      void setKinResolution(const pat::CandKinResolution &resol, const std::string &label="") ;

      /// Resolution on eta, possibly with a label to specify which resolution to use
      double resolEta(const std::string &label="") const { return getKinResolution(label).resolEta(this->p4()); }

      /// Resolution on theta, possibly with a label to specify which resolution to use
      double resolTheta(const std::string &label="") const { return getKinResolution(label).resolTheta(this->p4()); }

      /// Resolution on phi, possibly with a label to specify which resolution to use
      double resolPhi(const std::string &label="") const { return getKinResolution(label).resolPhi(this->p4()); }

      /// Resolution on energy, possibly with a label to specify which resolution to use
      double resolE(const std::string &label="") const { return getKinResolution(label).resolE(this->p4()); }

      /// Resolution on et, possibly with a label to specify which resolution to use
      double resolEt(const std::string &label="") const { return getKinResolution(label).resolEt(this->p4()); }

      /// Resolution on p, possibly with a label to specify which resolution to use
      double resolP(const std::string &label="") const { return getKinResolution(label).resolP(this->p4()); }

      /// Resolution on pt, possibly with a label to specify which resolution to use
      double resolPt(const std::string &label="") const { return getKinResolution(label).resolPt(this->p4()); }

      /// Resolution on 1/p, possibly with a label to specify which resolution to use
      double resolPInv(const std::string &label="") const { return getKinResolution(label).resolPInv(this->p4()); }

      /// Resolution on px, possibly with a label to specify which resolution to use
      double resolPx(const std::string &label="") const { return getKinResolution(label).resolPx(this->p4()); }

      /// Resolution on py, possibly with a label to specify which resolution to use
      double resolPy(const std::string &label="") const { return getKinResolution(label).resolPy(this->p4()); }

      /// Resolution on pz, possibly with a label to specify which resolution to use
      double resolPz(const std::string &label="") const { return getKinResolution(label).resolPz(this->p4()); }

      /// Resolution on mass, possibly with a label to specify which resolution to use
      /// Note: this will be zero if a mass-constrained parametrization is used for this object
      double resolM(const std::string &label="") const { return getKinResolution(label).resolM(this->p4()); }



    protected:
      // reference back to the original object
      edm::Ptr<reco::Candidate> refToOrig_;

      /// vector of trigger matches
      TriggerObjectStandAloneCollection triggerObjectMatchesEmbedded_;

      /// vector of the efficiencies (values)
      std::vector<pat::LookupTableRecord> efficiencyValues_;
      /// vector of the efficiencies (names)
      std::vector<std::string> efficiencyNames_;

      /// Reference to a generator level particle
      std::vector<reco::GenParticleRef> genParticleRef_;
      /// vector to hold an embedded generator level particle
      std::vector<reco::GenParticle>    genParticleEmbedded_;

      /// Overlapping test labels (only if there are any overlaps)
      std::vector<std::string> overlapLabels_;
      /// Overlapping items (sorted by distance)
      std::vector<reco::CandidatePtrVector> overlapItems_;

      /// User data object
      std::vector<std::string>      userDataLabels_;
      pat::UserDataCollection       userDataObjects_;
      // User float values
      std::vector<std::string>      userFloatLabels_;
      std::vector<float>            userFloats_;
      // User int values
      std::vector<std::string>      userIntLabels_;
      std::vector<int32_t>          userInts_;
      // User candidate matches
      std::vector<std::string>        userCandLabels_;
      std::vector<reco::CandidatePtr> userCands_;

      /// Kinematic resolutions.
      std::vector<pat::CandKinResolution> kinResolutions_;
      /// Labels for the kinematic resolutions.
      /// if (kinResolutions_.size() == kinResolutionLabels_.size()+1), then the first resolution has no label.
      std::vector<std::string>            kinResolutionLabels_;

      void addUserDataObject_( const std::string & label, std::unique_ptr<pat::UserData> value, bool overwrite = false ) ;

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
      return nullptr;
    } else if (!refToOrig_.isAvailable()) {
      throw edm::Exception(edm::errors::ProductNotFound) << "The original collection from which this PAT object was made is not present any more in the event, hence you cannot access the originating object anymore.";
    } else {
      return refToOrig_.get();
    }
  }

  template <class ObjectType>
  const edm::Ptr<reco::Candidate> & PATObject<ObjectType>::originalObjectRef() const { return refToOrig_; }

  template <class ObjectType>
  const TriggerObjectStandAlone * PATObject<ObjectType>::triggerObjectMatch( const size_t idx ) const {
    if ( idx >= triggerObjectMatches().size() ) return nullptr;
    TriggerObjectStandAloneRef ref( &triggerObjectMatchesEmbedded_, idx );
    return ref.isNonnull() ? ref.get() : nullptr;
  }

  template <class ObjectType>
  const TriggerObjectStandAloneCollection PATObject<ObjectType>::triggerObjectMatchesByType( const trigger::TriggerObjectType triggerObjectType ) const {
    TriggerObjectStandAloneCollection matches;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasTriggerObjectType( triggerObjectType ) ) matches.push_back( *( triggerObjectMatch( i ) ) );
    }
    return matches;
  }

  template <class ObjectType>
  const TriggerObjectStandAlone * PATObject<ObjectType>::triggerObjectMatchByType( const trigger::TriggerObjectType triggerObjectType, const size_t idx ) const {
    std::vector< size_t > refs;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasTriggerObjectType( triggerObjectType ) ) refs.push_back( i );
    }
    if ( idx >= refs.size() ) return nullptr;
    TriggerObjectStandAloneRef ref( &triggerObjectMatchesEmbedded_, refs.at( idx ) );
    return ref.isNonnull() ? ref.get() : nullptr;
  }

  template <class ObjectType>
  const TriggerObjectStandAloneCollection PATObject<ObjectType>::triggerObjectMatchesByCollection( const std::string & coll ) const {
    TriggerObjectStandAloneCollection matches;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasCollection( coll ) ) matches.push_back( *( triggerObjectMatch( i ) ) );
    }
    return matches;
  }

  template <class ObjectType>
  const TriggerObjectStandAlone * PATObject<ObjectType>::triggerObjectMatchByCollection( const std::string & coll, const size_t idx ) const {
    std::vector< size_t > refs;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasCollection( coll ) ) {
        refs.push_back( i );
      }
    }
    if ( idx >= refs.size() ) return nullptr;
    TriggerObjectStandAloneRef ref( &triggerObjectMatchesEmbedded_, refs.at( idx ) );
    return ref.isNonnull() ? ref.get() : nullptr;
  }

  template <class ObjectType>
  const TriggerObjectStandAloneCollection PATObject<ObjectType>::triggerObjectMatchesByCondition( const std::string & nameCondition ) const {
    TriggerObjectStandAloneCollection matches;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasConditionName( nameCondition ) ) matches.push_back( *( triggerObjectMatch( i ) ) );
    }
    return matches;
  }

  template <class ObjectType>
  const TriggerObjectStandAlone * PATObject<ObjectType>::triggerObjectMatchByCondition( const std::string & nameCondition, const size_t idx ) const {
    std::vector< size_t > refs;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasConditionName( nameCondition ) ) refs.push_back( i );
    }
    if ( idx >= refs.size() ) return nullptr;
    TriggerObjectStandAloneRef ref( &triggerObjectMatchesEmbedded_, refs.at( idx ) );
    return ref.isNonnull() ? ref.get() : nullptr;
  }

  template <class ObjectType>
  const TriggerObjectStandAloneCollection PATObject<ObjectType>::triggerObjectMatchesByAlgorithm( const std::string & nameAlgorithm, const bool algoCondAccepted ) const {
    TriggerObjectStandAloneCollection matches;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasAlgorithmName( nameAlgorithm, algoCondAccepted ) ) matches.push_back( *( triggerObjectMatch( i ) ) );
    }
    return matches;
  }

  template <class ObjectType>
  const TriggerObjectStandAlone * PATObject<ObjectType>::triggerObjectMatchByAlgorithm( const std::string & nameAlgorithm, const bool algoCondAccepted, const size_t idx ) const {
    std::vector< size_t > refs;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasAlgorithmName( nameAlgorithm, algoCondAccepted ) ) refs.push_back( i );
    }
    if ( idx >= refs.size() ) return nullptr;
    TriggerObjectStandAloneRef ref( &triggerObjectMatchesEmbedded_, refs.at( idx ) );
    return ref.isNonnull() ? ref.get() : nullptr;
  }

  template <class ObjectType>
  const TriggerObjectStandAloneCollection PATObject<ObjectType>::triggerObjectMatchesByFilter( const std::string & labelFilter ) const {
    TriggerObjectStandAloneCollection matches;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasFilterLabel( labelFilter ) ) matches.push_back( *( triggerObjectMatch( i ) ) );
    }
    return matches;
  }

  template <class ObjectType>
  const TriggerObjectStandAlone * PATObject<ObjectType>::triggerObjectMatchByFilter( const std::string & labelFilter, const size_t idx ) const {
    std::vector< size_t > refs;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasFilterLabel( labelFilter ) ) refs.push_back( i );
    }
    if ( idx >= refs.size() ) return nullptr;
    TriggerObjectStandAloneRef ref( &triggerObjectMatchesEmbedded_, refs.at( idx ) );
    return ref.isNonnull() ? ref.get() : nullptr;
  }

  template <class ObjectType>
  const TriggerObjectStandAloneCollection PATObject<ObjectType>::triggerObjectMatchesByPath( const std::string & namePath, const bool pathLastFilterAccepted, const bool pathL3FilterAccepted ) const {
    TriggerObjectStandAloneCollection matches;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != nullptr && triggerObjectMatch( i )->hasPathName( namePath, pathLastFilterAccepted, pathL3FilterAccepted ) ) matches.push_back( *( triggerObjectMatch( i ) ) );
    }
    return matches;
  }

  template <class ObjectType>
  const TriggerObjectStandAlone * PATObject<ObjectType>::triggerObjectMatchByPath( const std::string & namePath, const bool pathLastFilterAccepted, const bool pathL3FilterAccepted, const size_t idx ) const {
    std::vector< size_t > refs;
    for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
      if ( triggerObjectMatch( i ) != 0 && triggerObjectMatch( i )->hasPathName( namePath, pathLastFilterAccepted, pathL3FilterAccepted ) ) refs.push_back( i );
    }
    if ( idx >= refs.size() ) return nullptr;
    TriggerObjectStandAloneRef ref( &triggerObjectMatchesEmbedded_, refs.at( idx ) );
    return ref.isNonnull() ? ref.get() : nullptr;
  }

  template <class ObjectType>
  const pat::LookupTableRecord &
  PATObject<ObjectType>::efficiency(const std::string &name) const {
    // find the name in the (sorted) list of names
    auto it = std::lower_bound(efficiencyNames_.cbegin(), efficiencyNames_.cend(), name);
    if ((it == efficiencyNames_.end()) || (*it != name)) {
        throw cms::Exception("Invalid Label") << "There is no efficiency with name '" << name << "' in this PAT Object\n";
    }
    return efficiencyValues_[std::distance(efficiencyNames_.cbegin(),it)];
  }

  template <class ObjectType>
  std::vector<std::pair<std::string,pat::LookupTableRecord> >
  PATObject<ObjectType>::efficiencies() const {
    std::vector<std::pair<std::string,pat::LookupTableRecord> > ret;
    std::vector<std::string>::const_iterator itn = efficiencyNames_.begin(), edn = efficiencyNames_.end();
    std::vector<pat::LookupTableRecord>::const_iterator itv = efficiencyValues_.begin();
    for ( ; itn != edn; ++itn, ++itv) {
        ret.emplace_back( *itn, *itv);
    }
    return ret;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::setEfficiency(const std::string &name, const pat::LookupTableRecord & value) {
    // look for the name, or to the place where we can insert it without violating the alphabetic order
    auto it = std::lower_bound(efficiencyNames_.begin(), efficiencyNames_.end(), name);
    const auto dist = std::distance(efficiencyNames_.begin(),it);
    if (it == efficiencyNames_.end()) { // insert at the end
        efficiencyNames_.push_back(name);
        efficiencyValues_.push_back(value);
    } else if (*it == name) {           // replace existing
        efficiencyValues_[dist] = value;
    } else {                            // insert in the middle :-(
        efficiencyNames_. insert(it, name);
        efficiencyValues_.insert( efficiencyValues_.begin() + dist, value );
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
  reco::GenParticleRef PATObject<ObjectType>::genParticleById(int pdgId, int status, uint8_t autoCharge) const {
        // get a vector, avoiding an unneeded copy if there is no embedding
        const std::vector<reco::GenParticleRef> & vec = (genParticleEmbedded_.empty() ? genParticleRef_ : genParticleRefs());
        for (std::vector<reco::GenParticleRef>::const_iterator ref = vec.begin(), end = vec.end(); ref != end; ++ref) {
            if (ref->isNonnull()) {
                const reco::GenParticle & g = **ref;
                if ((status != 0) && (g.status() != status)) continue;
                if (pdgId == 0) {
                    return *ref;
                } else if (!autoCharge) {
                    if (pdgId == g.pdgId()) return *ref;
                } else if (abs(pdgId) == abs(g.pdgId())) {
                    // I want pdgId > 0 to match "correct charge" (for charged particles)
                    if (g.charge() == 0) return *ref;
                    else if ((this->charge() == 0) && (pdgId == g.pdgId())) return *ref;
                    else if (g.charge()*this->charge()*pdgId > 0) return *ref;
                }
            }
        }
        return reco::GenParticleRef();
  }

  template <class ObjectType>
  bool PATObject<ObjectType>::hasOverlaps(const std::string &label) const {
        auto match = std::lower_bound(overlapLabels_.cbegin(), overlapLabels_.cend(), label);
        return ( match != overlapLabels_.end() && *match == label );
  }

  template <class ObjectType>
  const reco::CandidatePtrVector & PATObject<ObjectType>::overlaps(const std::string &label) const {
        
        auto match = std::lower_bound(overlapLabels_.cbegin(), overlapLabels_.cend(), label);
        if( match == overlapLabels_.cend() || *match != label ) return pat_statics::EMPTY_CPV;
        return overlapItems_[std::distance(overlapLabels_.begin(),match)];
  }

  template <class ObjectType>
  void PATObject<ObjectType>::setOverlaps(const std::string &label, const reco::CandidatePtrVector & overlaps) {
      
        auto match = std::lower_bound(overlapLabels_.begin(), overlapLabels_.end(), label);
        const auto dist = std::distance(overlapLabels_.begin(),match);
        if ( match == overlapLabels_.end() || *match != label ) {
          overlapLabels_.insert(match,label);
          overlapItems_.insert(overlapItems_.begin()+dist,overlaps);
        } else {
          overlapItems_[dist] = overlaps;
        }        
  }

  template <class ObjectType>
  const pat::UserData * PATObject<ObjectType>::userDataObject_( const std::string & key ) const
  {
    auto it = std::lower_bound(userDataLabels_.cbegin(),userDataLabels_.cend(),key);
    if ( it != userDataLabels_.cend() && *it == key) {
      return &userDataObjects_[std::distance(userDataLabels_.cbegin(),it)];
    }
    return nullptr;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::addUserDataObject_( const std::string & label, std::unique_ptr<pat::UserData> data, bool overwrite ) 
  {
    auto it = std::lower_bound(userDataLabels_.begin(), userDataLabels_.end(), label);
    const auto dist = std::distance(userDataLabels_.begin(), it);
    if( it == userDataLabels_.end() || *it != label ) {
        userDataLabels_.insert(it,label);
        userDataObjects_.insert(userDataObjects_.begin()+dist, std::move(data));
    } else if( overwrite ) {
        userDataObjects_.set(dist, std::move(data));
    } else {
        //create a range by adding behind the first entry
        userDataLabels_.insert(it+1,label);
        userDataObjects_.insert(userDataObjects_.begin()+dist+1, std::move(data));
    }
  }


  template <class ObjectType>
  float PATObject<ObjectType>::userFloat( const std::string &key ) const
  {
    auto it = std::lower_bound(userFloatLabels_.cbegin(),userFloatLabels_.cend(),key);
    if( it != userFloatLabels_.cend() && *it == key ) {
      return userFloats_[std::distance(userFloatLabels_.cbegin(),it)];
    }
    throwMissingLabel("UserFloat",key,userFloatLabels_);
    return std::numeric_limits<float>::quiet_NaN();
  }

  template<class ObjectType>
  std::vector<float> PATObject<ObjectType>::userFloatRange( const std::string& key ) const {
    auto range = std::equal_range(userFloatLabels_.cbegin(),userFloatLabels_.cend(),key);
    std::vector<float> result;
    result.reserve(std::distance(range.first,range.second));
    for( auto it = range.first; it != range.second; ++it ) {
      result.push_back(userFloats_[std::distance(userFloatLabels_.cbegin(),it)]);
    }
    return result;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::addUserFloat( const std::string & label,
					    float data,
                                            const bool overwrite )
  {
    auto it = std::lower_bound(userFloatLabels_.begin(),userFloatLabels_.end(),label);
    const auto dist = std::distance(userFloatLabels_.begin(),it);
    if( it == userFloatLabels_.end() || *it != label ) {      
      userFloatLabels_.insert(it,label);
      userFloats_.insert(userFloats_.begin()+dist,data);
    } else if( overwrite ) {
      userFloats_[ dist ] = data;
    } else {
      //create a range by adding behind the first entry
      userFloatLabels_.insert(it+1,label);
      userFloats_.insert(userFloats_.begin()+dist+1, data); 
    }
  }


  template <class ObjectType>
  int PATObject<ObjectType>::userInt( const std::string & key ) const
  {
    auto it = std::lower_bound(userIntLabels_.cbegin(), userIntLabels_.cend(), key);
    if ( it != userIntLabels_.cend() && *it == key ) {
      return userInts_[std::distance(userIntLabels_.cbegin(),it)];
    }
    throwMissingLabel("UserInt",key,userIntLabels_);
    return std::numeric_limits<int>::max();
  }

  template<class ObjectType>
  std::vector<int> PATObject<ObjectType>::userIntRange( const std::string& key ) const{
    auto range = std::equal_range(userIntLabels_.cbegin(),userIntLabels_.cend(),key);
    std::vector<int> result;
    result.reserve(std::distance(range.first,range.second));
    for( auto it = range.first; it != range.second; ++it ) {
      result.push_back(userInts_[std::distance(userIntLabels_.cbegin(),it)]);
    }
    return result;
  }

  template <class ObjectType>
  void PATObject<ObjectType>::addUserInt( const std::string &label,
                                          int data,
                                          bool overwrite )
  {
    auto it = std::lower_bound(userIntLabels_.begin(),userIntLabels_.end(),label);
    const auto dist = std::distance(userIntLabels_.begin(),it);
    if( it == userIntLabels_.end() || *it != label ) { 
      userIntLabels_.insert(it,label);
      userInts_.insert(userInts_.begin()+dist,data);
    } else if( overwrite ) {
      userInts_[dist] = data;
    } else {
      //create a range by adding behind the first entry
      userIntLabels_.insert(it+1, label);
      userInts_.insert(userInts_.begin()+dist+1,data);
    }
  }

  template <class ObjectType>
  reco::CandidatePtr PATObject<ObjectType>::userCand( const std::string & key ) const
  {
    auto it = std::lower_bound(userCandLabels_.cbegin(), userCandLabels_.cend(), key);
    if (it != userCandLabels_.cend()) {
      return userCands_[std::distance(userCandLabels_.begin(),it)];
    }
    return reco::CandidatePtr();
  }

  template <class ObjectType>
  void PATObject<ObjectType>::addUserCand( const std::string &label,
					   const reco::CandidatePtr & data,
                                           const bool overwrite )
  {
    auto it = std::lower_bound(userCandLabels_.begin(),userCandLabels_.end(),label);
    const auto dist = std::distance(userCandLabels_.begin(),it);
    if( it == userCandLabels_.end() || *it != label ) {      
      userCandLabels_.insert(it,label);
      userCands_.insert(userCands_.begin()+dist,data);
    } else if( overwrite ) {
      userCands_[dist] = data;
    } else {
      userCandLabels_.insert(it+1,label);
      userCands_.insert(userCands_.begin()+dist+1,data);
    }    
  }


  template <class ObjectType>
  const pat::CandKinResolution & PATObject<ObjectType>::getKinResolution(const std::string &label) const {
    const bool has_unlabelled = (kinResolutionLabels_.size()+1 == kinResolutions_.size());
    if (label.empty()) {
        if ( has_unlabelled ) {
            return kinResolutions_[0];
        } else {
            throw cms::Exception("Missing Data", "This object does not contain an un-labelled kinematic resolution");
        }
    } else {
        auto match = std::lower_bound(kinResolutionLabels_.cbegin(), kinResolutionLabels_.cend(), label);
        const auto dist = std::distance(kinResolutionLabels_.begin(),match);
        const size_t increment = ( has_unlabelled ? 1 : 0 );
        if ( match == kinResolutionLabels_.end() || *match != label ) {
            cms::Exception ex("Missing Data");
            ex << "This object does not contain a kinematic resolution with name '" << label << "'.\n";
            ex << "The known labels are: " ;
            for (std::vector<std::string>::const_iterator it = kinResolutionLabels_.cbegin(); it != kinResolutionLabels_.cend(); ++it) {
                ex << "'" << *it << "' ";
            }
            ex << "\n";
            throw ex;
        } else {           
            return kinResolutions_[dist+increment];          
        }
    }
  }

  template <class ObjectType>
  bool PATObject<ObjectType>::hasKinResolution(const std::string &label) const {
    if (label.empty()) {
        return (kinResolutionLabels_.size()+1 == kinResolutions_.size());
    } else {
        auto match = std::lower_bound(kinResolutionLabels_.cbegin(), kinResolutionLabels_.cend(), label);
        return ( match != kinResolutionLabels_.cend() && *match == label );
    }
  }

  template <class ObjectType>
  void PATObject<ObjectType>::setKinResolution(const pat::CandKinResolution &resol, const std::string &label) {
    const bool has_unlabelled = (kinResolutionLabels_.size()+1 == kinResolutions_.size());
    if (label.empty()) {
        if (has_unlabelled) {
            // There is already an un-labelled object. Replace it
            kinResolutions_[0] = resol;
        } else {
            // Insert. Note that the un-labelled is always the first, so we need to insert before begin()
            // (for an empty vector, this should not cost more than push_back)
            kinResolutions_.insert(kinResolutions_.begin(), resol);
        }
    } else {
        auto match = std::lower_bound(kinResolutionLabels_.begin(), kinResolutionLabels_.end(), label);
        const auto dist = std::distance(kinResolutionLabels_.begin(),match);
        const size_t increment = ( has_unlabelled ? 1 : 0 );
        if ( match != kinResolutionLabels_.end() && *match == label ) {
            // Existing object: replace
            kinResolutions_[dist+increment] = resol;
        } else {
            kinResolutionLabels_.insert(match,label);
            kinResolutions_.insert(kinResolutions_.begin()+dist+increment,resol);
        }
    }
  }




}

#endif
