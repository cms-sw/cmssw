//
// $Id: GenericParticle.h,v 1.9 2010/03/02 21:02:46 rwolf Exp $
//

#ifndef DataFormats_PatCandidates_GenericParticle_h
#define DataFormats_PatCandidates_GenericParticle_h

/**
  \class    pat::GenericParticle GenericParticle.h "DataFormats/PatCandidates/interface/GenericParticle.h"
  \brief    Analysis-level Generic Particle class (e.g. for hadron or muon not fully reconstructed)

   GenericParticle implements the analysis-level generic particle class within the 'pat'
   namespace.

  \author   Giovanni Petrucciani
  \version  $Id: GenericParticle.h,v 1.9 2010/03/02 21:02:46 rwolf Exp $
*/

#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/PatCandidates/interface/Vertexing.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// Define typedefs for convenience
namespace pat {
  class GenericParticle;
  typedef std::vector<GenericParticle>              GenericParticleCollection; 
  typedef edm::Ref<GenericParticleCollection>       GenericParticleRef; 
  typedef edm::RefVector<GenericParticleCollection> GenericParticleRefVector; 
}


// Class definition
namespace pat {


  class GenericParticle : public PATObject<reco::RecoCandidate> {

    public:

      /// default constructor
      GenericParticle();
      /// constructor from Candidate
      GenericParticle(const reco::Candidate & aGenericParticle);
      /// constructor from ref to Candidate
      GenericParticle(const edm::RefToBase<reco::Candidate> & aGenericParticleRef);
      /// constructor from ref to Candidate
      GenericParticle(const edm::Ptr<reco::Candidate> & aGenericParticleRef);
      /// destructor
      virtual ~GenericParticle();

      /// required reimplementation of the Candidate's clone method
      virtual GenericParticle * clone() const { return new GenericParticle(*this); }

      /// Checks for overlap with another candidate. 
      /// It will return 'true' if the other candidate is a RecoCandidate, 
      /// and if they reference to at least one same non null track, supercluster or calotower (except for the multiple tracks)
      /// NOTE: It won't work with embedded references
      virtual bool overlap( const Candidate & ) const ;

      /// reference to a master track (might be transient refs if Tracks are embedded)
      /// returns null ref if there is no master track
      virtual reco::TrackRef track() const    { return track_.empty() ? trackRef_ : reco::TrackRef(&track_, 0); }
      /// reference to one of a set of multiple tracks (might be transient refs if Tracks are embedded)
      /// throws exception if idx >= numberOfTracks()
      virtual reco::TrackRef track( size_t idx ) const {  
            if (idx >= numberOfTracks()) throw cms::Exception("Index out of bounds") << "Requested track " << idx << " out of " << numberOfTracks() << ".\n";
            return (tracks_.empty() ? trackRefs_[idx] : reco::TrackRef(&tracks_, idx) );
      }
      /// number of multiple tracks (not including the master one)
      virtual size_t numberOfTracks() const { return tracks_.empty() ? trackRefs_.size() : tracks_.size(); }
      /// reference to a GsfTrack (might be transient ref if SuperCluster is embedded)
      /// returns null ref if there is no gsf track
      virtual reco::GsfTrackRef gsfTrack() const    { return (gsfTrack_.empty() ? gsfTrackRef_ : reco::GsfTrackRef(&gsfTrack_, 0)); }
      /// reference to a stand-alone muon Track (might be transient ref if SuperCluster is embedded)
      /// returns null ref if there is no stand-alone muon track
      virtual reco::TrackRef standAloneMuon() const { return (standaloneTrack_.empty() ? standaloneTrackRef_ : reco::TrackRef(&standaloneTrack_, 0)); }
      /// reference to a combined muon Track (might be transient ref if SuperCluster is embedded)
      /// returns null ref if there is no combined muon track
      virtual reco::TrackRef combinedMuon()   const { return (combinedTrack_.empty() ? combinedTrackRef_ : reco::TrackRef(&combinedTrack_, 0)); }
      /// reference to a SuperCluster (might be transient ref if SuperCluster is embedded)
      /// returns null ref if there is no supercluster
      virtual reco::SuperClusterRef superCluster() const { return superCluster_.empty() ? superClusterRef_ : reco::SuperClusterRef(&superCluster_, 0); }
      /// reference to a CaloTower  (might be transient ref if CaloTower is embedded)
      /// returns null ref if there is no calotower
      virtual CaloTowerRef caloTower() const { return caloTower_.empty() ? caloTowerRef_ : CaloTowerRef(&caloTower_, 0); }

      /// sets master track reference (or even embed it into the object)
      virtual void setTrack(const reco::TrackRef &ref, bool embed=false) ;
      /// sets multiple track references (or even embed the tracks into the object - whatch out for disk size issues!)
      virtual void setTracks(const reco::TrackRefVector &refs, bool embed=false) ;
      /// sets stand-alone muon track reference (or even embed it into the object)
      virtual void setStandAloneMuon(const reco::TrackRef &ref, bool embed=false) ;
      /// sets combined muon track reference (or even embed it into the object)
      virtual void setCombinedMuon(const reco::TrackRef &ref, bool embed=false) ;
      /// sets gsf track reference (or even embed it into the object)
      virtual void setGsfTrack(const reco::GsfTrackRef &ref, bool embed=false) ;
      /// sets supercluster reference (or even embed it into the object)
      virtual void setSuperCluster(const reco::SuperClusterRef &ref, bool embed=false) ;
      /// sets calotower reference (or even embed it into the object)
      virtual void setCaloTower(const CaloTowerRef &ref, bool embed=false) ;

      /// embeds the master track instead of keeping a reference to it      
      void embedTrack() ; 
      /// embeds the other tracks instead of keeping references
      void embedTracks() ; 
      /// embeds the stand-alone track instead of keeping a reference to it      
      void embedStandalone() ; 
      /// embeds the combined track instead of keeping a reference to it      
      void embedCombined() ; 
      /// embeds the gsf track instead of keeping a reference to it      
      void embedGsfTrack() ; 
      /// embeds the supercluster instead of keeping a reference to it      
      void embedSuperCluster() ; 
      /// embeds the calotower instead of keeping a reference to it      
      void embedCaloTower() ; 

      /// returns a user defined quality value, if set by the user to some meaningful value
      float quality() { return quality_; }
      /// sets a user defined quality value   
      void setQuality(float quality) { quality_ = quality; }
      
      //============ BEGIN ISOLATION BLOCK =====
      /// Returns the isolation variable for a specifc key (or 
      /// pseudo-key like CaloIso), or -1.0 if not available
      float userIsolation(IsolationKeys key) const { 
          if (key >= 0) {
              //if (key >= isolations_.size()) throw cms::Excepton("Missing Data") 
	      //<< "Isolation corresponding to key " 
	      //<< key << " was not stored for this particle.";
              if (size_t(key) >= isolations_.size()) return -1.0;
              return isolations_[key];
          } else switch (key) {
	  case pat::CaloIso:  
	    //if (isolations_.size() <= pat::HcalIso) throw cms::Excepton("Missing Data") 
	    //<< "CalIsoo Isolation was not stored for this particle.";
	    if (isolations_.size() <= pat::HcalIso) return -1.0; 
	    return isolations_[pat::EcalIso] + isolations_[pat::HcalIso];
	  default:
	    return -1.0;
	    //throw cms::Excepton("Missing Data") << "Isolation corresponding to key " 
	    //<< key << " was not stored for this particle.";
          }
      }
      /// Returns the isolation variable for string type function arguments
      /// (to be used with the cut-string parser);
      /// the possible values of the strings are the enums defined in
      /// DataFormats/PatCandidates/interface/Isolation.h
      float userIsolation(const std::string& key) const {
	// remove leading namespace specifier
	std::string prunedKey = ( key.find("pat::") == 0 ) ? std::string(key, 5) : key;
	if ( prunedKey == "TrackIso" ) return userIsolation(pat::TrackIso);
	if ( prunedKey == "EcalIso" ) return userIsolation(pat::EcalIso);
	if ( prunedKey == "HcalIso" ) return userIsolation(pat::HcalIso);
	if ( prunedKey == "PfAllParticleIso" ) return userIsolation(pat::PfAllParticleIso);
	if ( prunedKey == "PfChargedHadronIso" ) return userIsolation(pat::PfChargedHadronIso);
	if ( prunedKey == "PfNeutralHadronIso" ) return userIsolation(pat::PfNeutralHadronIso);
	if ( prunedKey == "PfGammaIso" ) return userIsolation(pat::PfGammaIso);
	if ( prunedKey == "User1Iso" ) return userIsolation(pat::User1Iso);
	if ( prunedKey == "User2Iso" ) return userIsolation(pat::User2Iso);
	if ( prunedKey == "User3Iso" ) return userIsolation(pat::User3Iso);
	if ( prunedKey == "User4Iso" ) return userIsolation(pat::User4Iso);
	if ( prunedKey == "User5Iso" ) return userIsolation(pat::User5Iso);
	if ( prunedKey == "UserBaseIso" ) return userIsolation(pat::UserBaseIso);
	if ( prunedKey == "CaloIso" ) return userIsolation(pat::CaloIso);
	//throw cms::Excepton("Missing Data")
	//<< "Isolation corresponding to key " 
	//<< key << " was not stored for this particle.";
	return -1.0;
      }
      /// Sets the isolation variable for a specifc key.
      /// Note that you can't set isolation for a pseudo-key like CaloIso
      void setIsolation(IsolationKeys key, float value) {
          if (key >= 0) {
              if (size_t(key) >= isolations_.size()) isolations_.resize(key+1, -1.0);
              isolations_[key] = value;
          } else {
              throw cms::Exception("Illegal Argument") << 
                  "The key for which you're setting isolation does not correspond " <<
                  "to an individual isolation but to the sum of more independent isolations " <<
                  "(e.g. Calo = Ecal + Hcal), so you can't SET the value, just GET it.\n" <<
                  "Please set up each component independly.\n";
          }
      }

      // ---- specific getters ----
      /// Return the tracker isolation variable that was stored in this 
      /// object when produced, or -1.0 if there is none
      float trackIso() const { return userIsolation(pat::TrackIso); }
      /// Return the sum of ecal and hcal isolation variable that were 
      /// stored in this object when produced, or -1.0 if at least one 
      /// is missing
      float caloIso()  const { return userIsolation(pat::CaloIso); }
      /// Return the ecal isolation variable that was stored in this 
      /// object when produced, or -1.0 if there is none
      float ecalIso()  const { return userIsolation(pat::EcalIso); }
      /// Return the hcal isolation variable that was stored in this 
      /// object when produced, or -1.0 if there is none
      float hcalIso()  const { return userIsolation(pat::HcalIso); }

      // ---- specific setters ----
      /// Sets tracker isolation variable
      void setTrackIso(float trackIso) { setIsolation(pat::TrackIso, trackIso); }
      /// Sets ecal isolation variable
      void setEcalIso(float caloIso)   { setIsolation(pat::EcalIso, caloIso);  } 
      /// Sets hcal isolation variable
      void setHcalIso(float caloIso)   { setIsolation(pat::HcalIso, caloIso);  }
      /// Sets user isolation variable #index
      void setUserIso(float value, uint8_t index=0)  { setIsolation(IsolationKeys(UserBaseIso + index), value); }


      //============ BEGIN ISODEPOSIT BLOCK =====
      /// Returns the IsoDeposit associated with some key, or a null pointer if it is not available
      const IsoDeposit * isoDeposit(IsolationKeys key) const {
          for (IsoDepositPairs::const_iterator it = isoDeposits_.begin(), ed = isoDeposits_.end(); 
                  it != ed; ++it) 
          {
              if (it->first == key) return & it->second;
          }
          return 0;
      } 

      /// Sets the IsoDeposit associated with some key; if it is already existent, it is overwritten.
      void setIsoDeposit(IsolationKeys key, const IsoDeposit &dep) {
          IsoDepositPairs::iterator it = isoDeposits_.begin(), ed = isoDeposits_.end();
          for (; it != ed; ++it) {
              if (it->first == key) { it->second = dep; return; }
          }
          isoDeposits_.push_back(std::make_pair(key,dep));
      } 

      // ---- specific getters ----
      const IsoDeposit * trackIsoDeposit() const { return isoDeposit(pat::TrackIso); }
      const IsoDeposit * ecalIsoDeposit()  const { return isoDeposit(pat::EcalIso ); }
      const IsoDeposit * hcalIsoDeposit()  const { return isoDeposit(pat::HcalIso ); }
      const IsoDeposit * userIsoDeposit(uint8_t index=0) const { return isoDeposit(IsolationKeys(UserBaseIso + index)); }

      // ---- specific setters ----
      void trackIsoDeposit(const IsoDeposit &dep) { setIsoDeposit(pat::TrackIso, dep); }
      void ecalIsoDeposit(const IsoDeposit &dep)  { setIsoDeposit(pat::EcalIso,  dep); }
      void hcalIsoDeposit(const IsoDeposit &dep)  { setIsoDeposit(pat::HcalIso,  dep); }
      void userIsoDeposit(const IsoDeposit &dep, uint8_t index=0) { setIsoDeposit(IsolationKeys(UserBaseIso + index), dep); }

      /// Vertex association (or associations, if any). Return null pointer if none has been set
      const pat::VertexAssociation              * vertexAssociation(size_t index=0) const { return vtxAss_.size() > index ? & vtxAss_[index] : 0; }
      /// Vertex associations. Can be empty if it was not enabled in the config file
      const std::vector<pat::VertexAssociation> & vertexAssociations()              const { return vtxAss_; }
      /// Set a single vertex association
      void  setVertexAssociation(const pat::VertexAssociation &assoc)  { vtxAss_ = std::vector<pat::VertexAssociation>(1, assoc); }
      /// Set multiple vertex associations
      void  setVertexAssociations(const std::vector<pat::VertexAssociation> &assocs) { vtxAss_ = assocs; }
    protected:
      // Any sort of single tracks
      reco::TrackRef           trackRef_, standaloneTrackRef_, combinedTrackRef_; // ref
      reco::TrackCollection    track_,    standaloneTrack_,    combinedTrack_;    // embedded

      // GsfTrack
      reco::GsfTrackRef        gsfTrackRef_; // normal
      reco::GsfTrackCollection gsfTrack_;    // embedded

      // CaloTower
      CaloTowerRef           caloTowerRef_;    // ref
      CaloTowerCollection    caloTower_;       // embedded

      // SuperCluster
      reco::SuperClusterRef        superClusterRef_; // ref
      reco::SuperClusterCollection superCluster_;    // embedded

      // Multiple tracks
      reco::TrackRefVector  trackRefs_; // by ref
      reco::TrackCollection tracks_;    // embedded

      // information originally in external branches
      // some quality variable
      float quality_;

      // --- Isolation and IsoDeposit related datamebers ---
      typedef std::vector<std::pair<IsolationKeys, pat::IsoDeposit> > IsoDepositPairs;
      IsoDepositPairs    isoDeposits_;
      std::vector<float> isolations_;

      // --- Vertexing information
      std::vector<pat::VertexAssociation> vtxAss_;

      void fillInFrom(const reco::Candidate &cand); 
    
  };


}

#endif
