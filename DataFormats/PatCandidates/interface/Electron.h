//
// $Id: Electron.h,v 1.12 2008/06/03 22:28:07 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    pat::Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   Electron implements the analysis-level electron class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Electron.h,v 1.12 2008/06/03 22:28:07 gpetrucc Exp $
*/


#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"

//#define PAT_patElectron_Default_eID    1 // allow a 'leptonID()' method with no argument, for a (configurable) default eID
//#define PAT_patElectron_Hardcoded_eIDs 1 // provide hard-coded methods as shortcuts for some standard eIDs
#define   PAT_patElectron_eID_Throw      1 // electron ID method will throw exception if the eID is missing 
                                           // (if you comment this line out, requests for missing IDs will just return -1.0)


namespace pat {


  typedef reco::GsfElectron ElectronType;
  typedef reco::GsfElectronCollection ElectronTypeCollection;


  class Electron : public Lepton<ElectronType> {

    public:
      typedef std::pair<std::string,float> IdPair; 

      Electron();
      Electron(const ElectronType & anElectron);
      Electron(const edm::RefToBase<ElectronType> & anElectronRef);
      Electron(const edm::Ptr<ElectronType> & anElectronRef);
      virtual ~Electron();

      virtual Electron * clone() const { return new Electron(*this); }

      /// override the ElectronType::gsfTrack method, to access the internal storage of the supercluster
      reco::GsfTrackRef gsfTrack() const;
      /// override the ElectronType::superCluster method, to access the internal storage of the supercluster
      reco::SuperClusterRef superCluster() const;
      /// override the ElectronType::track method, to access the internal storage of the track
      reco::TrackRef track() const;
      /// method to store the electron's supercluster internally
      void embedGsfTrack();
      /// method to store the electron's supercluster internally
      void embedSuperCluster();
      /// method to store the electron's supercluster internally
      void embedTrack();


// ========== Methods for electron ID ===================
      /// Returns a specific electron ID associated to the pat::Electron given its name
      /// For cut-based IDs, the value is 1.0 for good, 0.0 for bad.
      /// Note: an exception is thrown if the specified ID is not available
      float leptonID(const std::string & name) const ;
      /// Returns true if a specific ID is available in this pat::Electron
      bool isLeptonIDAvailable(const std::string & name) const;
      /// Returns all the electron IDs in the form of <name,value> pairs
      /// The 'default' ID is the first in the list
      const std::vector<IdPair> &  leptonIDs() const { return leptonIDs_; }
      /// Store multiple electron ID values, discarding existing ones
      /// The first one in the list becomes the 'default' electron id 
      void setLeptonIDs(const std::vector<IdPair> & ids) { leptonIDs_ = ids; }
#ifdef PAT_patElectron_Default_eID
      /// Returns the electron ID associated to the pat::Electron as 'default' ID
      /// For cut-based IDs, the value is 1.0 for good, 0.0 for bad.
      /// Note: an exception is thrown if no ID has been written in this pat::Electron
      float leptonID() const ;
      /// Return the name of the default electron ID name stored in this pat::Electron
      /// If no ID was stored, it returns the string "NULL" 
      const std::string & leptonIDname() const ;
      /// Store a single electron ID value, discarding existing ones
      /// This becomes the 'default' electron ID
      void setLeptonID(float id, const std::string & name = "") { leptonIDs_.clear(); leptonIDs_.push_back(IdPair(name,id)); }
#endif
#ifdef PAT_patElectron_Hardcoded_eIDs
      /// Checks if the electron has passed the 'robust' ID
      /// Note: an exception is thrown if no 'robust' ID is stored in the pat::Electron
      bool isRobustElectron() { return leptonID("robust") > 0.5; }   
      /// Checks if the electron has passed the 'loose' ID
      /// Note: an exception is thrown if no 'loose' ID is stored in the pat::Electron
      bool isLooseElectron() { return leptonID("loose") > 0.5; }   
      /// Checks if the electron has passed the 'tight' ID
      /// Note: an exception is thrown if no 'tight' ID is stored in the pat::Electron
      bool isTightElectron() { return leptonID("tight") > 0.5; }   
      /// Returns the value of the 'likelihood' electron ID
      /// Note: an exception is thrown if no 'likelihood' ID is stored in the pat::Electron
      float electronLikelihood() { return leptonID("likelihood"); }   
#endif

    protected:

      bool embeddedGsfTrack_;
      std::vector<reco::GsfTrack> gsfTrack_;
      bool embeddedSuperCluster_;
      std::vector<reco::SuperCluster> superCluster_;
      bool embeddedTrack_;
      std::vector<reco::Track> track_;

      std::vector<IdPair> leptonIDs_;

  };


}

#endif
