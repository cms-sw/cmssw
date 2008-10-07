//
// $Id: Electron.h,v 1.13 2008/06/13 09:55:35 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    pat::Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   Electron implements the analysis-level electron class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Electron.h,v 1.13 2008/06/13 09:55:35 gpetrucc Exp $
*/


#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


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
