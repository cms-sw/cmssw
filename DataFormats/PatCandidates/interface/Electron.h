//
// $Id: Electron.h,v 1.14 2008/10/07 18:04:58 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    pat::Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   pat::Electron implements the analysis-level electron class within the
   'pat' namespace.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author   Steven Lowette, Giovanni Petrucciani, Frederic Ronga
  \version  $Id: Electron.h,v 1.14 2008/10/07 18:04:58 gpetrucc Exp $
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

      /// default constructor
      Electron();
      /// constructor from a reco electron
      Electron(const ElectronType & anElectron);
      /// constructor from a RefToBase to a reco electron (to be superseded by Ptr counterpart)
      Electron(const edm::RefToBase<ElectronType> & anElectronRef);
      /// constructor from a Ptr to a reco electron
      Electron(const edm::Ptr<ElectronType> & anElectronRef);
      /// destructor
      virtual ~Electron();

      /// required reimplementation of the Candidate's clone method
      virtual Electron * clone() const { return new Electron(*this); }

      // ---- methods for content embedding ----
      /// override the ElectronType::gsfTrack method, to access the internal storage of the supercluster
      reco::GsfTrackRef gsfTrack() const;
      /// override the ElectronType::superCluster method, to access the internal storage of the supercluster
      reco::SuperClusterRef superCluster() const;
      /// override the ElectronType::track method, to access the internal storage of the track
      reco::TrackRef track() const;
      /// method to store the electron's GsfTrack internally
      void embedGsfTrack();
      /// method to store the electron's SuperCluster internally
      void embedSuperCluster();
      /// method to store the electron's Track internally
      void embedTrack();

      // ---- methods for electron ID ----
      /// Returns a specific electron ID associated to the pat::Electron given its name
      /// For cut-based IDs, the value is 1.0 for good, 0.0 for bad.
      /// Note: an exception is thrown if the specified ID is not available
      float leptonID(const std::string & name) const;
      /// Returns true if a specific ID is available in this pat::Electron
      bool isLeptonIDAvailable(const std::string & name) const;
      /// Returns all the electron IDs in the form of <name,value> pairs
      /// The 'default' ID is the first in the list
      const std::vector<IdPair> &  leptonIDs() const { return leptonIDs_; }
      /// Store multiple electron ID values, discarding existing ones
      /// The first one in the list becomes the 'default' electron id 
      void setLeptonIDs(const std::vector<IdPair> & ids) { leptonIDs_ = ids; }

    protected:

      // ---- for content embedding ----
      bool embeddedGsfTrack_;
      std::vector<reco::GsfTrack> gsfTrack_;
      bool embeddedSuperCluster_;
      std::vector<reco::SuperCluster> superCluster_;
      bool embeddedTrack_;
      std::vector<reco::Track> track_;
      // ---- electron ID's holder ----
      std::vector<IdPair> leptonIDs_;

  };


}

#endif
