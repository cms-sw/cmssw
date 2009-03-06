//
// $Id: Electron.h,v 1.18 2008/11/13 15:33:21 salerno Exp $
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
  \version  $Id: Electron.h,v 1.18 2008/11/13 15:33:21 salerno Exp $
*/


#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


// Define typedefs for convenience
namespace pat {
  class Electron;
  typedef std::vector<Electron>              ElectronCollection; 
  typedef edm::Ref<ElectronCollection>       ElectronRef; 
  typedef edm::RefVector<ElectronCollection> ElectronRefVector; 
}


// Class definition
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
      float electronID(const std::string & name) const;
      /// deprecated !!!
      float leptonID(const std::string & name) const {
        // avoid messagelogger in dataformats
        std::cout << "The pat::Electron::leptonID() method is deprecated. Please use the electronID() one instead" << std::endl;
        return electronID(name);
      }
      /// Returns true if a specific ID is available in this pat::Electron
      bool isElectronIDAvailable(const std::string & name) const;
      /// deprecated !!!
      bool isLeptonIDAvailable(const std::string & name) const {
        // avoid messagelogger in dataformats
        std::cout << "The pat::Electron::isLeptonIDAvailable() method is deprecated. Please use the isElectronIDAvailable() one instead" << std::endl;
        return isElectronIDAvailable(name);
      }
      /// Returns all the electron IDs in the form of <name,value> pairs
      /// The 'default' ID is the first in the list
      const std::vector<IdPair> &  electronIDs() const { return electronIDs_; }
      /// deprecated !!!
      const std::vector<IdPair> &  leptonIDs() const {
        // avoid messagelogger in dataformats
        std::cout << "The pat::Electron::leptonIDs() method is deprecated. Please use the electronIDs() one instead" << std::endl;
        return electronIDs();
      }
      /// Store multiple electron ID values, discarding existing ones
      /// The first one in the list becomes the 'default' electron id 
      void setElectronIDs(const std::vector<IdPair> & ids) { electronIDs_ = ids; }
      /// deprecated !!!
      void setLeptonIDs(const std::vector<IdPair> & ids) {
        // avoid messagelogger in dataformats
        std::cout << "The pat::Electron::setLeptonIDs() method is deprecated. Please use the setElectronIDs() one instead" << std::endl;
        setElectronIDs(ids);
      }
      /// Store the cluster shape variables associated to the electron
      void setClusterShapes ( const float& , const float& , const float& , const float& , const float& ) ;
      const float scSigmaEtaEta()   const { return  scSigmaEtaEta_ ; }
      const float scSigmaIEtaIEta() const { return  scSigmaIEtaIEta_ ; }  
      const float scE1x5()          const { return  scE1x5_ ; }
      const float scE2x5Max()       const { return  scE2x5Max_ ; }        
      const float scE5x5()          const { return  scE5x5_ ; }             

    protected:

      // ---- for content embedding ----
      bool embeddedGsfTrack_;
      std::vector<reco::GsfTrack> gsfTrack_;
      bool embeddedSuperCluster_;
      std::vector<reco::SuperCluster> superCluster_;
      bool embeddedTrack_;
      std::vector<reco::Track> track_;
      // ---- electron ID's holder ----
      std::vector<IdPair> electronIDs_;

      float scSigmaEtaEta_ ;
      float scSigmaIEtaIEta_ ; 
      float scE1x5_ ;
      float scE2x5Max_ ; 
      float scE5x5_ ; 

  };


}

#endif
