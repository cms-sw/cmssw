//
// $Id$
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    pat::Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   Electron implements the analysis-level electron class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id$
*/

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


namespace pat {


  typedef reco::GsfElectron ElectronType;
  typedef reco::GsfElectronCollection ElectronTypeCollection;


  class Electron : public Lepton<ElectronType> {

    public:

      Electron();
      Electron(const ElectronType & anElectron);
      Electron(const edm::RefToBase<ElectronType> & anElectronRef);
      virtual ~Electron();

      virtual Electron * clone() const { return new Electron(*this); }

      /// override the ElectronType::gsfTrack method, to access the internal storage of the supercluster
      reco::GsfTrackRef gsfTrack() const;
      /// override the ElectronType::superCluster method, to access the internal storage of the supercluster
      reco::SuperClusterRef superCluster() const;
      /// override the ElectronType::track method, to access the internal storage of the track
      reco::TrackRef track() const;
      float leptonID() const;
      float electronIDRobust() const;

      /// method to store the electron's supercluster internally
      void embedGsfTrack();
      /// method to store the electron's supercluster internally
      void embedSuperCluster();
      /// method to store the electron's supercluster internally
      void embedTrack();
      void setLeptonID(float id);
      void setElectronIDRobust(float id);

    protected:

      bool embeddedGsfTrack_;
      std::vector<reco::GsfTrack> gsfTrack_;
      bool embeddedSuperCluster_;
      std::vector<reco::SuperCluster> superCluster_;
      bool embeddedTrack_;
      std::vector<reco::Track> track_;
      float leptonID_;
      float electronIDRobust_;

  };


}

#endif
