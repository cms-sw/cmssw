//
// $Id: Electron.h,v 1.6 2008/02/11 15:20:48 llista Exp $
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    pat::Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   Electron implements the analysis-level electron class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Electron.h,v 1.6 2008/02/11 15:20:48 llista Exp $
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

      float trackIso() const;
      float caloIso() const;
      float leptonID() const;
      float electronIDRobust() const;
      float egammaTkIso() const;
      int   egammaTkNumIso() const;
      float egammaEcalIso() const;
      float egammaHcalIso() const;

      void setTrackIso(float trackIso);
      void setCaloIso(float caloIso);
      void setLeptonID(float id);
      void setElectronIDRobust(float id);
      void setEgammaTkIso(float tkIso);
      void setEgammaTkNumIso(int tkNumIso);
      void setEgammaEcalIso(float ecalIso);
      void setEgammaHcalIso(float hcalIso);

    protected:

      float trackIso_;
      float caloIso_;
      float leptonID_;
      float electronIDRobust_;
      float egammaTkIso_;
      int   egammaTkNumIso_;
      float egammaEcalIso_;
      float egammaHcalIso_;

  };


}

#endif
