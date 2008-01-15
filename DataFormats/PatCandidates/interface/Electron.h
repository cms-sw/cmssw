//
// $Id: Electron.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   Electron implements the analysis-level electron class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Electron.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
*/

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


namespace pat {


  typedef reco::PixelMatchGsfElectron ElectronType;
  typedef reco::PixelMatchGsfElectronCollection ElectronTypeCollection;


  class Electron : public Lepton<ElectronType> {

    friend class PATElectronProducer;

    public:

      Electron();
      Electron(const ElectronType & anElectron);
      virtual ~Electron();

      float getTrackIso() const;
      float getCaloIso() const;
      float getLeptonID() const;
      float getElectronIDRobust() const;
      float getEgammaTkIso() const;
      int getEgammaTkNumIso() const;
      float getEgammaEcalIso() const;
      float getEgammaHcalIso() const;

    protected:

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
      int egammaTkNumIso_;
      float egammaEcalIso_;
      float egammaHcalIso_;

  };


}

#endif
