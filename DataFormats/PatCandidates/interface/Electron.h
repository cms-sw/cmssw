//
// $Id: Electron.h,v 1.8 2008/04/03 12:29:08 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    pat::Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   Electron implements the analysis-level electron class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Electron.h,v 1.8 2008/04/03 12:29:08 gpetrucc Exp $
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
      float leptonID() const;
      float electronIDRobust() const;

      void setLeptonID(float id);
      void setElectronIDRobust(float id);

    protected:

      float leptonID_;
      float electronIDRobust_;

  };


}

#endif
