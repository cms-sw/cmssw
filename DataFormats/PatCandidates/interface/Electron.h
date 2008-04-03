//
// $Id: Electron.h,v 1.7 2008/03/05 14:47:33 fronga Exp $
//

#ifndef DataFormats_PatCandidates_Electron_h
#define DataFormats_PatCandidates_Electron_h

/**
  \class    pat::Electron Electron.h "DataFormats/PatCandidates/interface/Electron.h"
  \brief    Analysis-level electron class

   Electron implements the analysis-level electron class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Electron.h,v 1.7 2008/03/05 14:47:33 fronga Exp $
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
