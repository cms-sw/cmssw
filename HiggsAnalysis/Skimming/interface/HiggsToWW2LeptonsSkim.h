#ifndef HiggsToWW2LeptonsSkim_h
#define HiggsToWW2LeptonsSkim_h

/** \class HWWFilter
 *
 *  
 *  This class is an EDFilter choosing reconstructed di-tracks
 *  Allows extended requirements for tighter skim options (bool beTight=true)
 *
 *  $Date: 2009/06/16 10:29:37 $
 *  $Revision: 1.5 $
 *
 *  \author Ezio Torassa  -  INFN Padova
 *  \revised J. Fernandez  -  Univ. Oviedo
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>


using namespace edm;
using namespace std;


class HiggsToWW2LeptonsSkim : public edm::EDFilter {
    public:
       explicit HiggsToWW2LeptonsSkim(const edm::ParameterSet&);
       ~HiggsToWW2LeptonsSkim();
       virtual void endJob() ;

       virtual bool filter(Event&, const EventSetup&);

   private:
      double singleLeptonPtMin_;
      double diLeptonPtMin_;
      int    nLeptons_;
      double etaMin_;
      double etaMax_;
      bool   beTight_;
      double dilepM_;
      double eleHadronicOverEm_;
      unsigned int  nEvents_;
      unsigned int nAccepted_;

  // Reco samples
  edm::InputTag theGLBMuonLabel;
  edm::InputTag theGsfELabel;

};
#endif


   
