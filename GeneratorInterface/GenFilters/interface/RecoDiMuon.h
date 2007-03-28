#ifndef RecoDiMuon_h
#define RecoDiMuon_h

/** \class RecoDiMuon
 *
 *  
 *  This class is an EDFilter choosing reconstructed di-muons
 *
 *  $Date: 2006/10/18 18:06:25 $
 *  $Revision: 1.2 $
 *
 *  \author Chang Liu  -  Purdue University
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


using namespace edm;
using namespace std;

class RecoDiMuon : public edm::EDFilter {
    public:
       explicit RecoDiMuon(const edm::ParameterSet&);
       ~RecoDiMuon();
       virtual void endJob() ;

       virtual bool filter(Event&, const EventSetup&);

   private:
      InputTag muonLabel_;
      double singleMuonPtMin_;
      double diMuonPtMin_;
      unsigned int  nEvents_;
      unsigned int nAccepted_;
};
#endif
