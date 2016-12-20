// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/TrackReco/interface/TrackBase.h>
#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>

//
// class declaration
//

class EcalTangentFilter : public edm::EDFilter {
   public:
      explicit EcalTangentFilter(const edm::ParameterSet&);
      ~EcalTangentFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      
      // ----------member data ---------------------------
		int fNgood, fNtot, fEvt;
		std::string fMuLabel;
		double fMuonD0Min;
		double fMuonD0Max;
		bool fVerbose;
};
