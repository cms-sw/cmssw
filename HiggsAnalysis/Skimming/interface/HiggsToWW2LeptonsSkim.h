#ifndef HiggsToWW2LeptonsSkim_h
#define HiggsToWW2LeptonsSkim_h

/** \class HWWFilter
 *
 *
 *  This class is an EDFilter choosing reconstructed di-tracks
 *
 *
 *  \author Ezio Torassa  -  INFN Padova
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

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

class HiggsToWW2LeptonsSkim : public edm::EDFilter {
    public:
       explicit HiggsToWW2LeptonsSkim(const edm::ParameterSet&);
       ~HiggsToWW2LeptonsSkim();
       virtual void endJob() ;

       virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      double singleTrackPtMin_;
      double diTrackPtMin_;
      double etaMin_;
      double etaMax_;
      unsigned int  nEvents_;
      unsigned int nAccepted_;

  // Reco samples
  edm::EDGetTokenT<reco::TrackCollection> theGLBMuonToken;
  edm::EDGetTokenT<reco::GsfElectronCollection> theGsfEToken;

};
#endif



