#ifndef PF_PU_FirstVertexTracks_h
#define PF_PU_FirstVertexTracks_h

// -*- C++ -*-
//
// Package:    PF_PU_AssoMap
// Class:      PF_PU_FirstVertexTracks
// 
/**\class PF_PU_AssoMap PF_PU_FirstVertexTracks.cc CommonTools/RecoUtils/plugins/PF_PU_FirstVertexTracks.cc

  Description: Produces collection of tracks associated to the first vertex based on the pf_pu Association Map 
*/
//

// Original Author:  Matthias Geisler
//         Created:  Wed Apr 18 14:48:37 CEST 2012
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"


//
// class declaration
//

class PF_PU_FirstVertexTracks : public edm::EDProducer {
   public:
      explicit PF_PU_FirstVertexTracks(const edm::ParameterSet&);
      ~PF_PU_FirstVertexTracks();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual bool TrackMatch(const reco::Track&,const reco::Track&);

      // ----------member data ---------------------------

      edm::InputTag input_AssociationType_;

      edm::InputTag input_AssociationMap_;
      edm::InputTag input_generalTracksCollection_;
      edm::InputTag input_VertexCollection_;

      int input_MinQuality_;
};


#endif

