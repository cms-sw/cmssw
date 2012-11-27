#ifndef PFCand_NoPU_WithAM_h
#define PFCand_NoPU_WithAM_h

// -*- C++ -*-
//
// Package:    PFCand_NoPU_WithAM
// Class:      PFCand_NoPU_WithAM
// 
/**\class PF_PU_AssoMap PFCand_NoPU_WithAM.cc CommonTools/RecoUtils/plugins/PFCand_NoPU_WithAM.cc

 Description: Produces a collection of PFCandidates associated to the first vertex based on the association map

*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//         Created:  Thu Dec  1 16:07:41 CET 2011
// $Id: PFCand_NoPU_WithAM.h,v 1.1 2012/04/17 11:53:55 mgeisler Exp $
//
//

#include "CommonTools/RecoUtils/interface/PFCand_NoPU_WithAM_Algos.h"

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

//
// class declaration
//

class PFCand_NoPU_WithAM : public edm::EDProducer {
   public:
      explicit PFCand_NoPU_WithAM(const edm::ParameterSet&);
      ~PFCand_NoPU_WithAM();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------

      edm::InputTag input_VertexPFCandAssociationMap_;
};


#endif
