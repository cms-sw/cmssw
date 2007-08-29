// \class JetToTracksAssociator JetToTracksAssociator.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetToTracksAssociator.h,v 1.1 2007/07/31 00:34:54 fedor Exp $
//
//
#ifndef JetToTracksAssociator_h
#define JetToTracksAssociator_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"

class JetToTracksAssociator : public edm::EDProducer {
   public:
      JetToTracksAssociator(const edm::ParameterSet&);
      virtual ~JetToTracksAssociator();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::InputTag mJets;
     edm::InputTag mTracks;
     JetTracksAssociationDRVertex mAssociator;
};

#endif
