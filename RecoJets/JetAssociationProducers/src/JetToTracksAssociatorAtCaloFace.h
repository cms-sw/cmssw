// \class JetToTracksAssociator JetToTracksAssociatorAtCaloFace.cc 
// Associate jet with tracks extrapolated to CALO face
// Accommodated for Jet Package by: Fedor Ratnikov Sep.7, 2007
// $Id: JetToTracksAssociator.h,v 1.1 2007/08/29 17:53:15 fedor Exp $
//
//
#ifndef JetToTracksAssociatorAtCaloFace_h
#define JetToTracksAssociatorAtCaloFace_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRCalo.h"

class JetToTracksAssociatorAtCaloFace : public edm::EDProducer {
   public:
      JetToTracksAssociatorAtCaloFace(const edm::ParameterSet&);
      virtual ~JetToTracksAssociatorAtCaloFace();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::InputTag mJets;
     edm::InputTag mTracks;
     JetTracksAssociationDRCalo mAssociator;
};

#endif
