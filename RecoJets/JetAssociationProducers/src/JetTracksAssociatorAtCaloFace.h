// \class JetTracksAssociatorAtCaloFace JetTracksAssociatorAtCaloFace.cc 
// Associate jet with tracks extrapolated to CALO face
// Accommodated for Jet Package by: Fedor Ratnikov Sep.7, 2007
//
//
#ifndef JetTracksAssociatorAtCaloFace_h
#define JetTracksAssociatorAtCaloFace_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationXtrpCalo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

class JetTracksAssociatorAtCaloFace : public edm::stream::EDProducer<> {
   public:
      JetTracksAssociatorAtCaloFace(const edm::ParameterSet&);
      virtual ~JetTracksAssociatorAtCaloFace() {}

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
      
     JetTracksAssociatorAtCaloFace(){}
      
     edm::EDGetTokenT<edm::View <reco::Jet>> mJets;
     edm::EDGetTokenT<std::vector<reco::TrackExtrapolation> > mExtrapolations;
     JetTracksAssociationXtrpCalo mAssociator;
     edm::ESHandle<CaloGeometry> pGeo;
     bool firstRun;
     double dR_;
};

#endif
