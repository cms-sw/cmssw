// \class JetTracksAssociatorAtCaloFace JetTracksAssociatorAtCaloFace.cc 
// Associate jet with tracks extrapolated to CALO face
// Accommodated for Jet Package by: Fedor Ratnikov Sep.7, 2007
// $Id: JetTracksAssociatorAtCaloFace.h,v 1.4 2010/03/16 21:45:55 srappocc Exp $
//
//
#ifndef JetTracksAssociatorAtCaloFace_h
#define JetTracksAssociatorAtCaloFace_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationXtrpCalo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class JetTracksAssociatorAtCaloFace : public edm::EDProducer {
   public:
      JetTracksAssociatorAtCaloFace(const edm::ParameterSet&);
      virtual ~JetTracksAssociatorAtCaloFace() {}

      virtual void beginRun( edm::Run const & run, edm::EventSetup const & setup);

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
      
     JetTracksAssociatorAtCaloFace(){}
      
     edm::InputTag mJets;
     edm::InputTag mExtrapolations;
     JetTracksAssociationXtrpCalo mAssociator;
     edm::ESHandle<CaloGeometry> pGeo;
     bool firstRun;
     double dR_;
};

#endif
