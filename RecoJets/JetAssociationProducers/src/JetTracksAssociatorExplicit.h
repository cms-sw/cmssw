// \class JetTracksAssociatorExplicit JetTracksAssociatorExplicit.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetTracksAssociatorExplicit.h,v 1.1 2012/01/13 21:11:04 srappocc Exp $
//
//
#ifndef JetTracksAssociatorExplicit_h
#define JetTracksAssociatorExplicit_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationExplicit.h"
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationExplicit.h"

class JetTracksAssociatorExplicit : public edm::EDProducer {
   public:
      JetTracksAssociatorExplicit(const edm::ParameterSet&);
      virtual ~JetTracksAssociatorExplicit();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::InputTag mJets;
     edm::InputTag mTracks;
     JetTracksAssociationExplicit mAssociatorExplicit;
};

#endif
