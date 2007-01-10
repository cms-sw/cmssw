#ifndef EcalIsolation_h
#define EcalIsolation_h

// -*- C++ -*-
//
// Package:    EcalIsolation
// Class:      EcalIsolation
// 
/**\class EcalIsolation EcalIsolation.cc RecoTauTag/EcalIsolation/src/EcalIsolation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Artur Kalinowski
//         Created:  Mon Sep 11 12:48:02 CEST 2006
// $Id: EcalIsolation.h,v 1.1 2006/11/06 13:49:17 akalinow Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

//
// class decleration
//

class EcalIsolation : public edm::EDProducer {
   public:
      explicit EcalIsolation(const edm::ParameterSet&);
      ~EcalIsolation();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:



      // ----------member data ---------------------------

      /// label of tau trigger type analysis
      edm::InputTag mJetForFilter;
      /// label for tower builder module
      double mPisol;
      /// size of the small cone
      double mSmallCone;
      /// size of the big cone
      double mBigCone,pIsolCut;
      
};

//
// constants, enums and typedefs
//

                                            
//
// static data member definitions
//

#endif
