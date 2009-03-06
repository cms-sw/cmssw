// -*- C++ -*-
//
// Package:    PatShapeAna
// Class:      PatShapeAna
// 
/**\class PatShapeAna PatShapeAna.h PhysicsTools/PatShapeAna/interface/PatShapeAna.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Christian AUTERMANN
//         Created:  Sat Mar 22 12:58:04 CET 2008
// $Id: PATHemisphereProducer.h,v 1.3 2008/04/08 09:02:18 trommers Exp $
//
//

#ifndef PATHemisphereProducer_h
#define PATHemisphereProducer_h

// system include files
#include <memory>
#include <map>
#include <utility>//pair
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

#include "PhysicsTools/PatAlgos/interface/HemisphereAlgo.h" 
//
// class decleration
//

class PATHemisphereProducer : public edm::EDProducer {
   public:
      explicit PATHemisphereProducer(const edm::ParameterSet&);
      ~PATHemisphereProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      /// Input: All PAT objects that are to cross-clean  or needed for that
      edm::InputTag _patJets;
      edm::InputTag _patMets;
      edm::InputTag _patMuons;
      edm::InputTag _patElectrons;
      edm::InputTag _patPhotons;
      edm::InputTag _patTaus;

      int _seedMethod; 
      int _combinationMethod;

      HemisphereAlgo* myHemi;
      
      std::vector<float> vPx, vPy, vPz, vE; 
      std::vector<float> vA1, vA2;
      std::vector<int> vgroups;
  std::vector<reco::CandidateBaseRef> componentRefs_;

  
  typedef std::vector<float> HemiAxis;
 
      
   
    
};

#endif


