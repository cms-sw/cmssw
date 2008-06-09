#ifndef PhysicsTools_StarterKit_PatAnalyzerSkeleton_h
#define PhysicsTools_StarterKit_PatAnalyzerSkeleton_h

// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      PatAnalyzerSkeleton
// 
/**\class PatAnalyzerSkeleton PatAnalyzerSkeleton.cc PhysicsTools/StarterKit/plugins/PatAnalyzerSkeleton.cc

 Description: <A very (very) simple CMSSW analyzer for PAT objects>

 Implementation:
 
 this analyzer shows how to loop over PAT output. 
*/
//
// Original Author:  Freya Blekman
//         Created:  Mon Apr 21 10:03:50 CEST 2008
// $Id: PatAnalyzerSkeleton.cc,v 1.2 2008/05/13 10:25:05 fblekman Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "TH1D.h"
#include <map>

#include "DataFormats/Common/interface/View.h"
#include <string>
//
// class decleration
//

class PatAnalyzerSkeleton : public edm::EDAnalyzer {
   public:
      explicit PatAnalyzerSkeleton(const edm::ParameterSet&);
      ~PatAnalyzerSkeleton();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  
  std::map<std::string,TH1D*> histocontainer_; // simple map to contain all histograms. Histograms are booked in the beginJob() method
  
  edm::InputTag eleLabel_;
  edm::InputTag muoLabel_;
  edm::InputTag jetLabel_;
  edm::InputTag tauLabel_;
  edm::InputTag metLabel_;
  edm::InputTag phoLabel_;
};

#endif
