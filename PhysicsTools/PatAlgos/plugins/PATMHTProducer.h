// -*- C++ -*-
//
// Package:    PATMHTProducer
// Class:      PATMHTProducer
// 
/**\class PATMHTProducer 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Xin Shi & Freya Blekman, Cornell University
//         Created:  Fri Sep 12 17:58:29 CEST 2008
// $Id: PATMHTProducer.h,v 1.1.2.3 2008/10/15 18:54:30 xshi Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/InputTag.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/PatCandidates/interface/MHT.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"


#include "RecoMET/METAlgorithms/interface/SigInputObj.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"



//
// class declaration
//

namespace pat {
class PATMHTProducer : public edm::EDProducer {
   public:
      explicit PATMHTProducer(const edm::ParameterSet&);
      ~PATMHTProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void beginRun(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  // input tags.
  edm::InputTag mhtLabel_;
  edm::InputTag jetLabel_;
  edm::InputTag eleLabel_;
  edm::InputTag muoLabel_;
  edm::InputTag tauLabel_;
  edm::InputTag phoLabel_;
  
  std::vector<metsig::SigInputObj> physobjvector_ ;

  double uncertaintyScaleFactor_; // scale factor for the uncertainty parameters.

};
//define this as a plug-in

} //end of namespace

