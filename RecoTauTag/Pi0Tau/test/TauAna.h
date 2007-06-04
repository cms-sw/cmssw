#ifndef RecoTauTag_Pi0Tau_TauAna_h
#define RecoTauTag_Pi0Tau_TauAna_h
// -*- C++ -*-
//
// Package:    TauAna
// Class:      TauAna
// 
/**\class TauAna

Description: To study MC Tau properties

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Dongwook Jang
//         Created:  Wed Oct 11 11:08:40 CDT 2006
// $Id: TauAna.h,v 1.1 2007/03/27 21:32:03 dwjang Exp $
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TTree.h>
#include <TLorentzVector.h>

//
// class decleration
//

class TauAna : public edm::EDAnalyzer {
//class TauAna : public edm::EDProducer {
 public:
  explicit TauAna(const edm::ParameterSet&);
  ~TauAna();

 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::string trackCollectionName_;
  std::string tauCollectionName_;
  std::string pFCandidateProducerName_;
  std::string pFCandidateCollectionName_;
  std::string histFileName_;

  TTree *tree_;
  Int_t t_nSignalTracks;
  Int_t t_nSignalPi0s;
  Int_t t_nIsolationTracks;
  Int_t t_nIsolationPi0s;
  TLorentzVector *t_tracksMomentum;
  TLorentzVector *t_pi0sMomentum;
  TLorentzVector *t_momentum;

};

#endif
