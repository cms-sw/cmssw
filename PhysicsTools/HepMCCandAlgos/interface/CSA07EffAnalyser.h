//
// $Id: CSA07EffAnalyser.h,v 1.2 2008/03/13 16:22:39 lowette Exp $
//

#ifndef PhysicsTools_HepMCCandAlgos_CSA07ProcessId_h
#define PhysicsTools_HepMCCandAlgos_CSA07ProcessId_h


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

#include <string>


class CSA07EffAnalyser : public edm::EDAnalyzer {

  typedef struct {
    Int_t procId;
    Float_t ptHat;
    Float_t filterEff;
    Float_t weight;
    Int_t trigBits[90];
  } CSA07Info;

  public:

    CSA07EffAnalyser(const edm::ParameterSet & iConfig);
    virtual ~CSA07EffAnalyser();

    void analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup);

  private: 

    bool runOnChowder_;
    std::string rootFileName_;
    TFile * rootFile_;
    TTree * csa07T_;
    TBranch * csa07B_;
    CSA07Info csa07Info_;

};


#endif
