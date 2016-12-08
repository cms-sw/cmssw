#ifndef RecoParticleFlow_PFPatProducer_PFChargedHadronAnalyzer_
#define RecoParticleFlow_PFPatProducer_PFChargedHadronAnalyzer_

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <TFile.h>
#include <TTree.h>
#include <TVector3.h>

#include <math.h>



#include "DataFormats/PatCandidates/interface/MET.h"

class METReader : public edm::EDAnalyzer {
 public:
  
  explicit METReader(const edm::ParameterSet&);

  ~METReader();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run & r, const edm::EventSetup & c);

 private:
  
  edm::EDGetTokenT<pat::METCollection> _origMetLabel;
  edm::EDGetTokenT<pat::METCollection> _newMetLabel;

  // The root tuple ==============================
  std::string _outputfile;
  TFile* _file;
  TTree* _tree;

  float _origCalo;
  float _origRaw;
  float _newRaw;
  float _origT1;
  float _newT1;
  
  float _newT1Phi;
  float _newT1Px;
  float _newT1Py;
  float _newT1SumEt;

  float _newNoShiftT1;
  float _newT1JERUp;
  float _newT1JERDo;
  float _newT1JESUp;
  float _newT1JESDo;
  float _newT1MESUp;
  float _newT1MESDo;
  float _newT1EESUp;
  float _newT1EESDo;
  float _newT1TESUp;
  float _newT1TESDo;
  float _newT1UESUp;
  float _newT1UESDo;


  int _n;

};

#endif
