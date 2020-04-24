// -*- C++ -*-
//
// Package:    L1Trigger/L1Ntuples
// Class:      L1HOTreeProducer
// 


// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// data formats
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1HO.h"

//
// class declaration
//

class L1HOTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1HOTreeProducer(const edm::ParameterSet&);
  ~L1HOTreeProducer();
  
private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

public:
  
  L1Analysis::L1AnalysisL1HO l1HO;
  L1Analysis::L1AnalysisL1HODataFormat * l1HOData; 

private:

  // output file
  edm::Service<TFileService> fs_;
  
  // tree
  TTree * tree_;
 
  // EDM input tags
  edm::EDGetTokenT<edm::SortedCollection<HODataFrame>> hoDataFrameToken_;
};



L1HOTreeProducer::L1HOTreeProducer(const edm::ParameterSet& iConfig)
{

  hoDataFrameToken_ = consumes<edm::SortedCollection<HODataFrame>>(iConfig.getUntrackedParameter<edm::InputTag>("hoDataFrameToken"));

  l1HOData = l1HO.getData();
  
  // set up output
  tree_=fs_->make<TTree>("L1HOTree", "L1HOTree");
  tree_->Branch("L1HO", "L1Analysis::L1AnalysisL1HODataFormat", &l1HOData, 32000, 3);
}


L1HOTreeProducer::~L1HOTreeProducer()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1HOTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  l1HO.Reset();

  edm::Handle<edm::SortedCollection<HODataFrame>> hoDataFrame;

  iEvent.getByToken(hoDataFrameToken_, hoDataFrame);

  if (hoDataFrame.isValid()){ 
    l1HO.SetHO(*hoDataFrame);
  } else {
    edm::LogWarning("MissingProduct") << "HODataFrame not found. Branch will not be filled" << std::endl;
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1HOTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1HOTreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1HOTreeProducer);
