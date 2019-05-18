// -*- C++ -*-
//
// Package:    CSCOverlapsBeamSplashCut
// Class:      CSCOverlapsBeamSplashCut
//
/**\class CSCOverlapsBeamSplashCut CSCOverlapsBeamSplashCut.cc Alignment/CSCOverlapsBeamSplashCut/src/CSCOverlapsBeamSplashCut.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Sat Nov 21 21:18:04 CET 2009
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// references
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

//
// class decleration
//

class CSCOverlapsBeamSplashCut : public edm::EDFilter {
public:
  explicit CSCOverlapsBeamSplashCut(const edm::ParameterSet&);
  ~CSCOverlapsBeamSplashCut() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::InputTag m_src;
  int m_maxSegments;
  TH1F* m_numSegments;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CSCOverlapsBeamSplashCut::CSCOverlapsBeamSplashCut(const edm::ParameterSet& iConfig)
    : m_src(iConfig.getParameter<edm::InputTag>("src")), m_maxSegments(iConfig.getParameter<int>("maxSegments")) {
  edm::Service<TFileService> tFileService;
  m_numSegments = tFileService->make<TH1F>("numSegments", "", 201, -0.5, 200.5);
}

CSCOverlapsBeamSplashCut::~CSCOverlapsBeamSplashCut() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool CSCOverlapsBeamSplashCut::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<CSCSegmentCollection> cscSegments;
  iEvent.getByLabel(m_src, cscSegments);

  m_numSegments->Fill(cscSegments->size());

  if (m_maxSegments < 0)
    return true;

  else if (int(cscSegments->size()) <= m_maxSegments)
    return true;

  else
    return false;
}

// ------------ method called once each job just before starting event loop  ------------
void CSCOverlapsBeamSplashCut::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void CSCOverlapsBeamSplashCut::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCOverlapsBeamSplashCut);
