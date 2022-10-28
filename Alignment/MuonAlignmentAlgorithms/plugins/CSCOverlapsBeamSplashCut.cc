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
#include "FWCore/Framework/interface/one/EDFilter.h"
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

class CSCOverlapsBeamSplashCut : public edm::one::EDFilter<edm::one::SharedResources> {
public:
  explicit CSCOverlapsBeamSplashCut(const edm::ParameterSet&);
  ~CSCOverlapsBeamSplashCut() override = default;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  // ----------member data ---------------------------
  edm::InputTag m_src;
  int m_maxSegments;
  TH1F* m_numSegments;
};

//
// constructors and destructor
//
CSCOverlapsBeamSplashCut::CSCOverlapsBeamSplashCut(const edm::ParameterSet& iConfig)
    : m_src(iConfig.getParameter<edm::InputTag>("src")), m_maxSegments(iConfig.getParameter<int>("maxSegments")) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> tFileService;
  m_numSegments = tFileService->make<TH1F>("numSegments", "", 201, -0.5, 200.5);
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

//define this as a plug-in
DEFINE_FWK_MODULE(CSCOverlapsBeamSplashCut);
