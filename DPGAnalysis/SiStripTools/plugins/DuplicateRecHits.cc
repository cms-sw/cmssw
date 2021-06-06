// -*- C++ -*-
//
// Package:    DuplicateRecHits
// Class:      DuplicateRecHits
//
/**\class DuplicateRecHits DuplicateRecHits.cc trackCount/DuplicateRecHits/src/DuplicateRecHits.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Thu Sep 25 16:32:56 CEST 2008
// $Id: DuplicateRecHits.cc,v 1.15 2011/11/15 10:09:24 venturia Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// my includes

#include <string>
#include <set>
#include <numeric>

#include "TH1F.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
//
// class decleration
//

class DuplicateRecHits : public edm::EDAnalyzer {
public:
  explicit DuplicateRecHits(const edm::ParameterSet&);
  ~DuplicateRecHits() override;

private:
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<reco::TrackCollection> m_trkcollToken;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> m_builderToken;
  const TransientTrackingRecHitBuilder* m_builder = nullptr;

  TH1F* m_nduplicate;
  TH1F* m_nduplmod;
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
DuplicateRecHits::DuplicateRecHits(const edm::ParameterSet& iConfig)
    : m_trkcollToken(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackCollection"))),
      m_builderToken(esConsumes<edm::Transition::BeginRun>(
          edm::ESInputTag{"", iConfig.getParameter<std::string>("TTRHBuilder")})) {
  //now do what ever initialization is needed

  // histogram parameters

  edm::LogInfo("TrackCollection") << "Using collection "
                                  << iConfig.getParameter<edm::InputTag>("trackCollection").label().c_str();

  edm::Service<TFileService> tfserv;

  m_nduplicate = tfserv->make<TH1F>("nduplicate", "Number of duplicated clusters per track", 10, -0.5, 9.5);
  m_nduplmod = tfserv->make<TH1F>("nduplmod", "Number of duplicated clusters per module", 10, -0.5, 9.5);
  m_nduplmod->SetCanExtend(TH1::kXaxis);
}

DuplicateRecHits::~DuplicateRecHits() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void DuplicateRecHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  edm::Service<TFileService> tfserv;

  Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(m_trkcollToken, tracks);

  for (reco::TrackCollection::const_iterator it = tracks->begin(); it != tracks->end(); it++) {
    std::set<SiPixelRecHit::ClusterRef::key_type> clusters;
    int nduplicate = 0;
    for (trackingRecHit_iterator rh = it->recHitsBegin(); rh != it->recHitsEnd(); ++rh) {
      TransientTrackingRecHit::RecHitPointer ttrh = m_builder->build(&**rh);
      const SiPixelRecHit* pxrh = dynamic_cast<const SiPixelRecHit*>(ttrh->hit());
      if (pxrh) {
        //	  LogTrace("DuplicateHitFinder") << ttrh->det()->geographicalId() << " " << pxrh->cluster().index();
        if (clusters.find(pxrh->cluster().index()) != clusters.end()) {
          nduplicate++;
          std::stringstream detidstr;
          detidstr << ttrh->det()->geographicalId().rawId();
          m_nduplmod->Fill(detidstr.str().c_str(), 1.);
          LogDebug("DuplicateHitFinder") << "Track with " << it->recHitsSize() << " RecHits";
          LogTrace("DuplicateHitFinder") << "Duplicate found " << ttrh->det()->geographicalId().rawId() << " "
                                         << pxrh->cluster().index();
        }
        clusters.insert(pxrh->cluster().index());
      }
      m_nduplicate->Fill(nduplicate);
    }
  }
}

void DuplicateRecHits::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  m_builder = &iSetup.getData(m_builderToken);
}

void DuplicateRecHits::endRun(const edm::Run& iRun, const edm::EventSetup&) {}

// ------------ method called once each job just before starting event loop  ------------
void DuplicateRecHits::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void DuplicateRecHits::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(DuplicateRecHits);
