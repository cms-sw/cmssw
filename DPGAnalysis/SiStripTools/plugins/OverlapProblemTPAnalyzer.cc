// -*- C++ -*-
//
// Package:    OverlapProblemTPAnalyzer
// Class:      OverlapProblemTPAnalyzer
// 
/**\class OverlapProblemTPAnalyzer OverlapProblemTPAnalyzer.cc DebugTools/OverlapProblem/plugins/OverlapProblemTPAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Thu Dec 16 16:32:56 CEST 2010
// $Id: OverlapProblemTPAnalyzer.cc,v 1.1 2011/01/22 17:59:36 venturia Exp $
//
//


// system include files
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "TH1F.h"
#include "TH2F.h"
//
// class decleration
//


class OverlapProblemTPAnalyzer : public edm::EDAnalyzer {
public:
  explicit OverlapProblemTPAnalyzer(const edm::ParameterSet&);
  ~OverlapProblemTPAnalyzer();
  
private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  
      // ----------member data ---------------------------

  TH1F* m_ptp;
  TH1F* m_etatp;
  TH1F* m_nhits;
  TH1F* m_nrechits;
  TH2F* m_nrecvssimhits;
  TH1F* m_nassotk;
  TH1F* m_pdgid;
  TH1F* m_llbit;
  TH1F* m_statustp;

  std::vector<TH1F*> m_simhitytecr;
  std::vector<TH1F*> m_assosimhitytecr;


  edm::EDGetTokenT<TrackingParticleCollection> m_tpcollToken;
  edm::EDGetTokenT<edm::View<reco::Track> > m_trkcollToken;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> m_associatorToken;
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
OverlapProblemTPAnalyzer::OverlapProblemTPAnalyzer(const edm::ParameterSet& iConfig):
  m_simhitytecr(),  m_assosimhitytecr(),
  m_tpcollToken(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticlesCollection"))),
  m_trkcollToken(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("trackCollection"))),
  m_associatorToken(consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag("trackAssociatorByHits")))
{
   //now do what ever initialization is needed



  edm::Service<TFileService> tfserv;

  m_ptp = tfserv->make<TH1F>("tpmomentum","Tracking Particle momentum",100,0.,200.);
  m_etatp = tfserv->make<TH1F>("tpeta","Tracking Particle pseudorapidity",100,-4.,4.);
  m_nhits = tfserv->make<TH1F>("nhits","Tracking Particle associated hits",100,-0.5,99.5);
  m_nrechits = tfserv->make<TH1F>("nrechits","Tracking Particle associated rec hits",100,-0.5,99.5);
  m_nrecvssimhits = tfserv->make<TH2F>("nrecvssimhits","Tracking Particle associated rec hits vs sim hits",
				       100,-0.5,99.5,100,-0.5,99.5);
  m_nassotk = tfserv->make<TH1F>("nassotk","Number of assocated reco tracks",10,-0.5,9.5);

  m_pdgid = tfserv->make<TH1F>("pdgid","Tracking Particle PDG id",1000,-500.5,499.5);
  m_llbit = tfserv->make<TH1F>("llbit","Tracking Particle LongLived bit",2,-0.5,1.5);
  m_statustp = tfserv->make<TH1F>("statustp","Tracking Particle status",2000,-1000.5,999.5);

  for(unsigned int ring=0;ring<7;++ring) {

    char name[100];
    char title[100];

    sprintf(name,"simytecr_%d",ring+1);
    sprintf(title,"SimHit local Y TEC ring %d",ring+1);

    m_simhitytecr.push_back(tfserv->make<TH1F>(name,title,200,-20.,20.));

    sprintf(name,"assosimytecr_%d",ring+1);
    sprintf(title,"SimHit local Y TEC ring %d with associated RecHit",ring+1);

    m_assosimhitytecr.push_back(tfserv->make<TH1F>(name,title,200,-20.,20.));

  }

}


OverlapProblemTPAnalyzer::~OverlapProblemTPAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
OverlapProblemTPAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

   // reco track Handle

   //   Handle<reco::TrackCollection> trkcoll;
  Handle<edm::View<reco::Track> > trkcoll;
  iEvent.getByToken(m_trkcollToken,trkcoll);
  
  Handle<TrackingParticleCollection> tpcoll;
  iEvent.getByToken(m_tpcollToken,tpcoll);
    
  Handle<reco::TrackToTrackingParticleAssociator> tahandle;
  iEvent.getByToken(m_associatorToken,tahandle);
  
  // associate reco to sim tracks
  
  reco::SimToRecoCollection srcoll = tahandle->associateSimToReco(trkcoll,tpcoll);
  
  // loop on Handle with index and use find
  
  for(unsigned int index=0 ; index != tpcoll->size() ; ++index) {
    
    // get TrackingParticleRef
    
    const TrackingParticleRef tp(tpcoll,index);
    
    if(std::abs(tp->pdgId())!=13) continue;
    
    // get the SimHIt from tracker only
    
    // Commented since the new TP's do not have this method
    //    std::vector<PSimHit> tksimhits = tp->trackPSimHit(DetId::Tracker);
    
    
    m_ptp->Fill(tp->p());
    m_etatp->Fill(tp->eta());
    //     m_nhits->Fill(tp->matchedHit());
    // With the new Tracking Particles I have to use a different method
    //    m_nhits->Fill(tksimhits.size());
    m_nhits->Fill(tp->numberOfTrackerHits());
    
    
    m_pdgid->Fill(tp->pdgId());
    m_llbit->Fill(tp->longLived());
    m_statustp->Fill(tp->status());
    
    // prepare a vector of TrackingRecHit from the associated reco tracks
    
    std::vector<DetId> rechits;
    
    // look at associated tracks
    
    if(srcoll.find(tp) != srcoll.end()) {
      reco::SimToRecoCollection::result_type trks = srcoll[tp];
      m_nassotk->Fill(trks.size());
      
      // loop on associated tracks and fill TrackingRecHit vector
      for(reco::SimToRecoCollection::result_type::const_iterator trk = trks.begin();trk!=trks.end();++trk) {
	for(trackingRecHit_iterator rh = trk->first->recHitsBegin() ; rh!=trk->first->recHitsEnd() ; ++rh) {
	  rechits.push_back((*rh)->geographicalId());
	}
	
      }
      
    }
    else {
      m_nassotk->Fill(0.);
      edm::LogInfo("NoAssociatedRecoTrack") << "No associated reco track for TP with p = " << tp->p() 
					    << " and eta = " << tp->eta() ; 
    }
    
    m_nrechits->Fill(rechits.size());
    // new method used to be compliant with the new TP's
    m_nrecvssimhits->Fill(tp->numberOfTrackerHits(),rechits.size());
    
    LogDebug("RecHitDetId") << "List of " << rechits.size() << " rechits detid from muon with p = " << tp->p() 
			    << "and eta = " << tp->eta();
    for(unsigned int i=0;i<rechits.size();++i) {
      LogTrace("RecHitDetId") << rechits[i].rawId();
    }
    
    
    // loop on sim hits
    
    
    LogDebug("SimHitDetId") << "List of " << tp->numberOfTrackerHits() << " simhits detid from muon with p = " << tp->p() 
			    << "and eta = " << tp->eta();
    
    // commented since with the new TP's I don't know how to loop on PSimHits

    /*
    for( std::vector<PSimHit>::const_iterator sh = tksimhits.begin(); sh!= tksimhits.end(); ++sh) {
      
      // check if the SimHit is Tracker and TEC
      
      LogTrace("SimHitDetId") << sh->detUnitId();
      
      TECDetId det(sh->detUnitId());
      if(det.subDetector() == SiStripDetId::TEC) {
	
	unsigned int iring = det.ring() - 1;
	m_simhitytecr[iring]->Fill(sh->entryPoint().y());
	
	// check if there is a TrackingRecHit in the same detid and if this is the case fill the histos
	
	for(std::vector<DetId>::const_iterator rhdet = rechits.begin(); rhdet!= rechits.end() ; ++rhdet) {
	  if(det==*rhdet) {
	    m_assosimhitytecr[iring]->Fill(sh->entryPoint().y());
	    break;
	  }
	}
	
      }
    }
    */    
  }
  
}

void 
OverlapProblemTPAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{
}

void 
OverlapProblemTPAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}



//define this as a plug-in
DEFINE_FWK_MODULE(OverlapProblemTPAnalyzer);
