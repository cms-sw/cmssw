// -*- C++ -*-
//
// Package:    OverlapProblemTSOSAnalyzer
// Class:      OverlapProblemTSOSAnalyzer
// 
/**\class OverlapProblemTSOSAnalyzer OverlapProblemTSOSAnalyzer.cc DebugTools/OverlapProblem/plugins/OverlapProblemTSOSAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Thu Dec 16 16:32:56 CEST 2010
// $Id: OverlapProblemTSOSAnalyzer.cc,v 1.2 2013/04/10 21:08:01 venturia Exp $
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

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "TH1F.h"

#include "DPGAnalysis/SiStripTools/interface/TSOSHistogramMaker.h"
//
// class decleration
//


class OverlapProblemTSOSAnalyzer : public edm::EDAnalyzer {
public:
  explicit OverlapProblemTSOSAnalyzer(const edm::ParameterSet&);
  ~OverlapProblemTSOSAnalyzer();
  
private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  
      // ----------member data ---------------------------

  TH1F* m_ptrk;
  TH1F* m_etatrk;

  bool m_validOnly;
  edm::EDGetTokenT<TrajTrackAssociationCollection> m_ttacollToken;
  const bool m_debug;

  TSOSHistogramMaker m_tsoshm;

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
OverlapProblemTSOSAnalyzer::OverlapProblemTSOSAnalyzer(const edm::ParameterSet& iConfig):
  m_validOnly(iConfig.getParameter<bool>("onlyValidRecHit")),
  m_ttacollToken(consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajTrackAssoCollection"))),
  m_debug(iConfig.getUntrackedParameter<bool>("debugMode",false)),
  m_tsoshm(iConfig.getParameter<edm::ParameterSet>("tsosHMConf"))
{
   //now do what ever initialization is needed



  edm::Service<TFileService> tfserv;

  m_ptrk = tfserv->make<TH1F>("trkmomentum","Refitted Track  momentum",100,0.,200.);
  m_etatrk = tfserv->make<TH1F>("trketa","Refitted Track pseudorapidity",100,-4.,4.);


}


OverlapProblemTSOSAnalyzer::~OverlapProblemTSOSAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
OverlapProblemTSOSAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // loop on trajectories and plot TSOS local coordinate
  
  TrajectoryStateCombiner tsoscomb;
  
  // Trajectory Handle

  DetIdSelector selector("0x1fbff004-0x14ac1004");
  
  Handle<TrajTrackAssociationCollection> ttac;
  iEvent.getByToken(m_ttacollToken,ttac);
  
  for(TrajTrackAssociationCollection::const_iterator pair=ttac->begin();pair!=ttac->end();++pair) {
    
    const edm::Ref<std::vector<Trajectory> > & traj = pair->key;
    const reco::TrackRef & trk = pair->val;
    const std::vector<TrajectoryMeasurement> & tmcoll = traj->measurements();
    
    m_ptrk->Fill(trk->p());
    m_etatrk->Fill(trk->eta());


    for(std::vector<TrajectoryMeasurement>::const_iterator meas = tmcoll.begin() ; meas!= tmcoll.end() ; ++meas) {
      
      if(!meas->updatedState().isValid()) continue;
      
      TrajectoryStateOnSurface tsos = tsoscomb(meas->forwardPredictedState(), meas->backwardPredictedState());
      TransientTrackingRecHit::ConstRecHitPointer hit = meas->recHit();
      
      m_tsoshm.fill(tsos,hit);

      if(!hit->isValid() && m_validOnly) continue;

      if(m_debug) {
	if(selector.isSelected(hit->geographicalId())) {
	  const SiPixelRecHit* pixelrh = dynamic_cast<const SiPixelRecHit*>(hit->hit());
	  if(pixelrh) {
	    edm::LogInfo("ClusterFound") << "Cluster reference" << pixelrh->cluster().key(); 
	  }
	  else {
	    edm::LogInfo("NoCluster") << "No cluster found!";
	  }
	}
      }

    }
    
  }
  
  
}

void 
OverlapProblemTSOSAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{
}

void 
OverlapProblemTSOSAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}



//define this as a plug-in
DEFINE_FWK_MODULE(OverlapProblemTSOSAnalyzer);
