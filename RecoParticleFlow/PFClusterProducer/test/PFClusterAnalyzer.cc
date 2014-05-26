#include "RecoParticleFlow/PFClusterProducer/test/PFClusterAnalyzer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

using namespace std;
using namespace edm;
using namespace reco;

PFClusterAnalyzer::PFClusterAnalyzer(const edm::ParameterSet& iConfig) {
  


  inputTagPFClusters_ 
    = iConfig.getParameter<InputTag>("PFClusters");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  printBlocks_ = 
    iConfig.getUntrackedParameter<bool>("printBlocks",false);



  LogDebug("PFClusterAnalyzer")
    <<" input collection : "<<inputTagPFClusters_ ;
 
  hack = TFile::Open("dump.root","RECREATE");
  
  deltaEnergy = new TH1F("e_reso","Shashlik Energy Resolution (Et = 45 GeV, #eta = 2.0)",100,0.0,1.2);

}



PFClusterAnalyzer::~PFClusterAnalyzer() { 
  hack->cd();
  deltaEnergy->Write();
  hack->Close();
  delete hack;
}



void 
PFClusterAnalyzer::beginRun(const edm::Run& run,
			    const edm::EventSetup & es) { }


void PFClusterAnalyzer::analyze(const Event& iEvent, 
				  const EventSetup& iSetup) {
  
  LogDebug("PFClusterAnalyzer")<<"START event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<endl;
  
  edm::Handle<reco::GenParticleCollection> genps;
  iEvent.getByLabel("genParticles",genps);

  edm::Handle<std::vector<SimTrack> > simTracks;
  iEvent.getByLabel("g4SimHits",simTracks);

  edm::Handle<std::vector<PCaloHit> > simHits;
  iEvent.getByLabel("g4SimHits","EcalHitsEK",simHits);

  for( const auto& simtrack : *simTracks ) {
    //if( simtrack.genpartIndex() != -1 ) {
      std::cout << simtrack.trackerSurfacePosition().eta() << ' ' 
		<< simtrack.trackerSurfacePosition().phi() << std::endl;      
      //}
  }
  
  math::XYZPoint vtx;

  for( const auto& genp : *genps ) {
    if( std::abs(genp.pdgId()) == 11 ||
	std::abs(genp.pdgId()) == 22 ) {
      vtx = genp.vertex();
      std::cout << genp.eta() << ' ' << genp.phi() << ' ' << genp.vertex() << std::endl;
    }
  }

  
  // get PFClusters

  Handle<PFClusterCollection> pfClusters;
  fetchCandidateCollection(pfClusters, 
			   inputTagPFClusters_, 
			   iEvent );

  // get PFClusters for isolation
  
  
  hack->cd();

  for( unsigned i=0; i<pfClusters->size(); i++ ) {
    
    const reco::PFCluster& cluster = (*pfClusters)[i];
    
    deltaEnergy->Fill(cluster.energy()/276.0);

    if( verbose_ ) {
      cout<<"PFCluster "<<endl;
      cout<<cluster<<endl;
      cout<<"CaloCluster "<<endl;
      
      auto direction = cluster.position() - vtx;
      std::cout << "PFCluster position with vertex" << direction.eta() 
		<< ' ' << direction.phi() << std::endl;

      const CaloCluster* caloc = dynamic_cast<const CaloCluster*> (&cluster);
      assert(caloc);
      cout<<*caloc<<endl;
      cout<<endl;
    }    
  }
    
  LogDebug("PFClusterAnalyzer")<<"STOP event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<endl;
}


  
void 
PFClusterAnalyzer::fetchCandidateCollection(Handle<reco::PFClusterCollection>& c, 
					    const InputTag& tag, 
					    const Event& iEvent) const {
  
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found ) {
    ostringstream  err;
    err<<" cannot get PFClusters: "
       <<tag<<endl;
    LogError("PFClusters")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }
  
}



DEFINE_FWK_MODULE(PFClusterAnalyzer);
