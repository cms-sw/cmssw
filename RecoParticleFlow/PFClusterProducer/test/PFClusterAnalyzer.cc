#include "RecoParticleFlow/PFClusterProducer/test/PFClusterAnalyzer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


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
