#include "RecoParticleFlow/PFClusterProducer/test/PFClusterComparator.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iomanip>

using namespace std;
using namespace edm;
using namespace reco;

PFClusterComparator::PFClusterComparator(const edm::ParameterSet& iConfig) {
  


  inputTagPFClusters_ 
    = iConfig.getParameter<InputTag>("PFClusters");

  inputTagPFClustersCompare_ 
    = iConfig.getParameter<InputTag>("PFClustersCompare");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  printBlocks_ = 
    iConfig.getUntrackedParameter<bool>("printBlocks",false);

  log10E_old = fs_->make<TH1F>("log10E_old","log10(E cluster)",500,-5,5);
  log10E_new = fs_->make<TH1F>("log10E_new","log10(E cluster)",500,-5,5);
  deltaEnergy = fs_->make<TH1F>("delta_energy","E_{old} - E_{new}",5000,-5,5);

  posX_old = fs_->make<TH1F>("posX_old","log10(E cluster)",50000,0,500);
  posX_new = fs_->make<TH1F>("posX_new","log10(E cluster)",50000,0,500);
  deltaX = fs_->make<TH1F>("delta_X","X_{old} - X_{new}",5000,-5,5);

  posY_old = fs_->make<TH1F>("posY_old","log10(E cluster)",50000,0,500);
  posY_new = fs_->make<TH1F>("posY_new","log10(E cluster)",50000,0,500);
  deltaY = fs_->make<TH1F>("delta_Y","Y_{old} - Y_{new}",5000,-5,5);
  
  posZ_old = fs_->make<TH1F>("posZ_old","log10(E cluster)",50000,0,500);
  posZ_new = fs_->make<TH1F>("posZ_new","log10(E cluster)",50000,0,500);
  deltaZ = fs_->make<TH1F>("delta_Z","Z_{old} - Z_{new}",5000,-5,5);


  LogDebug("PFClusterComparator")
    <<" input collection : "<<inputTagPFClusters_ ;
   
}



PFClusterComparator::~PFClusterComparator() { }



void 
PFClusterComparator::beginRun(const edm::Run& run,
			    const edm::EventSetup & es) { }


void PFClusterComparator::analyze(const Event& iEvent, 
				  const EventSetup& iSetup) {
  std::map<unsigned,unsigned> detId_count;
    
  // get PFClusters

  Handle<PFClusterCollection> pfClusters;
  fetchCandidateCollection(pfClusters, 
			   inputTagPFClusters_, 
			   iEvent );

  Handle<PFClusterCollection> pfClustersCompare;
  fetchCandidateCollection(pfClustersCompare, 
			   inputTagPFClustersCompare_, 
			   iEvent );

  // get PFClusters for isolation
  
  std::cout << "There are " << pfClusters->size() << " PFClusters in the original cluster collection." << std::endl;
  std::cout << "There are " << pfClustersCompare->size() << " PFClusters in the new cluster collection." << std::endl;
  
  std::cout << std::flush << "---- COMPARING OLD TO NEW ----"
	    << std::endl  << std::flush;

  for( unsigned i=0; i<pfClusters->size(); i++ ) {  
    const reco::PFCluster& cluster = pfClusters->at(i);   
    detId_count[cluster.seed().rawId()] += 1;
    log10E_old->Fill(std::log10(cluster.energy()));
    posX_old->Fill(std::abs(cluster.position().x()));
    posY_old->Fill(std::abs(cluster.position().y()));
    posZ_old->Fill(std::abs(cluster.position().z()));
    bool foundmatch = false;
    for( unsigned k=0; k<pfClustersCompare->size(); ++k ) {
      const reco::PFCluster& clustercomp = pfClustersCompare->at(k);      
      if( cluster.seed().rawId() == clustercomp.seed().rawId() ) {
	foundmatch = true;
	const double denergy = std::abs(cluster.energy() - 
					clustercomp.energy());
	const double dcenergy = std::abs(cluster.correctedEnergy() - 
					clustercomp.correctedEnergy());
	const double dx = std::abs(cluster.position().x() - 
				     clustercomp.position().x());
	const double dy = std::abs(cluster.position().y() - 
				     clustercomp.position().y());
	const double dz = std::abs(cluster.position().z() - 
				     clustercomp.position().z());
	deltaEnergy->Fill((cluster.energy() - clustercomp.energy())/cluster.energy());
	deltaX->Fill((cluster.position().x() - clustercomp.position().x())/cluster.position().x());
	deltaY->Fill((cluster.position().y() - clustercomp.position().y())/cluster.position().y());
	deltaZ->Fill((cluster.position().z() - clustercomp.position().z())/cluster.position().z());
	
	if( denergy/std::abs(cluster.energy()) >  1e-5 ) {
	  std::cout << "   " << cluster.seed() 
		    << " Energies different by larger than tolerance! "
		    << "( "<< denergy << " )"
		    << "[ " << detId_count[cluster.seed().rawId()] << " ]" 
		    << " Old: " << std::setprecision(7) 
		    << cluster.energy() << " GeV , New: "
		    << clustercomp.energy() << " GeV" << std::endl;	  
	}
	if( dcenergy/std::abs(cluster.correctedEnergy()) >  1e-5 ) {
	  std::cout << "   " << cluster.seed() 
		    << " Corrected energies different by larger than tolerance! "
		    << "( "<< dcenergy << " )"
		    << "[ " << detId_count[cluster.seed().rawId()] << " ]" 
		    << " Old: " << std::setprecision(7) 
		    << cluster.correctedEnergy() << " GeV , New: "
		    << clustercomp.correctedEnergy() << " GeV" << std::endl;	  
	}
	std::cout << std::flush;
	if( dx/std::abs(cluster.position().x()) > 1e-5 ) {
	  std::cout << "***" << cluster.seed() 
		    << " X's different by larger than tolerance! "
		    << "( "<< dx << " )"
		    << "[ " << detId_count[cluster.seed().rawId()] << " ]" 
		    << " Old: " << std::setprecision(7) 
		    << cluster.position().x() << " , New: "
		    << clustercomp.position().x() << std::endl;
	}
	std::cout << std::flush;
	if( dy/std::abs(cluster.position().y()) > 1e-5 ) {
	  std::cout << "---" << cluster.seed() 
		    << " Y's different by larger than tolerance! "
		    << "( "<< dy << " )"
		    << "[ " << detId_count[cluster.seed().rawId()] << " ]" 
		    << " Old: " << std::setprecision(7) 
		    << cluster.position().y() << " , New: "
		    << clustercomp.position().y() << std::endl;
	}
	std::cout << std::flush;
	if( dz/std::abs(cluster.position().z()) > 1e-5 ) {
	  std::cout << "+++" << cluster.seed() 
		    << " Z's different by larger than tolerance! "
		    << "( "<< dz << " )"
		    << "[ " << detId_count[cluster.seed().rawId()] <<" ]" 
		    << " Old: " << std::setprecision(7) 
		    << cluster.position().z() << " , New: "
		    << clustercomp.position().z() << std::endl;
	}
	std::cout << std::flush;
      }      
    }
    if( !foundmatch ) {      
      std::cout << "Seed in old clusters and not new: " 
		<< cluster << std::endl;
    }
  }
   
  std::cout << std::flush << "---- COMPARING NEW TO OLD ----"
	    << std::endl  << std::flush;

  for( unsigned i=0; i<pfClustersCompare->size(); i++ ) {     
    const reco::PFCluster& cluster = pfClustersCompare->at(i);   
    log10E_new->Fill(std::log10(cluster.energy()));
    posX_new->Fill(std::abs(cluster.position().x()));
    posY_new->Fill(std::abs(cluster.position().y()));
    posZ_new->Fill(std::abs(cluster.position().z()));
    bool foundmatch = false;
    for( unsigned k=0; k<pfClusters->size(); ++k ) {
      const reco::PFCluster& clustercomp = pfClusters->at(k);      
      if( cluster.seed() == clustercomp.seed() ) {
	foundmatch = true;

      }      
    }
    if( !foundmatch ) {      
      std::cout << "Seed in new clusters and not old: " 
		<< cluster << std::endl;
    }
  }
  std::cout << std::flush;
}


  
void 
PFClusterComparator::fetchCandidateCollection(Handle<reco::PFClusterCollection>& c, 
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



DEFINE_FWK_MODULE(PFClusterComparator);
