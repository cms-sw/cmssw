#include "RecoParticleFlow/PFProducer/interface/PFProducer.h"
#include "RecoParticleFlow/PFAlgo/interface/PFAlgo.h"
#include "RecoParticleFlow/PFAlgo/interface/PFAlgoTestBenchElectrons.h"
#include "RecoParticleFlow/PFAlgo/interface/PFAlgoTestBenchConversions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"


#include <sstream>

using namespace std;

using namespace boost;

using namespace edm;



PFProducer::PFProducer(const edm::ParameterSet& iConfig) {

 

  // use configuration file to setup input/output collection names
  inputTagBlocks_ 
    = iConfig.getParameter<InputTag>("blocks");

  // register products
  produces<reco::PFCandidateCollection>();


  double nSigmaECAL 
    = iConfig.getParameter<double>("pf_nsigma_ECAL");
  double nSigmaHCAL 
    = iConfig.getParameter<double>("pf_nsigma_HCAL");
  
  
  double e_slope
    = iConfig.getParameter<double>("pf_calib_ECAL_slope");
  double e_offset 
    = iConfig.getParameter<double>("pf_calib_ECAL_offset");
  
  double eh_eslope
    = iConfig.getParameter<double>("pf_calib_ECAL_HCAL_eslope");
  double eh_hslope 
    = iConfig.getParameter<double>("pf_calib_ECAL_HCAL_hslope");
  double eh_offset 
    = iConfig.getParameter<double>("pf_calib_ECAL_HCAL_offset");
  
  double h_slope
    = iConfig.getParameter<double>("pf_calib_HCAL_slope");
  double h_offset 
    = iConfig.getParameter<double>("pf_calib_HCAL_offset");
  double h_damping 
    = iConfig.getParameter<double>("pf_calib_HCAL_damping");
  
  

  shared_ptr<PFEnergyCalibration> 
    calibration( new PFEnergyCalibration( e_slope,
					  e_offset, 
					  eh_eslope,
					  eh_hslope,
					  eh_offset,
					  h_slope,
					  h_offset,
					  h_damping ) );

  bool   clusterRecovery 
    = iConfig.getParameter<bool>("pf_clusterRecovery");
  
  double mvaCut = iConfig.getParameter<double>("pf_mergedPhotons_mvaCut");
  string mvaWeightFile 
    = iConfig.getParameter<string>("pf_mergedPhotons_mvaWeightFile");
  edm::FileInPath path_mvaWeightFile( mvaWeightFile.c_str() );
  double PSCut = iConfig.getParameter<double>("pf_mergedPhotons_PSCut");
  
  int algoType 
    = iConfig.getParameter<unsigned>("algoType");
  
  switch(algoType) {
  case 0:
    pfAlgo_.reset( new PFAlgo);
    break;
  case 1:
    pfAlgo_.reset( new PFAlgoTestBenchElectrons);
    break;
  case 2:
    pfAlgo_.reset( new PFAlgoTestBenchConversions);
    break;
  default:
    assert(0);
  }

  pfAlgo_->setParameters( nSigmaECAL, 
			 nSigmaHCAL,
			 calibration,
			 clusterRecovery,
			 PSCut, 
			 mvaCut, 
			 path_mvaWeightFile.fullPath().c_str() );

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  bool debug_ = 
    iConfig.getUntrackedParameter<bool>("debug",false);

  pfAlgo_->setDebug( debug_ );

}



PFProducer::~PFProducer() {}


void PFProducer::beginJob(const edm::EventSetup & es) {}


void PFProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) {
  
  LogDebug("PFProducer")<<"START event: "
			<<iEvent.id().event()
			<<" in run "<<iEvent.id().run()<<endl;
  

  // get the collection of blocks 

  Handle< reco::PFBlockCollection > blocks;

  LogDebug("PFBlock")<<"getting blocks"<<endl;
  bool found = iEvent.getByLabel( inputTagBlocks_, blocks );  

  if(!found ) {

    ostringstream err;
    err<<"cannot find blocks: "<<inputTagBlocks_;
    LogError("PFSimParticleProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }

  
  LogDebug("PFProducer")<<"particle flow is starting"<<endl;

  assert( blocks.isValid() );
 
  pfAlgo_->reconstructParticles( blocks );


  if(verbose_) {
    ostringstream  str;
    str<<(*pfAlgo_)<<endl;
    LogInfo("PFProducer") <<str.str()<<endl;
  }  

  auto_ptr< reco::PFCandidateCollection > 
    pOutputCandidateCollection( pfAlgo_->transferCandidates() ); 
  
  LogDebug("PFProducer")<<"particle flow: putting products in the event"<<endl;
  if ( verbose_ ) std::cout <<"particle flow: putting products in the event. Here the full list"<<endl;
  int nC=0;
  for( reco::PFCandidateCollection::const_iterator  itCand =  (*pOutputCandidateCollection).begin(); itCand !=  (*pOutputCandidateCollection).end(); itCand++) {
    nC++;
      if (verbose_ ) std::cout << nC << ")" << (*itCand).particleId() << std::endl;

  }
  
  iEvent.put(pOutputCandidateCollection);
}

