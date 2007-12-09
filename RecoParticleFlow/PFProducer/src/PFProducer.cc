#include "RecoParticleFlow/PFProducer/interface/PFProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"


using namespace std;

using namespace boost;

using namespace edm;



PFProducer::PFProducer(const edm::ParameterSet& iConfig) {

 

  // use configuration file to setup input/output collection names
  blocksModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("BlocksModuleLabel","particleFlowBlock");

  blocksInstanceName_ 
    = iConfig.getUntrackedParameter<string>
    ("BlocksInstanceName","");  


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
  

  

  pfAlgo_.setParameters( nSigmaECAL, 
			 nSigmaHCAL,
			 calibration,
			 clusterRecovery,
			 PSCut, 
			 mvaCut, 
			 path_mvaWeightFile.fullPath().c_str() );

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

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
  try{      
    LogDebug("PFBlock")<<"getting blocks"<<endl;
    iEvent.getByLabel( blocksModuleLabel_, 
		       blocksInstanceName_, 
		       blocks );      

  } catch (cms::Exception& err) { 
    LogError("PFProducer")<<err
			  <<" cannot get collection "
			  <<blocksModuleLabel_<<":"
			  <<blocksInstanceName_
			  <<endl;
    
    throw;
  }

  
  LogDebug("PFProducer")<<"particle flow is starting"<<endl;

  assert( blocks.isValid() );
 
  pfAlgo_.reconstructParticles( blocks );


  if(verbose_) {
    ostringstream  str;
    str<<pfAlgo_<<endl;
    LogInfo("PFProducer") <<str.str()<<endl;
  }  

  auto_ptr< reco::PFCandidateCollection > 
    pOutputCandidateCollection( pfAlgo_.transferCandidates() ); 
  
  LogDebug("PFProducer")<<"particle flow: putting products in the event"<<endl;
  
  iEvent.put(pOutputCandidateCollection);
}

