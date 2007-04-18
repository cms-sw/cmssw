#include "RecoParticleFlow/PFProducer/interface/PFProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
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
  
  
  // energyCalibration_ = new PFEnergyCalibration(iConfig);
  // energyCalibration_ = new PFEnergyCalibration();

  double calibParamECAL_offset 
    = iConfig.getParameter<double>("pf_ECAL_calib_p0");
  double calibParamECAL_slope 
    = iConfig.getParameter<double>("pf_ECAL_calib_p1");

  // energyCalibration_->setCalibrationParametersEm(calibParamECAL_slope_, calibParamECAL_offset_); 
  // PFBlock::setEnergyCalibration(energyCalibration_);
  // energyResolution_ = new PFEnergyResolution(iConfig);
  // energyResolution_ = new PFEnergyResolution();
  // PFBlock::setEnergyResolution(energyResolution_);


  pfAlgo_.setParameters( calibParamECAL_offset, 
			 calibParamECAL_slope, 
			 nSigmaECAL, 
			 nSigmaHCAL );
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

  ostringstream  str;
  str<<pfAlgo_<<endl;
  LogInfo("PFProducer") <<str.str()<<endl;
  
  auto_ptr< reco::PFCandidateCollection > 
    pOutputCandidateCollection( pfAlgo_.transferCandidates() ); 
  
  LogDebug("PFProducer")<<"particle flow: putting products in the event"<<endl;
  
  iEvent.put(pOutputCandidateCollection);
}

