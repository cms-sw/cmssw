#include "RecoParticleFlow/PFBlockProducer/interface/PFBlockProducer.h"

// #include "RecoParticleFlow/PFAlgo/interface/PFBlock.h"
// #include "RecoParticleFlow/PFAlgo/interface/PFBlockElement.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <set>

using namespace std;
using namespace edm;

PFBlockProducer::PFBlockProducer(const edm::ParameterSet& iConfig) {
  

  // use configuration file to setup input/output collection names
  recTrackModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("RecTrackModuleLabel","particleFlowTrack");

  pfClusterModuleLabel_ 
    = iConfig.getUntrackedParameter<string>
    ("PFClusterModuleLabel","particleFlowCluster");  

  pfClusterECALInstanceName_ 
    = iConfig.getUntrackedParameter<string>
    ("PFClusterECALInstanceName","ECAL");  

  pfClusterHCALInstanceName_ 
    = iConfig.getUntrackedParameter<string>
    ("PFClusterHCALInstanceName","HCAL");  

  pfClusterPSInstanceName_ 
    = iConfig.getUntrackedParameter<string>
    ("PFClusterPSInstanceName","PS");  


  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);



  produces<reco::PFBlockCollection>();
  

  
  // particle flow parameters  -----------------------------------

  string map_ECAL_eta 
    = iConfig.getParameter<string>("pf_resolution_map_ECAL_eta");  
  string map_ECAL_phi 
    = iConfig.getParameter<string>("pf_resolution_map_ECAL_phi");  
  //   will be necessary when preshower is used:
  //   string map_ECALec_x 
  //     = iConfig.getParameter<string>("pf_resolution_map_ECALec_x");  
  //   string map_ECALec_y 
  //     = iConfig.getParameter<string>("pf_resolution_map_ECALec_y");  
  string map_HCAL_eta 
    = iConfig.getParameter<string>("pf_resolution_map_HCAL_eta");  
  string map_HCAL_phi 
    = iConfig.getParameter<string>("pf_resolution_map_HCAL_phi"); 
	
  double DPtovPtCut 
     = iConfig.getParameter<double>("pf_DPtoverPt_Cut");   

  double chi2_ECAL_PS 
     = iConfig.getParameter<double>("pf_chi2_ECAL_PS");  
//   double chi2_HCAL_PS 
//     = iConfig.getParameter<double>("pf_chi2_HCAL_PS");  

  double chi2_ECAL_Track 
    = iConfig.getParameter<double>("pf_chi2_ECAL_Track");  
  double chi2_HCAL_Track 
    = iConfig.getParameter<double>("pf_chi2_HCAL_Track");  
  double chi2_ECAL_HCAL 
    = iConfig.getParameter<double>("pf_chi2_ECAL_HCAL");  
  double chi2_PS_Track 
    = iConfig.getParameter<double>("pf_chi2_PS_Track");  
  double chi2_PSH_PSV 
    = iConfig.getParameter<double>("pf_chi2_PSH_PSV");  
  
  bool multiLink = 
    iConfig.getUntrackedParameter<bool>("pf_multilink",false);

  //energyCalibration_ = new PFEnergyCalibration(iConfig);

  
  //   PFBlock::setEnergyResolution(energyResolution_);

  edm::FileInPath path_ECAL_eta( map_ECAL_eta.c_str() );
  edm::FileInPath path_ECAL_phi( map_ECAL_phi.c_str() );
  edm::FileInPath path_HCAL_eta( map_HCAL_eta.c_str() );
  edm::FileInPath path_HCAL_phi( map_HCAL_phi.c_str() );
   
  pfBlockAlgo_.setParameters( path_ECAL_eta.fullPath().c_str(),
			      path_ECAL_phi.fullPath().c_str(),
			      path_HCAL_eta.fullPath().c_str(),
			      path_HCAL_phi.fullPath().c_str(),
			      DPtovPtCut,
			      chi2_ECAL_Track,
			      chi2_HCAL_Track,
			      chi2_ECAL_HCAL,
			      chi2_ECAL_PS,
			      chi2_PS_Track,
			      chi2_PSH_PSV,
			      multiLink );
  

//   energyCalibration_ = new PFEnergyCalibration();
//   double calibParamECAL_slope_ 
//     = iConfig.getParameter<double>("pf_ECAL_calib_p1");
//   double calibParamECAL_offset_ 
//     = iConfig.getParameter<double>("pf_ECAL_calib_p0");
  
//   energyCalibration_->setCalibrationParametersEm(calibParamECAL_slope_, calibParamECAL_offset_); 
  
//   //   PFBlock::setEnergyCalibration(energyCalibration_);
//   //energyResolution_ = new PFEnergyResolution(iConfig);

//   energyResolution_ = new PFEnergyResolution();


}



PFBlockProducer::~PFBlockProducer() { }



void PFBlockProducer::beginJob(const edm::EventSetup & es) { }


void PFBlockProducer::produce(Event& iEvent, 
			      const EventSetup& iSetup) {
  
  LogDebug("PFBlockProducer")<<"START event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;
  
  
  
  // get rectracks
  
  Handle< reco::PFRecTrackCollection > recTracks;
  try{      
    // LogDebug("PFBlockProducer")<<"get HCAL clusters"<<endl;
    iEvent.getByLabel(recTrackModuleLabel_.c_str(), "", recTracks);
    
  } catch (cms::Exception& err) { 
    LogError("PFBlockProducer")<<err
			       <<" cannot get collection "
			       <<"particleFlowBlock"<<":"
			       <<""
			       <<endl;
  }
  
  
  // get ECAL, HCAL and PS clusters
  
  
  Handle< reco::PFClusterCollection > clustersECAL;
  try{      
    // LogDebug("PFBlockProducer")<<"get ECAL clusters"<<endl;
    iEvent.getByLabel(pfClusterModuleLabel_, pfClusterECALInstanceName_, 
		      clustersECAL);      
  } 
  catch (cms::Exception& err) { 
    LogError("PFBlockProducer")<<err
			       <<" cannot get collection "
			       <<pfClusterModuleLabel_<<":"
			       <<pfClusterECALInstanceName_
			       <<endl;
  }
  
  
  Handle< reco::PFClusterCollection > clustersHCAL;
  try{      
    // LogDebug("PFBlockProducer")<<"get HCAL clusters"<<endl;
    iEvent.getByLabel(pfClusterModuleLabel_, pfClusterHCALInstanceName_, 
		      clustersHCAL);
    
  } catch (cms::Exception& err) { 
    LogError("PFBlockProducer")<<err
			       <<" cannot get collection "
			       <<pfClusterModuleLabel_<<":"
			       <<pfClusterHCALInstanceName_
			       <<endl;
  }
    



  Handle< reco::PFClusterCollection > clustersPS;
  try{      
    //       LogDebug("PFBlockProducer")<<"get PS clusters"<<endl;
    iEvent.getByLabel(pfClusterModuleLabel_, pfClusterPSInstanceName_, 
		      clustersPS);
  } catch (cms::Exception& err) { 
    LogError("PFBlockProducer")<<err
			       <<" cannot get collection "
			       <<pfClusterModuleLabel_<<":"
			       <<pfClusterPSInstanceName_
			       <<endl;
  }
  
  
  pfBlockAlgo_.setInput( recTracks, 
			 clustersECAL,
			 clustersHCAL,
			 clustersPS );
  pfBlockAlgo_.findBlocks();
  
  if(verbose_) {
    ostringstream  str;
    str<<pfBlockAlgo_<<endl;
    LogInfo("PFBlockProducer") << str.str()<<endl;
  }    
  
  auto_ptr< reco::PFBlockCollection > 
    pOutputBlockCollection( pfBlockAlgo_.transferBlocks() ); 
  
  
  iEvent.put(pOutputBlockCollection);
  
  LogDebug("PFBlockProducer")<<"STOP event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;
}
