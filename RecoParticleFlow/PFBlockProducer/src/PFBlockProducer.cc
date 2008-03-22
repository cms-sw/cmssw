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
  inputTagRecTracks_ 
    = iConfig.getParameter<InputTag>("RecTracks");


  inputTagPFClustersECAL_ 
    = iConfig.getParameter<InputTag>("PFClustersECAL");

  inputTagPFClustersHCAL_ 
    = iConfig.getParameter<InputTag>("PFClustersHCAL");

  inputTagPFClustersPS_ 
    = iConfig.getParameter<InputTag>("PFClustersPS");



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
    iConfig.getParameter<bool>("pf_multilink");
  
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
  
  // LogDebug("PFBlockProducer")<<"get HCAL clusters"<<endl;
  bool found = iEvent.getByLabel(inputTagRecTracks_, recTracks);
    
  if(!found )
    LogError("PFBlockProducer")<<" cannot get rectracks: "
			       <<inputTagRecTracks_<<endl;
  
  
  // get ECAL, HCAL and PS clusters
  
  
  Handle< reco::PFClusterCollection > clustersECAL;
  found = iEvent.getByLabel(inputTagPFClustersECAL_, 
			    clustersECAL);      
  if(!found )
    LogError("PFBlockProducer")<<" cannot get ECAL clusters: "
			       <<inputTagPFClustersECAL_<<endl;
    
  
  Handle< reco::PFClusterCollection > clustersHCAL;
  found = iEvent.getByLabel(inputTagPFClustersHCAL_, 
			    clustersHCAL);      
  if(!found )
    LogError("PFBlockProducer")<<" cannot get HCAL clusters: "
			       <<inputTagPFClustersHCAL_<<endl;
    

  Handle< reco::PFClusterCollection > clustersPS;
  found = iEvent.getByLabel(inputTagPFClustersPS_, 
			    clustersPS);      
  if(!found )
    LogError("PFBlockProducer")<<" cannot get PS clusters: "
			       <<inputTagPFClustersPS_<<endl;
    
  

  
  
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
