#include "RecoParticleFlow/PFProducer/plugins/PFProducer.h"
#include "RecoParticleFlow/PFProducer/interface/PFAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibrationHF.h"

#include <sstream>

using namespace std;

using namespace boost;

using namespace edm;



PFProducer::PFProducer(const edm::ParameterSet& iConfig) {
  
  unsigned int newCalib = 
    iConfig.getParameter<unsigned int>("pf_newCalib");

  // Create a PFClusterCalbration auto_ptr
  boost::shared_ptr<pftools::PFClusterCalibration> 
    clusterCalibration( new pftools::PFClusterCalibration() );

  // Initialise function parameters properly.
  double lowEP0 
    = iConfig.getParameter<double>("pfcluster_lowEP0");
  double lowEP1 
    = iConfig.getParameter<double>("pfcluster_lowEP1");
  double globalP0 
    = iConfig.getParameter<double>("pfcluster_globalP0");
  double globalP1 
    = iConfig.getParameter<double>("pfcluster_globalP1");
  // 
  clusterCalibration->setCorrections(lowEP0, lowEP1, globalP0, globalP1);
 
  unsigned int allowNegative    
	= iConfig.getParameter<unsigned int>("pfcluster_allowNegative");
  clusterCalibration->setAllowNegativeEnergy(allowNegative);
	
  unsigned int doCorrection
    = iConfig.getParameter<unsigned int>("pfcluster_doCorrection");
  clusterCalibration->setDoCorrection(doCorrection);
 
  double barrelEta
    = iConfig.getParameter<double>("pfcluster_barrelEndcapEtaDiv");
  clusterCalibration->setBarrelBoundary(barrelEta);
	
  /* Now obsolete 
  double ecalEcut = 
    iConfig.getParameter<double>("pfcluster_ecalECut");
  double hcalEcut = 
    iConfig.getParameter<double>("pfcluster_hcalECut");

  clusterCalibration->setEcalHcalEnergyCuts(ecalEcut,hcalEcut);
  */
 
  std::vector<std::string>* names = clusterCalibration->getKnownSectorNames();
  for(std::vector<std::string>::iterator i = names->begin(); i != names->end(); ++i) {
    std::string sector = *i;
    std::vector<double> params
    = iConfig.getParameter<std::vector<double> >(sector);
    clusterCalibration->setEvolutionParameters(sector, params);
  }

  //Finally set eta correction
  unsigned int doEtaCorrection = iConfig.getParameter<unsigned int>("pfcluster_doEtaCorrection");   
  clusterCalibration->setDoEtaCorrection(doEtaCorrection);

  std::vector<double> etaCorrectionParams = 
    iConfig.getParameter<std::vector<double> >("pfcluster_etaCorrection");
  clusterCalibration->setEtaCorrectionParameters(etaCorrectionParams);
  // use configuration file to setup input/output collection names
  //std::cout << "Finished initialisaing PFClusterCalibration: it looks like...\n";
  //std::cout  << *clusterCalibration << std::endl;

  //Done with PFClusterCalibration //

  //--ab: get calibration factors for HF:
  bool calibHF_use;
  std::vector<double>  calibHF_eta_step;
  std::vector<double>  calibHF_a_EMonly;
  std::vector<double>  calibHF_b_HADonly;
  std::vector<double>  calibHF_a_EMHAD;
  std::vector<double>  calibHF_b_EMHAD;
  calibHF_use =     iConfig.getParameter<bool>("calibHF_use");
  calibHF_eta_step  = iConfig.getParameter<std::vector<double> >("calibHF_eta_step");
  calibHF_a_EMonly  = iConfig.getParameter<std::vector<double> >("calibHF_a_EMonly");
  calibHF_b_HADonly = iConfig.getParameter<std::vector<double> >("calibHF_b_HADonly");
  calibHF_a_EMHAD   = iConfig.getParameter<std::vector<double> >("calibHF_a_EMHAD");
  calibHF_b_EMHAD   = iConfig.getParameter<std::vector<double> >("calibHF_b_EMHAD");
  boost::shared_ptr<PFEnergyCalibrationHF>  
    thepfEnergyCalibrationHF ( new PFEnergyCalibrationHF(calibHF_use,calibHF_eta_step,calibHF_a_EMonly,calibHF_b_HADonly,calibHF_a_EMHAD,calibHF_b_EMHAD) ) ;
  //-----------------

  inputTagBlocks_ 
    = iConfig.getParameter<InputTag>("blocks");


  usePFElectrons_
    = iConfig.getParameter<bool>("usePFElectrons");    

  electronOutputCol_
    = iConfig.getParameter<std::string>("pf_electron_output_col");


  // register products
  produces<reco::PFCandidateCollection>();

  if (usePFElectrons_)
    produces<reco::PFCandidateCollection>(electronOutputCol_);
  
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

  //PFElectrons Configuration
  double mvaEleCut
    = iConfig.getParameter<double>("pf_electron_mvaCut");

  string mvaWeightFileEleID
    = iConfig.getParameter<string>("pf_electronID_mvaWeightFile");

  string path_mvaWeightFileEleID;
  if(usePFElectrons_)
    {
      path_mvaWeightFileEleID = edm::FileInPath ( mvaWeightFileEleID.c_str() ).fullPath();
    }

  // End PFElectrons Configuration

  bool usePFConversions
    = iConfig.getParameter<bool>("usePFConversions");  
  


  boost::shared_ptr<PFEnergyCalibration> 
    calibration( new PFEnergyCalibration( e_slope,
					  e_offset, 
					  eh_eslope,
					  eh_hslope,
					  eh_offset,
					  h_slope,
					  h_offset,
					  h_damping,
					  newCalib ) );

  int algoType 
    = iConfig.getParameter<unsigned>("algoType");
  
  switch(algoType) {
  case 0:
    pfAlgo_.reset( new PFAlgo);
    break;
   default:
    assert(0);
  }
  
  pfAlgo_->setParameters( nSigmaECAL, 
			  nSigmaHCAL,
			  calibration,
			  clusterCalibration,
			  thepfEnergyCalibrationHF,
			  newCalib);

  //PFElectrons: call the method setpfeleparameters
  pfAlgo_->setPFEleParameters(mvaEleCut,
			      path_mvaWeightFileEleID,
			      usePFElectrons_);
  
  pfAlgo_->setPFConversionParameters(usePFConversions);
  
  // Muon parameters
  std::vector<double> muonHCAL
    = iConfig.getParameter<std::vector<double> >("muon_HCAL");  
  std::vector<double> muonECAL
    = iConfig.getParameter<std::vector<double> >("muon_ECAL");  
  assert ( muonHCAL.size() == 2 && muonECAL.size() == 2 );
  
  // Fake track parameters
  double nSigmaTRACK
    = iConfig.getParameter<double>("nsigma_TRACK");  
  
  double ptError
    = iConfig.getParameter<double>("pt_Error");  
  
  std::vector<double> factors45
    = iConfig.getParameter<std::vector<double> >("factors_45");  
  assert ( factors45.size() == 2 );
  
  // Set muon and fake track parameters
  pfAlgo_->setPFMuonAndFakeParameters(muonHCAL,
				      muonECAL,
				      nSigmaTRACK,
				      ptError,
				      factors45);
  
  
  //MIKE: Vertex Parameters
  vertices_ = iConfig.getParameter<edm::InputTag>("vertexCollection");
  useVerticesForNeutral_ = iConfig.getParameter<bool>("useVerticesForNeutral");



  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  bool debug_ = 
    iConfig.getUntrackedParameter<bool>("debug",false);

  pfAlgo_->setDebug( debug_ );

}



PFProducer::~PFProducer() {}


void 
PFProducer::beginJob() {}

void 
PFProducer::beginRun(edm::Run & run, 
		     const edm::EventSetup & es) {}


void 
PFProducer::produce(Event& iEvent, 
		    const EventSetup& iSetup) {
  
  LogDebug("PFProducer")<<"START event: "
			<<iEvent.id().event()
			<<" in run "<<iEvent.id().run()<<endl;
  

  // Get The vertices from the event
  // and assign dynamic vertex parameters
  edm::Handle<reco::VertexCollection> vertices;
  bool gotVertices = iEvent.getByLabel(vertices_,vertices);
  if(!gotVertices) {
    ostringstream err;
    err<<"Cannot find vertices for this event.Continuing Without them ";
    LogError("PFSimParticleProducer")<<err.str()<<endl;
  }

  //Assign the PFAlgo Parameters
  pfAlgo_->setPFVertexParameters(useVerticesForNeutral_,*vertices);

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
  if(usePFElectrons_)
    {
      auto_ptr< reco::PFCandidateCollection >  
	pOutputElectronCandidateCollection( pfAlgo_->transferElectronCandidates() ); 
      iEvent.put(pOutputElectronCandidateCollection,electronOutputCol_);
    }
}

