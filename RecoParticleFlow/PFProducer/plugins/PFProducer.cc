#include "RecoParticleFlow/PFProducer/plugins/PFProducer.h"
#include "RecoParticleFlow/PFProducer/interface/PFAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibrationHF.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

#include <sstream>

#include "TFile.h"

using namespace std;

using namespace boost;

using namespace edm;



PFProducer::PFProducer(const edm::ParameterSet& iConfig) {
  
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

  inputTagBlocks_ = consumes<reco::PFBlockCollection>(iConfig.getParameter<InputTag>("blocks"));
  
  //Post cleaning of the muons
  inputTagMuons_ = consumes<reco::MuonCollection>(iConfig.getParameter<InputTag>("muons"));
  postMuonCleaning_
    = iConfig.getParameter<bool>("postMuonCleaning");

  if( iConfig.existsAs<bool>("useEGammaFilters") ) {
    use_EGammaFilters_ =  iConfig.getParameter<bool>("useEGammaFilters");    
  } else {
    use_EGammaFilters_ = false;
  }

  usePFElectrons_
    = iConfig.getParameter<bool>("usePFElectrons");    

  usePFPhotons_
    = iConfig.getParameter<bool>("usePFPhotons");    
  
  // **************************** !! IMPORTANT !! ************************************
  // When you code is swithed on, automatically turn off the old PFElectrons/PFPhotons. 
  // The two algorithms can not run at the same time
  // *********************************************************************************
 
  if(use_EGammaFilters_) {
    usePFElectrons_ = false;
    usePFPhotons_ = false;
  }


  usePhotonReg_
    = (usePFPhotons_) ? iConfig.getParameter<bool>("usePhotonReg") : false ;

  useRegressionFromDB_
    = (usePFPhotons_) ? iConfig.getParameter<bool>("useRegressionFromDB") : false; 

  useEGammaElectrons_
    = iConfig.getParameter<bool>("useEGammaElectrons");    

  if(  useEGammaElectrons_) {
    inputTagEgammaElectrons_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("egammaElectrons"));
  }

  electronOutputCol_
    = iConfig.getParameter<std::string>("pf_electron_output_col");

  bool usePFSCEleCalib;
  std::vector<double>  calibPFSCEle_Fbrem_barrel; 
  std::vector<double>  calibPFSCEle_Fbrem_endcap;
  std::vector<double>  calibPFSCEle_barrel;
  std::vector<double>  calibPFSCEle_endcap;
  usePFSCEleCalib =     iConfig.getParameter<bool>("usePFSCEleCalib");
  calibPFSCEle_Fbrem_barrel = iConfig.getParameter<std::vector<double> >("calibPFSCEle_Fbrem_barrel");
  calibPFSCEle_Fbrem_endcap = iConfig.getParameter<std::vector<double> >("calibPFSCEle_Fbrem_endcap");
  calibPFSCEle_barrel = iConfig.getParameter<std::vector<double> >("calibPFSCEle_barrel");
  calibPFSCEle_endcap = iConfig.getParameter<std::vector<double> >("calibPFSCEle_endcap");
  boost::shared_ptr<PFSCEnergyCalibration>  
    thePFSCEnergyCalibration ( new PFSCEnergyCalibration(calibPFSCEle_Fbrem_barrel,calibPFSCEle_Fbrem_endcap,
							 calibPFSCEle_barrel,calibPFSCEle_endcap )); 
			       
  bool useEGammaSupercluster = iConfig.getParameter<bool>("useEGammaSupercluster");
  double sumEtEcalIsoForEgammaSC_barrel = iConfig.getParameter<double>("sumEtEcalIsoForEgammaSC_barrel");
  double sumEtEcalIsoForEgammaSC_endcap = iConfig.getParameter<double>("sumEtEcalIsoForEgammaSC_endcap");
  double coneEcalIsoForEgammaSC = iConfig.getParameter<double>("coneEcalIsoForEgammaSC");
  double sumPtTrackIsoForEgammaSC_barrel = iConfig.getParameter<double>("sumPtTrackIsoForEgammaSC_barrel");
  double sumPtTrackIsoForEgammaSC_endcap = iConfig.getParameter<double>("sumPtTrackIsoForEgammaSC_endcap");
  double coneTrackIsoForEgammaSC = iConfig.getParameter<double>("coneTrackIsoForEgammaSC");
  unsigned int nTrackIsoForEgammaSC  = iConfig.getParameter<unsigned int>("nTrackIsoForEgammaSC");


  // register products
  produces<reco::PFCandidateCollection>();
  produces<reco::PFCandidateCollection>("CleanedHF");
  produces<reco::PFCandidateCollection>("CleanedCosmicsMuons");
  produces<reco::PFCandidateCollection>("CleanedTrackerAndGlobalMuons");
  produces<reco::PFCandidateCollection>("CleanedFakeMuons");
  produces<reco::PFCandidateCollection>("CleanedPunchThroughMuons");
  produces<reco::PFCandidateCollection>("CleanedPunchThroughNeutralHadrons");
  produces<reco::PFCandidateCollection>("AddedMuonsAndHadrons");


  if (usePFElectrons_) {
    produces<reco::PFCandidateCollection>(electronOutputCol_);
    produces<reco::PFCandidateElectronExtraCollection>(electronExtraOutputCol_);
  }

  if (usePFPhotons_) {
    produces<reco::PFCandidatePhotonExtraCollection>(photonExtraOutputCol_);
  }


  double nSigmaECAL 
    = iConfig.getParameter<double>("pf_nsigma_ECAL");
  double nSigmaHCAL 
    = iConfig.getParameter<double>("pf_nsigma_HCAL");
  
  //PFElectrons Configuration
  double mvaEleCut
    = iConfig.getParameter<double>("pf_electron_mvaCut");

  
  string mvaWeightFileEleID
    = iConfig.getParameter<string>("pf_electronID_mvaWeightFile");

  bool applyCrackCorrectionsForElectrons
    = iConfig.getParameter<bool>("pf_electronID_crackCorrection");
  
  string path_mvaWeightFileEleID;
  if(usePFElectrons_)
    {
      path_mvaWeightFileEleID = edm::FileInPath ( mvaWeightFileEleID.c_str() ).fullPath();
     }

  //PFPhoton Configuration

  string path_mvaWeightFileConvID;
  string mvaWeightFileConvID;
  string path_mvaWeightFileGCorr;
  string path_mvaWeightFileLCorr;
  string path_X0_Map;
  string path_mvaWeightFileRes;
  double mvaConvCut=-99.;
  double sumPtTrackIsoForPhoton = 99.;
  double sumPtTrackIsoSlopeForPhoton = 99.;

  if(usePFPhotons_ )
    {
      mvaWeightFileConvID =iConfig.getParameter<string>("pf_convID_mvaWeightFile");
      mvaConvCut = iConfig.getParameter<double>("pf_conv_mvaCut");
      path_mvaWeightFileConvID = edm::FileInPath ( mvaWeightFileConvID.c_str() ).fullPath();  
      sumPtTrackIsoForPhoton = iConfig.getParameter<double>("sumPtTrackIsoForPhoton");
      sumPtTrackIsoSlopeForPhoton = iConfig.getParameter<double>("sumPtTrackIsoSlopeForPhoton");

      string X0_Map=iConfig.getParameter<string>("X0_Map");
      path_X0_Map = edm::FileInPath( X0_Map.c_str() ).fullPath();

      if(!useRegressionFromDB_) {
	string mvaWeightFileLCorr=iConfig.getParameter<string>("pf_locC_mvaWeightFile");
	path_mvaWeightFileLCorr = edm::FileInPath( mvaWeightFileLCorr.c_str() ).fullPath();
	string mvaWeightFileGCorr=iConfig.getParameter<string>("pf_GlobC_mvaWeightFile");
	path_mvaWeightFileGCorr = edm::FileInPath( mvaWeightFileGCorr.c_str() ).fullPath();
	string mvaWeightFileRes=iConfig.getParameter<string>("pf_Res_mvaWeightFile");
	path_mvaWeightFileRes=edm::FileInPath(mvaWeightFileRes.c_str()).fullPath();

	TFile *fgbr = new TFile(path_mvaWeightFileGCorr.c_str(),"READ");
	ReaderGC_  =(const GBRForest*)fgbr->Get("GBRForest");
	TFile *fgbr2 = new TFile(path_mvaWeightFileLCorr.c_str(),"READ");
	ReaderLC_  = (const GBRForest*)fgbr2->Get("GBRForest");
	TFile *fgbr3 = new TFile(path_mvaWeightFileRes.c_str(),"READ");
	ReaderRes_  = (const GBRForest*)fgbr3->Get("GBRForest");
	LogDebug("PFProducer")<<"Will set regressions from binary files " <<endl;
      }

    }
  

  // Reading new EGamma selection cuts
  bool useProtectionsForJetMET(false);
  double ele_iso_pt(0.0), ele_iso_mva_barrel(0.0), ele_iso_mva_endcap(0.0), 
    ele_iso_combIso_barrel(0.0), ele_iso_combIso_endcap(0.0), 
    ele_noniso_mva(0.0);
  unsigned int ele_missinghits(0);
  double ph_MinEt(0.0), ph_combIso(0.0), ph_HoE(0.0), 
    ph_sietaieta_eb(0.0),ph_sietaieta_ee(0.0);
  string ele_iso_mvaWeightFile(""), ele_iso_path_mvaWeightFile("");
  edm::ParameterSet ele_protectionsForJetMET,ph_protectionsForJetMET;

 // Reading new EGamma ubiased collections and value maps
 if(use_EGammaFilters_) {
   ele_iso_mvaWeightFile = iConfig.getParameter<string>("isolatedElectronID_mvaWeightFile");
   ele_iso_path_mvaWeightFile  = edm::FileInPath ( ele_iso_mvaWeightFile.c_str() ).fullPath();
   inputTagPFEGammaCandidates_ = consumes<edm::View<reco::PFCandidate> >((iConfig.getParameter<edm::InputTag>("PFEGammaCandidates")));
   inputTagValueMapGedElectrons_ = consumes<edm::ValueMap<reco::GsfElectronRef>>(iConfig.getParameter<edm::InputTag>("GedElectronValueMap")); 
   inputTagValueMapGedPhotons_ = consumes<edm::ValueMap<reco::PhotonRef> >(iConfig.getParameter<edm::InputTag>("GedPhotonValueMap")); 
   ele_iso_pt = iConfig.getParameter<double>("electron_iso_pt");
   ele_iso_mva_barrel  = iConfig.getParameter<double>("electron_iso_mva_barrel");
   ele_iso_mva_endcap = iConfig.getParameter<double>("electron_iso_mva_endcap");
   ele_iso_combIso_barrel = iConfig.getParameter<double>("electron_iso_combIso_barrel");
   ele_iso_combIso_endcap = iConfig.getParameter<double>("electron_iso_combIso_endcap");
   ele_noniso_mva = iConfig.getParameter<double>("electron_noniso_mvaCut");
   ele_missinghits = iConfig.getParameter<unsigned int>("electron_missinghits"); 
   ph_MinEt  = iConfig.getParameter<double>("photon_MinEt");
   ph_combIso  = iConfig.getParameter<double>("photon_combIso");
   ph_HoE = iConfig.getParameter<double>("photon_HoE");
   ph_sietaieta_eb = iConfig.getParameter<double>("photon_SigmaiEtaiEta_barrel");
   ph_sietaieta_ee = iConfig.getParameter<double>("photon_SigmaiEtaiEta_endcap");
   useProtectionsForJetMET = 
     iConfig.getParameter<bool>("useProtectionsForJetMET");
   ele_protectionsForJetMET = 
     iConfig.getParameter<edm::ParameterSet>("electron_protectionsForJetMET");
   ph_protectionsForJetMET = 
     iConfig.getParameter<edm::ParameterSet>("photon_protectionsForJetMET");
 }

  //Secondary tracks and displaced vertices parameters

  bool rejectTracks_Bad
    = iConfig.getParameter<bool>("rejectTracks_Bad");

  bool rejectTracks_Step45
    = iConfig.getParameter<bool>("rejectTracks_Step45");

  bool usePFNuclearInteractions
    = iConfig.getParameter<bool>("usePFNuclearInteractions");

  bool usePFConversions
    = iConfig.getParameter<bool>("usePFConversions");  

  bool usePFDecays
    = iConfig.getParameter<bool>("usePFDecays");

  double dptRel_DispVtx
    = iConfig.getParameter<double>("dptRel_DispVtx");

  edm::ParameterSet iCfgCandConnector 
    = iConfig.getParameter<edm::ParameterSet>("iCfgCandConnector");


  // fToRead =  iConfig.getUntrackedParameter<vector<string> >("toRead");

  useCalibrationsFromDB_
    = iConfig.getParameter<bool>("useCalibrationsFromDB");

  if (useCalibrationsFromDB_)
    calibrationsLabel_ = iConfig.getParameter<std::string>("calibrationsLabel");

  boost::shared_ptr<PFEnergyCalibration> 
    calibration( new PFEnergyCalibration() ); 

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
			  thepfEnergyCalibrationHF);

  //PFElectrons: call the method setpfeleparameters
  pfAlgo_->setPFEleParameters(mvaEleCut,
			      path_mvaWeightFileEleID,
			      usePFElectrons_,
			      thePFSCEnergyCalibration,
			      calibration,
			      sumEtEcalIsoForEgammaSC_barrel,
			      sumEtEcalIsoForEgammaSC_endcap,
			      coneEcalIsoForEgammaSC,
			      sumPtTrackIsoForEgammaSC_barrel,
			      sumPtTrackIsoForEgammaSC_endcap,
			      nTrackIsoForEgammaSC,
			      coneTrackIsoForEgammaSC,
			      applyCrackCorrectionsForElectrons,
			      usePFSCEleCalib,
			      useEGammaElectrons_,
			      useEGammaSupercluster);
  
  //  pfAlgo_->setPFConversionParameters(usePFConversions);

  // PFPhotons: 
  pfAlgo_->setPFPhotonParameters(usePFPhotons_,
				 path_mvaWeightFileConvID,
				 mvaConvCut,
				 usePhotonReg_,
				 path_X0_Map,
				 calibration,
				 sumPtTrackIsoForPhoton,
				 sumPtTrackIsoSlopeForPhoton);


  // NEW EGamma Filters
   pfAlgo_->setEGammaParameters(use_EGammaFilters_,
				ele_iso_path_mvaWeightFile,
				ele_iso_pt,
				ele_iso_mva_barrel,
				ele_iso_mva_endcap,
				ele_iso_combIso_barrel,
				ele_iso_combIso_endcap,
				ele_noniso_mva,
				ele_missinghits,
				useProtectionsForJetMET,
				ele_protectionsForJetMET,
				ph_MinEt,
				ph_combIso,
				ph_HoE,
				ph_sietaieta_eb,
				ph_sietaieta_ee,
				ph_protectionsForJetMET);

  //Secondary tracks and displaced vertices parameters
  
  pfAlgo_->setDisplacedVerticesParameters(rejectTracks_Bad,
					  rejectTracks_Step45,
					  usePFNuclearInteractions,
 					  usePFConversions,
	 				  usePFDecays,
					  dptRel_DispVtx);
  
  if (usePFNuclearInteractions)
    pfAlgo_->setCandConnectorParameters( iCfgCandConnector );

  

  // Set muon and fake track parameters
  pfAlgo_->setPFMuonAndFakeParameters(iConfig);
  
  //Post cleaning of the HF
  postHFCleaning_
    = iConfig.getParameter<bool>("postHFCleaning");
  double minHFCleaningPt 
    = iConfig.getParameter<double>("minHFCleaningPt");
  double minSignificance
    = iConfig.getParameter<double>("minSignificance");
  double maxSignificance
    = iConfig.getParameter<double>("maxSignificance");
  double minSignificanceReduction
    = iConfig.getParameter<double>("minSignificanceReduction");
  double maxDeltaPhiPt
    = iConfig.getParameter<double>("maxDeltaPhiPt");
  double minDeltaMet
    = iConfig.getParameter<double>("minDeltaMet");

  // Set post HF cleaning muon parameters
  pfAlgo_->setPostHFCleaningParameters(postHFCleaning_,
				       minHFCleaningPt,
				       minSignificance,
				       maxSignificance,
				       minSignificanceReduction,
				       maxDeltaPhiPt,
				       minDeltaMet);

  // Input tags for HF cleaned rechits
  std::vector<edm::InputTag> tags =iConfig.getParameter< std::vector<edm::InputTag> >("cleanedHF");
  for (unsigned int i=0;i<tags.size();++i)
    inputTagCleanedHF_.push_back(consumes<reco::PFRecHitCollection>(tags[i])); 
  //MIKE: Vertex Parameters
  vertices_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));
  useVerticesForNeutral_ = iConfig.getParameter<bool>("useVerticesForNeutral");

  // Use HO clusters and links in the PF reconstruction
  useHO_= iConfig.getParameter<bool>("useHO");
  pfAlgo_->setHOTag(useHO_);

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  bool debug_ = 
    iConfig.getUntrackedParameter<bool>("debug",false);

  pfAlgo_->setDebug( debug_ );

}



PFProducer::~PFProducer() {}

void 
PFProducer::beginRun(const edm::Run & run, 
		     const edm::EventSetup & es) 
{


  /*
  static map<string, PerformanceResult::ResultType> functType;

  functType["PFfa_BARREL"] = PerformanceResult::PFfa_BARREL;
  functType["PFfa_ENDCAP"] = PerformanceResult::PFfa_ENDCAP;
  functType["PFfb_BARREL"] = PerformanceResult::PFfb_BARREL;
  functType["PFfb_ENDCAP"] = PerformanceResult::PFfb_ENDCAP;
  functType["PFfc_BARREL"] = PerformanceResult::PFfc_BARREL;
  functType["PFfc_ENDCAP"] = PerformanceResult::PFfc_ENDCAP;
  functType["PFfaEta_BARREL"] = PerformanceResult::PFfaEta_BARREL;
  functType["PFfaEta_ENDCAP"] = PerformanceResult::PFfaEta_ENDCAP;
  functType["PFfbEta_BARREL"] = PerformanceResult::PFfbEta_BARREL;
  functType["PFfbEta_ENDCAP"] = PerformanceResult::PFfbEta_ENDCAP;
  */

  if ( useCalibrationsFromDB_ ) { 
    // read the PFCalibration functions from the global tags
    edm::ESHandle<PerformancePayload> perfH;
    es.get<PFCalibrationRcd>().get(calibrationsLabel_, perfH);

    PerformancePayloadFromTFormula const * pfCalibrations = static_cast< const PerformancePayloadFromTFormula *>(perfH.product());
    
    pfAlgo_->thePFEnergyCalibration()->setCalibrationFunctions(pfCalibrations);
  }
  
  /*
  for(vector<string>::const_iterator name = fToRead.begin(); name != fToRead.end(); ++name) {    
    
    cout << "Function: " << *name << endl;
    PerformanceResult::ResultType fType = functType[*name];
    pfCalibrations->printFormula(fType);
    
    // evaluate it @ 10 GeV
    float energy = 10.;
    
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, energy);
    
    if(pfCalibrations->isInPayload(fType, point)) {
      float value = pfCalibrations->getResult(fType, point);
      cout << "   Energy before:: " << energy << " after: " << value << endl;
    } else cout <<  "outside limits!" << endl;
    
  }
  */
  
  if(usePFPhotons_ && useRegressionFromDB_) {
    edm::ESHandle<GBRForest> readerPFLCEB;
    edm::ESHandle<GBRForest> readerPFLCEE;    
    edm::ESHandle<GBRForest> readerPFGCEB;
    edm::ESHandle<GBRForest> readerPFGCEEHR9;
    edm::ESHandle<GBRForest> readerPFGCEELR9;
    edm::ESHandle<GBRForest> readerPFRes;
    es.get<GBRWrapperRcd>().get("PFLCorrectionBar",readerPFLCEB);
    ReaderLCEB_=readerPFLCEB.product();
    es.get<GBRWrapperRcd>().get("PFLCorrectionEnd",readerPFLCEE);
    ReaderLCEE_=readerPFLCEE.product();
    es.get<GBRWrapperRcd>().get("PFGCorrectionBar",readerPFGCEB);	
    ReaderGCBarrel_=readerPFGCEB.product();
    es.get<GBRWrapperRcd>().get("PFGCorrectionEndHighR9",readerPFGCEEHR9);
    ReaderGCEndCapHighr9_=readerPFGCEEHR9.product();
    es.get<GBRWrapperRcd>().get("PFGCorrectionEndLowR9",readerPFGCEELR9);
    ReaderGCEndCapLowr9_=readerPFGCEELR9.product();
    es.get<GBRWrapperRcd>().get("PFEcalResolution",readerPFRes);
    ReaderEcalRes_=readerPFRes.product();
    
    /*
    LogDebug("PFProducer")<<"setting regressions from DB "<<endl;
    */
  } 

    if(usePFPhotons_){
      //pfAlgo_->setPFPhotonRegWeights(ReaderLC_, ReaderGC_, ReaderRes_);
      pfAlgo_->setPFPhotonRegWeights(ReaderLCEB_,ReaderLCEE_,ReaderGCBarrel_,ReaderGCEndCapHighr9_, ReaderGCEndCapLowr9_, ReaderEcalRes_ );
    }
}


void 
PFProducer::produce(Event& iEvent, 
		    const EventSetup& iSetup) {
  
  LogDebug("PFProducer")<<"START event: "
			<<iEvent.id().event()
			<<" in run "<<iEvent.id().run()<<endl;
  

  // Get The vertices from the event
  // and assign dynamic vertex parameters
  edm::Handle<reco::VertexCollection> vertices;
  bool gotVertices = iEvent.getByToken(vertices_,vertices);
  if(!gotVertices) {
    ostringstream err;
    err<<"Cannot find vertices for this event.Continuing Without them ";
    LogError("PFProducer")<<err.str()<<endl;
  }

  //Assign the PFAlgo Parameters
  pfAlgo_->setPFVertexParameters(useVerticesForNeutral_,vertices.product());

  // get the collection of blocks 

  Handle< reco::PFBlockCollection > blocks;

  iEvent.getByToken( inputTagBlocks_, blocks );  

  // get the collection of muons 

  Handle< reco::MuonCollection > muons;

  if ( postMuonCleaning_ ) {

    iEvent.getByToken( inputTagMuons_, muons );  
    pfAlgo_->setMuonHandle(muons);
  }

  if (useEGammaElectrons_) {
    Handle < reco::GsfElectronCollection > egelectrons;
    iEvent.getByToken( inputTagEgammaElectrons_, egelectrons );  
    pfAlgo_->setEGElectronCollection(*egelectrons);
  }

  if(use_EGammaFilters_) {

    // Read PFEGammaCandidates

    edm::Handle<edm::View<reco::PFCandidate> > pfEgammaCandidates;
    iEvent.getByToken(inputTagPFEGammaCandidates_,pfEgammaCandidates);

    // Get the value maps
    
    edm::Handle<edm::ValueMap<reco::GsfElectronRef> > valueMapGedElectrons;
    iEvent.getByToken(inputTagValueMapGedElectrons_,valueMapGedElectrons);

    edm::Handle<edm::ValueMap<reco::PhotonRef> > valueMapGedPhotons;
    iEvent.getByToken(inputTagValueMapGedPhotons_,valueMapGedPhotons);

    pfAlgo_->setEGammaCollections(*pfEgammaCandidates,
    				  *valueMapGedElectrons,
    				  *valueMapGedPhotons);

  }


  LogDebug("PFProducer")<<"particle flow is starting"<<endl;

  assert( blocks.isValid() );

  pfAlgo_->reconstructParticles( blocks );
  
  if(verbose_) {
    ostringstream  str;
    str<<(*pfAlgo_)<<endl;
    //    cout << (*pfAlgo_) << endl;
    LogInfo("PFProducer") <<str.str()<<endl;
  }  


  // Florian 5/01/2011
  // Save the PFElectron Extra Collection First as to be able to create valid References  
  if(usePFElectrons_)   {  
    std::unique_ptr<reco::PFCandidateElectronExtraCollection> pOutputElectronCandidateExtraCollection( pfAlgo_->transferElectronExtra() ); 

    const edm::OrphanHandle<reco::PFCandidateElectronExtraCollection > electronExtraProd=
      iEvent.put(std::move(pOutputElectronCandidateExtraCollection),electronExtraOutputCol_);      
    pfAlgo_->setElectronExtraRef(electronExtraProd);
  }

  // Daniele 18/05/2011
  // Save the PFPhoton Extra Collection First as to be able to create valid References  
  if(usePFPhotons_)   {  
    std::unique_ptr<reco::PFCandidatePhotonExtraCollection> pOutputPhotonCandidateExtraCollection( pfAlgo_->transferPhotonExtra() ); 

    const edm::OrphanHandle<reco::PFCandidatePhotonExtraCollection > photonExtraProd=
      iEvent.put(std::move(pOutputPhotonCandidateExtraCollection),photonExtraOutputCol_);      
    pfAlgo_->setPhotonExtraRef(photonExtraProd);
  }

   // Save cosmic cleaned muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pCosmicsMuonCleanedCandidateCollection( pfAlgo_->getPFMuonAlgo()->transferCleanedCosmicCandidates() ); 
    // Save tracker/global cleaned muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pTrackerAndGlobalCleanedMuonCandidateCollection( pfAlgo_->getPFMuonAlgo()->transferCleanedTrackerAndGlobalCandidates() ); 
    // Save fake cleaned muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pFakeCleanedMuonCandidateCollection( pfAlgo_->getPFMuonAlgo()->transferCleanedFakeCandidates() ); 
    // Save punch-through cleaned muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pPunchThroughMuonCleanedCandidateCollection( pfAlgo_->getPFMuonAlgo()->transferPunchThroughCleanedMuonCandidates() ); 
    // Save punch-through cleaned neutral hadron candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pPunchThroughHadronCleanedCandidateCollection( pfAlgo_->getPFMuonAlgo()->transferPunchThroughCleanedHadronCandidates() ); 
    // Save added muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pAddedMuonCandidateCollection( pfAlgo_->getPFMuonAlgo()->transferAddedMuonCandidates() ); 

  // Check HF overcleaning
  reco::PFRecHitCollection hfCopy;
  for ( unsigned ihf=0; ihf<inputTagCleanedHF_.size(); ++ihf ) {
    Handle< reco::PFRecHitCollection > hfCleaned;
    bool foundHF = iEvent.getByToken( inputTagCleanedHF_[ihf], hfCleaned );  
    if (!foundHF) continue;
    for ( unsigned jhf=0; jhf<(*hfCleaned).size(); ++jhf ) { 
      hfCopy.push_back( (*hfCleaned)[jhf] );
    }
  }

  if (postHFCleaning_)
    pfAlgo_->checkCleaning( hfCopy );

  // Save recovered HF candidates
  std::unique_ptr<reco::PFCandidateCollection> pCleanedCandidateCollection( pfAlgo_->transferCleanedCandidates() ); 

  
  // Save the final PFCandidate collection
  std::unique_ptr<reco::PFCandidateCollection> pOutputCandidateCollection( pfAlgo_->transferCandidates() ); 
  

  
  LogDebug("PFProducer")<<"particle flow: putting products in the event"<<endl;
  if ( verbose_ ) std::cout <<"particle flow: putting products in the event. Here the full list"<<endl;
  int nC=0;
  for( reco::PFCandidateCollection::const_iterator  itCand =  (*pOutputCandidateCollection).begin(); itCand !=  (*pOutputCandidateCollection).end(); itCand++) {
    nC++;
      if (verbose_ ) std::cout << nC << ")" << (*itCand).particleId() << std::endl;

  }



  // Write in the event
  iEvent.put(std::move(pOutputCandidateCollection));
  iEvent.put(std::move(pCleanedCandidateCollection),"CleanedHF");

    if ( postMuonCleaning_ ) { 
      iEvent.put(std::move(pCosmicsMuonCleanedCandidateCollection),"CleanedCosmicsMuons");
      iEvent.put(std::move(pTrackerAndGlobalCleanedMuonCandidateCollection),"CleanedTrackerAndGlobalMuons");
      iEvent.put(std::move(pFakeCleanedMuonCandidateCollection),"CleanedFakeMuons");
      iEvent.put(std::move(pPunchThroughMuonCleanedCandidateCollection),"CleanedPunchThroughMuons");
      iEvent.put(std::move(pPunchThroughHadronCleanedCandidateCollection),"CleanedPunchThroughNeutralHadrons");
      iEvent.put(std::move(pAddedMuonCandidateCollection),"AddedMuonsAndHadrons");
    }

  if(usePFElectrons_)
    {
      std::unique_ptr<reco::PFCandidateCollection>  
	pOutputElectronCandidateCollection( pfAlgo_->transferElectronCandidates() ); 
      iEvent.put(std::move(pOutputElectronCandidateCollection),electronOutputCol_);

    }
}

