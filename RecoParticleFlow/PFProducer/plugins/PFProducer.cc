#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoParticleFlow/PFProducer/interface/PFEGammaFilters.h"
#include "RecoParticleFlow/PFProducer/interface/PFAlgo.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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
#include <string>

#include "TFile.h"


/**\class PFProducer 
\brief Producer for particle flow reconstructed particles (PFCandidates)

This producer makes use of PFAlgo, the particle flow algorithm.

\author Colin Bernet
\date   July 2006
*/

class PFProducer : public edm::stream::EDProducer<> {
 public:
  explicit PFProducer(const edm::ParameterSet&);
  
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;

 private:
  const edm::EDPutTokenT<reco::PFCandidateCollection> putToken_;

  edm::EDGetTokenT<reco::PFBlockCollection>  inputTagBlocks_;
  edm::EDGetTokenT<reco::MuonCollection>     inputTagMuons_;
  edm::EDGetTokenT<reco::VertexCollection>   vertices_;
  edm::EDGetTokenT<reco::GsfElectronCollection> inputTagEgammaElectrons_;


  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection> >  inputTagCleanedHF_;
  std::string electronOutputCol_;
  std::string electronExtraOutputCol_;
  std::string photonExtraOutputCol_;

  // NEW EGamma Filters
  edm::EDGetTokenT<edm::ValueMap<reco::GsfElectronRef> >inputTagValueMapGedElectrons_;
  edm::EDGetTokenT<edm::ValueMap<reco::PhotonRef> > inputTagValueMapGedPhotons_;
  edm::EDGetTokenT<edm::View<reco::PFCandidate> > inputTagPFEGammaCandidates_;

  bool use_EGammaFilters_;
  std::unique_ptr<PFEGammaFilters> pfegamma_ = nullptr;


  //Use of HO clusters and links in PF Reconstruction
  bool useHO_;

  /// verbose ?
  bool  verbose_;

  // Post muon cleaning ?
  bool postMuonCleaning_;
  
  // what about e/g electrons ?
  bool useEGammaElectrons_;

  // Use vertices for Neutral particles ?
  bool useVerticesForNeutral_;

  // Take PF cluster calibrations from Global Tag ?
  bool useCalibrationsFromDB_;
  std::string calibrationsLabel_;

  bool postHFCleaning_;
  // Name of the calibration functions to read from the database
  // std::vector<std::string> fToRead;
  
  /// particle flow algorithm
  PFAlgo pfAlgo_;

};

DEFINE_FWK_MODULE(PFProducer);


using namespace std;
using namespace edm;


PFProducer::PFProducer(const edm::ParameterSet& iConfig)
  : putToken_{produces<reco::PFCandidateCollection>()}
  , pfAlgo_(iConfig.getUntrackedParameter<bool>("debug",false))
{
  //--ab: get calibration factors for HF:
  auto thepfEnergyCalibrationHF = std::make_shared<PFEnergyCalibrationHF>(
      iConfig.getParameter<bool>("calibHF_use"),
      iConfig.getParameter<std::vector<double> >("calibHF_eta_step"),
      iConfig.getParameter<std::vector<double> >("calibHF_a_EMonly"),
      iConfig.getParameter<std::vector<double> >("calibHF_b_HADonly"),
      iConfig.getParameter<std::vector<double> >("calibHF_a_EMHAD"),
      iConfig.getParameter<std::vector<double> >("calibHF_b_EMHAD")
  );
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

  useEGammaElectrons_
    = iConfig.getParameter<bool>("useEGammaElectrons");    

  if(  useEGammaElectrons_) {
    inputTagEgammaElectrons_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("egammaElectrons"));
  }

  electronOutputCol_
    = iConfig.getParameter<std::string>("pf_electron_output_col");

  std::vector<double>  calibPFSCEle_Fbrem_barrel; 
  std::vector<double>  calibPFSCEle_Fbrem_endcap;
  std::vector<double>  calibPFSCEle_barrel;
  std::vector<double>  calibPFSCEle_endcap;
  calibPFSCEle_Fbrem_barrel = iConfig.getParameter<std::vector<double> >("calibPFSCEle_Fbrem_barrel");
  calibPFSCEle_Fbrem_endcap = iConfig.getParameter<std::vector<double> >("calibPFSCEle_Fbrem_endcap");
  calibPFSCEle_barrel = iConfig.getParameter<std::vector<double> >("calibPFSCEle_barrel");
  calibPFSCEle_endcap = iConfig.getParameter<std::vector<double> >("calibPFSCEle_endcap");
  std::shared_ptr<PFSCEnergyCalibration>  
    thePFSCEnergyCalibration ( new PFSCEnergyCalibration(calibPFSCEle_Fbrem_barrel,calibPFSCEle_Fbrem_endcap,
							 calibPFSCEle_barrel,calibPFSCEle_endcap )); 


  // register products
  produces<reco::PFCandidateCollection>("CleanedHF");
  produces<reco::PFCandidateCollection>("CleanedCosmicsMuons");
  produces<reco::PFCandidateCollection>("CleanedTrackerAndGlobalMuons");
  produces<reco::PFCandidateCollection>("CleanedFakeMuons");
  produces<reco::PFCandidateCollection>("CleanedPunchThroughMuons");
  produces<reco::PFCandidateCollection>("CleanedPunchThroughNeutralHadrons");
  produces<reco::PFCandidateCollection>("AddedMuonsAndHadrons");


  double nSigmaECAL 
    = iConfig.getParameter<double>("pf_nsigma_ECAL");
  double nSigmaHCAL 
    = iConfig.getParameter<double>("pf_nsigma_HCAL");
  
  string mvaWeightFileEleID
    = iConfig.getParameter<string>("pf_electronID_mvaWeightFile");

  string path_mvaWeightFileEleID;

  //PFPhoton Configuration

  string path_mvaWeightFileConvID;
  string mvaWeightFileConvID;
  string path_mvaWeightFileGCorr;
  string path_mvaWeightFileLCorr;
  string path_X0_Map;
  string path_mvaWeightFileRes;


  // Reading new EGamma selection cuts
  bool useProtectionsForJetMET(false);
 // Reading new EGamma ubiased collections and value maps
 if(use_EGammaFilters_) {
   inputTagPFEGammaCandidates_ = consumes<edm::View<reco::PFCandidate> >((iConfig.getParameter<edm::InputTag>("PFEGammaCandidates")));
   inputTagValueMapGedElectrons_ = consumes<edm::ValueMap<reco::GsfElectronRef>>(iConfig.getParameter<edm::InputTag>("GedElectronValueMap")); 
   inputTagValueMapGedPhotons_ = consumes<edm::ValueMap<reco::PhotonRef> >(iConfig.getParameter<edm::InputTag>("GedPhotonValueMap")); 
   useProtectionsForJetMET = iConfig.getParameter<bool>("useProtectionsForJetMET");
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

  auto calibration = std::make_shared<PFEnergyCalibration>();
  
  pfAlgo_.setParameters( nSigmaECAL, 
			  nSigmaHCAL,
			  calibration,
			  thepfEnergyCalibrationHF);

  // NEW EGamma Filters
   pfAlgo_.setEGammaParameters(use_EGammaFilters_, useProtectionsForJetMET);

  if(use_EGammaFilters_) pfegamma_ = std::make_unique<PFEGammaFilters>(iConfig);


  //Secondary tracks and displaced vertices parameters
  
  pfAlgo_.setDisplacedVerticesParameters(rejectTracks_Bad,
					  rejectTracks_Step45,
					  usePFNuclearInteractions,
 					  usePFConversions,
	 				  usePFDecays,
					  dptRel_DispVtx);
  
  if (usePFNuclearInteractions)
    pfAlgo_.setCandConnectorParameters( iCfgCandConnector );

  

  // Set muon and fake track parameters
  pfAlgo_.setPFMuonAndFakeParameters(iConfig);
  pfAlgo_.setBadHcalTrackParams(iConfig);
  
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
  pfAlgo_.setPostHFCleaningParameters(postHFCleaning_,
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
  pfAlgo_.setHOTag(useHO_);

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);
}


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
    
    pfAlgo_.thePFEnergyCalibration()->setCalibrationFunctions(pfCalibrations);
  }
  
}


void 
PFProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  LogDebug("PFProducer")<<"START event: " <<iEvent.id().event() <<" in run "<<iEvent.id().run()<<endl;

  //Assign the PFAlgo Parameters
  pfAlgo_.setPFVertexParameters(useVerticesForNeutral_, iEvent.get(vertices_));

  // get the collection of blocks 
  auto blocks = iEvent.getHandle( inputTagBlocks_);
  assert( blocks.isValid() );

  // get the collection of muons 
  if ( postMuonCleaning_ ) pfAlgo_.setMuonHandle( iEvent.getHandle(inputTagMuons_) );

  if(use_EGammaFilters_) pfAlgo_.setEGammaCollections( iEvent.get(inputTagPFEGammaCandidates_),
                                                        iEvent.get(inputTagValueMapGedElectrons_),
                                                        iEvent.get(inputTagValueMapGedPhotons_));


  LogDebug("PFProducer")<<"particle flow is starting"<<endl;

  pfAlgo_.reconstructParticles( blocks, pfegamma_.get() );
  
  if(verbose_) {
    ostringstream  str;
    str<< pfAlgo_ <<endl;
    //    cout << pfAlgo_ << endl;
    LogInfo("PFProducer") <<str.str()<<endl;
  }  

   // Save cosmic cleaned muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pCosmicsMuonCleanedCandidateCollection( pfAlgo_.getPFMuonAlgo()->transferCleanedCosmicCandidates() ); 
    // Save tracker/global cleaned muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pTrackerAndGlobalCleanedMuonCandidateCollection( pfAlgo_.getPFMuonAlgo()->transferCleanedTrackerAndGlobalCandidates() ); 
    // Save fake cleaned muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pFakeCleanedMuonCandidateCollection( pfAlgo_.getPFMuonAlgo()->transferCleanedFakeCandidates() ); 
    // Save punch-through cleaned muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pPunchThroughMuonCleanedCandidateCollection( pfAlgo_.getPFMuonAlgo()->transferPunchThroughCleanedMuonCandidates() ); 
    // Save punch-through cleaned neutral hadron candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pPunchThroughHadronCleanedCandidateCollection( pfAlgo_.getPFMuonAlgo()->transferPunchThroughCleanedHadronCandidates() ); 
    // Save added muon candidates
    std::unique_ptr<reco::PFCandidateCollection> 
      pAddedMuonCandidateCollection( pfAlgo_.getPFMuonAlgo()->transferAddedMuonCandidates() ); 

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
    pfAlgo_.checkCleaning( hfCopy );

  // Save recovered HF candidates
  std::unique_ptr<reco::PFCandidateCollection> pCleanedCandidateCollection( pfAlgo_.transferCleanedCandidates() ); 

  
  // Save the final PFCandidate collection
  reco::PFCandidateCollection pOutputCandidateCollection = pfAlgo_.transferCandidates();
  

  
  LogDebug("PFProducer")<<"particle flow: putting products in the event"<<endl;
  if ( verbose_ ) std::cout <<"particle flow: putting products in the event. Here the full list"<<endl;
  int nC=0;
  for(auto const& cand : pOutputCandidateCollection) {
    nC++;
      if (verbose_ ) std::cout << nC << ")" << cand.particleId() << std::endl;

  }



  // Write in the event
  iEvent.emplace(putToken_,pOutputCandidateCollection);
  iEvent.put(std::move(pCleanedCandidateCollection),"CleanedHF");

    if ( postMuonCleaning_ ) { 
      iEvent.put(std::move(pCosmicsMuonCleanedCandidateCollection),"CleanedCosmicsMuons");
      iEvent.put(std::move(pTrackerAndGlobalCleanedMuonCandidateCollection),"CleanedTrackerAndGlobalMuons");
      iEvent.put(std::move(pFakeCleanedMuonCandidateCollection),"CleanedFakeMuons");
      iEvent.put(std::move(pPunchThroughMuonCleanedCandidateCollection),"CleanedPunchThroughMuons");
      iEvent.put(std::move(pPunchThroughHadronCleanedCandidateCollection),"CleanedPunchThroughNeutralHadrons");
      iEvent.put(std::move(pAddedMuonCandidateCollection),"AddedMuonsAndHadrons");
    }
}
