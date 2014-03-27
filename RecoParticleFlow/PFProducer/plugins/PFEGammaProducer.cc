#include "RecoParticleFlow/PFProducer/plugins/PFEGammaProducer.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibrationHF.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"

#include "DataFormats/Common/interface/RefToPtr.h"
#include <sstream>

#include "TFile.h"

//#define PFLOW_DEBUG

#ifdef PFLOW_DEBUG
#define docast(x,y) dynamic_cast<x>(y)
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x)  edm::LogInfo(x)
#else
#define docast(x,y) reinterpret_cast<x>(y)
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

namespace {
  typedef std::list< reco::PFBlockRef >::iterator IBR;
}

PFEGammaProducer::PFEGammaProducer(const edm::ParameterSet& iConfig):
  primaryVertex_(reco::Vertex()),
  ebeeClustersCollection_("EBEEClusters"),
  esClustersCollection_("ESClusters") {
    
  PFEGammaAlgo::PFEGConfigInfo algo_config;

  inputTagBlocks_ 
    = consumes<reco::PFBlockCollection>(iConfig.getParameter<edm::InputTag>("blocks"));

  eetopsSrc_ = consumes<reco::PFCluster::EEtoPSAssociation>(iConfig.getParameter<edm::InputTag>("EEtoPS_source"));

  algo_config.useReg
    =  iConfig.getParameter<bool>("usePhotonReg");

  useVerticesForNeutral_
    = iConfig.getParameter<bool>("useVerticesForNeutral"); 

  useRegressionFromDB_
    = iConfig.getParameter<bool>("useRegressionFromDB"); 


  std::vector<double>  calibPFSCEle_Fbrem_barrel; 
  std::vector<double>  calibPFSCEle_Fbrem_endcap;
  std::vector<double>  calibPFSCEle_barrel;
  std::vector<double>  calibPFSCEle_endcap;
  algo_config.usePFSCEleCalib = iConfig.getParameter<bool>("usePFSCEleCalib");
  calibPFSCEle_Fbrem_barrel = iConfig.getParameter<std::vector<double> >("calibPFSCEle_Fbrem_barrel");
  calibPFSCEle_Fbrem_endcap = iConfig.getParameter<std::vector<double> >("calibPFSCEle_Fbrem_endcap");
  calibPFSCEle_barrel = iConfig.getParameter<std::vector<double> >("calibPFSCEle_barrel");
  calibPFSCEle_endcap = iConfig.getParameter<std::vector<double> >("calibPFSCEle_endcap");
  std::shared_ptr<PFSCEnergyCalibration>  
    thePFSCEnergyCalibration ( new PFSCEnergyCalibration(calibPFSCEle_Fbrem_barrel,calibPFSCEle_Fbrem_endcap,
                                                         calibPFSCEle_barrel,calibPFSCEle_endcap )); 
                               
  algo_config.useEGammaSupercluster = 
    iConfig.getParameter<bool>("useEGammaSupercluster");
  algo_config.produceEGCandsWithNoSuperCluster = 
    iConfig.getParameter<bool>("produceEGCandsWithNoSuperCluster");
  algo_config.sumEtEcalIsoForEgammaSC_barrel = 
    iConfig.getParameter<double>("sumEtEcalIsoForEgammaSC_barrel");
  algo_config.sumEtEcalIsoForEgammaSC_endcap = 
    iConfig.getParameter<double>("sumEtEcalIsoForEgammaSC_endcap");
  algo_config.coneEcalIsoForEgammaSC = 
    iConfig.getParameter<double>("coneEcalIsoForEgammaSC");
  algo_config.sumPtTrackIsoForEgammaSC_barrel = 
    iConfig.getParameter<double>("sumPtTrackIsoForEgammaSC_barrel");
  algo_config.sumPtTrackIsoForEgammaSC_endcap = 
    iConfig.getParameter<double>("sumPtTrackIsoForEgammaSC_endcap");
  algo_config.coneTrackIsoForEgammaSC = 
    iConfig.getParameter<double>("coneTrackIsoForEgammaSC");
  algo_config.nTrackIsoForEgammaSC  = 
    iConfig.getParameter<unsigned int>("nTrackIsoForEgammaSC");


  // register products
  produces<reco::PFCandidateCollection>();
  produces<reco::PFCandidateEGammaExtraCollection>();  
  produces<reco::SuperClusterCollection>();
  produces<reco::CaloClusterCollection>(ebeeClustersCollection_);
  produces<reco::CaloClusterCollection>(esClustersCollection_);  
  
  //PFElectrons Configuration
  algo_config.mvaEleCut
    = iConfig.getParameter<double>("pf_electron_mvaCut");

  
  algo_config. mvaWeightFileEleID
    = iConfig.getParameter<std::string>("pf_electronID_mvaWeightFile");

  algo_config.applyCrackCorrections
    = iConfig.getParameter<bool>("pf_electronID_crackCorrection");
  
  std::string path_mvaWeightFileEleID;

  algo_config.mvaWeightFileEleID = 
    edm::FileInPath ( algo_config.mvaWeightFileEleID.c_str() ).fullPath();
     

  //PFPhoton Configuration

  std::string path_mvaWeightFileConvID;
  std::string mvaWeightFileConvID;
  std::string path_mvaWeightFileGCorr;
  std::string path_mvaWeightFileLCorr;
  std::string path_X0_Map;
  std::string path_mvaWeightFileRes;

  algo_config.mvaweightfile =
    iConfig.getParameter<std::string>("pf_convID_mvaWeightFile");
  algo_config.mvaConvCut = iConfig.getParameter<double>("pf_conv_mvaCut");
  algo_config.mvaweightfile = 
    edm::FileInPath ( algo_config.mvaweightfile.c_str() ).fullPath();  
  algo_config.sumPtTrackIsoForPhoton = 
    iConfig.getParameter<double>("sumPtTrackIsoForPhoton");
  algo_config.sumPtTrackIsoSlopeForPhoton = 
    iConfig.getParameter<double>("sumPtTrackIsoSlopeForPhoton");

  algo_config.X0_Map = iConfig.getParameter<std::string>("X0_Map");
  algo_config.X0_Map = 
    edm::FileInPath( algo_config.X0_Map.c_str() ).fullPath();

  if(!useRegressionFromDB_) {
    std::string mvaWeightFileLCorr=iConfig.getParameter<std::string>("pf_locC_mvaWeightFile");
    path_mvaWeightFileLCorr = edm::FileInPath( mvaWeightFileLCorr.c_str() ).fullPath();
    std::string mvaWeightFileGCorr=iConfig.getParameter<std::string>("pf_GlobC_mvaWeightFile");
    path_mvaWeightFileGCorr = edm::FileInPath( mvaWeightFileGCorr.c_str() ).fullPath();
    std::string mvaWeightFileRes=iConfig.getParameter<std::string>("pf_Res_mvaWeightFile");
    path_mvaWeightFileRes=edm::FileInPath(mvaWeightFileRes.c_str()).fullPath();

    TFile *fgbr = new TFile(path_mvaWeightFileGCorr.c_str(),"READ");
    ReaderGC_  =(const GBRForest*)fgbr->Get("GBRForest");
    TFile *fgbr2 = new TFile(path_mvaWeightFileLCorr.c_str(),"READ");
    ReaderLC_  = (const GBRForest*)fgbr2->Get("GBRForest");
    TFile *fgbr3 = new TFile(path_mvaWeightFileRes.c_str(),"READ");
    ReaderRes_  = (const GBRForest*)fgbr3->Get("GBRForest");
    LogDebug("PFEGammaProducer")<<"Will set regressions from binary files " <<std::endl;
  }

  edm::ParameterSet iCfgCandConnector 
    = iConfig.getParameter<edm::ParameterSet>("iCfgCandConnector");


  // fToRead =  iConfig.getUntrackedParameter<std::vector<std::string> >("toRead");

  useCalibrationsFromDB_
    = iConfig.getParameter<bool>("useCalibrationsFromDB");    

  std::shared_ptr<PFEnergyCalibration> calibration(new PFEnergyCalibration()); 

  int algoType 
    = iConfig.getParameter<unsigned>("algoType");
  
  switch(algoType) {
  case 0:
    //pfAlgo_.reset( new PFAlgo);
    break;
   default:
    assert(0);
  }
  
  //PFEGamma
  setPFEGParameters(algo_config);  

  //MIKE: Vertex Parameters
  vertices_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

//   bool debug_ = 
//     iConfig.getUntrackedParameter<bool>("debug",false);

}



PFEGammaProducer::~PFEGammaProducer() {}

void 
PFEGammaProducer::beginRun(const edm::Run & run, 
                     const edm::EventSetup & es) 
{

  if(useRegressionFromDB_) {
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
    LogDebug("PFEGammaProducer")<<"setting regressions from DB "<<std::endl;
    */
  } 


  //pfAlgo_->setPFPhotonRegWeights(ReaderLC_, ReaderGC_, ReaderRes_);
    
}


void 
PFEGammaProducer::produce(edm::Event& iEvent, 
			     const edm::EventSetup& iSetup) {
  
  LOGDRESSED("PFEGammaProducer")
    <<"START event: "
    <<iEvent.id().event()
    <<" in run "<<iEvent.id().run()<<std::endl;
  

  // reset output collection  
  egCandidates_.reset( new reco::PFCandidateCollection );   
  egExtra_.reset( new reco::PFCandidateEGammaExtraCollection ); 
  sClusters_.reset( new reco::SuperClusterCollection );      
    
  // Get the EE-PS associations
  edm::Handle<reco::PFCluster::EEtoPSAssociation> eetops;
  iEvent.getByToken(eetopsSrc_,eetops);
  pfeg_->setEEtoPSAssociation(eetops);

  // Get The vertices from the event
  // and assign dynamic vertex parameters
  edm::Handle<reco::VertexCollection> vertices;
  bool gotVertices = iEvent.getByToken(vertices_,vertices);
  if(!gotVertices) {
    std::ostringstream err;
    err<<"Cannot find vertices for this event.Continuing Without them ";
    edm::LogError("PFEGammaProducer")<<err.str()<<std::endl;
  }

  //Assign the PFAlgo Parameters
  setPFVertexParameters(useVerticesForNeutral_,vertices.product());

  // get the collection of blocks 

  edm::Handle< reco::PFBlockCollection > blocks;

  LOGDRESSED("PFEGammaProducer")<<"getting blocks"<<std::endl;
  bool found = iEvent.getByToken( inputTagBlocks_, blocks );  

  if(!found ) {

    std::ostringstream err;
    err<<"cannot find blocks: (tag index)"
       << std::hex<< inputTagBlocks_.index() << std::dec;
    edm::LogError("PFEGammaProducer")<<err.str()<<std::endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }
  
  LOGDRESSED("PFEGammaProducer")
    <<"EGPFlow is starting..."<<std::endl;

#ifdef PFLOW_DEBUG
  assert( blocks.isValid() && "edm::Handle to blocks was null!");
  std::ostringstream  str;
  //str<<(*pfAlgo_)<<std::endl;
  //    cout << (*pfAlgo_) << std::endl;
  LOGDRESSED("PFEGammaProducer") <<str.str()<<std::endl;
#endif  

  // sort elements in three lists:
  std::list< reco::PFBlockRef > hcalBlockRefs;
  std::list< reco::PFBlockRef > ecalBlockRefs;
  std::list< reco::PFBlockRef > hoBlockRefs;
  std::list< reco::PFBlockRef > otherBlockRefs;
  
  for( unsigned i=0; i<blocks->size(); ++i ) {
    // reco::PFBlockRef blockref( blockh,i );
    //reco::PFBlockRef blockref = createBlockRef( *blocks, i);
    reco::PFBlockRef blockref(blocks, i);    
    
    const edm::OwnVector< reco::PFBlockElement >& 
      elements = blockref->elements();
   
    LOGDRESSED("PFEGammaProducer") 
      << "Found " << elements.size() 
      << " PFBlockElements in block: " << i << std::endl;
    
    bool singleEcalOrHcal = false;
    if( elements.size() == 1 ){
      switch( elements[0].type() ) {
      case reco::PFBlockElement::SC:
	edm::LogError("PFEGammaProducer")
	  << "PFBLOCKALGO BUG!!!! Found a SuperCluster in a block by itself!";
      case reco::PFBlockElement::PS1:
      case reco::PFBlockElement::PS2:
      case reco::PFBlockElement::ECAL:
        ecalBlockRefs.push_back( blockref );
        singleEcalOrHcal = true;
	break;
      case reco::PFBlockElement::HFEM:
      case reco::PFBlockElement::HFHAD:
      case reco::PFBlockElement::HCAL:
        hcalBlockRefs.push_back( blockref );
        singleEcalOrHcal = true;
	break;
      case reco::PFBlockElement::HO:
        // Single HO elements are likely to be noise. Not considered for now.
        hoBlockRefs.push_back( blockref );
        singleEcalOrHcal = true;
	break;
      default:
	break;
      }
    }
    
    if(!singleEcalOrHcal) {
      otherBlockRefs.push_back( blockref );
    }
  }//loop blocks
  
  // loop on blocks that are not single ecal, single ps1, single ps2 , or
  // single hcal and produce unbiased collection of EGamma Candidates

  //printf("loop over blocks\n");
  //unsigned nblcks = 0;

  // this auto is a const reco::PFBlockRef&
  for( const auto& blockref : otherBlockRefs ) {   
    // this auto is a: const edm::OwnVector< reco::PFBlockElement >&
    const auto& elements = blockref->elements();
    // make a copy of the link data, which will be edited.
    //PFBlock::LinkData linkData =  block.linkData();
    
    // keep track of the elements which are still active.
    std::vector<bool> active( elements.size(), true );      
    
    pfeg_->RunPFEG(blockref,active);
    
    edm::LogInfo("PFEGammaProducer")
      << "Block with " << elements.size() 
      << " elements produced " 
      << pfeg_->getCandidates().size() 
      << " e-g candidates!" << std::endl;

    const size_t egsize = egCandidates_->size();
    egCandidates_->resize(egsize + pfeg_->getCandidates().size());
    reco::PFCandidateCollection::iterator eginsertfrom = 
      egCandidates_->begin() + egsize;
    std::move(pfeg_->getCandidates().begin(),
	      pfeg_->getCandidates().end(),
	      eginsertfrom);
    
    const size_t egxsize = egExtra_->size();
    egExtra_->resize(egxsize + pfeg_->getEGExtra().size());
    reco::PFCandidateEGammaExtraCollection::iterator egxinsertfrom = 
      egExtra_->begin() + egxsize;
    std::move(pfeg_->getEGExtra().begin(),
	      pfeg_->getEGExtra().end(),
	      egxinsertfrom);

    const size_t rscsize = sClusters_->size();
    sClusters_->resize(rscsize + pfeg_->getRefinedSCs().size());
    reco::SuperClusterCollection::iterator rscinsertfrom = 
      sClusters_->begin() + rscsize;
    std::move(pfeg_->getRefinedSCs().begin(),
	      pfeg_->getRefinedSCs().end(),
	      rscinsertfrom);    
  }
  
  edm::LogInfo("PFEGammaProducer")
      << "Running PFEGammaAlgo on all blocks produced = " 
      << egCandidates_->size() << " e-g candidates!"
      << std::endl;

  edm::RefProd<reco::SuperClusterCollection> sClusterProd = 
    iEvent.getRefBeforePut<reco::SuperClusterCollection>();

  edm::RefProd<reco::PFCandidateEGammaExtraCollection> egXtraProd = 
    iEvent.getRefBeforePut<reco::PFCandidateEGammaExtraCollection>();
  
  size_t non_zero_sc_idx = 0; 
  
  //set the correct references to refined SC and EG extra using the refprods
  for (unsigned int i=0; i < egCandidates_->size(); ++i) {
    reco::PFCandidate &cand = egCandidates_->at(i);
    reco::PFCandidateEGammaExtra &xtra = egExtra_->at(i);
    reco::SuperCluster& rsc = sClusters_->at(non_zero_sc_idx);
    
    reco::PFCandidateEGammaExtraRef extraref(egXtraProd,i);
    reco::SuperClusterRef refinedSCRef;
    // only set refined SC refs where the is valid!
    if( rsc.energy() != 0.0 ) {      
      refinedSCRef = reco::SuperClusterRef(sClusterProd,non_zero_sc_idx++);
    } else {
      sClusters_->erase(sClusters_->begin() + non_zero_sc_idx);
    }
    xtra.setSuperClusterRef(refinedSCRef); 
    cand.setSuperClusterRef(refinedSCRef);
    cand.setPFEGammaExtraRef(extraref);    
  }
  
  //build collections of output CaloClusters from the used PFClusters
  std::auto_ptr<reco::CaloClusterCollection> caloClustersEBEE(new reco::CaloClusterCollection);
  std::auto_ptr<reco::CaloClusterCollection> caloClustersES(new reco::CaloClusterCollection);
  
  std::map<edm::Ptr<reco::CaloCluster>, unsigned int> pfClusterMapEBEE; //maps of pfclusters to caloclusters 
  std::map<edm::Ptr<reco::CaloCluster>, unsigned int> pfClusterMapES;  
  
  for( const auto& sc : *sClusters_ ) {
    for (reco::CaloCluster_iterator pfclus = sc.clustersBegin(); pfclus!=sc.clustersEnd(); ++pfclus) {
      if (!pfClusterMapEBEE.count(*pfclus)) {
        reco::CaloCluster caloclus(**pfclus);
        caloClustersEBEE->push_back(caloclus);
        pfClusterMapEBEE[*pfclus] = caloClustersEBEE->size() - 1;
      }
      else {
        throw cms::Exception("PFEgammaProducer::produce")
            << "Found an EB/EE pfcluster matched to more than one supercluster!" 
            << std::dec << std::endl;
      }
    }
    for (reco::CaloCluster_iterator pfclus = sc.preshowerClustersBegin(); pfclus!=sc.preshowerClustersEnd(); ++pfclus) {
      if (!pfClusterMapES.count(*pfclus)) {
        reco::CaloCluster caloclus(**pfclus);
        caloClustersES->push_back(caloclus);
        pfClusterMapES[*pfclus] = caloClustersES->size() - 1;
      }
      else {
        throw cms::Exception("PFEgammaProducer::produce")
            << "Found an ES pfcluster matched to more than one supercluster!" 
            << std::dec << std::endl;
      }
    }
  }
  
  //put calocluster output collections in event and get orphan handles to create ptrs
  const edm::OrphanHandle<reco::CaloClusterCollection> &caloClusHandleEBEE = iEvent.put(caloClustersEBEE,ebeeClustersCollection_);
  const edm::OrphanHandle<reco::CaloClusterCollection> &caloClusHandleES = iEvent.put(caloClustersES,esClustersCollection_);
  
  //relink superclusters to output caloclusters
  for( auto& sc : *sClusters_ ) {
    edm::Ptr<reco::CaloCluster> seedptr(caloClusHandleEBEE,pfClusterMapEBEE[sc.seed()]);
    sc.setSeed(seedptr);
    
    reco::CaloClusterPtrVector clusters;
    for (reco::CaloCluster_iterator pfclus = sc.clustersBegin(); pfclus!=sc.clustersEnd(); ++pfclus) {
      edm::Ptr<reco::CaloCluster> clusptr(caloClusHandleEBEE,pfClusterMapEBEE[*pfclus]);
      clusters.push_back(clusptr);
    }
    sc.setClusters(clusters);
    
    reco::CaloClusterPtrVector psclusters;
    for (reco::CaloCluster_iterator pfclus = sc.preshowerClustersBegin(); pfclus!=sc.preshowerClustersEnd(); ++pfclus) {
      edm::Ptr<reco::CaloCluster> clusptr(caloClusHandleES,pfClusterMapES[*pfclus]);
      psclusters.push_back(clusptr);
    }
    sc.setPreshowerClusters(psclusters);  
  }
  
  // release our demonspawn into the wild to cause havoc
  iEvent.put(sClusters_);
  iEvent.put(egExtra_);  
  iEvent.put(egCandidates_); 
}

//PFEGammaAlgo: a new method added to set the parameters for electron and photon reconstruction. 
void 
PFEGammaProducer::setPFEGParameters(PFEGammaAlgo::PFEGConfigInfo& cfg) {  
  
  FILE * fileEleID = fopen(cfg.mvaWeightFileEleID.c_str(), "r");
  if (fileEleID) {
    fclose(fileEleID);
  }
  else {
    std::string err = "PFAlgo: cannot open weight file '";
    err += cfg.mvaWeightFileEleID;
    err += "'";
    throw std::invalid_argument( err );
  }

  //for MVA pass PV if there is one in the collection otherwise pass a dummy  
  if(!useVerticesForNeutral_) { // create a dummy PV  
    reco::Vertex::Error e;  
    e(0, 0) = 0.0015 * 0.0015;  
    e(1, 1) = 0.0015 * 0.0015;  
    e(2, 2) = 15. * 15.;  
    reco::Vertex::Point p(0, 0, 0);  
    primaryVertex_ = reco::Vertex(p, e, 0, 0, 0);  
  }  
  // pv=&dummy;  
  //if(! usePFPhotons_) return;  
  FILE * filePhotonConvID = fopen(cfg.mvaweightfile.c_str(), "r");  
  if (filePhotonConvID) {  
    fclose(filePhotonConvID);  
  }  
  else {  
    std::string err = "PFAlgo: cannot open weight file '";  
    err += cfg.mvaweightfile;
    err += "'";  
    throw std::invalid_argument( err );  
  }  
  cfg.primaryVtx = &primaryVertex_;  
  pfeg_.reset(new PFEGammaAlgo(cfg));
}

/*
void PFAlgo::setPFPhotonRegWeights(
                  const GBRForest *LCorrForest,
                  const GBRForest *GCorrForest,
                  const GBRForest *ResForest
                  ) {                                                           
  if(usePFPhotons_) 
    pfpho_->setGBRForest(LCorrForest, GCorrForest, ResForest);
} 
*/

void
PFEGammaProducer::setPFVertexParameters(bool useVertex,
                              const reco::VertexCollection*  primaryVertices) {
  useVerticesForNeutral_ = useVertex;

  //Set the vertices for muon cleaning
//  pfmu_->setInputsForCleaning(primaryVertices);


  //Now find the primary vertex!
  //bool primaryVertexFound = false;
  int nVtx=primaryVertices->size();
  pfeg_->setnPU(nVtx);
//   if(usePFPhotons_){
//     pfpho_->setnPU(nVtx);
//   }
  primaryVertex_ = primaryVertices->front();
  for (unsigned short i=0 ;i<primaryVertices->size();++i)
    {
      if(primaryVertices->at(i).isValid()&&(!primaryVertices->at(i).isFake()))
        {
          primaryVertex_ = primaryVertices->at(i);
          //primaryVertexFound = true;
          break;
        }
    }
  
  pfeg_->setPhotonPrimaryVtx(primaryVertex_ );
  
}
