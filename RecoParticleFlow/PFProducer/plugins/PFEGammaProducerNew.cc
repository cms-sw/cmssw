#include "RecoParticleFlow/PFProducer/plugins/PFEGammaProducerNew.h"
#include "RecoParticleFlow/PFProducer/interface/PFEGammaAlgoNew.h"

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
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include <sstream>

#include "TFile.h"


namespace {
  typedef std::list< reco::PFBlockRef >::iterator IBR;
}

PFEGammaProducerNew::PFEGammaProducerNew(const edm::ParameterSet& iConfig) {
  

  inputTagBlocks_ 
    = iConfig.getParameter<edm::InputTag>("blocks");

  usePhotonReg_
    =  iConfig.getParameter<bool>("usePhotonReg");

  useRegressionFromDB_
    = iConfig.getParameter<bool>("useRegressionFromDB"); 


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
  std::shared_ptr<PFSCEnergyCalibration>  
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
  produces<reco::PFCandidateEGammaExtraCollection>();
  produces<reco::CaloClusterCollection>("EBEEClusters");
  produces<reco::CaloClusterCollection>("ESClusters");
  produces<reco::SuperClusterCollection>();
  
  //PFElectrons Configuration
  double mvaEleCut
    = iConfig.getParameter<double>("pf_electron_mvaCut");

  
  std::string mvaWeightFileEleID
    = iConfig.getParameter<std::string>("pf_electronID_mvaWeightFile");

  bool applyCrackCorrectionsForElectrons
    = iConfig.getParameter<bool>("pf_electronID_crackCorrection");
  
  std::string path_mvaWeightFileEleID;

  path_mvaWeightFileEleID = edm::FileInPath ( mvaWeightFileEleID.c_str() ).fullPath();
     

  //PFPhoton Configuration

  std::string path_mvaWeightFileConvID;
  std::string mvaWeightFileConvID;
  std::string path_mvaWeightFileGCorr;
  std::string path_mvaWeightFileLCorr;
  std::string path_X0_Map;
  std::string path_mvaWeightFileRes;
  double mvaConvCut=-99.;
  double sumPtTrackIsoForPhoton = 99.;
  double sumPtTrackIsoSlopeForPhoton = 99.;


  mvaWeightFileConvID =iConfig.getParameter<std::string>("pf_convID_mvaWeightFile");
  mvaConvCut = iConfig.getParameter<double>("pf_conv_mvaCut");
  path_mvaWeightFileConvID = edm::FileInPath ( mvaWeightFileConvID.c_str() ).fullPath();  
  sumPtTrackIsoForPhoton = iConfig.getParameter<double>("sumPtTrackIsoForPhoton");
  sumPtTrackIsoSlopeForPhoton = iConfig.getParameter<double>("sumPtTrackIsoSlopeForPhoton");

  std::string X0_Map=iConfig.getParameter<std::string>("X0_Map");
  path_X0_Map = edm::FileInPath( X0_Map.c_str() ).fullPath();

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
    LogDebug("PFEGammaProducerNew")<<"Will set regressions from binary files " <<std::endl;
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
  setPFEGParameters(mvaEleCut,
		    path_mvaWeightFileEleID,
		    true,
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
		    useEGammaSupercluster,
		    true,
		    path_mvaWeightFileConvID,
		    mvaConvCut,
		    usePhotonReg_,
		    path_X0_Map,
		    sumPtTrackIsoForPhoton,
		    sumPtTrackIsoSlopeForPhoton);  

  //MIKE: Vertex Parameters
  vertices_ = iConfig.getParameter<edm::InputTag>("vertexCollection");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

//   bool debug_ = 
//     iConfig.getUntrackedParameter<bool>("debug",false);

}



PFEGammaProducerNew::~PFEGammaProducerNew() {}

void 
PFEGammaProducerNew::beginRun(const edm::Run & run, 
                     const edm::EventSetup & es) 
{


  /*
  static map<std::string, PerformanceResult::ResultType> functType;

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
  
  /*
  for(std::vector<std::string>::const_iterator name = fToRead.begin(); name != fToRead.end(); ++name) {    
    
    cout << "Function: " << *name << std::endl;
    PerformanceResult::ResultType fType = functType[*name];
    pfCalibrations->printFormula(fType);
    
    // evaluate it @ 10 GeV
    float energy = 10.;
    
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, energy);
    
    if(pfCalibrations->isInPayload(fType, point)) {
      float value = pfCalibrations->getResult(fType, point);
      cout << "   Energy before:: " << energy << " after: " << value << std::endl;
    } else cout <<  "outside limits!" << std::endl;
    
  }
  */
  
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
    LogDebug("PFEGammaProducerNew")<<"setting regressions from DB "<<std::endl;
    */
  } 


  //pfAlgo_->setPFPhotonRegWeights(ReaderLC_, ReaderGC_, ReaderRes_);
  setPFPhotonRegWeights(ReaderLCEB_,ReaderLCEE_,ReaderGCBarrel_,ReaderGCEndCapHighr9_, ReaderGCEndCapLowr9_, ReaderEcalRes_ );
    
}


void 
PFEGammaProducerNew::produce(edm::Event& iEvent, 
			     const edm::EventSetup& iSetup) {
  
  edm::LogInfo("PFEGammaProducerNew")
    <<"START event: "
    <<iEvent.id().event()
    <<" in run "<<iEvent.id().run()<<std::endl;
  

  // reset output collection
  if(egCandidates_.get() )
    egCandidates_->clear();
  else 
    egCandidates_.reset( new reco::PFCandidateCollection ); 

  if(ebeeClusters_.get() )
    ebeeClusters_->clear();
  else 
    ebeeClusters_.reset( new reco::CaloClusterCollection );  

  //printf("ebeeclusters->size() = %i\n",int(ebeeClusters_->size()));
  
  if(esClusters_.get() )
    esClusters_->clear();
  else 
    esClusters_.reset( new reco::CaloClusterCollection );    
  
  if(sClusters_.get() )
    sClusters_->clear();
  else 
    sClusters_.reset( new reco::SuperClusterCollection );   
  
  if(egExtra_.get() )
    egExtra_->clear();
  else 
    egExtra_.reset( new reco::PFCandidateEGammaExtraCollection );  
  
  // Get The vertices from the event
  // and assign dynamic vertex parameters
  edm::Handle<reco::VertexCollection> vertices;
  bool gotVertices = iEvent.getByLabel(vertices_,vertices);
  if(!gotVertices) {
    std::ostringstream err;
    err<<"Cannot find vertices for this event.Continuing Without them ";
    edm::LogError("PFEGammaProducerNew")<<err.str()<<std::endl;
  }

  //Assign the PFAlgo Parameters
  setPFVertexParameters(useVerticesForNeutral_,vertices.product());

  // get the collection of blocks 

  edm::Handle< reco::PFBlockCollection > blocks;

  edm::LogInfo("PFEGammaProducerNew")<<"getting blocks"<<std::endl;
  bool found = iEvent.getByLabel( inputTagBlocks_, blocks );  

  if(!found ) {

    std::ostringstream err;
    err<<"cannot find blocks: "<<inputTagBlocks_;
    edm::LogError("PFEGammaProducerNew")<<err.str()<<std::endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }


  
  edm::LogInfo("PFEGammaProducerNew")<<"EGPFlow is starting"<<std::endl;

  assert( blocks.isValid() && "edm::Handle to blocks was null!");

  //pfAlgo_->reconstructParticles( blocks );
  
  if(verbose_) {
    std::ostringstream  str;
    //str<<(*pfAlgo_)<<std::endl;
    //    cout << (*pfAlgo_) << std::endl;
    edm::LogInfo("PFEGammaProducerNew") <<str.str()<<std::endl;
  }  

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
   
    edm::LogInfo("PFEGammaProducerNew") 
      << "Found " << elements.size() 
      << " PFBlockElements in block: " << i << std::endl;

    bool singleEcalOrHcal = false;
    if( elements.size() == 1 ){
      switch( elements[0].type() ) {
      case reco::PFBlockElement::ECAL:
        ecalBlockRefs.push_back( blockref );
        singleEcalOrHcal = true;
	break;
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
  
  // loop on blocks that are not single ecal, 
  // and not single hcal and produce unbiased collection of EGamma Candidates

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
    egCandidates_->insert( egCandidates_->end(),
			   pfeg_->getCandidates().begin(), 
			   pfeg_->getCandidates().end()    );
    egExtra_->insert( egExtra_->end(), 
		      pfeg_->getEGExtra().begin(), 
		      pfeg_->getEGExtra().end()    );   

    printf("post algo: egCandidates size = %i\n",int(egCandidates_->size()));
  }
  
//   edm::RefProd<reco::CaloClusterCollection> ebeeClusterProd = iEvent.getRefBeforePut<reco::CaloClusterCollection>("EBEEClusters");
//   edm::RefProd<reco::CaloClusterCollection> esClusterProd = iEvent.getRefBeforePut<reco::CaloClusterCollection>("ESClusters");
  edm::RefProd<reco::SuperClusterCollection> sClusterProd = 
    iEvent.getRefBeforePut<reco::SuperClusterCollection>();
  
  //printf("loop over candidates\n");
  //make CaloClusters for Refined SuperClusters
  // LGRAY -- why do we need to do this... can't we just keep the references
  //          of the original EGcand??
  // This is super bloaty other wise.... we should fix this...
  std::vector<std::vector<int> > ebeeidxs(egCandidates_->size());
  std::vector<std::vector<int> > esidxs(egCandidates_->size());;
  for (unsigned int icand=0; icand<egCandidates_->size(); ++icand) {
    reco::PFCandidate &cand = egCandidates_->at(icand);
    //reco::PFCandidateEGammaExtra &extra = egExtra_->at(icand);

    //loop over blockelements
   // printf("loop over blockelements\n");
    // this auto is a reco::PFCandidate::ElementsInBlocks
    for( const auto& used : cand.elementsInBlocks() ) {    
      const reco::PFBlockElement& element(used.first->elements()[used.second]);
      switch( element.type() ) {
      case reco::PFBlockElement::ECAL:
        ebeeClusters_->push_back( *(element.clusterRef()) );
        ebeeidxs[icand].push_back(ebeeClusters_->size()-1);     
	break;
      case reco::PFBlockElement::PS1:
      case reco::PFBlockElement::PS2:
	esClusters_->push_back( *(element.clusterRef()) );	 
	esidxs[icand].push_back(esClusters_->size()-1);
	break;
      default:
	break;
      }
    }
  }
    
  //put cluster products
  std::auto_ptr< reco::CaloClusterCollection >
    pOutputEBEEClusters( ebeeClusters_ ); 
  edm::OrphanHandle<reco::CaloClusterCollection > ebeeClusterProd=
    iEvent.put(pOutputEBEEClusters,"EBEEClusters");        
    
  std::auto_ptr< reco::CaloClusterCollection >
    pOutputESClusters( esClusters_ ); 
  edm::OrphanHandle<reco::CaloClusterCollection > esClusterProd=
    iEvent.put(pOutputESClusters,"ESClusters");         
    
  //loop over sets of clusters to make superclusters
  for (unsigned int iclus=0; iclus<egCandidates_->size(); ++iclus) {
    reco::PFCandidate &cand = egCandidates_->at(iclus);
    reco::PFCandidateEGammaExtra &extra = egExtra_->at(iclus);    

    const std::vector<int> &ebeeidx = ebeeidxs[iclus];
    const std::vector<int> &esidx = esidxs[iclus];
    reco::CaloClusterPtr seed;
    
    reco::CaloClusterPtrVector ebeeclusters;
    reco::CaloClusterPtrVector esclusters;
    
    double maxenergy = 0.;
    double rawenergy = 0.;
    double energy = 0.;

    double posX = 0.;
    double posY = 0.;
    double posZ = 0.;
    for (unsigned int icaloclus=0; icaloclus<ebeeidx.size(); ++icaloclus) {
      const reco::CaloCluster &cluster = ebeeClusterProd->at(ebeeidx[icaloclus]);
      reco::CaloClusterPtr caloptr(ebeeClusterProd,ebeeidx[icaloclus]);
      ebeeclusters.push_back(caloptr);
            
      rawenergy += cluster.energy();        
      energy += cluster.energy();   
      
      posX += cluster.energy()*cluster.position().x();
      posY += cluster.energy()*cluster.position().y();
      posZ += cluster.energy()*cluster.position().z();
      
      if (cluster.energy()>maxenergy) {
        maxenergy = cluster.energy();
        seed = caloptr;
      }      
    }
    
    for (unsigned int icaloclus=0; icaloclus<esidx.size(); ++icaloclus) {
      const reco::CaloCluster &cluster = esClusterProd->at(esidx[icaloclus]);
      
      reco::CaloClusterPtr caloptr(esClusterProd,esidx[icaloclus]);
      esclusters.push_back(caloptr);
      
      energy += cluster.energy();        
    } 
        
    posX /= rawenergy;
    posY /= rawenergy;
    posZ /= rawenergy;
    
    math::XYZPoint scposition(posX,posY,posZ);
    
    reco::SuperCluster refinedSC(rawenergy,scposition,seed,ebeeclusters,esclusters);
    sClusters_->push_back(refinedSC);
    
    reco::SuperClusterRef scref(sClusterProd,sClusters_->size()-1);
    cand.setSuperClusterRef(scref);
    extra.setSuperClusterRef(scref);
  }
    
// Save the PFEGamma Extra Collection First as to be able to create valid References  
  std::auto_ptr< reco::PFCandidateEGammaExtraCollection >
    pOutputEGammaCandidateExtraCollection( egExtra_ );    
  const edm::OrphanHandle<reco::PFCandidateEGammaExtraCollection > egammaExtraProd=
    iEvent.put(pOutputEGammaCandidateExtraCollection);      
  //pfAlgo_->setEGammaExtraRef(egammaExtraProd);
   
  //final loop over Candidates to set PFCandidateEGammaExtra references
  for (unsigned int icand=0; icand<egCandidates_->size(); ++icand) {
    reco::PFCandidate &cand = egCandidates_->at(icand);
    
    reco::PFCandidateEGammaExtraRef extraref(egammaExtraProd,icand);
    cand.setPFEGammaExtraRef(extraref);    
  }
     
    
  std::auto_ptr< reco::SuperClusterCollection >
    pOutputSClusters( sClusters_ ); 
  //edm::OrphanHandle<reco::SuperClusterCollection > sClusterProd=
    iEvent.put(pOutputSClusters);    
    
  // Save the final PFCandidate collection
  std::auto_ptr< reco::PFCandidateCollection > 
    pOutputCandidateCollection( egCandidates_ ); 
  

  
//   LogDebug("PFEGammaProducerNew")<<"particle flow: putting products in the event"<<std::endl;
//   if ( verbose_ ) std::cout <<"particle flow: putting products in the event. Here the full list"<<std::endl;
//   int nC=0;
//   for( reco::PFCandidateCollection::const_iterator  itCand =  (*pOutputCandidateCollection).begin(); itCand !=  (*pOutputCandidateCollection).end(); itCand++) {
//     nC++;
//       if (verbose_ ) std::cout << nC << ")" << (*itCand).particleId() << std::std::endl;
// 
//   }
// 
//   // Write in the event
   iEvent.put(pOutputCandidateCollection);
 
}

//PFEGammaAlgo: a new method added to set the parameters for electron and photon reconstruction. 
void 
PFEGammaProducerNew::setPFEGParameters(double mvaEleCut,
				       std::string mvaWeightFileEleID,
                           bool usePFElectrons,
                           const std::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
                           const std::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
                           double sumEtEcalIsoForEgammaSC_barrel,
                           double sumEtEcalIsoForEgammaSC_endcap,
                           double coneEcalIsoForEgammaSC,
                           double sumPtTrackIsoForEgammaSC_barrel,
                           double sumPtTrackIsoForEgammaSC_endcap,
                           unsigned int nTrackIsoForEgammaSC,
                           double coneTrackIsoForEgammaSC,
                           bool applyCrackCorrections,
                           bool usePFSCEleCalib,
                           bool useEGElectrons,
                           bool useEGammaSupercluster,
                           bool usePFPhotons,  
                           std::string mvaWeightFileConvID, 
                           double mvaConvCut,
                           bool useReg,
                           std::string X0_Map,
                           double sumPtTrackIsoForPhoton,
                           double sumPtTrackIsoSlopeForPhoton                      
                        ) {
  
  mvaEleCut_ = mvaEleCut;
  usePFElectrons_ = usePFElectrons;
  applyCrackCorrectionsElectrons_ = applyCrackCorrections;  
  usePFSCEleCalib_ = usePFSCEleCalib;
  thePFSCEnergyCalibration_ = thePFSCEnergyCalibration;
  useEGElectrons_ = useEGElectrons;
  useEGammaSupercluster_ = useEGammaSupercluster;
  sumEtEcalIsoForEgammaSC_barrel_ = sumEtEcalIsoForEgammaSC_barrel;
  sumEtEcalIsoForEgammaSC_endcap_ = sumEtEcalIsoForEgammaSC_endcap;
  coneEcalIsoForEgammaSC_ = coneEcalIsoForEgammaSC;
  sumPtTrackIsoForEgammaSC_barrel_ = sumPtTrackIsoForEgammaSC_barrel;
  sumPtTrackIsoForEgammaSC_endcap_ = sumPtTrackIsoForEgammaSC_endcap;
  coneTrackIsoForEgammaSC_ = coneTrackIsoForEgammaSC;
  nTrackIsoForEgammaSC_ = nTrackIsoForEgammaSC;


  if(!usePFElectrons_) return;
  mvaWeightFileEleID_ = mvaWeightFileEleID;
  FILE * fileEleID = fopen(mvaWeightFileEleID_.c_str(), "r");
  if (fileEleID) {
    fclose(fileEleID);
  }
  else {
    std::string err = "PFAlgo: cannot open weight file '";
    err += mvaWeightFileEleID;
    err += "'";
    throw std::invalid_argument( err );
  }
  
  usePFPhotons_ = usePFPhotons;

  //for MVA pass PV if there is one in the collection otherwise pass a dummy    
  reco::Vertex dummy;  
  if(useVertices_)  
    {  
      dummy = primaryVertex_;  
    }  
  else { // create a dummy PV  
    reco::Vertex::Error e;  
    e(0, 0) = 0.0015 * 0.0015;  
    e(1, 1) = 0.0015 * 0.0015;  
    e(2, 2) = 15. * 15.;  
    reco::Vertex::Point p(0, 0, 0);  
    dummy = reco::Vertex(p, e, 0, 0, 0);  
  }  
  // pv=&dummy;  
  //if(! usePFPhotons_) return;  
  FILE * filePhotonConvID = fopen(mvaWeightFileConvID.c_str(), "r");  
  if (filePhotonConvID) {  
    fclose(filePhotonConvID);  
  }  
  else {  
    std::string err = "PFAlgo: cannot open weight file '";  
    err += mvaWeightFileConvID;  
    err += "'";  
    throw std::invalid_argument( err );  
  }  
  const reco::Vertex* pv=&dummy;  
  pfeg_.reset(new PFEGammaAlgoNew(mvaEleCut_,mvaWeightFileEleID_,
                             thePFSCEnergyCalibration_,
                             thePFEnergyCalibration,
                             applyCrackCorrectionsElectrons_,
                             usePFSCEleCalib_,
                             useEGElectrons_,
                             useEGammaSupercluster_,
                             sumEtEcalIsoForEgammaSC_barrel_,
                             sumEtEcalIsoForEgammaSC_endcap_,
                             coneEcalIsoForEgammaSC_,
                             sumPtTrackIsoForEgammaSC_barrel_,
                             sumPtTrackIsoForEgammaSC_endcap_,
                             nTrackIsoForEgammaSC_,
                             coneTrackIsoForEgammaSC_,
                            mvaWeightFileConvID, 
                            mvaConvCut, 
                            useReg,
                            X0_Map,  
                            *pv,
                            sumPtTrackIsoForPhoton,
                            sumPtTrackIsoSlopeForPhoton
                            ));
  return;  
  
//   pfele_= new PFElectronAlgo(mvaEleCut_,mvaWeightFileEleID_,
//                           thePFSCEnergyCalibration_,
//                           thePFEnergyCalibration,
//                           applyCrackCorrectionsElectrons_,
//                           usePFSCEleCalib_,
//                           useEGElectrons_,
//                           useEGammaSupercluster_,
//                           sumEtEcalIsoForEgammaSC_barrel_,
//                           sumEtEcalIsoForEgammaSC_endcap_,
//                           coneEcalIsoForEgammaSC_,
//                           sumPtTrackIsoForEgammaSC_barrel_,
//                           sumPtTrackIsoForEgammaSC_endcap_,
//                           nTrackIsoForEgammaSC_,
//                           coneTrackIsoForEgammaSC_);
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
void PFEGammaProducerNew::setPFPhotonRegWeights(
                                   const GBRForest *LCorrForestEB,
                                   const GBRForest *LCorrForestEE,
                                   const GBRForest *GCorrForestBarrel,
                                   const GBRForest *GCorrForestEndcapHr9,
                                   const GBRForest *GCorrForestEndcapLr9,                                          const GBRForest *PFEcalResolution
                                   ){
  
  pfeg_->setGBRForest(LCorrForestEB,LCorrForestEE,
                       GCorrForestBarrel, GCorrForestEndcapHr9, 
                       GCorrForestEndcapLr9, PFEcalResolution);
}

void
PFEGammaProducerNew::setPFVertexParameters(bool useVertex,
                              const reco::VertexCollection*  primaryVertices) {
  useVertices_ = useVertex;

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
