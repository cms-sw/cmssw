#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoParticleFlow/PFProducer/interface/PFAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"  //PFMuons
#include "RecoParticleFlow/PFProducer/interface/PFElectronAlgo.h"  
#include "RecoParticleFlow/PFProducer/interface/PFPhotonAlgo.h"    
#include "RecoParticleFlow/PFProducer/interface/PFElectronExtraEqual.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibrationHF.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"

// #include "FWCore/Framework/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
// #include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Math/interface/LorentzVector.h"


#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"


#include "Math/PxPyPzM4D.h"
#include "Math/LorentzVector.h"
#include "Math/DisplacementVector3D.h"
#include "Math/SMatrix.h"
#include "TDecompChol.h"

#include "boost/graph/adjacency_matrix.hpp" 
#include "boost/graph/graph_utility.hpp" 



using namespace std;
using namespace reco;
using namespace boost;

typedef std::list< reco::PFBlockRef >::iterator IBR;



PFAlgo::PFAlgo()
  : pfCandidates_( new PFCandidateCollection),
    nSigmaECAL_(0), 
    nSigmaHCAL_(1),
    algo_(1),
    debug_(false),
    pfele_(0),
    pfpho_(0),
    pfegamma_(0),
    useVertices_(false)
{}

PFAlgo::~PFAlgo() {
  if (usePFElectrons_) delete pfele_;
  if (usePFPhotons_)     delete pfpho_;
  if (useEGammaFilters_)  delete pfegamma_;
}


void
PFAlgo::setParameters(double nSigmaECAL,
                      double nSigmaHCAL, 
                      const boost::shared_ptr<PFEnergyCalibration>& calibration,
		      const boost::shared_ptr<PFEnergyCalibrationHF>&  thepfEnergyCalibrationHF) {

  nSigmaECAL_ = nSigmaECAL;
  nSigmaHCAL_ = nSigmaHCAL;

  calibration_ = calibration;
  thepfEnergyCalibrationHF_ = thepfEnergyCalibrationHF;

}


PFMuonAlgo* PFAlgo::getPFMuonAlgo() {
  return pfmu_;
}

//PFElectrons: a new method added to set the parameters for electron reconstruction. 
void 
PFAlgo::setPFEleParameters(double mvaEleCut,
			   string mvaWeightFileEleID,
			   bool usePFElectrons,
			   const boost::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
			   const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
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
			   bool useEGammaSupercluster) {
  
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
    string err = "PFAlgo: cannot open weight file '";
    err += mvaWeightFileEleID;
    err += "'";
    throw invalid_argument( err );
  }
  pfele_= new PFElectronAlgo(mvaEleCut_,mvaWeightFileEleID_,
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
			     coneTrackIsoForEgammaSC_);
}

void 
PFAlgo::setPFPhotonParameters(bool usePFPhotons,  
			      std::string mvaWeightFileConvID, 
			      double mvaConvCut,
			      bool useReg,
			      std::string X0_Map,
			      const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
			      double sumPtTrackIsoForPhoton,
			      double sumPtTrackIsoSlopeForPhoton)
 {

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
  if(! usePFPhotons_) return;  
  FILE * filePhotonConvID = fopen(mvaWeightFileConvID.c_str(), "r");  
  if (filePhotonConvID) {  
    fclose(filePhotonConvID);  
  }  
  else {  
    string err = "PFAlgo: cannot open weight file '";  
    err += mvaWeightFileConvID;  
    err += "'";  
    throw invalid_argument( err );  
  }  
  const reco::Vertex* pv=&dummy;  
  pfpho_ = new PFPhotonAlgo(mvaWeightFileConvID, 
			    mvaConvCut, 
			    useReg,
			    X0_Map,  
			    *pv,
			    thePFEnergyCalibration,
                            sumPtTrackIsoForPhoton,
                            sumPtTrackIsoSlopeForPhoton
			    );
  return;
}

void PFAlgo::setEGammaParameters(bool use_EGammaFilters,
				 std::string ele_iso_path_mvaWeightFile,
				 double ele_iso_pt,
				 double ele_iso_mva_barrel,
				 double ele_iso_mva_endcap,
				 double ele_iso_combIso_barrel,
				 double ele_iso_combIso_endcap,
				 double ele_noniso_mva,
				 unsigned int ele_missinghits,
				 bool useProtectionsForJetMET,
				 const edm::ParameterSet& ele_protectionsForJetMET,
				 double ph_MinEt,
				 double ph_combIso,
				 double ph_HoE,
				 double ph_sietaieta_eb,
				 double ph_sietaieta_ee,
				 const edm::ParameterSet& ph_protectionsForJetMET
				 )
{
  
  useEGammaFilters_ = use_EGammaFilters;

  if(!useEGammaFilters_ ) return;
  FILE * fileEGamma_ele_iso_ID = fopen(ele_iso_path_mvaWeightFile.c_str(), "r");  
  if (fileEGamma_ele_iso_ID) {  
    fclose(fileEGamma_ele_iso_ID);  
  }  
  else {  
    string err = "PFAlgo: cannot open weight file '";  
    err += ele_iso_path_mvaWeightFile;  
    err += "'";  
    throw invalid_argument( err );  
  }  

  //  ele_iso_mvaID_ = new ElectronMVAEstimator(ele_iso_path_mvaWeightFile_);
  useProtectionsForJetMET_ = useProtectionsForJetMET;

  pfegamma_ =  new PFEGammaFilters(ph_MinEt,
				   ph_combIso,
				   ph_HoE,
				   ph_sietaieta_eb,
				   ph_sietaieta_ee,
				   ph_protectionsForJetMET,
				   ele_iso_pt,
				   ele_iso_mva_barrel,
				   ele_iso_mva_endcap,
				   ele_iso_combIso_barrel,
				   ele_iso_combIso_endcap,
				   ele_noniso_mva,
				   ele_missinghits,
				   ele_iso_path_mvaWeightFile,
				   ele_protectionsForJetMET);

  return;
}
void  PFAlgo::setEGammaCollections(const edm::View<reco::PFCandidate> & pfEgammaCandidates,
				   const edm::ValueMap<reco::GsfElectronRef> & valueMapGedElectrons,
				   const edm::ValueMap<reco::PhotonRef> & valueMapGedPhotons){
  if(useEGammaFilters_) {
    pfEgammaCandidates_ = & pfEgammaCandidates;
    valueMapGedElectrons_ = & valueMapGedElectrons;
    valueMapGedPhotons_ = & valueMapGedPhotons;
  }
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
void PFAlgo::setPFPhotonRegWeights(
				   const GBRForest *LCorrForestEB,
				   const GBRForest *LCorrForestEE,
				   const GBRForest *GCorrForestBarrel,
				   const GBRForest *GCorrForestEndcapHr9,
				   const GBRForest *GCorrForestEndcapLr9,			     		   const GBRForest *PFEcalResolution
				   ){
  
  pfpho_->setGBRForest(LCorrForestEB,LCorrForestEE,
		       GCorrForestBarrel, GCorrForestEndcapHr9, 
		       GCorrForestEndcapLr9, PFEcalResolution);
}
void 
PFAlgo::setPFMuonAndFakeParameters(const edm::ParameterSet& pset) 
				   //				   std::vector<double> muonHCAL,
				   //				   std::vector<double> muonECAL,
				   //				   std::vector<double> muonHO,
				   //				   double nSigmaTRACK,
				   //				   double ptError,
				   //				   std::vector<double> factors45)
{




  pfmu_ = new PFMuonAlgo();
  pfmu_->setParameters(pset);

  // Muon parameters
  muonHCAL_= pset.getParameter<std::vector<double> >("muon_HCAL");  
  muonECAL_= pset.getParameter<std::vector<double> >("muon_ECAL");  
  muonHO_= pset.getParameter<std::vector<double> >("muon_HO");  
  assert ( muonHCAL_.size() == 2 && muonECAL_.size() == 2 && muonHO_.size() == 2);
  nSigmaTRACK_= pset.getParameter<double>("nsigma_TRACK");  
  ptError_= pset.getParameter<double>("pt_Error");  
  factors45_ = pset.getParameter<std::vector<double> >("factors_45");  
  assert ( factors45_.size() == 2 );

}


void 
PFAlgo::setMuonHandle(const edm::Handle<reco::MuonCollection>& muons) {
  muonHandle_ = muons;
}

  
void 
PFAlgo::setPostHFCleaningParameters(bool postHFCleaning,
				    double minHFCleaningPt,
				    double minSignificance,
				    double maxSignificance,
				    double minSignificanceReduction,
				    double maxDeltaPhiPt,
				    double minDeltaMet) { 
  postHFCleaning_ = postHFCleaning;
  minHFCleaningPt_ = minHFCleaningPt;
  minSignificance_ = minSignificance;
  maxSignificance_ = maxSignificance;
  minSignificanceReduction_= minSignificanceReduction;
  maxDeltaPhiPt_ = maxDeltaPhiPt;
  minDeltaMet_ = minDeltaMet;
}

void 
PFAlgo::setDisplacedVerticesParameters(bool rejectTracks_Bad,
				       bool rejectTracks_Step45,
				       bool usePFNuclearInteractions,
				       bool usePFConversions,
				       bool usePFDecays, 
				       double dptRel_DispVtx){

  rejectTracks_Bad_ = rejectTracks_Bad;
  rejectTracks_Step45_ = rejectTracks_Step45;
  usePFNuclearInteractions_ = usePFNuclearInteractions;
  usePFConversions_ = usePFConversions;
  usePFDecays_ = usePFDecays;
  dptRel_DispVtx_ = dptRel_DispVtx;

}


void
PFAlgo::setPFVertexParameters(bool useVertex,
			      const reco::VertexCollection*  primaryVertices) {
  useVertices_ = useVertex;

  //Set the vertices for muon cleaning
  pfmu_->setInputsForCleaning(primaryVertices);


  //Now find the primary vertex!
  bool primaryVertexFound = false;
  nVtx_ = primaryVertices->size();
  if(usePFPhotons_){
    pfpho_->setnPU(nVtx_);
  }
  for (unsigned short i=0 ;i<primaryVertices->size();++i)
    {
      if(primaryVertices->at(i).isValid()&&(!primaryVertices->at(i).isFake()))
	{
	  primaryVertex_ = primaryVertices->at(i);
	  primaryVertexFound = true;
          break;
	}
    }
  //Use vertices if the user wants to but only if it exists a good vertex 
  useVertices_ = useVertex && primaryVertexFound; 
  if(usePFPhotons_) {
    if (useVertices_ ){
      pfpho_->setPhotonPrimaryVtx(primaryVertex_ );
    }
    else{
      reco::Vertex::Error e;  
      e(0, 0) = 0.0015 * 0.0015;  
      e(1, 1) = 0.0015 * 0.0015;  
      e(2, 2) = 15. * 15.;  
      reco::Vertex::Point p(0, 0, 0);  
      reco::Vertex dummy = reco::Vertex(p, e, 0, 0, 0); 
      //      std::cout << " PFPho " << pfpho_ << std::endl;
      pfpho_->setPhotonPrimaryVtx(dummy);
    }
  }
}


void PFAlgo::reconstructParticles( const reco::PFBlockHandle& blockHandle ) {

  blockHandle_ = blockHandle;
  reconstructParticles( *blockHandle_ ); 
}



void PFAlgo::reconstructParticles( const reco::PFBlockCollection& blocks ) {

  // reset output collection
  if(pfCandidates_.get() )
    pfCandidates_->clear();
  else 
    pfCandidates_.reset( new reco::PFCandidateCollection );

  if(pfElectronCandidates_.get() )
    pfElectronCandidates_->clear();
  else
    pfElectronCandidates_.reset( new reco::PFCandidateCollection);

  // Clearing pfPhotonCandidates
  if( pfPhotonCandidates_.get() )
    pfPhotonCandidates_->clear();
  else
    pfPhotonCandidates_.reset( new reco::PFCandidateCollection);

  if(pfCleanedCandidates_.get() ) 
    pfCleanedCandidates_->clear();
  else
    pfCleanedCandidates_.reset( new reco::PFCandidateCollection );
  
  // not a auto_ptr; shout not be deleted after transfer
  pfElectronExtra_.clear();
  pfPhotonExtra_.clear();
  
  if( debug_ ) {
    cout<<"*********************************************************"<<endl;
    cout<<"*****           Particle flow algorithm             *****"<<endl;
    cout<<"*********************************************************"<<endl;
  }

  // sort elements in three lists:
  std::list< reco::PFBlockRef > hcalBlockRefs;
  std::list< reco::PFBlockRef > ecalBlockRefs;
  std::list< reco::PFBlockRef > hoBlockRefs;
  std::list< reco::PFBlockRef > otherBlockRefs;
  
  for( unsigned i=0; i<blocks.size(); ++i ) {
    // reco::PFBlockRef blockref( blockh,i );
    reco::PFBlockRef blockref = createBlockRef( blocks, i);
    
    const reco::PFBlock& block = *blockref;
    const edm::OwnVector< reco::PFBlockElement >& 
      elements = block.elements();
   
    bool singleEcalOrHcal = false;
    if( elements.size() == 1 ){
      if( elements[0].type() == reco::PFBlockElement::ECAL ){
	ecalBlockRefs.push_back( blockref );
	singleEcalOrHcal = true;
      }
      if( elements[0].type() == reco::PFBlockElement::HCAL ){
	hcalBlockRefs.push_back( blockref );
	singleEcalOrHcal = true;
      }
      if( elements[0].type() == reco::PFBlockElement::HO ){
	// Single HO elements are likely to be noise. Not considered for now.
	hoBlockRefs.push_back( blockref );
	singleEcalOrHcal = true;
      }
    }
    
    if(!singleEcalOrHcal) {
      otherBlockRefs.push_back( blockref );
    }
  }//loop blocks
  
  if( debug_ ){
    cout<<"# Ecal blocks: "<<ecalBlockRefs.size()
        <<", # Hcal blocks: "<<hcalBlockRefs.size()
        <<", # HO blocks: "<<hoBlockRefs.size()
        <<", # Other blocks: "<<otherBlockRefs.size()<<endl;
  }


  // loop on blocks that are not single ecal, 
  // and not single hcal.

  unsigned nblcks = 0;
  for( IBR io = otherBlockRefs.begin(); io!=otherBlockRefs.end(); ++io) {
    if ( debug_ ) std::cout << "Block number " << nblcks++ << std::endl; 
    processBlock( *io, hcalBlockRefs, ecalBlockRefs );
  }

  std::list< reco::PFBlockRef > empty;

  unsigned hblcks = 0;
  // process remaining single hcal blocks
  for( IBR ih = hcalBlockRefs.begin(); ih!=hcalBlockRefs.end(); ++ih) {
    if ( debug_ ) std::cout << "HCAL block number " << hblcks++ << std::endl;
    processBlock( *ih, empty, empty );
  }

  unsigned eblcks = 0;
  // process remaining single ecal blocks
  for( IBR ie = ecalBlockRefs.begin(); ie!=ecalBlockRefs.end(); ++ie) {
    if ( debug_ ) std::cout << "ECAL block number " << eblcks++ << std::endl;
    processBlock( *ie, empty, empty );
  }

  // Post HF Cleaning
  postCleaning();

  //Muon post cleaning
  pfmu_->postClean(pfCandidates_.get());

  //Add Missing muons
   if( muonHandle_.isValid())
     pfmu_->addMissingMuons(muonHandle_,pfCandidates_.get());
}


void PFAlgo::processBlock( const reco::PFBlockRef& blockref,
                           std::list<reco::PFBlockRef>& hcalBlockRefs, 
                           std::list<reco::PFBlockRef>& ecalBlockRefs ) { 
  
  // debug_ = false;
  assert(!blockref.isNull() );
  const reco::PFBlock& block = *blockref;

  typedef std::multimap<double, unsigned>::iterator IE;
  typedef std::multimap<double, std::pair<unsigned,::math::XYZVector> >::iterator IS;
  typedef std::multimap<double, std::pair<unsigned,bool> >::iterator IT;
  typedef std::multimap< unsigned, std::pair<double, unsigned> >::iterator II;

  if(debug_) {
    cout<<"#########################################################"<<endl;
    cout<<"#####           Process Block:                      #####"<<endl;
    cout<<"#########################################################"<<endl;
    cout<<block<<endl;
  }


  const edm::OwnVector< reco::PFBlockElement >& elements = block.elements();
  // make a copy of the link data, which will be edited.
  PFBlock::LinkData linkData =  block.linkData();
  
  // keep track of the elements which are still active.
  vector<bool>   active( elements.size(), true );


  // //PFElectrons:
  // usePFElectrons_ external configurable parameter to set the usage of pf electron
  std::vector<reco::PFCandidate> tempElectronCandidates;
  tempElectronCandidates.clear();
  if (usePFElectrons_) {
    if (pfele_->isElectronValidCandidate(blockref,active, primaryVertex_ )){
      // if there is at least a valid candidate is get the vector of pfcandidates
      const std::vector<reco::PFCandidate> PFElectCandidates_(pfele_->getElectronCandidates());
      for ( std::vector<reco::PFCandidate>::const_iterator ec=PFElectCandidates_.begin();   ec != PFElectCandidates_.end(); ++ec )tempElectronCandidates.push_back(*ec);
      
      // (***) We're filling the ElectronCandidates into the PFCandiate collection
      // ..... Once we let PFPhotonAlgo over-write electron-decision, we need to move this to
      // ..... after the PhotonAlgo has run (Fabian)
    }    
    // The vector active is automatically changed (it is passed by ref) in PFElectronAlgo
    // for all the electron candidate      
    pfElectronCandidates_->insert(pfElectronCandidates_->end(), 
				  pfele_->getAllElectronCandidates().begin(), 
				  pfele_->getAllElectronCandidates().end()); 

    pfElectronExtra_.insert(pfElectronExtra_.end(), 
			    pfele_->getElectronExtra().begin(),
			    pfele_->getElectronExtra().end());
    
  }
  if( /* --- */ usePFPhotons_ /* --- */ ) {    
    
    if(debug_)
      cout<<endl<<"--------------- entering PFPhotonAlgo ----------------"<<endl;
    vector<PFCandidatePhotonExtra> pfPhotonExtraCand;
    if ( pfpho_->isPhotonValidCandidate(blockref,               // passing the reference to the PFBlock
					active,                 // std::vector<bool> containing information about acitivity
					pfPhotonCandidates_,    // pointer to candidate vector, to be filled by the routine
					pfPhotonExtraCand,      // candidate extra vector, to be filled by the routine   
					tempElectronCandidates
					//pfElectronCandidates_   // pointer to some auziliary UNTOUCHED FOR NOW
					) ) {
      if(debug_)
	std::cout<< " In this PFBlock we found "<<pfPhotonCandidates_->size()<<" Photon Candidates."<<std::endl;
      
      // CAUTION: In case we want to allow the PhotonAlgo to 'over-write' what the ElectronAlgo did above
      // ........ we should NOT fill the PFCandidate-vector with the electrons above (***)
      
      // Here we need to add all the photon cands to the pfCandidate list
      unsigned int extracand =0;
      PFCandidateCollection::const_iterator cand = pfPhotonCandidates_->begin();      
      for( ; cand != pfPhotonCandidates_->end(); ++cand, ++extracand) {
	pfCandidates_->push_back(*cand);
	pfPhotonExtra_.push_back(pfPhotonExtraCand[extracand]);
      }
      
    } // end of 'if' in case photons are found    
    pfPhotonExtraCand.clear();
    pfPhotonCandidates_->clear();
  } // end of Photon algo
  
  if (usePFElectrons_) {
    for ( std::vector<reco::PFCandidate>::const_iterator ec=tempElectronCandidates.begin();   ec != tempElectronCandidates.end(); ++ec ){
      pfCandidates_->push_back(*ec);  
    } 
    tempElectronCandidates.clear();
  }


  // New EGamma Reconstruction 10/10/2013
  if(useEGammaFilters_) {

    // const edm::ValueMap<reco::GsfElectronRef> & myGedElectronValMap(*valueMapGedElectrons_);
    bool egmLocalDebug = false;
    bool egmLocalBlockDebug = false;

    unsigned int negmcandidates = pfEgammaCandidates_->size();
    for(unsigned int ieg=0 ; ieg <  negmcandidates; ++ieg) {
      //      const reco::PFCandidate & egmcand((*pfEgammaCandidates_)[ieg]);
      reco::PFCandidateRef pfEgmRef = pfEgammaCandidates_->refAt(ieg).castTo<reco::PFCandidateRef>(); 

      const PFCandidate::ElementsInBlocks& theElements = (*pfEgmRef).elementsInBlocks();
      PFCandidate::ElementsInBlocks::const_iterator iegfirst = theElements.begin(); 
      bool sameBlock = false;
      bool isGoodElectron = false;
      bool isGoodPhoton = false;
      bool isPrimaryElectron = false;
      if(iegfirst->first == blockref) 
	sameBlock = true;
      if(sameBlock) {

	if(egmLocalDebug)
	  cout << " I am in looping on EGamma Candidates: pt " << (*pfEgmRef).pt() 
	       << " eta,phi " << (*pfEgmRef).eta() << ", " << (*pfEgmRef).phi() 
	       << " charge " << (*pfEgmRef).charge() << endl;

	if((*pfEgmRef).gsfTrackRef().isNonnull()) {

	  reco::GsfElectronRef gedEleRef = (*valueMapGedElectrons_)[pfEgmRef];
	  if(gedEleRef.isNonnull()) {
	    isGoodElectron = pfegamma_->passElectronSelection(*gedEleRef,*pfEgmRef,nVtx_);
	    isPrimaryElectron = pfegamma_->isElectron(*gedEleRef);
	    if(egmLocalDebug){
	      if(isGoodElectron) 
		cout << "** Good Electron, pt " << gedEleRef->pt()
		     << " eta, phi " << gedEleRef->eta() << ", " << gedEleRef->phi() 
		     << " charge " << gedEleRef->charge() 
		     << " isPrimary " << isPrimaryElectron  << endl;
	    }

	   
	  }
	}
	if((*pfEgmRef).superClusterRef().isNonnull()) {

	  reco::PhotonRef gedPhoRef = (*valueMapGedPhotons_)[pfEgmRef];   
	  if(gedPhoRef.isNonnull()) {
	    isGoodPhoton =  pfegamma_->passPhotonSelection(*gedPhoRef);
	    if(egmLocalDebug) {
	      if(isGoodPhoton) 
		cout << "** Good Photon, pt " << gedPhoRef->pt()
		     << " eta, phi " << gedPhoRef->eta() << ", " << gedPhoRef->phi() << endl;
	    }
	  }
	}
      } // end same block
      
      if(isGoodElectron && isGoodPhoton) {
	if(isPrimaryElectron)
	  isGoodPhoton = false;
	else
	  isGoodElectron = false;
      }

      // isElectron
      if(isGoodElectron) {
	reco::GsfElectronRef gedEleRef = (*valueMapGedElectrons_)[pfEgmRef];
	reco::PFCandidate myPFElectron = *pfEgmRef;
	// run protections
	bool lockTracks = false;
	bool isSafe = true;
	if( useProtectionsForJetMET_) {
	  lockTracks = true;
	  isSafe = pfegamma_->isElectronSafeForJetMET(*gedEleRef,myPFElectron,primaryVertex_,lockTracks);
	}

	if(isSafe) {
	  reco::PFCandidate::ParticleType particleType = reco::PFCandidate::e;
	  myPFElectron.setParticleType(particleType);
	  myPFElectron.setCharge(gedEleRef->charge());
	  myPFElectron.setP4(gedEleRef->p4());
	  myPFElectron.set_mva_e_pi(gedEleRef->mva_e_pi());
          myPFElectron.set_mva_Isolated(gedEleRef->mva_Isolated());

	  if(egmLocalDebug) {
	    cout << " PFAlgo: found an electron with NEW EGamma code " << endl;
	    cout << " myPFElectron: pt " << myPFElectron.pt() 
		 << " eta,phi " << myPFElectron.eta() << ", " <<myPFElectron.phi()
		 << " mva " << myPFElectron.mva_e_pi() 
		 << " charge " << myPFElectron.charge() << endl;
	  }
	  
	  
	  // Lock all the elements
	  if(egmLocalBlockDebug)
	    cout << " THE BLOCK " << *blockref << endl;
	  for (PFCandidate::ElementsInBlocks::const_iterator ieb = theElements.begin(); 
	       ieb<theElements.end(); ++ieb) {
	    active[ieb->second] = false;
	    if(egmLocalBlockDebug)
	      cout << " Elements used " <<  ieb->second << endl;
	  }
	  
	  // The electron is considered safe for JetMET and the additional tracks pointing to it are locked
	  if(lockTracks) {
	    const PFCandidate::ElementsInBlocks& extraTracks = myPFElectron.egammaExtraRef()->extraNonConvTracks();
	    for (PFCandidate::ElementsInBlocks::const_iterator itrk = extraTracks.begin(); 
		 itrk<extraTracks.end(); ++itrk) {
	      active[itrk->second] = false;
	    }
	  }

	  pfCandidates_->push_back(myPFElectron);

	}
	else {
	  if(egmLocalDebug)
	    cout << "PFAlgo: Electron DISCARDED, NOT SAFE FOR JETMET " << endl;
	} 
      }
      if(isGoodPhoton) {
	reco::PhotonRef gedPhoRef = (*valueMapGedPhotons_)[pfEgmRef];   
	reco::PFCandidate myPFPhoton = *pfEgmRef;
	bool isSafe = true;
	if( useProtectionsForJetMET_) {
	  isSafe = pfegamma_->isPhotonSafeForJetMET(*gedPhoRef,myPFPhoton);
	}



	if(isSafe) {
	  reco::PFCandidate::ParticleType particleType = reco::PFCandidate::gamma;
	  myPFPhoton.setParticleType(particleType);
	  myPFPhoton.setCharge(0);
	  myPFPhoton.set_mva_nothing_gamma(1.);
	  ::math::XYZPoint v(primaryVertex_.x(), primaryVertex_.y(), primaryVertex_.z());
	  myPFPhoton.setVertex( v  );  
	  myPFPhoton.setP4(gedPhoRef->p4());
	  if(egmLocalDebug) {
	    cout << " PFAlgo: found a photon with NEW EGamma code " << endl;
	    cout << " myPFPhoton: pt " << myPFPhoton.pt() 
		 << " eta,phi " << myPFPhoton.eta() << ", " <<myPFPhoton.phi() 
		 << " charge " << myPFPhoton.charge() << endl;
	  }
	  
	  // Lock all the elements
	  if(egmLocalBlockDebug)
	    cout << " THE BLOCK " << *blockref << endl;
	  for (PFCandidate::ElementsInBlocks::const_iterator ieb = theElements.begin(); 
	       ieb<theElements.end(); ++ieb) {
	    active[ieb->second] = false;
	    if(egmLocalBlockDebug)
	      cout << " Elements used " <<  ieb->second << endl;
	  }
	  pfCandidates_->push_back(myPFPhoton);

	} // end isSafe
      } // end isGoodPhoton
    } // end loop on EGM candidates
  } // end if use EGammaFilters



  //Lock extra conversion tracks not used by Photon Algo
  if (usePFConversions_  )
    {
      for(unsigned iEle=0; iEle<elements.size(); iEle++) {
	PFBlockElement::Type type = elements[iEle].type();
	if(type==PFBlockElement::TRACK)
	  {
	    if(elements[iEle].trackRef()->algo() == 12) // should not be reco::TrackBase::conversionStep ?
	      active[iEle]=false;	
	    if(elements[iEle].trackRef()->quality(reco::TrackBase::highPurity))continue;
	    const reco::PFBlockElementTrack * trackRef = dynamic_cast<const reco::PFBlockElementTrack*>((&elements[iEle]));
	    if(!(trackRef->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)))continue;
	    if(elements[iEle].convRefs().size())active[iEle]=false;
	  }
      }
    }




  
  if(debug_) 
    cout<<endl<<"--------------- loop 1 ------------------"<<endl;

  //COLINFEB16
  // In loop 1, we loop on all elements. 

  // The primary goal is to deal with tracks that are:
  // - not associated to an HCAL cluster
  // - not identified as an electron. 
  // Those tracks should be predominantly relatively low energy charged 
  // hadrons which are not detected in the ECAL. 

  // The secondary goal is to prepare for the next loops
  // - The ecal and hcal elements are sorted in separate vectors
  // which will be used as a base for the corresponding loops.
  // - For tracks which are connected to more than one HCAL cluster, 
  // the links between the track and the cluster are cut for all clusters 
  // but the closest one. 
  // - HF only blocks ( HFEM, HFHAD, HFEM+HFAD) are identified 

  // obsolete comments? 
  // loop1: 
  // - sort ecal and hcal elements in separate vectors
  // - for tracks:
  //       - lock closest ecal cluster 
  //       - cut link to farthest hcal cluster, if more than 1.

  // vectors to store indices to ho, hcal and ecal elements 
  vector<unsigned> hcalIs;
  vector<unsigned> hoIs;
  vector<unsigned> ecalIs;
  vector<unsigned> trackIs;
  vector<unsigned> ps1Is;
  vector<unsigned> ps2Is;

  vector<unsigned> hfEmIs;
  vector<unsigned> hfHadIs;
  

  for(unsigned iEle=0; iEle<elements.size(); iEle++) {
    PFBlockElement::Type type = elements[iEle].type();

    if(debug_ && type != PFBlockElement::BREM ) cout<<endl<<elements[iEle];   

    switch( type ) {
    case PFBlockElement::TRACK:
      if ( active[iEle] ) { 
	trackIs.push_back( iEle );
	if(debug_) cout<<"TRACK, stored index, continue"<<endl;
      }
      break;
    case PFBlockElement::ECAL: 
      if ( active[iEle]  ) { 
	ecalIs.push_back( iEle );
	if(debug_) cout<<"ECAL, stored index, continue"<<endl;
      }
      continue;
    case PFBlockElement::HCAL:
      if ( active[iEle] ) { 
	hcalIs.push_back( iEle );
	if(debug_) cout<<"HCAL, stored index, continue"<<endl;
      }
      continue;
    case PFBlockElement::HO:
      if (useHO_) {
	if ( active[iEle] ) { 
	  hoIs.push_back( iEle );
	  if(debug_) cout<<"HO, stored index, continue"<<endl;
	}
      }
      continue;
    case PFBlockElement::HFEM:
      if ( active[iEle] ) { 
	hfEmIs.push_back( iEle );
	if(debug_) cout<<"HFEM, stored index, continue"<<endl;
      }
      continue;
    case PFBlockElement::HFHAD:
      if ( active[iEle] ) { 
	hfHadIs.push_back( iEle );
	if(debug_) cout<<"HFHAD, stored index, continue"<<endl;
      }
      continue;
    default:
      continue;
    }

    // we're now dealing with a track
    unsigned iTrack = iEle;
    reco::MuonRef muonRef = elements[iTrack].muonRef();    

    //Check if the track is a primary track of a secondary interaction
    //If that is the case reconstruct a charged hadron noly using that
    //track
    if (active[iTrack] && isFromSecInt(elements[iEle], "primary")){
      bool isPrimaryTrack = elements[iEle].displacedVertexRef(PFBlockElement::T_TO_DISP)->displacedVertexRef()->isTherePrimaryTracks();
      if (isPrimaryTrack) {
	if (debug_) cout << "Primary Track reconstructed alone" << endl;

	unsigned tmpi = reconstructTrack(elements[iEle]);
	(*pfCandidates_)[tmpi].addElementInBlock( blockref, iEle );
	active[iTrack] = false;
      }
    }


    if(debug_) { 
      if ( !active[iTrack] ) 
	cout << "Already used by electrons, muons, conversions" << endl;
    }
    
    // Track already used as electron, muon, conversion?
    // Added a check on the activated element
    if ( ! active[iTrack] ) continue;
  
    reco::TrackRef trackRef = elements[iTrack].trackRef();
    assert( !trackRef.isNull() ); 

    if (debug_ ) {
      cout <<"PFAlgo:processBlock "<<" "<<trackIs.size()<<" "<<ecalIs.size()<<" "<<hcalIs.size()<<" "<<hoIs.size()<<endl;
    }

    // look for associated elements of all types
    //COLINFEB16 
    // all types of links are considered. 
    // the elements are sorted by increasing distance
    std::multimap<double, unsigned> ecalElems;
    block.associatedElements( iTrack,  linkData,
                              ecalElems ,
                              reco::PFBlockElement::ECAL,
			      reco::PFBlock::LINKTEST_ALL );
    
    std::multimap<double, unsigned> hcalElems;
    block.associatedElements( iTrack,  linkData,
                              hcalElems,
                              reco::PFBlockElement::HCAL,
                              reco::PFBlock::LINKTEST_ALL );

    // When a track has no HCAL cluster linked, but another track is linked to the same
    // ECAL cluster and an HCAL cluster, link the track to the HCAL cluster for 
    // later analysis
    if ( hcalElems.empty() && !ecalElems.empty() ) {
      // debug_ = true;
      unsigned ntt = 1;
      unsigned index = ecalElems.begin()->second;
      std::multimap<double, unsigned> sortedTracks;
      block.associatedElements( index,  linkData,
				sortedTracks,
				reco::PFBlockElement::TRACK,
				reco::PFBlock::LINKTEST_ALL );
      // std::cout << "The ECAL is linked to " << sortedTracks.size() << std::endl;

      // Loop over all tracks
      for(IE ie = sortedTracks.begin(); ie != sortedTracks.end(); ++ie ) {
	unsigned jTrack = ie->second;
	// std::cout << "Track " << jTrack << std::endl;
	// Track must be active
	if ( !active[jTrack] ) continue;
	//std::cout << "Active " << std::endl;
	
	// The loop is on the other tracks !
	if ( jTrack == iTrack ) continue;
	//std::cout << "A different track ! " << std::endl;
	
	// Check if the ECAL closest to this track is the current ECAL
	// Otherwise ignore this track in the neutral energy determination
	std::multimap<double, unsigned> sortedECAL;
	block.associatedElements( jTrack,  linkData,
				  sortedECAL,
				  reco::PFBlockElement::ECAL,
				  reco::PFBlock::LINKTEST_ALL );
	if ( sortedECAL.begin()->second != index ) continue;
	//std::cout << "With closest ECAL identical " << std::endl;

	
	// Check if this track is also linked to an HCAL 
	std::multimap<double, unsigned> sortedHCAL;
	block.associatedElements( jTrack,  linkData,
				  sortedHCAL,
				  reco::PFBlockElement::HCAL,
				  reco::PFBlock::LINKTEST_ALL );
	if ( !sortedHCAL.size() ) continue;
	//std::cout << "With an HCAL cluster " << sortedHCAL.begin()->first << std::endl;
	ntt++;
	
	// In that case establish a link with the first track 
	block.setLink( iTrack, 
		       sortedHCAL.begin()->second, 
		       sortedECAL.begin()->first, 
		       linkData,
		       PFBlock::LINKTEST_RECHIT );
	
      } // End other tracks
      
      // Redefine HCAL elements 
      block.associatedElements( iTrack,  linkData,
				hcalElems,
				reco::PFBlockElement::HCAL,
				reco::PFBlock::LINKTEST_ALL );

      if ( debug_ && hcalElems.size() ) 
	std::cout << "Track linked back to HCAL due to ECAL sharing with other tracks" << std::endl;
    }
    
    //MICHELE 
    //TEMPORARY SOLUTION FOR ELECTRON REJECTION IN PFTAU
    //COLINFEB16 
    // in case particle flow electrons are not reconstructed, 
    // the mva_e_pi of the charged hadron will be set to 1 
    // if a GSF element is associated to the current TRACK element
    // This information will be used in the electron rejection for tau ID.
    std::multimap<double,unsigned> gsfElems;
    if (usePFElectrons_ == false) {
      block.associatedElements( iTrack,  linkData,
				gsfElems ,
				reco::PFBlockElement::GSF );
    }
    //

    if(hcalElems.empty() && debug_) {
      cout<<"no hcal element connected to track "<<iTrack<<endl;
    }

    // will now loop on associated elements ... 
    // typedef std::multimap<double, unsigned>::iterator IE;
   
    bool hcalFound = false;

    if(debug_) 
      cout<<"now looping on elements associated to the track"<<endl;
    
    // ... first on associated ECAL elements
    // Check if there is still a free ECAL for this track
    for(IE ie = ecalElems.begin(); ie != ecalElems.end(); ++ie ) {
      
      unsigned index = ie->second;
      // Sanity checks and optional printout
      PFBlockElement::Type type = elements[index].type();
      if(debug_) {
	double  dist  = ie->first;
        cout<<"\telement "<<elements[index]<<" linked with distance = "<< dist <<endl;
	if ( ! active[index] ) cout << "This ECAL is already used - skip it" << endl;      
      }
      assert( type == PFBlockElement::ECAL );

      // This ECAL is not free (taken by an electron?) - just skip it
      if ( ! active[index] ) continue;      

      // Flag ECAL clusters for which the corresponding track is not linked to an HCAL cluster

      //reco::PFClusterRef clusterRef = elements[index].clusterRef();
      //assert( !clusterRef.isNull() );
      if( !hcalElems.empty() && debug_)
	cout<<"\t\tat least one hcal element connected to the track."
	    <<" Sparing Ecal cluster for the hcal loop"<<endl; 

    } //loop print ecal elements

  
    // tracks which are not linked to an HCAL 
    // are reconstructed now. 

    if( hcalElems.empty() ) {
      
      if ( debug_ ) 
	std::cout << "Now deals with tracks linked to no HCAL clusters" << std::endl; 
      // vector<unsigned> elementIndices;
      // elementIndices.push_back(iTrack); 

      // The track momentum
      double trackMomentum = elements[iTrack].trackRef()->p();      
      if ( debug_ ) 
	std::cout << elements[iTrack] << std::endl; 
      
      // Is it a "tight" muon ? In that case assume no 
      //Track momentum corresponds to the calorimeter
      //energy for now
      bool thisIsAMuon = PFMuonAlgo::isMuon(elements[iTrack]);
      if ( thisIsAMuon ) trackMomentum = 0.;

      // If it is not a muon check if Is it a fake track ?
      //Michalis: I wonder if we should convert this to dpt/pt?
      bool rejectFake = false;
      if ( !thisIsAMuon  && elements[iTrack].trackRef()->ptError() > ptError_ ) { 

	double deficit = trackMomentum; 
	double resol = neutralHadronEnergyResolution(trackMomentum,
						     elements[iTrack].trackRef()->eta());
	resol *= trackMomentum;

	if ( !ecalElems.empty() ) { 
	  unsigned thisEcal = ecalElems.begin()->second;
	  reco::PFClusterRef clusterRef = elements[thisEcal].clusterRef();
	  deficit -= clusterRef->energy();
	  resol = neutralHadronEnergyResolution(trackMomentum,
						clusterRef->positionREP().Eta());
	  resol *= trackMomentum;
	}

	bool isPrimary = isFromSecInt(elements[iTrack], "primary");

	if ( deficit > nSigmaTRACK_*resol && !isPrimary) { 
	  rejectFake = true;
	  active[iTrack] = false;
	  if ( debug_ )  
	    std::cout << elements[iTrack] << std::endl
		      << "is probably a fake (1) --> lock the track" 
		      << std::endl;
	}
      }

      if ( rejectFake ) continue;

      // Create a track candidate       
      // unsigned tmpi = reconstructTrack( elements[iTrack] );
      //active[iTrack] = false;
      std::vector<unsigned> tmpi;
      std::vector<unsigned> kTrack;
      
      // Some cleaning : secondary tracks without calo's and large momentum must be fake
      double Dpt = trackRef->ptError();

      if ( rejectTracks_Step45_ && ecalElems.empty() && 
	   trackMomentum > 30. && Dpt > 0.5 && 
	   ( trackRef->algo() == TrackBase::mixedTripletStep || 
	     trackRef->algo() == TrackBase::pixelLessStep || 
	     trackRef->algo() == TrackBase::tobTecStep ) ) {

	double dptRel = Dpt/trackRef->pt()*100;
	bool isPrimaryOrSecondary = isFromSecInt(elements[iTrack], "all");

	if ( isPrimaryOrSecondary && dptRel < dptRel_DispVtx_) continue;
	unsigned nHits =  elements[iTrack].trackRef()->hitPattern().trackerLayersWithMeasurement();
	unsigned int NLostHit = trackRef->hitPattern().trackerLayersWithoutMeasurement(HitPattern::TRACK_HITS);

	if ( debug_ ) 
	  std::cout << "A track (algo = " << trackRef->algo() << ") with momentum " << trackMomentum 
		    << " / " << elements[iTrack].trackRef()->pt() << " +/- " << Dpt 
		    << " / " << elements[iTrack].trackRef()->eta() 
		    << " without any link to ECAL/HCAL and with " << nHits << " (" << NLostHit 
		    << ") hits (lost hits) has been cleaned" << std::endl;
	active[iTrack] = false;
	continue;
      }


      tmpi.push_back(reconstructTrack( elements[iTrack]));

      kTrack.push_back(iTrack);
      active[iTrack] = false;

      // No ECAL cluster either ... continue...
      if ( ecalElems.empty() ) { 
	(*pfCandidates_)[tmpi[0]].setEcalEnergy( 0., 0. );
	(*pfCandidates_)[tmpi[0]].setHcalEnergy( 0., 0. );
	(*pfCandidates_)[tmpi[0]].setHoEnergy( 0., 0. );
	(*pfCandidates_)[tmpi[0]].setPs1Energy( 0 );
	(*pfCandidates_)[tmpi[0]].setPs2Energy( 0 );
	(*pfCandidates_)[tmpi[0]].addElementInBlock( blockref, kTrack[0] );
	continue;
      }
          
      // Look for closest ECAL cluster
      unsigned thisEcal = ecalElems.begin()->second;
      reco::PFClusterRef clusterRef = elements[thisEcal].clusterRef();
      if ( debug_ ) std::cout << " is associated to " << elements[thisEcal] << std::endl;


      // Set ECAL energy for muons
      if ( thisIsAMuon ) { 
	(*pfCandidates_)[tmpi[0]].setEcalEnergy( clusterRef->energy(),
						 std::min(clusterRef->energy(), muonECAL_[0]) );
	(*pfCandidates_)[tmpi[0]].setHcalEnergy( 0., 0. );
	(*pfCandidates_)[tmpi[0]].setHoEnergy( 0., 0. );
	(*pfCandidates_)[tmpi[0]].setPs1Energy( 0 );
	(*pfCandidates_)[tmpi[0]].setPs2Energy( 0 );
	(*pfCandidates_)[tmpi[0]].addElementInBlock( blockref, kTrack[0] );
      }
      
      double slopeEcal = 1.;
      bool connectedToEcal = false;
      unsigned iEcal = 99999;
      double calibEcal = 0.;
      double calibHcal = 0.;
      double totalEcal = thisIsAMuon ? -muonECAL_[0] : 0.;

      // Consider charged particles closest to the same ECAL cluster
      std::multimap<double, unsigned> sortedTracks;
      block.associatedElements( thisEcal,  linkData,
				sortedTracks,
				reco::PFBlockElement::TRACK,
				reco::PFBlock::LINKTEST_ALL );
    
      for(IE ie = sortedTracks.begin(); ie != sortedTracks.end(); ++ie ) {
	unsigned jTrack = ie->second;

	// Don't consider already used tracks
	if ( !active[jTrack] ) continue;

	// The loop is on the other tracks !
	if ( jTrack == iTrack ) continue;
	
	// Check if the ECAL cluster closest to this track is 
	// indeed the current ECAL cluster. Only tracks not 
	// linked to an HCAL cluster are considered here
	std::multimap<double, unsigned> sortedECAL;
	block.associatedElements( jTrack,  linkData,
				  sortedECAL,
				  reco::PFBlockElement::ECAL,
				  reco::PFBlock::LINKTEST_ALL );
	if ( sortedECAL.begin()->second != thisEcal ) continue;

	// Check if this track is a muon
	bool thatIsAMuon = PFMuonAlgo::isMuon(elements[jTrack]);


	// Check if this track is not a fake
	bool rejectFake = false;
	reco::TrackRef trackRef = elements[jTrack].trackRef();
	if ( !thatIsAMuon && trackRef->ptError() > ptError_) { 
	  double deficit = trackMomentum + trackRef->p() - clusterRef->energy();
	  double resol = nSigmaTRACK_*neutralHadronEnergyResolution(trackMomentum+trackRef->p(),
								    clusterRef->positionREP().Eta());
	  resol *= (trackMomentum+trackRef->p());
	  if ( deficit > nSigmaTRACK_*resol ) { 
 	    rejectFake = true;
	    kTrack.push_back(jTrack);
	    active[jTrack] = false;
	    if ( debug_ )  
	      std::cout << elements[jTrack] << std::endl
			<< "is probably a fake (2) --> lock the track" 
			<< std::endl; 
	  }
	}
	if ( rejectFake ) continue;
	
	
	// Otherwise, add this track momentum to the total track momentum
	/* */
	// reco::TrackRef trackRef = elements[jTrack].trackRef();
	if ( !thatIsAMuon ) {	
	  if ( debug_ ) 
	    std::cout << "Track momentum increased from " << trackMomentum << " GeV "; 
	  trackMomentum += trackRef->p();
	  if ( debug_ ) {
	    std::cout << "to " << trackMomentum << " GeV." << std::endl; 
	    std::cout << "with " << elements[jTrack] << std::endl;
	  }
	} else { 
	  totalEcal -= muonECAL_[0];
	  totalEcal = std::max(totalEcal, 0.);
	}

	// And create a charged particle candidate !

	tmpi.push_back(reconstructTrack( elements[jTrack] ));


	kTrack.push_back(jTrack);
	active[jTrack] = false;

	if ( thatIsAMuon ) { 
	  (*pfCandidates_)[tmpi.back()].setEcalEnergy(clusterRef->energy(),
						      std::min(clusterRef->energy(),muonECAL_[0]));
	  (*pfCandidates_)[tmpi.back()].setHcalEnergy( 0., 0. );
	  (*pfCandidates_)[tmpi.back()].setHoEnergy( 0., 0. );
	  (*pfCandidates_)[tmpi.back()].setPs1Energy( 0 );
	  (*pfCandidates_)[tmpi.back()].setPs2Energy( 0 );
	  (*pfCandidates_)[tmpi.back()].addElementInBlock( blockref, kTrack.back() );
	}
      }


      if ( debug_ ) std::cout << "Loop over all associated ECAL clusters" << std::endl; 
      // Loop over all ECAL linked clusters ordered by increasing distance.
      for(IE ie = ecalElems.begin(); ie != ecalElems.end(); ++ie ) {
	
	unsigned index = ie->second;
	PFBlockElement::Type type = elements[index].type();
	assert( type == PFBlockElement::ECAL );
	if ( debug_ ) std::cout << elements[index] << std::endl;
	
	// Just skip clusters already taken
	if ( debug_ && ! active[index] ) std::cout << "is not active  - ignore " << std::endl;
	if ( ! active[index] ) continue;

	// Just skip this cluster if it's closer to another track
	/* */
	block.associatedElements( index,  linkData,
				  sortedTracks,
			 	  reco::PFBlockElement::TRACK,
				  reco::PFBlock::LINKTEST_ALL );
	bool skip = true;
	for (unsigned ic=0; ic<kTrack.size();++ic) {
	  if ( sortedTracks.begin()->second == kTrack[ic] ) { 
	    skip = false;
	    break;
	  }
	}
	if ( debug_ && skip ) std::cout << "is closer to another track - ignore " << std::endl;
	if ( skip ) continue;
	/* */
	  	
	// The corresponding PFCluster and energy
	reco::PFClusterRef clusterRef = elements[index].clusterRef();
	assert( !clusterRef.isNull() );

	if ( debug_ ) {
	  double dist = ie->first;
	  std::cout <<"Ecal cluster with raw energy = " << clusterRef->energy() 
		    <<" linked with distance = " << dist << std::endl;
	}
	/*
	double dist = ie->first;
	if ( !connectedToEcal && dist > 0.1 ) {
	  std::cout << "Warning - first ECAL cluster at a distance of " << dist << "from the closest track!" << std::endl;
	  cout<<"\telement "<<elements[index]<<" linked with distance = "<< dist <<endl;
	  cout<<"\telement "<<elements[iTrack]<<" linked with distance = "<< dist <<endl;
	}
	*/ 

        // Check the presence of preshower clusters in the vicinity
        // Preshower cluster closer to another ECAL cluster are ignored.
        //JOSH: This should use the association map already produced by the cluster corrector for consistency,
        //but should be ok for now
        vector<double> ps1Ene(1,static_cast<double>(0.));
        associatePSClusters(index, reco::PFBlockElement::PS1, block, elements, linkData, active, ps1Ene);
        vector<double> ps2Ene(1,static_cast<double>(0.));
        associatePSClusters(index, reco::PFBlockElement::PS2, block, elements, linkData, active, ps2Ene);        
        
	double ecalEnergy = clusterRef->correctedEnergy();
	if ( debug_ )
	  std::cout << "Corrected ECAL(+PS) energy = " << ecalEnergy << std::endl;

	// Since the electrons were found beforehand, this track must be a hadron. Calibrate 
	// the energy under the hadron hypothesis.
	totalEcal += ecalEnergy;
	double previousCalibEcal = calibEcal;
	double previousSlopeEcal = slopeEcal;
	calibEcal = std::max(totalEcal,0.);
	calibHcal = 0.;
	calibration_->energyEmHad(trackMomentum,calibEcal,calibHcal,
				  clusterRef->positionREP().Eta(),
				  clusterRef->positionREP().Phi());
	if ( totalEcal > 0.) slopeEcal = calibEcal/totalEcal;

	if ( debug_ )
	  std::cout << "The total calibrated energy so far amounts to = " << calibEcal << std::endl;
	

	// Stop the loop when adding more ECAL clusters ruins the compatibility
	if ( connectedToEcal && calibEcal - trackMomentum >= 0. ) {
	// if ( connectedToEcal && calibEcal - trackMomentum >=
	//     nSigmaECAL_*neutralHadronEnergyResolution(trackMomentum,clusterRef->positionREP().Eta())  ) { 
	  calibEcal = previousCalibEcal;
	  slopeEcal = previousSlopeEcal;
	  totalEcal = calibEcal/slopeEcal;

	  // Turn this last cluster in a photon 
	  // (The PS clusters are already locked in "associatePSClusters")
	  active[index] = false;

	  // Find the associated tracks
	  std::multimap<double, unsigned> assTracks;
	  block.associatedElements( index,  linkData,
				    assTracks,
				    reco::PFBlockElement::TRACK,
				    reco::PFBlock::LINKTEST_ALL );


	  unsigned tmpe = reconstructCluster( *clusterRef, ecalEnergy ); 
	  (*pfCandidates_)[tmpe].setEcalEnergy( clusterRef->energy(), ecalEnergy );
	  (*pfCandidates_)[tmpe].setHcalEnergy( 0., 0. );
	  (*pfCandidates_)[tmpe].setHoEnergy( 0., 0. );
	  (*pfCandidates_)[tmpe].setPs1Energy( ps1Ene[0] );
	  (*pfCandidates_)[tmpe].setPs2Energy( ps2Ene[0] );
	  (*pfCandidates_)[tmpe].addElementInBlock( blockref, index );
	  // Check that there is at least one track
	  if(assTracks.size()) {
	    (*pfCandidates_)[tmpe].addElementInBlock( blockref, assTracks.begin()->second );
	    
	    // Assign the position of the track at the ECAL entrance
	    const ::math::XYZPointF& chargedPosition = 
	      dynamic_cast<const reco::PFBlockElementTrack*>(&elements[assTracks.begin()->second])->positionAtECALEntrance();
	    (*pfCandidates_)[tmpe].setPositionAtECALEntrance(chargedPosition);
	  }
	  break;
	}

	// Lock used clusters.
	connectedToEcal = true;
	iEcal = index;
	active[index] = false;
	for (unsigned ic=0; ic<tmpi.size();++ic)  
	  (*pfCandidates_)[tmpi[ic]].addElementInBlock( blockref, iEcal ); 


      } // Loop ecal elements
    
      bool bNeutralProduced = false;
  
      // Create a photon if the ecal energy is too large
      if( connectedToEcal ) {
	reco::PFClusterRef pivotalRef = elements[iEcal].clusterRef();

	double neutralEnergy = calibEcal - trackMomentum;

	/*
	// Check if there are other tracks linked to that ECAL
	for(IE ie = sortedTracks.begin(); ie != sortedTracks.end() && neutralEnergy > 0; ++ie ) {
	  unsigned jTrack = ie->second;
	  
	  // The loop is on the other tracks !
	  if ( jTrack == iTrack ) continue;

	  // Check if the ECAL closest to this track is the current ECAL
	  // Otherwise ignore this track in the neutral energy determination
	  std::multimap<double, unsigned> sortedECAL;
	  block.associatedElements( jTrack,  linkData,
				    sortedECAL,
				    reco::PFBlockElement::ECAL,
				    reco::PFBlock::LINKTEST_ALL );
	  if ( sortedECAL.begin()->second != iEcal ) continue;
	  
	  // Check if this track is also linked to an HCAL 
	  // (in which case it goes to the next loop and is ignored here)
	  std::multimap<double, unsigned> sortedHCAL;
	  block.associatedElements( jTrack,  linkData,
				    sortedHCAL,
				    reco::PFBlockElement::HCAL,
				    reco::PFBlock::LINKTEST_ALL );
	  if ( sortedHCAL.size() ) continue;
	  
	  // Otherwise, subtract this track momentum from the ECAL energy 
	  reco::TrackRef trackRef = elements[jTrack].trackRef();
	  neutralEnergy -= trackRef->p();

	} // End other tracks
	*/
	
	// Add a photon is the energy excess is large enough
	double resol = neutralHadronEnergyResolution(trackMomentum,pivotalRef->positionREP().Eta());
	resol *= trackMomentum;
	if ( neutralEnergy > std::max(0.5,nSigmaECAL_*resol) ) {
	  neutralEnergy /= slopeEcal;
	  unsigned tmpj = reconstructCluster( *pivotalRef, neutralEnergy ); 
	  (*pfCandidates_)[tmpj].setEcalEnergy( pivotalRef->energy(), neutralEnergy );
	  (*pfCandidates_)[tmpj].setHcalEnergy( 0., 0. );
	  (*pfCandidates_)[tmpj].setHoEnergy( 0., 0. );
	  (*pfCandidates_)[tmpj].setPs1Energy( 0. );
	  (*pfCandidates_)[tmpj].setPs2Energy( 0. );
	  (*pfCandidates_)[tmpj].addElementInBlock(blockref, iEcal);
	  bNeutralProduced = true;
	  for (unsigned ic=0; ic<kTrack.size();++ic) 
	    (*pfCandidates_)[tmpj].addElementInBlock( blockref, kTrack[ic] ); 
	} // End neutral energy

	// Set elements in blocks and ECAL energies to all tracks
      	for (unsigned ic=0; ic<tmpi.size();++ic) { 
	  
	  // Skip muons
	  if ( (*pfCandidates_)[tmpi[ic]].particleId() == reco::PFCandidate::mu ) continue; 

	  double fraction = trackMomentum > 0 ? (*pfCandidates_)[tmpi[ic]].trackRef()->p()/trackMomentum : 0;
	  double ecalCal = bNeutralProduced ? 
	    (calibEcal-neutralEnergy*slopeEcal)*fraction : calibEcal*fraction;
	  double ecalRaw = totalEcal*fraction;

	  if (debug_) cout << "The fraction after photon supression is " << fraction << " calibrated ecal = " << ecalCal << endl;

	  (*pfCandidates_)[tmpi[ic]].setEcalEnergy( ecalRaw, ecalCal );
	  (*pfCandidates_)[tmpi[ic]].setHcalEnergy( 0., 0. );
	  (*pfCandidates_)[tmpi[ic]].setHoEnergy( 0., 0. );
	  (*pfCandidates_)[tmpi[ic]].setPs1Energy( 0 );
	  (*pfCandidates_)[tmpi[ic]].setPs2Energy( 0 );
	  (*pfCandidates_)[tmpi[ic]].addElementInBlock( blockref, kTrack[ic] );
	}

      } // End connected ECAL

      // Fill the element_in_block for tracks that are eventually linked to no ECAL clusters at all.
      for (unsigned ic=0; ic<tmpi.size();++ic) { 
	const PFCandidate& pfc = (*pfCandidates_)[tmpi[ic]];
	const PFCandidate::ElementsInBlocks& eleInBlocks = pfc.elementsInBlocks();
	if ( eleInBlocks.size() == 0 ) { 
	  if ( debug_ )std::cout << "Single track / Fill element in block! " << std::endl;
	  (*pfCandidates_)[tmpi[ic]].addElementInBlock( blockref, kTrack[ic] );
	}
      }

    }


    // In case several HCAL elements are linked to this track, 
    // unlinking all of them except the closest. 
    for(IE ie = hcalElems.begin(); ie != hcalElems.end(); ++ie ) {
      
      unsigned index = ie->second;
      
      PFBlockElement::Type type = elements[index].type();

      if(debug_) {
	double dist = block.dist(iTrack,index,linkData,reco::PFBlock::LINKTEST_ALL);
        cout<<"\telement "<<elements[index]<<" linked with distance "<< dist <<endl;
      }

      assert( type == PFBlockElement::HCAL );
      
      // all hcal clusters except the closest 
      // will be unlinked from the track
      if( !hcalFound ) { // closest hcal
        if(debug_) 
          cout<<"\t\tclosest hcal cluster, doing nothing"<<endl;
        
        hcalFound = true;
        
        // active[index] = false;
        // hcalUsed.push_back( index );
      }
      else { // other associated hcal
        // unlink from the track
        if(debug_) 
          cout<<"\t\tsecondary hcal cluster. unlinking"<<endl;
        block.setLink( iTrack, index, -1., linkData,
                       PFBlock::LINKTEST_RECHIT );
      }
    } //loop hcal elements   
  } // end of loop 1 on elements iEle of any type



  // deal with HF. 
  if( !(hfEmIs.empty() &&  hfHadIs.empty() ) ) {
    // there is at least one HF element in this block. 
    // so all elements must be HF.
    assert( hfEmIs.size() + hfHadIs.size() == elements.size() );

    if( elements.size() == 1 ) {
      //Auguste: HAD-only calibration here
      reco::PFClusterRef clusterRef = elements[0].clusterRef();
      double energyHF = 0.;
      double uncalibratedenergyHF = 0.;
      unsigned tmpi = 0;
      switch( clusterRef->layer() ) {
      case PFLayer::HF_EM:
	// do EM-only calibration here
	energyHF = clusterRef->energy();
	uncalibratedenergyHF = energyHF;
	if(thepfEnergyCalibrationHF_->getcalibHF_use()==true){
	  energyHF = thepfEnergyCalibrationHF_->energyEm(uncalibratedenergyHF,
							 clusterRef->positionREP().Eta(),
							 clusterRef->positionREP().Phi()); 
	}
	tmpi = reconstructCluster( *clusterRef, energyHF );     
	(*pfCandidates_)[tmpi].setEcalEnergy( uncalibratedenergyHF, energyHF );
	(*pfCandidates_)[tmpi].setHcalEnergy( 0., 0.);
	(*pfCandidates_)[tmpi].setHoEnergy( 0., 0.);
	(*pfCandidates_)[tmpi].setPs1Energy( 0. );
	(*pfCandidates_)[tmpi].setPs2Energy( 0. );
	(*pfCandidates_)[tmpi].addElementInBlock( blockref, hfEmIs[0] );
	//std::cout << "HF EM alone ! " << energyHF << std::endl;
	break;
      case PFLayer::HF_HAD:
	// do HAD-only calibration here
	energyHF = clusterRef->energy();
	uncalibratedenergyHF = energyHF;
	if(thepfEnergyCalibrationHF_->getcalibHF_use()==true){
	  energyHF = thepfEnergyCalibrationHF_->energyHad(uncalibratedenergyHF,
							 clusterRef->positionREP().Eta(),
							 clusterRef->positionREP().Phi()); 
	}
	tmpi = reconstructCluster( *clusterRef, energyHF );     
	(*pfCandidates_)[tmpi].setHcalEnergy( uncalibratedenergyHF, energyHF );
	(*pfCandidates_)[tmpi].setEcalEnergy( 0., 0.);
	(*pfCandidates_)[tmpi].setHoEnergy( 0., 0.);
	(*pfCandidates_)[tmpi].setPs1Energy( 0. );
	(*pfCandidates_)[tmpi].setPs2Energy( 0. );
	(*pfCandidates_)[tmpi].addElementInBlock( blockref, hfHadIs[0] );
	//std::cout << "HF Had alone ! " << energyHF << std::endl;
	break;
      default:
	assert(0);
      }
    }
    else if(  elements.size() == 2 ) {
      //Auguste: EM + HAD calibration here
      reco::PFClusterRef c0 = elements[0].clusterRef();
      reco::PFClusterRef c1 = elements[1].clusterRef();
      // 2 HF elements. Must be in each layer.
      reco::PFClusterRef cem =  ( c0->layer() == PFLayer::HF_EM ? c0 : c1 );
      reco::PFClusterRef chad = ( c1->layer() == PFLayer::HF_HAD ? c1 : c0 );
      
      if( cem->layer() != PFLayer::HF_EM || chad->layer() != PFLayer::HF_HAD ) {
	cerr<<"Error: 2 elements, but not 1 HFEM and 1 HFHAD"<<endl;
	cerr<<block<<endl;
	assert(0);
// 	assert( c1->layer()== PFLayer::HF_EM &&
// 		c0->layer()== PFLayer::HF_HAD );
      }    
      // do EM+HAD calibration here
      double energyHfEm = cem->energy();
      double energyHfHad = chad->energy();
      double uncalibratedenergyHFEm = energyHfEm;
      double uncalibratedenergyHFHad = energyHfHad;
      if(thepfEnergyCalibrationHF_->getcalibHF_use()==true){
	
	energyHfEm = thepfEnergyCalibrationHF_->energyEmHad(uncalibratedenergyHFEm,
							    0.0,
							    c0->positionREP().Eta(),
							    c0->positionREP().Phi()); 
	energyHfHad = thepfEnergyCalibrationHF_->energyEmHad(0.0,
							     uncalibratedenergyHFHad,
							     c1->positionREP().Eta(),
							     c1->positionREP().Phi()); 
      }
      unsigned tmpi = reconstructCluster( *chad, energyHfEm+energyHfHad );     
      (*pfCandidates_)[tmpi].setEcalEnergy( uncalibratedenergyHFEm, energyHfEm );
      (*pfCandidates_)[tmpi].setHcalEnergy( uncalibratedenergyHFHad, energyHfHad);
      (*pfCandidates_)[tmpi].setHoEnergy( 0., 0.);
      (*pfCandidates_)[tmpi].setPs1Energy( 0. );
      (*pfCandidates_)[tmpi].setPs2Energy( 0. );
      (*pfCandidates_)[tmpi].addElementInBlock( blockref, hfEmIs[0] );
      (*pfCandidates_)[tmpi].addElementInBlock( blockref, hfHadIs[0] );
      //std::cout << "HF EM+HAD found ! " << energyHfEm << " " << energyHfHad << std::endl;     
    }
    else {
      // 1 HF element in the block, 
      // but number of elements not equal to 1 or 2 
      cerr<<"Warning: HF, but n elem different from 1 or 2"<<endl;
      cerr<<block<<endl;
//       assert(0);
//       cerr<<"not ready for navigation in the HF!"<<endl;
    }
  }



  if(debug_) { 
    cout<<endl;
    cout<<endl<<"--------------- loop hcal ---------------------"<<endl;
  }
  
  // track information:
  // trackRef
  // ecal energy
  // hcal energy
  // rescale 

  for(unsigned i=0; i<hcalIs.size(); i++) {
    
    unsigned iHcal= hcalIs[i];    
    PFBlockElement::Type type = elements[iHcal].type();

    assert( type == PFBlockElement::HCAL );

    if(debug_) cout<<endl<<elements[iHcal]<<endl;

    // vector<unsigned> elementIndices;
    // elementIndices.push_back(iHcal);

    // associated tracks
    std::multimap<double, unsigned> sortedTracks;
    block.associatedElements( iHcal,  linkData,
			      sortedTracks,
			      reco::PFBlockElement::TRACK,
			      reco::PFBlock::LINKTEST_ALL );

    std::multimap< unsigned, std::pair<double, unsigned> > associatedEcals;

    std::map< unsigned, std::pair<double, double> > associatedPSs;
 
    std::multimap<double, std::pair<unsigned,bool> > associatedTracks;
    
    // A temporary maps for ECAL satellite clusters
    std::multimap<double,std::pair<unsigned,::math::XYZVector> > ecalSatellites;
    std::pair<unsigned,::math::XYZVector> fakeSatellite = make_pair(iHcal,::math::XYZVector(0.,0.,0.));
    ecalSatellites.insert( make_pair(-1., fakeSatellite) );

    std::multimap< unsigned, std::pair<double, unsigned> > associatedHOs;
    
    PFClusterRef hclusterref = elements[iHcal].clusterRef();
    assert(!hclusterref.isNull() );


    //if there is no track attached to that HCAL, then do not
    //reconstruct an HCAL alone, check if it can be recovered 
    //first
    
    // if no associated tracks, create a neutral hadron
    //if(sortedTracks.empty() ) {

    if( sortedTracks.empty() ) {
      if(debug_) 
        cout<<"\tno associated tracks, keep for later"<<endl;
      continue;
    }

    // Lock the HCAL cluster
    active[iHcal] = false;


    // in the following, tracks are associated to this hcal cluster.
    // will look for an excess of energy in the calorimeters w/r to 
    // the charged energy, and turn this excess into a neutral hadron or 
    // a photon.

    if(debug_) cout<<"\t"<<sortedTracks.size()<<" associated tracks:"<<endl;

    double totalChargedMomentum = 0;
    double sumpError2 = 0.;
    double totalHO = 0.;
    double totalEcal = 0.;
    double totalHcal = hclusterref->energy();
    vector<double> hcalP;
    vector<double> hcalDP;
    vector<unsigned> tkIs;
    double maxDPovP = -9999.;
    
    //Keep track of how much energy is assigned to calorimeter-vs-track energy/momentum excess
    vector< unsigned > chargedHadronsIndices;
    vector< unsigned > chargedHadronsInBlock;
    double mergedNeutralHadronEnergy = 0;
    double mergedPhotonEnergy = 0;
    double muonHCALEnergy = 0.;
    double muonECALEnergy = 0.;
    double muonHCALError = 0.;
    double muonECALError = 0.;
    unsigned nMuons = 0; 


    ::math::XYZVector photonAtECAL(0.,0.,0.);
    std::vector<std::pair<unsigned,::math::XYZVector> > ecalClusters;
    double sumEcalClusters=0;
    ::math::XYZVector hadronDirection(hclusterref->position().X(),
				    hclusterref->position().Y(),
				    hclusterref->position().Z());
    hadronDirection = hadronDirection.Unit();
    ::math::XYZVector hadronAtECAL = totalHcal * hadronDirection;

    // Loop over all tracks associated to this HCAL cluster
    for(IE ie = sortedTracks.begin(); ie != sortedTracks.end(); ++ie ) {

      unsigned iTrack = ie->second;

      // Check the track has not already been used (e.g., in electrons, conversions...)
      if ( !active[iTrack] ) continue;
      // Sanity check 1
      PFBlockElement::Type type = elements[iTrack].type();
      assert( type == reco::PFBlockElement::TRACK );
      // Sanity check 2
      reco::TrackRef trackRef = elements[iTrack].trackRef();
      assert( !trackRef.isNull() ); 

      // The direction at ECAL entrance
      const ::math::XYZPointF& chargedPosition = 
	dynamic_cast<const reco::PFBlockElementTrack*>(&elements[iTrack])->positionAtECALEntrance();
      ::math::XYZVector chargedDirection(chargedPosition.X(),chargedPosition.Y(),chargedPosition.Z());
      chargedDirection = chargedDirection.Unit();

      // look for ECAL elements associated to iTrack (associated to iHcal)
      std::multimap<double, unsigned> sortedEcals;
      block.associatedElements( iTrack,  linkData,
                                sortedEcals,
				reco::PFBlockElement::ECAL,
				reco::PFBlock::LINKTEST_ALL );

      if(debug_) cout<<"\t\t\tnumber of Ecal elements linked to this track: "
                     <<sortedEcals.size()<<endl;     

      // look for HO elements associated to iTrack (associated to iHcal)
      std::multimap<double, unsigned> sortedHOs;
      if (useHO_) {
	block.associatedElements( iTrack,  linkData,
				  sortedHOs,
				  reco::PFBlockElement::HO,
				  reco::PFBlock::LINKTEST_ALL );

      }
      if(debug_) 
	cout<<"PFAlgo : number of HO elements linked to this track: "
	    <<sortedHOs.size()<<endl;
      
      // Create a PF Candidate right away if the track is a tight muon
      reco::MuonRef muonRef = elements[iTrack].muonRef();


      bool thisIsAMuon = PFMuonAlgo::isMuon(elements[iTrack]);
      bool thisIsAnIsolatedMuon = PFMuonAlgo::isIsolatedMuon(elements[iTrack]);
      bool thisIsALooseMuon = false;


      if(!thisIsAMuon ) {
	thisIsALooseMuon = PFMuonAlgo::isLooseMuon(elements[iTrack]);
      }


      if ( thisIsAMuon ) {
	if ( debug_ ) { 
	  std::cout << "\t\tThis track is identified as a muon - remove it from the stack" << std::endl;
	  std::cout << "\t\t" << elements[iTrack] << std::endl;
	}

	// Create a muon.

	unsigned tmpi = reconstructTrack( elements[iTrack] );


	(*pfCandidates_)[tmpi].addElementInBlock( blockref, iTrack );
	(*pfCandidates_)[tmpi].addElementInBlock( blockref, iHcal );
	double muonHcal = std::min(muonHCAL_[0]+muonHCAL_[1],totalHcal);

	// if muon is isolated and muon momentum exceeds the calo energy, absorb the calo energy	
	// rationale : there has been a EM showering by the muon in the calorimeter - or the coil -
	// and we don't want to double count the energy 
	bool letMuonEatCaloEnergy = false;

	if(thisIsAnIsolatedMuon){
	  // The factor 1.3 is the e/pi factor in HCAL...
	  double totalCaloEnergy = totalHcal / 1.30;
	  unsigned iEcal = 0;
	  if( !sortedEcals.empty() ) { 
	    iEcal = sortedEcals.begin()->second; 
	    PFClusterRef eclusterref = elements[iEcal].clusterRef();
	    totalCaloEnergy += eclusterref->energy();
	  }

	  if (useHO_) {
	    // The factor 1.3 is assumed to be the e/pi factor in HO, too.
	    unsigned iHO = 0;
	    if( !sortedHOs.empty() ) { 
	      iHO = sortedHOs.begin()->second; 
	      PFClusterRef eclusterref = elements[iHO].clusterRef();
	      totalCaloEnergy += eclusterref->energy() / 1.30;
	    }
	  }

	  // std::cout << "muon p / total calo = " << muonRef->p() << " "  << (pfCandidates_->back()).p() << " " << totalCaloEnergy << std::endl;
	  //if(muonRef->p() > totalCaloEnergy ) letMuonEatCaloEnergy = true;
	  if( (pfCandidates_->back()).p() > totalCaloEnergy ) letMuonEatCaloEnergy = true;
	}

	if(letMuonEatCaloEnergy) muonHcal = totalHcal;
	double muonEcal =0.;
	unsigned iEcal = 0;
	if( !sortedEcals.empty() ) { 
	  iEcal = sortedEcals.begin()->second; 
	  PFClusterRef eclusterref = elements[iEcal].clusterRef();
	  (*pfCandidates_)[tmpi].addElementInBlock( blockref, iEcal);
	  muonEcal = std::min(muonECAL_[0]+muonECAL_[1],eclusterref->energy());
	  if(letMuonEatCaloEnergy) muonEcal = eclusterref->energy();
	  // If the muon expected energy accounts for the whole ecal cluster energy, lock the ecal cluster
	  if ( eclusterref->energy() - muonEcal  < 0.2 ) active[iEcal] = false;
	  (*pfCandidates_)[tmpi].setEcalEnergy(eclusterref->energy(), muonEcal);
	}
	unsigned iHO = 0;
	double muonHO =0.;
	if (useHO_) {
	  if( !sortedHOs.empty() ) { 
	    iHO = sortedHOs.begin()->second; 
	    PFClusterRef hoclusterref = elements[iHO].clusterRef();
	    (*pfCandidates_)[tmpi].addElementInBlock( blockref, iHO);
	    muonHO = std::min(muonHO_[0]+muonHO_[1],hoclusterref->energy());
	    if(letMuonEatCaloEnergy) muonHO = hoclusterref->energy();
	    // If the muon expected energy accounts for the whole HO cluster energy, lock the HO cluster
	    if ( hoclusterref->energy() - muonHO  < 0.2 ) active[iHO] = false;	    
	    (*pfCandidates_)[tmpi].setHcalEnergy(totalHcal, muonHcal);
	    (*pfCandidates_)[tmpi].setHoEnergy(hoclusterref->energy(), muonHO);
	  }
	} else {
	  (*pfCandidates_)[tmpi].setHcalEnergy(totalHcal, muonHcal);
	}

	if(letMuonEatCaloEnergy){
	  muonHCALEnergy += totalHcal;
	  if (useHO_) muonHCALEnergy +=muonHO;
	  muonHCALError += 0.;
	  muonECALEnergy += muonEcal;
	  muonECALError += 0.;
	  photonAtECAL -= muonEcal*chargedDirection;
	  hadronAtECAL -= totalHcal*chargedDirection;
	  if ( !sortedEcals.empty() ) active[iEcal] = false;
	  active[iHcal] = false;
	  if (useHO_ && !sortedHOs.empty() ) active[iHO] = false;
	}
	else{
	// Estimate of the energy deposit & resolution in the calorimeters
	  muonHCALEnergy += muonHCAL_[0];
	  muonHCALError += muonHCAL_[1]*muonHCAL_[1];
	  if ( muonHO > 0. ) { 
	    muonHCALEnergy += muonHO_[0];
	    muonHCALError += muonHO_[1]*muonHO_[1];
	  }
	  muonECALEnergy += muonECAL_[0];
	  muonECALError += muonECAL_[1]*muonECAL_[1];
	  // ... as well as the equivalent "momentum" at ECAL entrance
	  photonAtECAL -= muonECAL_[0]*chargedDirection;
	  hadronAtECAL -= muonHCAL_[0]*chargedDirection;
	}

	// Remove it from the stack
	active[iTrack] = false;
	// Go to next track
	continue;
      }
 
      //

      if(debug_) cout<<"\t\t"<<elements[iTrack]<<endl;

      // introduce  tracking errors 
      double trackMomentum = trackRef->p();
      totalChargedMomentum += trackMomentum; 

      // If the track is not a tight muon, but still resembles a muon
      // keep it for later for blocks with too large a charged energy
      if ( thisIsALooseMuon && !thisIsAMuon ) nMuons += 1;	

      // ... and keep anyway the pt error for possible fake rejection
      // ... blow up errors of 5th anf 4th iteration, to reject those
      // ... tracks first (in case it's needed)
      double Dpt = trackRef->ptError();
      double blowError = 1.;
      switch (trackRef->algo()) {
      case TrackBase::ctf:
      case TrackBase::initialStep:
      case TrackBase::lowPtTripletStep:
      case TrackBase::pixelPairStep:
      case TrackBase::detachedTripletStep:
      case TrackBase::mixedTripletStep:
      case TrackBase::jetCoreRegionalStep:
      case TrackBase::muonSeededStepInOut:
      case TrackBase::muonSeededStepOutIn:
	blowError = 1.;
	break;
      case TrackBase::pixelLessStep:
	blowError = factors45_[0];
	break;
      case TrackBase::tobTecStep:
	blowError = factors45_[1];
	break;
      case reco::TrackBase::hltIter0:
      case reco::TrackBase::hltIter1:
      case reco::TrackBase::hltIter2:
      case reco::TrackBase::hltIter3:
	blowError = 1.;
	break;
      case reco::TrackBase::hltIter4:
	blowError = factors45_[0];
	break;
      case reco::TrackBase::hltIterX:
	blowError = 1.;
	break;
      default:
	blowError = 1E9;
	break;
      }
      // except if it is from an interaction
      bool isPrimaryOrSecondary = isFromSecInt(elements[iTrack], "all");

      if ( isPrimaryOrSecondary ) blowError = 1.;

      std::pair<unsigned,bool> tkmuon = make_pair(iTrack,thisIsALooseMuon);
      associatedTracks.insert( make_pair(-Dpt*blowError, tkmuon) );     
 
      // Also keep the total track momentum error for comparison with the calo energy
      double Dp = trackRef->qoverpError()*trackMomentum*trackMomentum;
      sumpError2 += Dp*Dp;

      bool connectedToEcal = false; // Will become true if there is at least one ECAL cluster connected
      if( !sortedEcals.empty() ) 
        { // start case: at least one ecal element associated to iTrack
	  
	  // Loop over all connected ECAL clusters
	  for ( IE iec=sortedEcals.begin(); 
		iec!=sortedEcals.end(); ++iec ) {

	    unsigned iEcal = iec->second;
	    double dist = iec->first;
	    
	    // Ignore ECAL cluters already used
	    if( !active[iEcal] ) { 
	      if(debug_) cout<<"\t\tcluster locked"<<endl;          
	      continue;
	    }

	    // Sanity checks
	    PFBlockElement::Type type = elements[ iEcal ].type();
	    assert( type == PFBlockElement::ECAL );
	    PFClusterRef eclusterref = elements[iEcal].clusterRef();
	    assert(!eclusterref.isNull() );
	      
	    // Check if this ECAL is not closer to another track - ignore it in that case
	    std::multimap<double, unsigned> sortedTracksEcal;
	    block.associatedElements( iEcal,  linkData,
				      sortedTracksEcal,
				      reco::PFBlockElement::TRACK,
				      reco::PFBlock::LINKTEST_ALL );
	    unsigned jTrack = sortedTracksEcal.begin()->second;
	    if ( jTrack != iTrack ) continue; 

	    // double chi2Ecal = block.chi2(jTrack,iEcal,linkData,
	    //                              reco::PFBlock::LINKTEST_ALL);
	    double distEcal = block.dist(jTrack,iEcal,linkData,
					 reco::PFBlock::LINKTEST_ALL);
					 
	    // totalEcal += eclusterref->energy();
	    // float ecalEnergyUnCalibrated = eclusterref->energy();
	    //std::cout << "Ecal Uncalibrated " << ecalEnergyUnCalibrated << std::endl;

	    float ecalEnergyCalibrated = eclusterref->correctedEnergy();
	    ::math::XYZVector photonDirection(eclusterref->position().X(),
					    eclusterref->position().Y(),
					    eclusterref->position().Z());
	    photonDirection = photonDirection.Unit();

	    if ( !connectedToEcal ) { // This is the closest ECAL cluster - will add its energy later
	      
	      if(debug_) cout<<"\t\t\tclosest: "
			     <<elements[iEcal]<<endl;
	      
	      connectedToEcal = true;
	      // PJ 1st-April-09 : To be done somewhere !!! (Had to comment it, but it is needed)
	      // currentChargedHadron.addElementInBlock( blockref, iEcal );
       
	      std::pair<unsigned,::math::XYZVector> satellite = 
		make_pair(iEcal,ecalEnergyCalibrated*photonDirection);
	      ecalSatellites.insert( make_pair(-1., satellite) );

	    } else { // Keep satellite clusters for later
	      
	      std::pair<unsigned,::math::XYZVector> satellite = 
		make_pair(iEcal,ecalEnergyCalibrated*photonDirection);
	      ecalSatellites.insert( make_pair(dist, satellite) );
	      
	    }
	    
	    std::pair<double, unsigned> associatedEcal 
	      = make_pair( distEcal, iEcal );
	    associatedEcals.insert( make_pair(iTrack, associatedEcal) );

	  } // End loop ecal associated to iTrack
	} // end case: at least one ecal element associated to iTrack

      if( useHO_ && !sortedHOs.empty() ) 
        { // start case: at least one ho element associated to iTrack
	  
	  // Loop over all connected HO clusters
	  for ( IE ieho=sortedHOs.begin(); ieho!=sortedHOs.end(); ++ieho ) {

	    unsigned iHO = ieho->second;
	    double distHO = ieho->first;
	    
	    // Ignore HO cluters already used
	    if( !active[iHO] ) { 
	      if(debug_) cout<<"\t\tcluster locked"<<endl;          
	      continue;
	    }
	    
	    // Sanity checks
	    PFBlockElement::Type type = elements[ iHO ].type();
	    assert( type == PFBlockElement::HO );
	    PFClusterRef hoclusterref = elements[iHO].clusterRef();
	    assert(!hoclusterref.isNull() );
	    
	    // Check if this HO is not closer to another track - ignore it in that case
	    std::multimap<double, unsigned> sortedTracksHO;
	    block.associatedElements( iHO,  linkData,
				      sortedTracksHO,
				      reco::PFBlockElement::TRACK,
				      reco::PFBlock::LINKTEST_ALL );
	    unsigned jTrack = sortedTracksHO.begin()->second;
	    if ( jTrack != iTrack ) continue; 

	    // double chi2HO = block.chi2(jTrack,iHO,linkData,
	    //                              reco::PFBlock::LINKTEST_ALL);
	    //double distHO = block.dist(jTrack,iHO,linkData,
	    //		       reco::PFBlock::LINKTEST_ALL);
	    
	    // Increment the total energy by the energy of this HO cluster
	    totalHO += hoclusterref->energy();
	    active[iHO] = false;
	    // Keep track for later reference in the PFCandidate.
	    std::pair<double, unsigned> associatedHO
	      = make_pair( distHO, iHO );
	    associatedHOs.insert( make_pair(iTrack, associatedHO) );
	    
	  } // End loop ho associated to iTrack
	} // end case: at least one ho element associated to iTrack
      
      
    } // end loop on tracks associated to hcal element iHcal

    // Include totalHO in totalHCAL for the time being (it will be calibrated as HCAL energy)
    totalHcal += totalHO;
    
    // test compatibility between calo and tracker. //////////////

    double caloEnergy = 0.;
    double slopeEcal = 1.0;
    double calibEcal = 0.;
    double calibHcal = 0.;
    hadronDirection = hadronAtECAL.Unit();

    // Determine the expected calo resolution from the total charged momentum
    double Caloresolution = neutralHadronEnergyResolution( totalChargedMomentum, hclusterref->positionREP().Eta());    
    Caloresolution *= totalChargedMomentum;
    // Account for muons
    Caloresolution = std::sqrt(Caloresolution*Caloresolution + muonHCALError + muonECALError);
    totalEcal -= std::min(totalEcal,muonECALEnergy);
    totalHcal -= std::min(totalHcal,muonHCALEnergy);
    if ( totalEcal < 1E-9 ) photonAtECAL = ::math::XYZVector(0.,0.,0.);
    if ( totalHcal < 1E-9 ) hadronAtECAL = ::math::XYZVector(0.,0.,0.);

    // Loop over all ECAL satellites, starting for the closest to the various tracks
    // and adding other satellites until saturation of the total track momentum
    // Note : for code simplicity, the first element of the loop is the HCAL cluster
    // with 0 energy in the ECAL
    for ( IS is = ecalSatellites.begin(); is != ecalSatellites.end(); ++is ) { 

      // Add the energy of this ECAL cluster
      double previousCalibEcal = calibEcal;
      double previousCalibHcal = calibHcal;
      double previousCaloEnergy = caloEnergy;
      double previousSlopeEcal = slopeEcal;
      ::math::XYZVector previousHadronAtECAL = hadronAtECAL;
      //
      totalEcal += sqrt(is->second.second.Mag2());
      photonAtECAL += is->second.second;
      calibEcal = std::max(0.,totalEcal);
      calibHcal = std::max(0.,totalHcal);
      hadronAtECAL = calibHcal * hadronDirection;
      // Calibrate ECAL and HCAL energy under the hadron hypothesis.
      calibration_->energyEmHad(totalChargedMomentum,calibEcal,calibHcal,
				hclusterref->positionREP().Eta(),
				hclusterref->positionREP().Phi());
      caloEnergy = calibEcal+calibHcal;
      if ( totalEcal > 0.) slopeEcal = calibEcal/totalEcal;

      hadronAtECAL = calibHcal * hadronDirection; 
      
      // Continue looping until all closest clusters are exhausted and as long as
      // the calorimetric energy does not saturate the total momentum.
      if ( is->first < 0. || caloEnergy - totalChargedMomentum <= 0. ) { 
	if(debug_) cout<<"\t\t\tactive, adding "<<is->second.second
		       <<" to ECAL energy, and locking"<<endl;
	active[is->second.first] = false;
        double clusterEnergy=sqrt(is->second.second.Mag2());
        if(clusterEnergy>50) {
          ecalClusters.push_back(is->second);
          sumEcalClusters+=clusterEnergy;
	}
        continue;
      }

      // Otherwise, do not consider the last cluster examined and exit.
      // active[is->second.first] = true;
      totalEcal -= sqrt(is->second.second.Mag2());
      photonAtECAL -= is->second.second;
      calibEcal = previousCalibEcal;
      calibHcal = previousCalibHcal;
      hadronAtECAL = previousHadronAtECAL;
      slopeEcal = previousSlopeEcal;
      caloEnergy = previousCaloEnergy;

      break;

    }

    // Sanity check !
    assert(caloEnergy>=0);


    // And now check for hadronic energy excess...

    //colin: resolution should be measured on the ecal+hcal case. 
    // however, the result will be close. 
    // double Caloresolution = neutralHadronEnergyResolution( caloEnergy );
    // Caloresolution *= caloEnergy;
    // PJ The resolution is on the expected charged calo energy !
    //double Caloresolution = neutralHadronEnergyResolution( totalChargedMomentum, hclusterref->positionREP().Eta());
    //Caloresolution *= totalChargedMomentum;
    // that of the charged particles linked to the cluster!

    /* */
    ////////////////////// TRACKER MUCH LARGER THAN CALO /////////////////////////
    if ( totalChargedMomentum - caloEnergy > nSigmaTRACK_*Caloresolution ) {    

      /* 
      cout<<"\tCompare Calo Energy to total charged momentum "<<endl;
      cout<<"\t\tsum p    = "<<totalChargedMomentum<<" +- "<<sqrt(sumpError2)<<endl;
      cout<<"\t\tsum ecal = "<<totalEcal<<endl;
      cout<<"\t\tsum hcal = "<<totalHcal<<endl;
      cout<<"\t\t => Calo Energy = "<<caloEnergy<<" +- "<<Caloresolution<<endl;
      cout<<"\t\t => Calo Energy- total charged momentum = "
         <<caloEnergy-totalChargedMomentum<<" +- "<<TotalError<<endl;
      cout<<endl;
      cout << "\t\tNumber/momentum of muons found in the block : " << nMuons << std::endl;
      */
      // First consider loose muons
      if ( nMuons > 0 ) { 
	for ( IT it = associatedTracks.begin(); it != associatedTracks.end(); ++it ) { 
	  unsigned iTrack = it->second.first;
	  // Only active tracks
	  if ( !active[iTrack] ) continue;
	  // Only muons
	  if ( !it->second.second ) continue;

	  double trackMomentum = elements[it->second.first].trackRef()->p();

	  // look for ECAL elements associated to iTrack (associated to iHcal)
	  std::multimap<double, unsigned> sortedEcals;
	  block.associatedElements( iTrack,  linkData,
				    sortedEcals,
				    reco::PFBlockElement::ECAL,
				    reco::PFBlock::LINKTEST_ALL );
	  std::multimap<double, unsigned> sortedHOs;
	  block.associatedElements( iTrack,  linkData,
				    sortedHOs,
				    reco::PFBlockElement::HO,
				    reco::PFBlock::LINKTEST_ALL );

	  //Here allow for loose muons! 
	  unsigned tmpi = reconstructTrack( elements[iTrack],true);

	  (*pfCandidates_)[tmpi].addElementInBlock( blockref, iTrack );
	  (*pfCandidates_)[tmpi].addElementInBlock( blockref, iHcal );
	  double muonHcal = std::min(muonHCAL_[0]+muonHCAL_[1],totalHcal-totalHO);
	  double muonHO = 0.;
	  (*pfCandidates_)[tmpi].setHcalEnergy(totalHcal,muonHcal);
	  if( !sortedEcals.empty() ) { 
	    unsigned iEcal = sortedEcals.begin()->second; 
	    PFClusterRef eclusterref = elements[iEcal].clusterRef();
	    (*pfCandidates_)[tmpi].addElementInBlock( blockref, iEcal);
	    double muonEcal = std::min(muonECAL_[0]+muonECAL_[1],eclusterref->energy());
	    (*pfCandidates_)[tmpi].setEcalEnergy(eclusterref->energy(),muonEcal);
	  }
	  if( useHO_ && !sortedHOs.empty() ) { 
	    unsigned iHO = sortedHOs.begin()->second; 
	    PFClusterRef hoclusterref = elements[iHO].clusterRef();
	    (*pfCandidates_)[tmpi].addElementInBlock( blockref, iHO);
	    muonHO = std::min(muonHO_[0]+muonHO_[1],hoclusterref->energy());
	    (*pfCandidates_)[tmpi].setHcalEnergy(max(totalHcal-totalHO,0.0),muonHcal);
	    (*pfCandidates_)[tmpi].setHoEnergy(hoclusterref->energy(),muonHO);
	  }
	  // Remove it from the block
	  const ::math::XYZPointF& chargedPosition = 
	    dynamic_cast<const reco::PFBlockElementTrack*>(&elements[it->second.first])->positionAtECALEntrance();	  
	  ::math::XYZVector chargedDirection(chargedPosition.X(), chargedPosition.Y(), chargedPosition.Z());
	  chargedDirection = chargedDirection.Unit();
	  totalChargedMomentum -= trackMomentum;
	  // Update the calo energies
	  if ( totalEcal > 0. ) 
	    calibEcal -= std::min(calibEcal,muonECAL_[0]*calibEcal/totalEcal);
	  if ( totalHcal > 0. ) 
	    calibHcal -= std::min(calibHcal,muonHCAL_[0]*calibHcal/totalHcal);
	  totalEcal -= std::min(totalEcal,muonECAL_[0]);
	  totalHcal -= std::min(totalHcal,muonHCAL_[0]);
	  if ( totalEcal > muonECAL_[0] ) photonAtECAL -= muonECAL_[0] * chargedDirection;
	  if ( totalHcal > muonHCAL_[0] ) hadronAtECAL -= muonHCAL_[0]*calibHcal/totalHcal * chargedDirection;
	  caloEnergy = calibEcal+calibHcal;
	  muonHCALEnergy += muonHCAL_[0];
	  muonHCALError += muonHCAL_[1]*muonHCAL_[1];
	  if ( muonHO > 0. ) { 
	    muonHCALEnergy += muonHO_[0];
	    muonHCALError += muonHO_[1]*muonHO_[1];
	    if ( totalHcal > 0. ) { 
	      calibHcal -= std::min(calibHcal,muonHO_[0]*calibHcal/totalHcal);
	      totalHcal -= std::min(totalHcal,muonHO_[0]);
	    }
	  }
	  muonECALEnergy += muonECAL_[0];
	  muonECALError += muonECAL_[1]*muonECAL_[1];
	  active[iTrack] = false;
	  // Stop the loop whenever enough muons are removed
	  //Commented out: Keep looking for muons since they often come in pairs -Matt
	  //if ( totalChargedMomentum < caloEnergy ) break;	
	}
	// New calo resolution.
	Caloresolution = neutralHadronEnergyResolution( totalChargedMomentum, hclusterref->positionREP().Eta());    
	Caloresolution *= totalChargedMomentum;
	Caloresolution = std::sqrt(Caloresolution*Caloresolution + muonHCALError + muonECALError);
      }
    }
    /*    
    if(debug_){
      cout<<"\tBefore Cleaning "<<endl;
      cout<<"\tCompare Calo Energy to total charged momentum "<<endl;
      cout<<"\t\tsum p    = "<<totalChargedMomentum<<" +- "<<sqrt(sumpError2)<<endl;
      cout<<"\t\tsum ecal = "<<totalEcal<<endl;
      cout<<"\t\tsum hcal = "<<totalHcal<<endl;
      cout<<"\t\t => Calo Energy = "<<caloEnergy<<" +- "<<Caloresolution<<endl;
      cout<<"\t\t => Calo Energy- total charged momentum = "
	  <<caloEnergy-totalChargedMomentum<<" +- "<<TotalError<<endl;
      cout<<endl;
    }
    */

    // Second consider bad tracks (if still needed after muon removal)
    unsigned corrTrack = 10000000;
    double corrFact = 1.;

    if (rejectTracks_Bad_ && 
	totalChargedMomentum - caloEnergy > nSigmaTRACK_*Caloresolution) { 
      
      for ( IT it = associatedTracks.begin(); it != associatedTracks.end(); ++it ) { 
	unsigned iTrack = it->second.first;
	// Only active tracks
	if ( !active[iTrack] ) continue;
	const reco::TrackRef& trackref = elements[it->second.first].trackRef();

	double dptRel = fabs(it->first)/trackref->pt()*100;
	bool isSecondary = isFromSecInt(elements[iTrack], "secondary");
	bool isPrimary = isFromSecInt(elements[iTrack], "primary");

	if ( isSecondary && dptRel < dptRel_DispVtx_ ) continue;
	// Consider only bad tracks
	if ( fabs(it->first) < ptError_ ) break;
	// What would become the block charged momentum if this track were removed
	double wouldBeTotalChargedMomentum = 
	  totalChargedMomentum - trackref->p();
	// Reject worst tracks, as long as the total charged momentum 
	// is larger than the calo energy

	if ( wouldBeTotalChargedMomentum > caloEnergy ) { 

	  if (debug_ && isSecondary) {
	    cout << "In bad track rejection step dptRel = " << dptRel << " dptRel_DispVtx_ = " << dptRel_DispVtx_ << endl;
	    cout << "The calo energy would be still smaller even without this track but it is attached to a NI"<< endl;
	  }


	  if(isPrimary || (isSecondary && dptRel < dptRel_DispVtx_)) continue;
	  active[iTrack] = false;
	  totalChargedMomentum = wouldBeTotalChargedMomentum;
	  if ( debug_ )
	    std::cout << "\tElement  " << elements[iTrack] 
		      << " rejected (Dpt = " << -it->first 
		      << " GeV/c, algo = " << trackref->algo() << ")" << std::endl;
	// Just rescale the nth worst track momentum to equalize the calo energy
	} else {	
	  if(isPrimary) break;
	  corrTrack = iTrack;
	  corrFact = (caloEnergy - wouldBeTotalChargedMomentum)/elements[it->second.first].trackRef()->p();
	  if ( trackref->p()*corrFact < 0.05 ) { 
	    corrFact = 0.;
	    active[iTrack] = false;
	  }
	  totalChargedMomentum -= trackref->p()*(1.-corrFact);
	  if ( debug_ ) 
	    std::cout << "\tElement  " << elements[iTrack] 
		      << " (Dpt = " << -it->first 
		      << " GeV/c, algo = " << trackref->algo() 
		      << ") rescaled by " << corrFact 
		      << " Now the total charged momentum is " << totalChargedMomentum  << endl;
	  break;
	}
      }
    }

    // New determination of the calo and track resolution avec track deletion/rescaling.
    Caloresolution = neutralHadronEnergyResolution( totalChargedMomentum, hclusterref->positionREP().Eta());    
    Caloresolution *= totalChargedMomentum;
    Caloresolution = std::sqrt(Caloresolution*Caloresolution + muonHCALError + muonECALError);

    // Check if the charged momentum is still very inconsistent with the calo measurement.
    // In this case, just drop all tracks from 4th and 5th iteration linked to this block
    
    if ( rejectTracks_Step45_ &&
	 sortedTracks.size() > 1 && 
	 totalChargedMomentum - caloEnergy > nSigmaTRACK_*Caloresolution ) { 
      for ( IT it = associatedTracks.begin(); it != associatedTracks.end(); ++it ) { 
	unsigned iTrack = it->second.first;
	reco::TrackRef trackref = elements[iTrack].trackRef();
	if ( !active[iTrack] ) continue;

	double dptRel = fabs(it->first)/trackref->pt()*100;
	bool isPrimaryOrSecondary = isFromSecInt(elements[iTrack], "all");




	if (isPrimaryOrSecondary && dptRel < dptRel_DispVtx_) continue;

	switch (trackref->algo()) {
	case TrackBase::ctf:
	case TrackBase::initialStep:
	case TrackBase::lowPtTripletStep:
	case TrackBase::pixelPairStep:
	case TrackBase::detachedTripletStep:
	case TrackBase::mixedTripletStep:
	case TrackBase::jetCoreRegionalStep:
	case TrackBase::muonSeededStepInOut:
	case TrackBase::muonSeededStepOutIn:
	  break;
	case TrackBase::pixelLessStep:
	case TrackBase::tobTecStep:
	  active[iTrack] = false;	
	  totalChargedMomentum -= trackref->p();
	  
	  if ( debug_ ) 
	    std::cout << "\tElement  " << elements[iTrack] 
		      << " rejected (Dpt = " << -it->first 
		      << " GeV/c, algo = " << trackref->algo() << ")" << std::endl;
	  break;
	case reco::TrackBase::hltIter0:
	case reco::TrackBase::hltIter1:
	case reco::TrackBase::hltIter2:
	case reco::TrackBase::hltIter3:
	case reco::TrackBase::hltIter4:
	case reco::TrackBase::hltIterX:
	  break;	  
	default:
	  break;
	}
      }
    }

    // New determination of the calo and track resolution avec track deletion/rescaling.
    Caloresolution = neutralHadronEnergyResolution( totalChargedMomentum, hclusterref->positionREP().Eta());    
    Caloresolution *= totalChargedMomentum;
    Caloresolution = std::sqrt(Caloresolution*Caloresolution + muonHCALError + muonECALError);

    // Make PF candidates with the remaining tracks in the block
    sumpError2 = 0.;
    for ( IT it = associatedTracks.begin(); it != associatedTracks.end(); ++it ) { 
      unsigned iTrack = it->second.first;
      if ( !active[iTrack] ) continue;
      reco::TrackRef trackRef = elements[iTrack].trackRef();
      double trackMomentum = trackRef->p();
      double Dp = trackRef->qoverpError()*trackMomentum*trackMomentum;
      unsigned tmpi = reconstructTrack( elements[iTrack] );


      (*pfCandidates_)[tmpi].addElementInBlock( blockref, iTrack );
      (*pfCandidates_)[tmpi].addElementInBlock( blockref, iHcal );
      std::pair<II,II> myEcals = associatedEcals.equal_range(iTrack);
      for (II ii=myEcals.first; ii!=myEcals.second; ++ii ) { 
	unsigned iEcal = ii->second.second;
	if ( active[iEcal] ) continue;
	(*pfCandidates_)[tmpi].addElementInBlock( blockref, iEcal );
      }
      
      if (useHO_) {
	std::pair<II,II> myHOs = associatedHOs.equal_range(iTrack);
	for (II ii=myHOs.first; ii!=myHOs.second; ++ii ) { 
	  unsigned iHO = ii->second.second;
	  if ( active[iHO] ) continue;
	  (*pfCandidates_)[tmpi].addElementInBlock( blockref, iHO );
	}
      }

      if ( iTrack == corrTrack ) { 
	(*pfCandidates_)[tmpi].rescaleMomentum(corrFact);
	trackMomentum *= corrFact;
      }
      chargedHadronsIndices.push_back( tmpi );
      chargedHadronsInBlock.push_back( iTrack );
      active[iTrack] = false;
      hcalP.push_back(trackMomentum);
      hcalDP.push_back(Dp);
      if (Dp/trackMomentum > maxDPovP) maxDPovP = Dp/trackMomentum;
      sumpError2 += Dp*Dp;
    }

    // The total uncertainty of the difference Calo-Track
    double TotalError = sqrt(sumpError2 + Caloresolution*Caloresolution);

    if ( debug_ ) {
      cout<<"\tCompare Calo Energy to total charged momentum "<<endl;
      cout<<"\t\tsum p    = "<<totalChargedMomentum<<" +- "<<sqrt(sumpError2)<<endl;
      cout<<"\t\tsum ecal = "<<totalEcal<<endl;
      cout<<"\t\tsum hcal = "<<totalHcal<<endl;
      cout<<"\t\t => Calo Energy = "<<caloEnergy<<" +- "<<Caloresolution<<endl;
      cout<<"\t\t => Calo Energy- total charged momentum = "
	  <<caloEnergy-totalChargedMomentum<<" +- "<<TotalError<<endl;
      cout<<endl;
    }

    /* */

    /////////////// TRACKER AND CALO COMPATIBLE  ////////////////
    double nsigma = nSigmaHCAL(totalChargedMomentum,hclusterref->positionREP().Eta());
    //double nsigma = nSigmaHCAL(caloEnergy,hclusterref->positionREP().Eta());
    if ( abs(totalChargedMomentum-caloEnergy)<nsigma*TotalError ) {

      // deposited caloEnergy compatible with total charged momentum
      // if tracking errors are large take weighted average
      
      if(debug_) {
	cout<<"\t\tcase 1: COMPATIBLE "
	    <<"|Calo Energy- total charged momentum| = "
	    <<abs(caloEnergy-totalChargedMomentum)
	    <<" < "<<nsigma<<" x "<<TotalError<<endl;
	if (maxDPovP < 0.1 )
	  cout<<"\t\t\tmax DP/P = "<< maxDPovP 
	      <<" less than 0.1: do nothing "<<endl;
	else 
	  cout<<"\t\t\tmax DP/P = "<< maxDPovP 
	      <<" >  0.1: take weighted averages "<<endl;
      }
      
      // if max DP/P < 10%  do nothing
      if (maxDPovP > 0.1) {
      
        // for each track associated to hcal
        //      int nrows = tkIs.size();
        int nrows = chargedHadronsIndices.size();
        TMatrixTSym<double> a (nrows);
        TVectorD b (nrows);
        TVectorD check(nrows);
        double sigma2E = Caloresolution*Caloresolution;
        for(int i=0; i<nrows; i++) {
          double sigma2i = hcalDP[i]*hcalDP[i];
          if (debug_){
            cout<<"\t\t\ttrack associated to hcal "<<i 
                <<" P = "<<hcalP[i]<<" +- "
                <<hcalDP[i]<<endl;
	  }
          a(i,i) = 1./sigma2i + 1./sigma2E;
          b(i) = hcalP[i]/sigma2i + caloEnergy/sigma2E;
          for(int j=0; j<nrows; j++) {
            if (i==j) continue;
            a(i,j) = 1./sigma2E;
          } // end loop on j
        } // end loop on i
        
        // solve ax = b
        //if (debug_){// debug1
        //cout<<" matrix a(i,j) "<<endl;
        //a.Print();
        //cout<<" vector b(i) "<<endl;
        //b.Print();
        //} // debug1
        TDecompChol decomp(a);
        bool ok = false;
        TVectorD x = decomp.Solve(b, ok);
        // for each track create a PFCandidate track
        // with a momentum rescaled to weighted average
        if (ok) {
          for (int i=0; i<nrows; i++){
            //      unsigned iTrack = trackInfos[i].index;
            unsigned ich = chargedHadronsIndices[i];
            double rescaleFactor =  x(i)/hcalP[i];
            (*pfCandidates_)[ich].rescaleMomentum( rescaleFactor );

            if(debug_){
              cout<<"\t\t\told p "<<hcalP[i]
                  <<" new p "<<x(i) 
                  <<" rescale "<<rescaleFactor<<endl;
            }
          }
        }
        else {
          cerr<<"not ok!"<<endl;
          assert(0);
        }
      }
    }

    /////////////// NEUTRAL DETECTION  ////////////////
    else if( caloEnergy > totalChargedMomentum ) {
      
      //case 2: caloEnergy > totalChargedMomentum + nsigma*TotalError
      //there is an excess of energy in the calos
      //create a neutral hadron or a photon

      /*        
      //If it's isolated don't create neutrals since the energy deposit is always coming from a showering muon
      bool thisIsAnIsolatedMuon = false;
      for(IE ie = sortedTracks.begin(); ie != sortedTracks.end(); ++ie ) {
        unsigned iTrack = ie->second;
	if(PFMuonAlgo::isIsolatedMuon(elements[iTrack].muonRef())) thisIsAnIsolatedMuon = true;	
      }
      
      if(thisIsAnIsolatedMuon){
	if(debug_)cout<<" Not looking for neutral b/c this is an isolated muon "<<endl;
	break;
      }
      */
      double eNeutralHadron = caloEnergy - totalChargedMomentum;
      double ePhoton = (caloEnergy - totalChargedMomentum) / slopeEcal;

      if(debug_) {
        if(!sortedTracks.empty() ){
          cout<<"\t\tcase 2: NEUTRAL DETECTION "
              <<caloEnergy<<" > "<<nsigma<<"x"<<TotalError
              <<" + "<<totalChargedMomentum<<endl;
          cout<<"\t\tneutral activity detected: "<<endl
              <<"\t\t\t           photon = "<<ePhoton<<endl
              <<"\t\t\tor neutral hadron = "<<eNeutralHadron<<endl;
            
          cout<<"\t\tphoton or hadron ?"<<endl;}
          
        if(sortedTracks.empty() )
          cout<<"\t\tno track -> hadron "<<endl;
        else 
          cout<<"\t\t"<<sortedTracks.size()
              <<"tracks -> check if the excess is photonic or hadronic"<<endl;
      }
        

      double ratioMax = 0.;
      reco::PFClusterRef maxEcalRef;
      unsigned maxiEcal= 9999;       

      // for each track associated to hcal: iterator IE ie :
        
      for(IE ie = sortedTracks.begin(); ie != sortedTracks.end(); ++ie ) {
          
        unsigned iTrack = ie->second;
          
        PFBlockElement::Type type = elements[iTrack].type();
        assert( type == reco::PFBlockElement::TRACK );
          
        reco::TrackRef trackRef = elements[iTrack].trackRef();
        assert( !trackRef.isNull() ); 
          
	II iae = associatedEcals.find(iTrack);
          
        if( iae == associatedEcals.end() ) continue;
          
        // double distECAL = iae->second.first;
        unsigned iEcal = iae->second.second;
          
        PFBlockElement::Type typeEcal = elements[iEcal].type();
        assert(typeEcal == reco::PFBlockElement::ECAL);
          
        reco::PFClusterRef clusterRef = elements[iEcal].clusterRef();
        assert( !clusterRef.isNull() );
          
        double pTrack = trackRef->p();
        double eECAL = clusterRef->energy();
        double eECALOverpTrack = eECAL / pTrack;

	if ( eECALOverpTrack > ratioMax ) { 
	  ratioMax = eECALOverpTrack;
	  maxEcalRef = clusterRef;
	  maxiEcal = iEcal;
	}          
	
      }  // end loop on tracks associated to hcal: iterator IE ie
    
      std::vector<reco::PFClusterRef> pivotalClusterRef;
      std::vector<unsigned> iPivotal;
      std::vector<double> particleEnergy, ecalEnergy, hcalEnergy, rawecalEnergy, rawhcalEnergy;
      std::vector<::math::XYZVector> particleDirection;

      // If the excess is smaller than the ecal energy, assign the whole 
      // excess to photons
      if ( ePhoton < totalEcal || eNeutralHadron-calibEcal < 1E-10 ) { 
        if ( !maxEcalRef.isNull() ) {
	 // So the merged photon energy is,
         mergedPhotonEnergy  = ePhoton;
        }
      } else { 
	// Otherwise assign the whole ECAL energy to the photons
        if ( !maxEcalRef.isNull() ) {
	 // So the merged photon energy is,
         mergedPhotonEnergy  = totalEcal;
        }
	// ... and assign the remaining excess to neutral hadrons using the direction of ecal clusters
        mergedNeutralHadronEnergy = eNeutralHadron-calibEcal;
      }

      if ( mergedPhotonEnergy > 0 ) {
          // Split merged photon into photons for each energetic ecal cluster (necessary for jet substructure reconstruction)
          // make only one merged photon if less than 2 ecal clusters
          if ( ecalClusters.size()<=1 ) {
            ecalClusters.clear();
            ecalClusters.push_back(make_pair(maxiEcal, photonAtECAL));
            sumEcalClusters=sqrt(photonAtECAL.Mag2());
          }
	  for(std::vector<std::pair<unsigned,::math::XYZVector> >::const_iterator pae = ecalClusters.begin(); pae != ecalClusters.end(); ++pae ) {
           double clusterEnergy=sqrt(pae->second.Mag2());
	   particleEnergy.push_back(mergedPhotonEnergy*clusterEnergy/sumEcalClusters);
           particleDirection.push_back(pae->second);
	   ecalEnergy.push_back(mergedPhotonEnergy*clusterEnergy/sumEcalClusters);
	   hcalEnergy.push_back(0.);
	   rawecalEnergy.push_back(totalEcal);
	   rawhcalEnergy.push_back(totalHcal);
	   pivotalClusterRef.push_back(elements[pae->first].clusterRef());
 	   iPivotal.push_back(pae->first);
          }
      }
	
      if ( mergedNeutralHadronEnergy > 1.0 ) { 
          // Split merged neutral hadrons according to directions of energetic ecal clusters (necessary for jet substructure reconstruction)
          // make only one merged neutral hadron if less than 2 ecal clusters
          if ( ecalClusters.size()<=1 ) {
              ecalClusters.clear();
              ecalClusters.push_back(make_pair(iHcal, hadronAtECAL));
              sumEcalClusters=sqrt(hadronAtECAL.Mag2());
          }
          for(std::vector<std::pair<unsigned,::math::XYZVector> >::const_iterator pae = ecalClusters.begin(); pae != ecalClusters.end(); ++pae ) {
           double clusterEnergy=sqrt(pae->second.Mag2());
	   particleEnergy.push_back(mergedNeutralHadronEnergy*clusterEnergy/sumEcalClusters);
           particleDirection.push_back(pae->second);
	   ecalEnergy.push_back(0.);
	   hcalEnergy.push_back(mergedNeutralHadronEnergy*clusterEnergy/sumEcalClusters);
	   rawecalEnergy.push_back(totalEcal);
	   rawhcalEnergy.push_back(totalHcal);
	   pivotalClusterRef.push_back(hclusterref);
	   iPivotal.push_back(iHcal);	
         }
      }

      // reconstructing a merged neutral
      // the type of PFCandidate is known from the 
      // reference to the pivotal cluster. 
      
      for ( unsigned iPivot=0; iPivot<iPivotal.size(); ++iPivot ) { 
	
	if ( particleEnergy[iPivot] < 0. ) 
	  std::cout << "ALARM = Negative energy ! " 
		    << particleEnergy[iPivot] << std::endl;
	
	bool useDirection = true;
	unsigned tmpi = reconstructCluster( *pivotalClusterRef[iPivot], 
					    particleEnergy[iPivot], 
					    useDirection,
	                                    particleDirection[iPivot].X(),
					    particleDirection[iPivot].Y(),
					    particleDirection[iPivot].Z()); 

      
	(*pfCandidates_)[tmpi].setEcalEnergy( rawecalEnergy[iPivot],ecalEnergy[iPivot] );
	if ( !useHO_ ) { 
	  (*pfCandidates_)[tmpi].setHcalEnergy( rawhcalEnergy[iPivot],hcalEnergy[iPivot] );
	  (*pfCandidates_)[tmpi].setHoEnergy(0., 0.);
	} else { 
	  (*pfCandidates_)[tmpi].setHcalEnergy( max(rawhcalEnergy[iPivot]-totalHO,0.0),hcalEnergy[iPivot]*(1.-totalHO/rawhcalEnergy[iPivot]));
	  (*pfCandidates_)[tmpi].setHoEnergy(totalHO, totalHO * hcalEnergy[iPivot]/rawhcalEnergy[iPivot]);
	} 
	(*pfCandidates_)[tmpi].setPs1Energy( 0. );
	(*pfCandidates_)[tmpi].setPs2Energy( 0. );
	(*pfCandidates_)[tmpi].set_mva_nothing_gamma( -1. );
	//       (*pfCandidates_)[tmpi].addElement(&elements[iPivotal]);
	// (*pfCandidates_)[tmpi].addElementInBlock(blockref, iPivotal[iPivot]);
	(*pfCandidates_)[tmpi].addElementInBlock( blockref, iHcal );
	for ( unsigned ich=0; ich<chargedHadronsInBlock.size(); ++ich) { 
	  unsigned iTrack = chargedHadronsInBlock[ich];
	  (*pfCandidates_)[tmpi].addElementInBlock( blockref, iTrack );
	  // Assign the position of the track at the ECAL entrance
	  const ::math::XYZPointF& chargedPosition = 
	    dynamic_cast<const reco::PFBlockElementTrack*>(&elements[iTrack])->positionAtECALEntrance();
	  (*pfCandidates_)[tmpi].setPositionAtECALEntrance(chargedPosition);

	  std::pair<II,II> myEcals = associatedEcals.equal_range(iTrack);
	  for (II ii=myEcals.first; ii!=myEcals.second; ++ii ) { 
	    unsigned iEcal = ii->second.second;
	    if ( active[iEcal] ) continue;
	    (*pfCandidates_)[tmpi].addElementInBlock( blockref, iEcal );
	  }
	}

      }

    } // excess of energy
  
  
    // will now share the hcal energy between the various charged hadron 
    // candidates, taking into account the potential neutral hadrons
    
    double totalHcalEnergyCalibrated = calibHcal;
    double totalEcalEnergyCalibrated = calibEcal;
    //JB: The question is: we've resolved the merged photons cleanly, but how should
    //the remaining hadrons be assigned the remaining ecal energy?
    //*Temporary solution*: follow HCAL example with fractions...    
 
   /*
    if(totalEcal>0) {
      // removing ecal energy from abc calibration
      totalHcalEnergyCalibrated  -= calibEcal;
      // totalHcalEnergyCalibrated -= calibration_->paramECALplusHCAL_slopeECAL() * totalEcal;
    }
    */
    // else caloEnergy = hcal only calibrated energy -> ok.

    // remove the energy of the potential neutral hadron
    totalHcalEnergyCalibrated -= mergedNeutralHadronEnergy;
    // similarly for the merged photons
    totalEcalEnergyCalibrated -= mergedPhotonEnergy;
    // share between the charged hadrons

    //COLIN can compute this before
    // not exactly equal to sum p, this is sum E
    double chargedHadronsTotalEnergy = 0;
    for( unsigned ich=0; ich<chargedHadronsIndices.size(); ++ich ) {
      unsigned index = chargedHadronsIndices[ich];
      reco::PFCandidate& chargedHadron = (*pfCandidates_)[index];
      chargedHadronsTotalEnergy += chargedHadron.energy();
    }

    for( unsigned ich=0; ich<chargedHadronsIndices.size(); ++ich ) {
      unsigned index = chargedHadronsIndices[ich];
      reco::PFCandidate& chargedHadron = (*pfCandidates_)[index];
      float fraction = chargedHadron.energy()/chargedHadronsTotalEnergy;

      if ( !useHO_ ) { 
	chargedHadron.setHcalEnergy(  fraction * totalHcal, fraction * totalHcalEnergyCalibrated );          
	chargedHadron.setHoEnergy(  0., 0. ); 
      } else { 
	chargedHadron.setHcalEnergy(  fraction * max(totalHcal-totalHO,0.0), fraction * totalHcalEnergyCalibrated * (1.-totalHO/totalHcal) );          
	chargedHadron.setHoEnergy( fraction * totalHO, fraction * totalHO * totalHcalEnergyCalibrated / totalHcal ); 
      }
      //JB: fixing up (previously omitted) setting of ECAL energy gouzevit
      chargedHadron.setEcalEnergy( fraction * totalEcal, fraction * totalEcalEnergyCalibrated );
    }

    // Finally treat unused ecal satellites as photons.
    for ( IS is = ecalSatellites.begin(); is != ecalSatellites.end(); ++is ) { 
      
      // Ignore satellites already taken
      unsigned iEcal = is->second.first;
      if ( !active[iEcal] ) continue;

      // Sanity checks again (well not useful, this time!)
      PFBlockElement::Type type = elements[ iEcal ].type();
      assert( type == PFBlockElement::ECAL );
      PFClusterRef eclusterref = elements[iEcal].clusterRef();
      assert(!eclusterref.isNull() );

      // Lock the cluster
      active[iEcal] = false;
      
      // Find the associated tracks
      std::multimap<double, unsigned> assTracks;
      block.associatedElements( iEcal,  linkData,
				assTracks,
				reco::PFBlockElement::TRACK,
				reco::PFBlock::LINKTEST_ALL );

      // Create a photon
      unsigned tmpi = reconstructCluster( *eclusterref, sqrt(is->second.second.Mag2()) ); 
      (*pfCandidates_)[tmpi].setEcalEnergy( eclusterref->energy(),sqrt(is->second.second.Mag2()) );
      (*pfCandidates_)[tmpi].setHcalEnergy( 0., 0. );
      (*pfCandidates_)[tmpi].setHoEnergy( 0., 0. );
      (*pfCandidates_)[tmpi].setPs1Energy( associatedPSs[iEcal].first );
      (*pfCandidates_)[tmpi].setPs2Energy( associatedPSs[iEcal].second );
      (*pfCandidates_)[tmpi].addElementInBlock( blockref, iEcal );
      (*pfCandidates_)[tmpi].addElementInBlock( blockref, sortedTracks.begin()->second) ;
    }


  }

   // end loop on hcal element iHcal= hcalIs[i] 


  // Processing the remaining HCAL clusters
  if(debug_) { 
    cout<<endl;
    if(debug_) 
      cout<<endl
          <<"---- loop remaining hcal ------- "<<endl;
  }
  

  //COLINFEB16 
  // now dealing with the HCAL elements that are not linked to any track 
  for(unsigned ihcluster=0; ihcluster<hcalIs.size(); ihcluster++) {
    unsigned iHcal = hcalIs[ihcluster];

    // Keep ECAL and HO elements for reference in the PFCandidate
    std::vector<unsigned> ecalRefs;
    std::vector<unsigned> hoRefs;
    
    if(debug_)
      cout<<endl<<elements[iHcal]<<" ";
    
    
    if( !active[iHcal] ) {
      if(debug_) 
        cout<<"not active"<<endl;         
      continue;
    }
    
    // Find the ECAL elements linked to it
    std::multimap<double, unsigned> ecalElems;
    block.associatedElements( iHcal,  linkData,
                              ecalElems ,
                              reco::PFBlockElement::ECAL,
			      reco::PFBlock::LINKTEST_ALL );

    // Loop on these ECAL elements
    float totalEcal = 0.;
    float ecalMax = 0.;
    reco::PFClusterRef eClusterRef;
    for(IE ie = ecalElems.begin(); ie != ecalElems.end(); ++ie ) {
      
      unsigned iEcal = ie->second;
      double dist = ie->first;
      PFBlockElement::Type type = elements[iEcal].type();
      assert( type == PFBlockElement::ECAL );
      
      // Check if already used
      if( !active[iEcal] ) continue;

      // Check the distance (one HCALPlusECAL tower, roughly)
      // if ( dist > 0.15 ) continue;

      //COLINFEB16 
      // what could be done is to
      // - link by rechit.
      // - take in the neutral hadron all the ECAL clusters 
      // which are within the same CaloTower, according to the distance, 
      // except the ones which are closer to another HCAL cluster.
      // - all the other ECAL linked to this HCAL are photons. 
      // 
      // about the closest HCAL cluster. 
      // it could maybe be easier to loop on the ECAL clusters first
      // to cut the links to all HCAL clusters except the closest, as is 
      // done in the first track loop. But maybe not!
      // or add an helper function to the PFAlgo class to ask 
      // if a given element is the closest of a given type to another one?
    
      // Check if not closer from another free HCAL 
      std::multimap<double, unsigned> hcalElems;
      block.associatedElements( iEcal,  linkData,
				hcalElems ,
				reco::PFBlockElement::HCAL,
				reco::PFBlock::LINKTEST_ALL );

      bool isClosest = true;
      for(IE ih = hcalElems.begin(); ih != hcalElems.end(); ++ih ) {

	unsigned jHcal = ih->second;
	double distH = ih->first;
	
	if ( !active[jHcal] ) continue;

	if ( distH < dist ) { 
	  isClosest = false;
	  break;
	}

      }

      if (!isClosest) continue;


      if(debug_) {
        cout<<"\telement "<<elements[iEcal]<<" linked with dist "<< dist<<endl;
	cout<<"Added to HCAL cluster to form a neutral hadron"<<endl;
      }
      
      reco::PFClusterRef eclusterRef = elements[iEcal].clusterRef();
      assert( !eclusterRef.isNull() );

      double ecalEnergy = eclusterRef->correctedEnergy();
      
      //std::cout << "EcalEnergy, ps1, ps2 = " << ecalEnergy 
      //          << ", " << ps1Ene[0] << ", " << ps2Ene[0] << std::endl;
      totalEcal += ecalEnergy;
      if ( ecalEnergy > ecalMax ) { 
	ecalMax = ecalEnergy;
	eClusterRef = eclusterRef;
      }
      
      ecalRefs.push_back(iEcal);
      active[iEcal] = false;


    }// End loop ECAL
    
    // Now find the HO clusters linked to the HCAL cluster
    double totalHO = 0.;
    double hoMax = 0.;
    //unsigned jHO = 0;
    if (useHO_) {
      std::multimap<double, unsigned> hoElems;
      block.associatedElements( iHcal,  linkData,
				hoElems ,
				reco::PFBlockElement::HO,
				reco::PFBlock::LINKTEST_ALL );
      
      // Loop on these HO elements
      //      double totalHO = 0.;
      //      double hoMax = 0.;
      //      unsigned jHO = 0;
      reco::PFClusterRef hoClusterRef;
      for(IE ie = hoElems.begin(); ie != hoElems.end(); ++ie ) {
	
	unsigned iHO = ie->second;
	double dist = ie->first;
	PFBlockElement::Type type = elements[iHO].type();
	assert( type == PFBlockElement::HO );
	
	// Check if already used
	if( !active[iHO] ) continue;
	
	// Check the distance (one HCALPlusHO tower, roughly)
	// if ( dist > 0.15 ) continue;
	
	// Check if not closer from another free HCAL 
	std::multimap<double, unsigned> hcalElems;
	block.associatedElements( iHO,  linkData,
				  hcalElems ,
				  reco::PFBlockElement::HCAL,
				  reco::PFBlock::LINKTEST_ALL );
	
	bool isClosest = true;
	for(IE ih = hcalElems.begin(); ih != hcalElems.end(); ++ih ) {
	  
	  unsigned jHcal = ih->second;
	  double distH = ih->first;
	  
	  if ( !active[jHcal] ) continue;
	  
	  if ( distH < dist ) { 
	    isClosest = false;
	    break;
	  }
	  
	}
	
	if (!isClosest) continue;
	
	if(debug_ && useHO_) {
	  cout<<"\telement "<<elements[iHO]<<" linked with dist "<< dist<<endl;
	  cout<<"Added to HCAL cluster to form a neutral hadron"<<endl;
	}
	
	reco::PFClusterRef hoclusterRef = elements[iHO].clusterRef();
	assert( !hoclusterRef.isNull() );
	
	double hoEnergy = hoclusterRef->energy(); // calibration_->energyEm(*hoclusterRef,ps1Ene,ps2Ene,crackCorrection);
	
	totalHO += hoEnergy;
	if ( hoEnergy > hoMax ) { 
	  hoMax = hoEnergy;
	  hoClusterRef = hoclusterRef;
	  //jHO = iHO;
	}
	
	hoRefs.push_back(iHO);
	active[iHO] = false;
	
      }// End loop HO
    }

    PFClusterRef hclusterRef 
      = elements[iHcal].clusterRef();
    assert( !hclusterRef.isNull() );
    
    // HCAL energy
    double totalHcal =  hclusterRef->energy();
    // Include the HO energy 
    if ( useHO_ ) totalHcal += totalHO;

    // std::cout << "Hcal Energy,eta : " << totalHcal 
    //          << ", " << hclusterRef->positionREP().Eta()
    //          << std::endl;
    // Calibration
    //double caloEnergy = totalHcal;
    // double slopeEcal = 1.0;
    double calibEcal = totalEcal > 0. ? totalEcal : 0.;
    double calibHcal = std::max(0.,totalHcal);
    if  (  hclusterRef->layer() == PFLayer::HF_HAD  ||
	   hclusterRef->layer() == PFLayer::HF_EM ) { 
      //caloEnergy = totalHcal/0.7;
      calibEcal = totalEcal;
    } else { 
      calibration_->energyEmHad(-1.,calibEcal,calibHcal,
				hclusterRef->positionREP().Eta(),
				hclusterRef->positionREP().Phi());      
      //caloEnergy = calibEcal+calibHcal;
    }

    // std::cout << "CalibEcal,HCal = " << calibEcal << ", " << calibHcal << std::endl;
    // std::cout << "-------------------------------------------------------------------" << std::endl;
    // ECAL energy : calibration

    // double particleEnergy = caloEnergy;
    // double particleEnergy = totalEcal + calibHcal;
    // particleEnergy /= (1.-0.724/sqrt(particleEnergy)-0.0226/particleEnergy);

    unsigned tmpi = reconstructCluster( *hclusterRef, 
                                        calibEcal+calibHcal ); 

    
    (*pfCandidates_)[tmpi].setEcalEnergy( totalEcal, calibEcal );
    if ( !useHO_ ) { 
      (*pfCandidates_)[tmpi].setHcalEnergy( totalHcal, calibHcal );
      (*pfCandidates_)[tmpi].setHoEnergy(0.,0.);
    } else { 
      (*pfCandidates_)[tmpi].setHcalEnergy( max(totalHcal-totalHO,0.0), calibHcal*(1.-totalHO/totalHcal));
      (*pfCandidates_)[tmpi].setHoEnergy(totalHO,totalHO*calibHcal/totalHcal);
    }
    (*pfCandidates_)[tmpi].setPs1Energy( 0. );
    (*pfCandidates_)[tmpi].setPs2Energy( 0. );
    (*pfCandidates_)[tmpi].addElementInBlock( blockref, iHcal );
    for (unsigned iec=0; iec<ecalRefs.size(); ++iec) 
      (*pfCandidates_)[tmpi].addElementInBlock( blockref, ecalRefs[iec] );
    for (unsigned iho=0; iho<hoRefs.size(); ++iho) 
      (*pfCandidates_)[tmpi].addElementInBlock( blockref, hoRefs[iho] );
      
  }//loop hcal elements




  if(debug_) { 
    cout<<endl;
    if(debug_) cout<<endl<<"---- loop ecal------- "<<endl;
  }
  
  // for each ecal element iEcal = ecalIs[i] in turn:

  for(unsigned i=0; i<ecalIs.size(); i++) {
    unsigned iEcal = ecalIs[i];
    
    if(debug_) 
      cout<<endl<<elements[iEcal]<<" ";
    
    if( ! active[iEcal] ) {
      if(debug_) 
        cout<<"not active"<<endl;         
      continue;
    }
    
    PFBlockElement::Type type = elements[ iEcal ].type();
    assert( type == PFBlockElement::ECAL );

    PFClusterRef clusterref = elements[iEcal].clusterRef();
    assert(!clusterref.isNull() );

    active[iEcal] = false;

    float ecalEnergy = clusterref->correctedEnergy();
    // float ecalEnergy = calibration_->energyEm( clusterref->energy() );
    double particleEnergy = ecalEnergy;
    
    unsigned tmpi = reconstructCluster( *clusterref, 
                                        particleEnergy );
 
    (*pfCandidates_)[tmpi].setEcalEnergy( clusterref->energy(),ecalEnergy );
    (*pfCandidates_)[tmpi].setHcalEnergy( 0., 0. );
    (*pfCandidates_)[tmpi].setHoEnergy( 0., 0. );
    (*pfCandidates_)[tmpi].setPs1Energy( 0. );
    (*pfCandidates_)[tmpi].setPs2Energy( 0. );
    (*pfCandidates_)[tmpi].addElementInBlock( blockref, iEcal );
    

  }  // end loop on ecal elements iEcal = ecalIs[i]

}  // end processBlock

/////////////////////////////////////////////////////////////////////
unsigned PFAlgo::reconstructTrack( const reco::PFBlockElement& elt, bool allowLoose) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  reco::TrackRef trackRef = eltTrack->trackRef();
  const reco::Track& track = *trackRef;
  reco::MuonRef muonRef = eltTrack->muonRef();
  int charge = track.charge()>0 ? 1 : -1;

  // Assume this particle is a charged Hadron
  double px = track.px();
  double py = track.py();
  double pz = track.pz();
  double energy = sqrt(track.p()*track.p() + 0.13957*0.13957);

  // Create a PF Candidate
  ::math::XYZTLorentzVector momentum(px,py,pz,energy);
  reco::PFCandidate::ParticleType particleType 
    = reco::PFCandidate::h;

  // Add it to the stack
  pfCandidates_->push_back( PFCandidate( charge, 
                                         momentum,
                                         particleType ) );
  //Set vertex and stuff like this
  pfCandidates_->back().setVertexSource( PFCandidate::kTrkVertex );
  pfCandidates_->back().setTrackRef( trackRef );
  pfCandidates_->back().setPositionAtECALEntrance( eltTrack->positionAtECALEntrance());
  if( muonRef.isNonnull())
    pfCandidates_->back().setMuonRef( muonRef );



  //OK Now try to reconstruct the particle as a muon
  bool isMuon=pfmu_->reconstructMuon(pfCandidates_->back(),muonRef,allowLoose);
  bool isFromDisp = isFromSecInt(elt, "secondary");


  if ((!isMuon) && isFromDisp) {
    double Dpt = trackRef->ptError();
    double dptRel = Dpt/trackRef->pt()*100;
    //If the track is ill measured it is better to not refit it, since the track information probably would not be used.
    //In the PFAlgo we use the trackref information. If the track error is too big the refitted information might be very different
    // from the not refitted one.
    if (dptRel < dptRel_DispVtx_){
      if (debug_) 
	cout << "Not refitted px = " << px << " py = " << py << " pz = " << pz << " energy = " << energy << endl; 
      //reco::TrackRef trackRef = eltTrack->trackRef();
      reco::PFDisplacedVertexRef vRef = eltTrack->displacedVertexRef(reco::PFBlockElement::T_FROM_DISP)->displacedVertexRef();
      reco::Track trackRefit = vRef->refittedTrack(trackRef);
      //change the momentum with the refitted track
      ::math::XYZTLorentzVector momentum(trackRefit.px(),
				       trackRefit.py(),
				       trackRefit.pz(),
				       sqrt(trackRefit.p()*trackRefit.p() + 0.13957*0.13957));
      if (debug_) 
	cout << "Refitted px = " << px << " py = " << py << " pz = " << pz << " energy = " << energy << endl; 
    }
    pfCandidates_->back().setFlag( reco::PFCandidate::T_FROM_DISP, true);
    pfCandidates_->back().setDisplacedVertexRef( eltTrack->displacedVertexRef(reco::PFBlockElement::T_FROM_DISP)->displacedVertexRef(), reco::PFCandidate::T_FROM_DISP);
  }

  // do not label as primary a track which would be recognised as a muon. A muon cannot produce NI. It is with high probability a fake
  if(isFromSecInt(elt, "primary") && !isMuon) {
    pfCandidates_->back().setFlag( reco::PFCandidate::T_TO_DISP, true);
    pfCandidates_->back().setDisplacedVertexRef( eltTrack->displacedVertexRef(reco::PFBlockElement::T_TO_DISP)->displacedVertexRef(), reco::PFCandidate::T_TO_DISP);
  }
  // returns index to the newly created PFCandidate
  return pfCandidates_->size()-1;
}


unsigned 
PFAlgo::reconstructCluster(const reco::PFCluster& cluster,
                           double particleEnergy, 
                           bool useDirection, 
			   double particleX,
			   double particleY, 
			   double particleZ) {
  
  reco::PFCandidate::ParticleType particleType = reco::PFCandidate::X;

  // need to convert the ::math::XYZPoint data member of the PFCluster class=
  // to a displacement vector: 

  // Transform particleX,Y,Z to a position at ECAL/HCAL entrance
  double factor = 1.;
  if ( useDirection ) { 
    switch( cluster.layer() ) {
    case PFLayer::ECAL_BARREL:
    case PFLayer::HCAL_BARREL1:
      factor = std::sqrt(cluster.position().Perp2()/(particleX*particleX+particleY*particleY));
      break;
    case PFLayer::ECAL_ENDCAP:
    case PFLayer::HCAL_ENDCAP:
    case PFLayer::HF_HAD:
    case PFLayer::HF_EM:
      factor = cluster.position().Z()/particleZ;
      break;
    default:
      assert(0);
    }
  }
  //MIKE First of all let's check if we have vertex.
  ::math::XYZPoint vertexPos; 
  if(useVertices_)
    vertexPos = ::math::XYZPoint(primaryVertex_.x(),primaryVertex_.y(),primaryVertex_.z());
  else
    vertexPos = ::math::XYZPoint(0.0,0.0,0.0);


  ::math::XYZVector clusterPos( cluster.position().X()-vertexPos.X(), 
			      cluster.position().Y()-vertexPos.Y(),
			      cluster.position().Z()-vertexPos.Z());
  ::math::XYZVector particleDirection ( particleX*factor-vertexPos.X(), 
				      particleY*factor-vertexPos.Y(), 
				      particleZ*factor-vertexPos.Z() );

  //::math::XYZVector clusterPos( cluster.position().X(), cluster.position().Y(),cluster.position().Z() );
  //::math::XYZVector particleDirection ( particleX, particleY, particleZ );

  clusterPos = useDirection ? particleDirection.Unit() : clusterPos.Unit();
  clusterPos *= particleEnergy;

  // clusterPos is now a vector along the cluster direction, 
  // with a magnitude equal to the cluster energy.
  
  double mass = 0;
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > 
    momentum( clusterPos.X(), clusterPos.Y(), clusterPos.Z(), mass); 
  // mathcore is a piece of #$%
  ::math::XYZTLorentzVector  tmp;
  // implicit constructor not allowed
  tmp = momentum;

  // Charge
  int charge = 0;

  // Type
  switch( cluster.layer() ) {
  case PFLayer::ECAL_BARREL:
  case PFLayer::ECAL_ENDCAP:
    particleType = PFCandidate::gamma;
    break;
  case PFLayer::HCAL_BARREL1:
  case PFLayer::HCAL_ENDCAP:    
    particleType = PFCandidate::h0;
    break;
  case PFLayer::HF_HAD:
    particleType = PFCandidate::h_HF;
    break;
  case PFLayer::HF_EM:
    particleType = PFCandidate::egamma_HF;
    break;
  default:
    assert(0);
  }

  // The pf candidate
  pfCandidates_->push_back( PFCandidate( charge, 
                                         tmp, 
                                         particleType ) );

  // The position at ECAL entrance (well: watch out, it is not true
  // for HCAL clusters... to be fixed)
  pfCandidates_->back().
    setPositionAtECALEntrance(::math::XYZPointF(cluster.position().X(),
					      cluster.position().Y(),
					      cluster.position().Z()));

  //Set the cnadidate Vertex
  pfCandidates_->back().setVertex(vertexPos);  

  if(debug_) 
    cout<<"** candidate: "<<pfCandidates_->back()<<endl; 

  // returns index to the newly created PFCandidate
  return pfCandidates_->size()-1;

}


//GMA need the followign two for HO also

double 
PFAlgo::neutralHadronEnergyResolution(double clusterEnergyHCAL, double eta) const {

  // Add a protection
  if ( clusterEnergyHCAL < 1. ) clusterEnergyHCAL = 1.;

  double resol =  fabs(eta) < 1.48 ? 
    sqrt (1.02*1.02/clusterEnergyHCAL + 0.065*0.065)
    :
    sqrt (1.20*1.20/clusterEnergyHCAL + 0.028*0.028);

  return resol;

}

double
PFAlgo::nSigmaHCAL(double clusterEnergyHCAL, double eta) const {
  double nS = fabs(eta) < 1.48 ? 
    nSigmaHCAL_ * (1. + exp(-clusterEnergyHCAL/100.))     
    :
    nSigmaHCAL_ * (1. + exp(-clusterEnergyHCAL/100.));     
  
  return nS;
}


ostream& operator<<(ostream& out, const PFAlgo& algo) {
  if(!out ) return out;

  out<<"====== Particle Flow Algorithm ======= ";
  out<<endl;
  out<<"nSigmaECAL_     "<<algo.nSigmaECAL_<<endl;
  out<<"nSigmaHCAL_     "<<algo.nSigmaHCAL_<<endl;
  out<<endl;
  out<<(*(algo.calibration_))<<endl;
  out<<endl;
  out<<"reconstructed particles: "<<endl;
   
  const std::auto_ptr< reco::PFCandidateCollection >& 
    candidates = algo.pfCandidates(); 

  if(!candidates.get() ) {
    out<<"candidates already transfered"<<endl;
    return out;
  }
  for(PFCandidateConstIterator ic=algo.pfCandidates_->begin(); 
      ic != algo.pfCandidates_->end(); ic++) {
    out<<(*ic)<<endl;
  }

  return out;
}

reco::PFBlockRef 
PFAlgo::createBlockRef( const reco::PFBlockCollection& blocks, 
			unsigned bi ) {

  if( blockHandle_.isValid() ) {
    return reco::PFBlockRef(  blockHandle_, bi );
  } 
  else {
    return reco::PFBlockRef(  &blocks, bi );
  }
}

void 
PFAlgo::associatePSClusters(unsigned iEcal,
			    reco::PFBlockElement::Type psElementType,
			    const reco::PFBlock& block,
			    const edm::OwnVector< reco::PFBlockElement >& elements, 
			    const reco::PFBlock::LinkData& linkData,
			    std::vector<bool>& active, 
			    std::vector<double>& psEne) {

  // Find all PS clusters with type psElement associated to ECAL cluster iEcal, 
  // within all PFBlockElement "elements" of a given PFBlock "block"
  // psElement can be reco::PFBlockElement::PS1 or reco::PFBlockElement::PS2
  // Returns a vector of PS cluster energies, and updates the "active" vector.

  // Find all PS clusters linked to the iEcal cluster
  std::multimap<double, unsigned> sortedPS;
  typedef std::multimap<double, unsigned>::iterator IE;
  block.associatedElements( iEcal,  linkData,
			    sortedPS, psElementType,
			    reco::PFBlock::LINKTEST_ALL );

  // Loop over these PS clusters
  double totalPS = 0.;
  for ( IE ips=sortedPS.begin(); ips!=sortedPS.end(); ++ips ) {

    // CLuster index and distance to iEcal
    unsigned iPS = ips->second;
    // double distPS = ips->first;

    // Ignore clusters already in use
    if (!active[iPS]) continue;

    // Check that this cluster is not closer to another ECAL cluster
    std::multimap<double, unsigned> sortedECAL;
    block.associatedElements( iPS,  linkData,
			      sortedECAL,
			      reco::PFBlockElement::ECAL,
			      reco::PFBlock::LINKTEST_ALL );
    unsigned jEcal = sortedECAL.begin()->second;
    if ( jEcal != iEcal ) continue; 
    
    // Update PS energy
    PFBlockElement::Type pstype = elements[ iPS ].type();
    assert( pstype == psElementType );
    PFClusterRef psclusterref = elements[iPS].clusterRef();
    assert(!psclusterref.isNull() );
    totalPS += psclusterref->energy(); 
    psEne[0] += psclusterref->energy();
    active[iPS] = false;
  }
	    

}


bool
PFAlgo::isFromSecInt(const reco::PFBlockElement& eTrack, string order) const {

  reco::PFBlockElement::TrackType T_TO_DISP = reco::PFBlockElement::T_TO_DISP;
  reco::PFBlockElement::TrackType T_FROM_DISP = reco::PFBlockElement::T_FROM_DISP;
  //  reco::PFBlockElement::TrackType T_FROM_GAMMACONV = reco::PFBlockElement::T_FROM_GAMMACONV;
  reco::PFBlockElement::TrackType T_FROM_V0 = reco::PFBlockElement::T_FROM_V0;

  bool bPrimary = (order.find("primary") != string::npos);
  bool bSecondary = (order.find("secondary") != string::npos);
  bool bAll = (order.find("all") != string::npos);

  bool isToDisp = usePFNuclearInteractions_ && eTrack.trackType(T_TO_DISP);
  bool isFromDisp = usePFNuclearInteractions_ && eTrack.trackType(T_FROM_DISP);

  if (bPrimary && isToDisp) return true;
  if (bSecondary && isFromDisp ) return true;
  if (bAll && ( isToDisp || isFromDisp ) ) return true;

//   bool isFromConv = usePFConversions_ && eTrack.trackType(T_FROM_GAMMACONV);

//   if ((bAll || bSecondary)&& isFromConv) return true;

  bool isFromDecay = (bAll || bSecondary) && usePFDecays_ && eTrack.trackType(T_FROM_V0);

  return isFromDecay;


}


void  
PFAlgo::setEGElectronCollection(const reco::GsfElectronCollection & egelectrons) {
  if(useEGElectrons_ && pfele_) pfele_->setEGElectronCollection(egelectrons);
}

void 
PFAlgo::postCleaning() { 
  
  // Check if the post HF Cleaning was requested - if not, do nothing
  if ( !postHFCleaning_ ) return;

  //Compute met and met significance (met/sqrt(SumEt))
  double metX = 0.;
  double metY = 0.;
  double sumet = 0;
  std::vector<unsigned int> pfCandidatesToBeRemoved;
  for(unsigned i=0; i<pfCandidates_->size(); i++) {
    const PFCandidate& pfc = (*pfCandidates_)[i];
    metX += pfc.px();
    metY += pfc.py();
    sumet += pfc.pt();
  }    
  double met2 = metX*metX+metY*metY;
  // Select events with large MET significance.
  double significance = std::sqrt(met2/sumet);
  double significanceCor = significance;
  if ( significance > minSignificance_ ) { 
    
    double metXCor = metX;
    double metYCor = metY;
    double sumetCor = sumet;
    double met2Cor = met2;
    double deltaPhi = 3.14159;
    double deltaPhiPt = 100.;
    bool next = true;
    unsigned iCor = 1E9;

    // Find the HF candidate with the largest effect on the MET
    while ( next ) { 

      double metReduc = -1.;
      // Loop on the candidates
      for(unsigned i=0; i<pfCandidates_->size(); ++i) {
	const PFCandidate& pfc = (*pfCandidates_)[i];

	// Check that the pfCandidate is in the HF
	if ( pfc.particleId() != reco::PFCandidate::h_HF && 
	     pfc.particleId() != reco::PFCandidate::egamma_HF ) continue;

	// Check if has meaningful pt
	if ( pfc.pt() < minHFCleaningPt_ ) continue;
	
	// Check that it is  not already scheculed to be cleaned
	bool skip = false;
	for(unsigned j=0; j<pfCandidatesToBeRemoved.size(); ++j) {
	  if ( i == pfCandidatesToBeRemoved[j] ) skip = true;
	  if ( skip ) break;
	}
	if ( skip ) continue;
	
	// Check that the pt and the MET are aligned
	deltaPhi = std::acos((metX*pfc.px()+metY*pfc.py())/(pfc.pt()*std::sqrt(met2)));
	deltaPhiPt = deltaPhi*pfc.pt();
	if ( deltaPhiPt > maxDeltaPhiPt_ ) continue;

	// Now remove the candidate from the MET
	double metXInt = metX - pfc.px();
	double metYInt = metY - pfc.py();
	double sumetInt = sumet - pfc.pt();
	double met2Int = metXInt*metXInt+metYInt*metYInt;
	if ( met2Int < met2Cor ) {
	  metXCor = metXInt;
	  metYCor = metYInt;
	  metReduc = (met2-met2Int)/met2Int; 
	  met2Cor = met2Int;
	  sumetCor = sumetInt;
	  significanceCor = std::sqrt(met2Cor/sumetCor);
	  iCor = i;
	}
      }
      //
      // If the MET must be significanly reduced, schedule the candidate to be cleaned
      if ( metReduc > minDeltaMet_ ) { 
	pfCandidatesToBeRemoved.push_back(iCor);
	metX = metXCor;
	metY = metYCor;
	sumet = sumetCor;
	met2 = met2Cor;
      } else { 
      // Otherwise just stop the loop
	next = false;
      }
    }
    //
    // The significance must be significantly reduced to indeed clean the candidates
    if ( significance - significanceCor > minSignificanceReduction_ && 
	 significanceCor < maxSignificance_ ) {
      std::cout << "Significance reduction = " << significance << " -> " 
		<< significanceCor << " = " << significanceCor - significance 
		<< std::endl;
      for(unsigned j=0; j<pfCandidatesToBeRemoved.size(); ++j) {
	std::cout << "Removed : " << (*pfCandidates_)[pfCandidatesToBeRemoved[j]] << std::endl;
	pfCleanedCandidates_->push_back( (*pfCandidates_)[ pfCandidatesToBeRemoved[j] ] );
	(*pfCandidates_)[pfCandidatesToBeRemoved[j]].rescaleMomentum(1E-6);
	//reco::PFCandidate::ParticleType unknown = reco::PFCandidate::X;
	//(*pfCandidates_)[pfCandidatesToBeRemoved[j]].setParticleType(unknown);
      }
    }
  }

}

void 
PFAlgo::checkCleaning( const reco::PFRecHitCollection& cleanedHits ) { 
  

  // No hits to recover, leave.
  if ( !cleanedHits.size() ) return;

  //Compute met and met significance (met/sqrt(SumEt))
  double metX = 0.;
  double metY = 0.;
  double sumet = 0;
  std::vector<unsigned int> hitsToBeAdded;
  for(unsigned i=0; i<pfCandidates_->size(); i++) {
    const PFCandidate& pfc = (*pfCandidates_)[i];
    metX += pfc.px();
    metY += pfc.py();
    sumet += pfc.pt();
  }
  double met2 = metX*metX+metY*metY;
  double met2_Original = met2;
  // Select events with large MET significance.
  // double significance = std::sqrt(met2/sumet);
  // double significanceCor = significance;
  double metXCor = metX;
  double metYCor = metY;
  double sumetCor = sumet;
  double met2Cor = met2;
  bool next = true;
  unsigned iCor = 1E9;
  
  // Find the cleaned hit with the largest effect on the MET
  while ( next ) { 
    
    double metReduc = -1.;
    // Loop on the candidates
    for(unsigned i=0; i<cleanedHits.size(); ++i) {
      const PFRecHit& hit = cleanedHits[i];
      double length = std::sqrt(hit.position().Mag2()); 
      double px = hit.energy() * hit.position().x()/length;
      double py = hit.energy() * hit.position().y()/length;
      double pt = std::sqrt(px*px + py*py);
      
      // Check that it is  not already scheculed to be cleaned
      bool skip = false;
      for(unsigned j=0; j<hitsToBeAdded.size(); ++j) {
	if ( i == hitsToBeAdded[j] ) skip = true;
	if ( skip ) break;
      }
      if ( skip ) continue;
      
      // Now add the candidate to the MET
      double metXInt = metX + px;
      double metYInt = metY + py;
      double sumetInt = sumet + pt;
      double met2Int = metXInt*metXInt+metYInt*metYInt;

      // And check if it could contribute to a MET reduction
      if ( met2Int < met2Cor ) {
	metXCor = metXInt;
	metYCor = metYInt;
	metReduc = (met2-met2Int)/met2Int; 
	met2Cor = met2Int;
	sumetCor = sumetInt;
	// significanceCor = std::sqrt(met2Cor/sumetCor);
	iCor = i;
      }
    }
    //
    // If the MET must be significanly reduced, schedule the candidate to be added
    //
    if ( metReduc > minDeltaMet_ ) { 
      hitsToBeAdded.push_back(iCor);
      metX = metXCor;
      metY = metYCor;
      sumet = sumetCor;
      met2 = met2Cor;
    } else { 
      // Otherwise just stop the loop
      next = false;
    }
  }
  //
  // At least 10 GeV MET reduction
  if ( std::sqrt(met2_Original) - std::sqrt(met2) > 5. ) { 
    if ( debug_ ) { 
      std::cout << hitsToBeAdded.size() << " hits were re-added " << std::endl;
      std::cout << "MET reduction = " << std::sqrt(met2_Original) << " -> " 
		<< std::sqrt(met2Cor) << " = " <<  std::sqrt(met2Cor) - std::sqrt(met2_Original)  
		<< std::endl;
      std::cout << "Added after cleaning check : " << std::endl; 
    }
    for(unsigned j=0; j<hitsToBeAdded.size(); ++j) {
      const PFRecHit& hit = cleanedHits[hitsToBeAdded[j]];
      PFCluster cluster(hit.layer(), hit.energy(),
			hit.position().x(), hit.position().y(), hit.position().z() );
      reconstructCluster(cluster,hit.energy());
      if ( debug_ ) { 
	std::cout << pfCandidates_->back() << ". time = " << hit.time() << std::endl;
      }
    }
  }

}



void PFAlgo::setElectronExtraRef(const edm::OrphanHandle<reco::PFCandidateElectronExtraCollection >& extrah) {
  if(!usePFElectrons_) return;
  //  std::cout << " setElectronExtraRef " << std::endl;
  unsigned size=pfCandidates_->size();

  for(unsigned ic=0;ic<size;++ic) {
    // select the electrons and add the extra
    if((*pfCandidates_)[ic].particleId()==PFCandidate::e) {
      
      PFElectronExtraEqual myExtraEqual((*pfCandidates_)[ic].gsfTrackRef());
      std::vector<PFCandidateElectronExtra>::const_iterator it=find_if(pfElectronExtra_.begin(),pfElectronExtra_.end(),myExtraEqual);
      if(it!=pfElectronExtra_.end()) {
	//	std::cout << " Index " << it-pfElectronExtra_.begin() << std::endl;
	reco::PFCandidateElectronExtraRef theRef(extrah,it-pfElectronExtra_.begin());
	(*pfCandidates_)[ic].setPFElectronExtraRef(theRef);
      }
      else {
	(*pfCandidates_)[ic].setPFElectronExtraRef(PFCandidateElectronExtraRef());
      }
    }
    else  // else save the mva and the extra as well ! 
      {
	if((*pfCandidates_)[ic].trackRef().isNonnull()) {
	  PFElectronExtraKfEqual myExtraEqual((*pfCandidates_)[ic].trackRef());
	  std::vector<PFCandidateElectronExtra>::const_iterator it=find_if(pfElectronExtra_.begin(),pfElectronExtra_.end(),myExtraEqual);
	  if(it!=pfElectronExtra_.end()) {
	    (*pfCandidates_)[ic].set_mva_e_pi(it->mvaVariable(PFCandidateElectronExtra::MVA_MVA));
	    reco::PFCandidateElectronExtraRef theRef(extrah,it-pfElectronExtra_.begin());
	    (*pfCandidates_)[ic].setPFElectronExtraRef(theRef);
	    (*pfCandidates_)[ic].setGsfTrackRef(it->gsfTrackRef());
	  }	
	}
      }

  }

  size=pfElectronCandidates_->size();
  for(unsigned ic=0;ic<size;++ic) {
    // select the electrons - this test is actually not needed for this collection
    if((*pfElectronCandidates_)[ic].particleId()==PFCandidate::e) {
      // find the corresponding extra
      PFElectronExtraEqual myExtraEqual((*pfElectronCandidates_)[ic].gsfTrackRef());
      std::vector<PFCandidateElectronExtra>::const_iterator it=find_if(pfElectronExtra_.begin(),pfElectronExtra_.end(),myExtraEqual);
      if(it!=pfElectronExtra_.end()) {
	reco::PFCandidateElectronExtraRef theRef(extrah,it-pfElectronExtra_.begin());
	(*pfElectronCandidates_)[ic].setPFElectronExtraRef(theRef);

      }
    }
  }

}
void PFAlgo::setPhotonExtraRef(const edm::OrphanHandle<reco::PFCandidatePhotonExtraCollection >& ph_extrah) {
  if(!usePFPhotons_) return;  
  unsigned int size=pfCandidates_->size();
  unsigned int sizePhExtra = pfPhotonExtra_.size();
  for(unsigned ic=0;ic<size;++ic) {
    // select the electrons and add the extra
    if((*pfCandidates_)[ic].particleId()==PFCandidate::gamma && (*pfCandidates_)[ic].mva_nothing_gamma() > 0.99) {
      if((*pfCandidates_)[ic].superClusterRef().isNonnull()) {
	bool found = false;
	for(unsigned pcextra=0;pcextra<sizePhExtra;++pcextra) {
	  if((*pfCandidates_)[ic].superClusterRef() == pfPhotonExtra_[pcextra].superClusterRef()) {
	    reco::PFCandidatePhotonExtraRef theRef(ph_extrah,pcextra);
	    (*pfCandidates_)[ic].setPFPhotonExtraRef(theRef);
	    found = true;
	    break;
	  }
	}
	if(!found) 
	  (*pfCandidates_)[ic].setPFPhotonExtraRef((PFCandidatePhotonExtraRef())); // null ref
      }
      else {
	(*pfCandidates_)[ic].setPFPhotonExtraRef((PFCandidatePhotonExtraRef())); // null ref
      }
    }
  }      
}
