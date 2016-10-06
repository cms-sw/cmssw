// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      PrimaryVertexValidation
// 
/**\class PrimaryVertexValidation PrimaryVertexValidation.cc Alignment/OfflineValidation/plugins/PrimaryVertexValidation.cc

 Description: Validate alignment constants using unbiased vertex residuals

 Implementation:
 <Notes on implementation>
*/
//
// Original Author:  Marco Musich
//         Created:  Tue Mar 02 10:39:34 CDT 2010
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "Alignment/OfflineValidation/plugins/PrimaryVertexValidation.h"

// ROOT includes
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TVector3.h"
#include "TFile.h"
#include "TMath.h"
#include "TROOT.h"
#include "TChain.h"
#include "TNtuple.h"
#include "TMatrixD.h"
#include "TVectorD.h"

// CMSSW includes
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ.h"
#include "RecoVertex/PrimaryVertexProducer/interface/GapClusterizerInZ.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

const int PrimaryVertexValidation::nMaxtracks_;

// Constructor
PrimaryVertexValidation::PrimaryVertexValidation(const edm::ParameterSet& iConfig):
  storeNtuple_(iConfig.getParameter<bool>("storeNtuple")),
  lightNtupleSwitch_(iConfig.getParameter<bool>("isLightNtuple")),
  useTracksFromRecoVtx_(iConfig.getParameter<bool>("useTracksFromRecoVtx")),
  vertexZMax_(iConfig.getUntrackedParameter<double>("vertexZMax",99.)),
  askFirstLayerHit_(iConfig.getParameter<bool>("askFirstLayerHit")),
  ptOfProbe_(iConfig.getUntrackedParameter<double>("probePt",0.)),
  etaOfProbe_(iConfig.getUntrackedParameter<double>("probeEta",2.4)),
  nBins_(iConfig.getUntrackedParameter<int>("numberOfBins",24)),
  debug_(iConfig.getParameter<bool>("Debug")),
  runControl_(iConfig.getUntrackedParameter<bool>("runControl",false))
{
  
  // now do what ever initialization is needed
  // initialize phase space boundaries
  
  usesResource(TFileService::kSharedResource);

  std::vector<unsigned int> defaultRuns;
  defaultRuns.push_back(0);
  runControlNumbers_ = iConfig.getUntrackedParameter<std::vector<unsigned int> >("runControlNumber",defaultRuns);

  phiSect_ = (2*TMath::Pi())/nBins_;
  etaSect_ = 5./nBins_;

  edm::InputTag TrackCollectionTag_ = iConfig.getParameter<edm::InputTag>("TrackCollectionTag");
  theTrackCollectionToken = consumes<reco::TrackCollection>(TrackCollectionTag_);

  edm::InputTag VertexCollectionTag_ = iConfig.getParameter<edm::InputTag>("VertexCollectionTag");
  theVertexCollectionToken = consumes<reco::VertexCollection>(VertexCollectionTag_);

  edm::InputTag BeamspotTag_ = edm::InputTag("offlineBeamSpot");
  theBeamspotToken = consumes<reco::BeamSpot>(BeamspotTag_);

  // select and configure the track filter 
  theTrackFilter_= new TrackFilterForPVFinding(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters") );
  // select and configure the track clusterizer  
  std::string clusteringAlgorithm=iConfig.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<std::string>("algorithm");
  if (clusteringAlgorithm=="gap"){
    theTrackClusterizer_ = new GapClusterizerInZ(iConfig.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkGapClusParameters"));
  }else if(clusteringAlgorithm=="DA"){
    theTrackClusterizer_ = new DAClusterizerInZ(iConfig.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
  }else{
    throw VertexException("PrimaryVertexProducerAlgorithm: unknown clustering algorithm: " + clusteringAlgorithm);  
  }
}
   
// Destructor
PrimaryVertexValidation::~PrimaryVertexValidation()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PrimaryVertexValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace std;
  using namespace reco;
  using namespace IPTools;

  if(nBins_!=24){ 
    edm::LogInfo("PrimaryVertexValidation")<<"Using: "<<nBins_<<" bins plots";
  }
  
  bool passesRunControl = false;

  if(runControl_){
    for(unsigned int j=0;j<runControlNumbers_.size();j++){
      if(iEvent.eventAuxiliary().run() == runControlNumbers_[j]){ 
	if (debug_){
	  edm::LogInfo("PrimaryVertexValidation")<<" run number: "<<iEvent.eventAuxiliary().run()<<" keeping run:"<<runControlNumbers_[j];
	}
	passesRunControl = true;
	break;
      }
    }
    if (!passesRunControl) return;
  }

  Nevt_++;  

  //=======================================================
  // Initialize Root-tuple variables
  //=======================================================

  SetVarToZero();
 
  //=======================================================
  // Retrieve the Magnetic Field information
  //=======================================================

  edm::ESHandle<MagneticField> theMGField;
  iSetup.get<IdealMagneticFieldRecord>().get( theMGField );

  //=======================================================
  // Retrieve the Tracking Geometry information
  //=======================================================

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get( theTrackingGeometry );

  //=======================================================
  // Retrieve the Transient Track Builder information
  //=======================================================

  edm::ESHandle<TransientTrackBuilder> theB_;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB_);
  double fBfield_=((*theB_).field()->inTesla(GlobalPoint(0.,0.,0.))).z();

  //=======================================================
  // Retrieve the Track information
  //=======================================================
  
  edm::Handle<TrackCollection>  trackCollectionHandle;
  iEvent.getByToken(theTrackCollectionToken, trackCollectionHandle);
  
  //=======================================================
  // Retrieve offline vartex information (only for reco)
  //=======================================================
 
  //edm::Handle<VertexCollection> vertices;
  edm::Handle<std::vector<Vertex> > vertices;

  try {
    iEvent.getByToken(theVertexCollectionToken, vertices);
  } catch ( cms::Exception& er ) {
    LogTrace("PrimaryVertexValidation")<<"caught std::exception "<<er.what()<<std::endl;  
  }

  std::vector<Vertex> vsorted = *(vertices);
  // sort the vertices by number of tracks in descending order
  // use chi2 as tiebreaker
  std::sort( vsorted.begin(), vsorted.end(), PrimaryVertexValidation::vtxSort );
  
  // skip events with no PV, this should not happen
  if( vsorted.size() == 0) return;
  // skip events failing vertex cut
  if( fabs(vsorted[0].z()) > vertexZMax_ ) return; 
  
  if ( vsorted[0].isValid() ) {
    xOfflineVertex_ = (vsorted)[0].x();
    yOfflineVertex_ = (vsorted)[0].y();
    zOfflineVertex_ = (vsorted)[0].z();

    xErrOfflineVertex_ = (vsorted)[0].xError();
    yErrOfflineVertex_ = (vsorted)[0].yError();
    zErrOfflineVertex_ = (vsorted)[0].zError();
  }

  h_xOfflineVertex->Fill(xOfflineVertex_);    
  h_yOfflineVertex->Fill(yOfflineVertex_);     
  h_zOfflineVertex->Fill(zOfflineVertex_);     
  h_xErrOfflineVertex->Fill(xErrOfflineVertex_);  
  h_yErrOfflineVertex->Fill(yErrOfflineVertex_);  
  h_zErrOfflineVertex->Fill(zErrOfflineVertex_);  

  unsigned int vertexCollectionSize = vsorted.size();
  int nvvertex = 0;
  
  for (unsigned int i=0; i<vertexCollectionSize; i++) {
    const Vertex& vertex = vsorted.at(i);
    if (vertex.isValid()) nvvertex++;
  }

  nOfflineVertices_ = nvvertex;
  h_nOfflineVertices->Fill(nvvertex);

  if ( vsorted.size() && useTracksFromRecoVtx_ ) {
   
    double sumpt    = 0;
    size_t ntracks  = 0;
    double chi2ndf  = 0.; 
    double chi2prob = 0.;

    if (!vsorted.at(0).isFake()) {
      
      Vertex pv = vsorted.at(0);
      
      ntracks  = pv.tracksSize();
      chi2ndf  = pv.normalizedChi2();
      chi2prob = TMath::Prob(pv.chi2(),(int)pv.ndof());
      
      h_recoVtxNtracks_->Fill(ntracks);   
      h_recoVtxChi2ndf_->Fill(chi2ndf);    
      h_recoVtxChi2Prob_->Fill(chi2prob);   
      
      for (Vertex::trackRef_iterator itrk = pv.tracks_begin();itrk != pv.tracks_end(); ++itrk) {
	double pt = (**itrk).pt();
	sumpt += pt*pt;
	
	const math::XYZPoint myVertex(pv.position().x(),pv.position().y(),pv.position().z());
	
	double dxyRes = (**itrk).dxy(myVertex);
	double dzRes  = (**itrk).dz(myVertex);
	
	double dxy_err = (**itrk).dxyError();
	double dz_err  = (**itrk).dzError();
	
	float trackphi = ((**itrk).phi())*(180/TMath::Pi());
	float tracketa = (**itrk).eta();
	
	for(int i=0; i<nBins_; i++){
	  
	  float phiF = (-TMath::Pi()+i*phiSect_)*(180/TMath::Pi());
	  float phiL = (-TMath::Pi()+(i+1)*phiSect_)*(180/TMath::Pi());
	  
	  float etaF=-2.5+i*etaSect_;
	  float etaL=-2.5+(i+1)*etaSect_;
	  
	  if(tracketa >= etaF && tracketa < etaL ){

	    a_dxyEtaBiasResiduals[i]->Fill(dxyRes*cmToum);
	    a_dzEtaBiasResiduals[i]->Fill(dzRes*cmToum); 
	    n_dxyEtaBiasResiduals[i]->Fill((dxyRes)/dxy_err);
	    n_dzEtaBiasResiduals[i]->Fill((dzRes)/dz_err);

	  }

	  if(trackphi >= phiF && trackphi < phiL ){ 

	    a_dxyPhiBiasResiduals[i]->Fill(dxyRes*cmToum);
	    a_dzPhiBiasResiduals[i]->Fill(dzRes*cmToum); 
	    n_dxyPhiBiasResiduals[i]->Fill((dxyRes)/dxy_err);
	    n_dzPhiBiasResiduals[i]->Fill((dzRes)/dz_err); 
	    
	    for(int j=0; j<nBins_; j++){
	      
	      float etaJ=-2.5+j*etaSect_;
	      float etaK=-2.5+(j+1)*etaSect_;
	      
	      if(tracketa >= etaJ && tracketa < etaK ){
		
		a_dxyBiasResidualsMap[i][j]->Fill(dxyRes*cmToum); 
		a_dzBiasResidualsMap[i][j]->Fill(dzRes*cmToum);   
		
		n_dxyBiasResidualsMap[i][j]->Fill((dxyRes)/dxy_err); 
		n_dzBiasResidualsMap[i][j]->Fill((dzRes)/dz_err);  
		
	      }
	    }
	  }		
	}
      }
      
      h_recoVtxSumPt_->Fill(sumpt);   

    }
  }

  //=======================================================
  // Retrieve Beamspot information
  //=======================================================

  BeamSpot beamSpot;
  edm::Handle<BeamSpot> beamSpotHandle;
  iEvent.getByToken(theBeamspotToken, beamSpotHandle);
    
  if ( beamSpotHandle.isValid() ) {
    beamSpot = *beamSpotHandle;
    BSx0_    = beamSpot.x0();
    BSy0_    = beamSpot.y0();
    BSz0_    = beamSpot.z0();
    Beamsigmaz_ = beamSpot.sigmaZ();    
    Beamdxdz_   = beamSpot.dxdz();	     
    BeamWidthX_ = beamSpot.BeamWidthX();
    BeamWidthY_ = beamSpot.BeamWidthY();

    wxy2_=TMath::Power(BeamWidthX_,2)+TMath::Power(BeamWidthY_,2);

  } else {
    edm::LogWarning("PrimaryVertexValidation")<<"No BeamSpot found!";
  }

  h_BSx0->Fill(BSx0_);      
  h_BSy0->Fill(BSy0_);     
  h_BSz0->Fill(BSz0_);      
  h_Beamsigmaz->Fill(Beamsigmaz_);
  h_BeamWidthX->Fill(BeamWidthX_);
  h_BeamWidthY->Fill(BeamWidthY_);

  if(debug_)
    edm::LogInfo("PrimaryVertexValidation")<<"Beamspot x:" <<BSx0_<<" y:"<<BSy0_<<" z:"<<BSz0_;
   
  //=======================================================
  // Starts here ananlysis
  //=======================================================
  
  RunNumber_=iEvent.eventAuxiliary().run();
  h_runNumber->Fill(RunNumber_);
  LuminosityBlockNumber_=iEvent.eventAuxiliary().luminosityBlock();
  EventNumber_=iEvent.eventAuxiliary().id().event();
    
  if(debug_)
    edm::LogInfo("PrimaryVertexValidation")<<" looping over "<<trackCollectionHandle->size()<< "tracks";

  h_nTracks->Fill(trackCollectionHandle->size()); 

  //======================================================
  // Interface RECO tracks to vertex reconstruction
  //======================================================
 
  std::vector<TransientTrack> t_tks;
  unsigned int k = 0;   
  for(TrackCollection::const_iterator track = trackCollectionHandle->begin(); track!= trackCollectionHandle->end(); ++track, ++k){
  
    TransientTrack tt = theB_->build(&(*track));  
    tt.setBeamSpot(beamSpot);
    t_tks.push_back(tt);
  
  }
  
  if(debug_) {
    edm::LogInfo("PrimaryVertexValidation") << "Found: " << t_tks.size() << " reconstructed tracks";
  }
  
  //======================================================
  // select the tracks
  //======================================================

  std::vector<TransientTrack> seltks = theTrackFilter_->select(t_tks);
    
  //======================================================
  // clusterize tracks in Z
  //======================================================

  vector< vector<TransientTrack> > clusters = theTrackClusterizer_->clusterize(seltks);
  
  if (debug_){
    edm::LogInfo("PrimaryVertexValidation")<<" looping over: "<< clusters.size() << " clusters  from " << t_tks.size() << " selected tracks";
  }
  
  nClus_=clusters.size();  
  h_nClus->Fill(nClus_);

  //======================================================
  // Starts loop on clusters 
  //======================================================

  for (vector< vector<TransientTrack> >::const_iterator iclus = clusters.begin(); iclus != clusters.end(); iclus++) {

    nTracksPerClus_=0;

    unsigned int i = 0;   
    for(vector<TransientTrack>::const_iterator theTTrack = iclus->begin(); theTTrack!= iclus->end(); ++theTTrack, ++i)
      {
	if ( nTracks_ >= nMaxtracks_ ) {
	  edm::LogError("PrimaryVertexValidation")<<" Warning - Number of tracks: " << nTracks_ << " , greater than " << nMaxtracks_;
	  continue;
	}
	
	const Track & theTrack = theTTrack->track();

	pt_[nTracks_]       = theTrack.pt();
	p_[nTracks_]        = theTrack.p();
	nhits_[nTracks_]    = theTrack.numberOfValidHits();
	eta_[nTracks_]      = theTrack.eta();
	theta_[nTracks_]    = theTrack.theta();
	phi_[nTracks_]      = theTrack.phi();
	chi2_[nTracks_]     = theTrack.chi2();
	chi2ndof_[nTracks_] = theTrack.normalizedChi2();
	charge_[nTracks_]   = theTrack.charge();
	qoverp_[nTracks_]   = theTrack.qoverp();
	dz_[nTracks_]       = theTrack.dz();
	dxy_[nTracks_]      = theTrack.dxy();
	
	TrackBase::TrackQuality _trackQuality = TrackBase::qualityByName("highPurity");
	isHighPurity_[nTracks_] = theTrack.quality(_trackQuality);
	
	math::XYZPoint point(BSx0_,BSy0_,BSz0_);
	dxyBs_[nTracks_]    = theTrack.dxy(point);
	dzBs_[nTracks_]     = theTrack.dz(point);

	xPCA_[nTracks_]     = theTrack.vertex().x();
	yPCA_[nTracks_]     = theTrack.vertex().y();
	zPCA_[nTracks_]     = theTrack.vertex().z(); 
	
	//=======================================================
	// Retrieve rechit information
	//=======================================================  
	
	int nRecHit1D=0;
	int nRecHit2D=0;
	int nhitinTIB=0; 
	int nhitinTOB=0; 
	int nhitinTID=0; 
	int nhitinTEC=0;
	int nhitinBPIX=0;
	int nhitinFPIX=0;
	
	for (trackingRecHit_iterator iHit = theTTrack->recHitsBegin(); iHit != theTTrack->recHitsEnd(); ++iHit) {
	  if((*iHit)->isValid()) {	
	    
	    if (this->isHit2D(**iHit)) {++nRecHit2D;}
	    else {++nRecHit1D; }
	    
	    int type =(*iHit)->geographicalId().subdetId();
	    
	    if(type==int(StripSubdetector::TIB)){++nhitinTIB;}
	    if(type==int(StripSubdetector::TOB)){++nhitinTOB;}
	    if(type==int(StripSubdetector::TID)){++nhitinTID;}
	    if(type==int(StripSubdetector::TEC)){++nhitinTEC;}
	    if(type==int(                kBPIX)){++nhitinBPIX;}
	    if(type==int(                kFPIX)){++nhitinFPIX;}
	  }
	}      

	nhits1D_[nTracks_]     = nRecHit1D;
	nhits2D_[nTracks_]     = nRecHit2D;
	nhitsBPIX_[nTracks_]   = nhitinBPIX;
	nhitsFPIX_[nTracks_]   = nhitinFPIX;
	nhitsTIB_[nTracks_]    = nhitinTIB;
	nhitsTID_[nTracks_]    = nhitinTID;
	nhitsTOB_[nTracks_]    = nhitinTOB;
	nhitsTEC_[nTracks_]    = nhitinTEC;
	
	//=======================================================
	// Good tracks for vertexing selection
	//=======================================================  

	bool pass = true;
	if(askFirstLayerHit_) pass = this->hasFirstLayerPixelHits((*theTTrack));
	if (pass && (theTrack.pt() >=ptOfProbe_) && fabs(theTrack.eta()) <= etaOfProbe_){
	  isGoodTrack_[nTracks_]=1;
	}
      
	//=======================================================
	// Fit unbiased vertex
	//=======================================================
	
	vector<TransientTrack> theFinalTracks;
	theFinalTracks.clear();

	for(vector<TransientTrack>::const_iterator tk = iclus->begin(); tk!= iclus->end(); ++tk){
	  
	  pass = this->hasFirstLayerPixelHits((*tk));
	  if (pass){
	    if( tk == theTTrack ) continue;
	    else {
	      theFinalTracks.push_back((*tk));
	    }
	  }
	}
	
	if(theFinalTracks.size() > 1){
	    
	  if(debug_)
	    edm::LogInfo("PrimaryVertexValidation")<<"Transient Track Collection size: "<<theFinalTracks.size();
	  try{
	      
	    VertexFitter<5>* theFitter = new AdaptiveVertexFitter;
	    TransientVertex theFittedVertex = theFitter->vertex(theFinalTracks);

	    //AdaptiveVertexFitter* theFitter = new AdaptiveVertexFitter;
	    //TransientVertex theFittedVertex = theFitter->vertex(theFinalTracks,beamSpot);  // if you want the beam constraint

	    double totalTrackWeights=0;
	    if(theFittedVertex.isValid ()){

	      
	      if(theFittedVertex.hasTrackWeight()){
		for(size_t rtracks= 0; rtracks < theFinalTracks.size(); rtracks++){
		  sumOfWeightsUnbiasedVertex_[nTracks_] += theFittedVertex.trackWeight(theFinalTracks[rtracks]);
		  totalTrackWeights+= theFittedVertex.trackWeight(theFinalTracks[rtracks]);
		  h_fitVtxTrackWeights_->Fill(theFittedVertex.trackWeight(theFinalTracks[rtracks]));
		}
	      }
	      
	      h_fitVtxTrackAverageWeight_->Fill(totalTrackWeights/theFinalTracks.size());

	      const math::XYZPoint theRecoVertex(xOfflineVertex_,yOfflineVertex_,zOfflineVertex_);      
	      const math::XYZPoint myVertex(theFittedVertex.position().x(),theFittedVertex.position().y(),theFittedVertex.position().z());

	      const Vertex vertex = theFittedVertex;
	      fillTrackHistos(hDA,"all",&(*theTTrack),vertex,beamSpot,fBfield_);

	      hasRecVertex_[nTracks_]    = 1;
	      xUnbiasedVertex_[nTracks_] = theFittedVertex.position().x();
	      yUnbiasedVertex_[nTracks_] = theFittedVertex.position().y();
	      zUnbiasedVertex_[nTracks_] = theFittedVertex.position().z();
	      
	      chi2normUnbiasedVertex_[nTracks_] = theFittedVertex.normalisedChiSquared();
	      chi2UnbiasedVertex_[nTracks_]     = theFittedVertex.totalChiSquared();
	      DOFUnbiasedVertex_[nTracks_]      = theFittedVertex.degreesOfFreedom();   
	      chi2ProbUnbiasedVertex_[nTracks_] = TMath::Prob(theFittedVertex.totalChiSquared(),(int)theFittedVertex.degreesOfFreedom());
	      tracksUsedForVertexing_[nTracks_] = theFinalTracks.size();

	      h_fitVtxNtracks_->Fill(theFinalTracks.size());        
	      h_fitVtxChi2_->Fill(theFittedVertex.totalChiSquared());
	      h_fitVtxNdof_->Fill(theFittedVertex.degreesOfFreedom());
	      h_fitVtxChi2ndf_->Fill(theFittedVertex.normalisedChiSquared());        
	      h_fitVtxChi2Prob_->Fill(TMath::Prob(theFittedVertex.totalChiSquared(),(int)theFittedVertex.degreesOfFreedom()));       
	            
	      // from my Vertex
	      double dxyFromMyVertex = theTrack.dxy(myVertex);
	      double dzFromMyVertex  = theTrack.dz(myVertex);
          
	      double dz_err = hypot(theTrack.dzError(),theFittedVertex.positionError().czz());

	      // PV2D 
	      std::pair<bool,Measurement1D> s_ip2dpv = signedTransverseImpactParameter(*theTTrack,
										       GlobalVector(theTrack.px(),
												    theTrack.py(),
												    theTrack.pz()),
										       theFittedVertex);            
	      
	      double s_ip2dpv_corr = s_ip2dpv.second.value();
	      double s_ip2dpv_err  = s_ip2dpv.second.error();
	      
	      // PV3D
	      std::pair<bool, Measurement1D> s_ip3dpv = signedImpactParameter3D(*theTTrack,		    
										GlobalVector(theTrack.px(),  
											     theTrack.py(),  
											     theTrack.pz()), 
										theFittedVertex);            
	      
	      double s_ip3dpv_corr = s_ip3dpv.second.value();
	      double s_ip3dpv_err  = s_ip3dpv.second.error();

	      // PV3D absolute
	      std::pair<bool,Measurement1D> ip3dpv = absoluteImpactParameter3D(*theTTrack,theFittedVertex);
	      double ip3d_corr = ip3dpv.second.value(); 
	      double ip3d_err  = ip3dpv.second.error(); 
	      
	      // with respect to any specified vertex, such as primary vertex
	      GlobalPoint vert(theFittedVertex.position().x(),theFittedVertex.position().y(),theFittedVertex.position().z());
	      TrajectoryStateClosestToPoint traj = (*theTTrack).trajectoryStateClosestToPoint(vert);

	      double d0 = traj.perigeeParameters().transverseImpactParameter();
	      //double d0_error = traj.perigeeError().transverseImpactParameterError();
	      double z0 = traj.perigeeParameters().longitudinalImpactParameter();
	      double z0_error = traj.perigeeError().longitudinalImpactParameterError();

	      // define IPs
	     
	      dxyFromMyVertex_[nTracks_]      = dxyFromMyVertex; 
	      dxyErrorFromMyVertex_[nTracks_] = s_ip2dpv_err;
	      IPTsigFromMyVertex_[nTracks_]   = dxyFromMyVertex/s_ip2dpv_err;
 
	      dzFromMyVertex_[nTracks_]       = dzFromMyVertex;  
	      dzErrorFromMyVertex_[nTracks_]  = dz_err;
	      IPLsigFromMyVertex_[nTracks_]   = dzFromMyVertex/dz_err;

	      d3DFromMyVertex_[nTracks_]      = ip3d_corr;
	      d3DErrorFromMyVertex_[nTracks_] = ip3d_err;
	      IP3DsigFromMyVertex_[nTracks_]  = (ip3d_corr/ip3d_err);
	     	      
	      // fill directly the histograms of residuals
	      
	      float trackphi = (theTrack.phi())*(180/TMath::Pi());
	      float tracketa = theTrack.eta();
	      float trackpt  = theTrack.pt();

	      // checks on the probe track quality
	      if(trackpt >= ptOfProbe_ && fabs(tracketa)<= etaOfProbe_){

		fillTrackHistos(hDA,"sel",&(*theTTrack),vertex,beamSpot,fBfield_);

		// probe checks
		h_probePt_->Fill(theTrack.pt());
		h_probeP_->Fill(theTrack.p());
		h_probeEta_->Fill(theTrack.eta());
		h_probePhi_->Fill(theTrack.phi());
		h2_probeEtaPhi_->Fill(theTrack.eta(),theTrack.phi());
		h2_probeEtaPt_->Fill(theTrack.eta(),theTrack.pt());

		h_probeChi2_->Fill(theTrack.chi2());
		h_probeNormChi2_->Fill(theTrack.normalizedChi2());
		h_probeCharge_->Fill(theTrack.charge());
		h_probeQoverP_->Fill(theTrack.qoverp());
		h_probeHits_->Fill(theTrack.numberOfValidHits());       
		h_probeHits1D_->Fill(nRecHit1D);
		h_probeHits2D_->Fill(nRecHit2D);
		h_probeHitsInTIB_->Fill(nhitinBPIX);
		h_probeHitsInTOB_->Fill(nhitinFPIX);
		h_probeHitsInTID_->Fill(nhitinTIB);
		h_probeHitsInTEC_->Fill(nhitinTID);
		h_probeHitsInBPIX_->Fill(nhitinTOB);
		h_probeHitsInFPIX_->Fill(nhitinTEC);
		
		float dxyRecoV = theTrack.dz(theRecoVertex);
		float dzRecoV  = theTrack.dxy(theRecoVertex);
		float dxysigmaRecoV = TMath::Sqrt(theTrack.d0Error()*theTrack.d0Error()+xErrOfflineVertex_*yErrOfflineVertex_);
		float dzsigmaRecoV  = TMath::Sqrt(theTrack.dzError()*theTrack.dzError()+zErrOfflineVertex_*zErrOfflineVertex_);

		double zTrack=(theTTrack->stateAtBeamLine().trackStateAtPCA()).position().z();
		double zVertex=theFittedVertex.position().z();
		double tantheta=tan((theTTrack->stateAtBeamLine().trackStateAtPCA()).momentum().theta());

		double dz2= pow(theTrack.dzError(),2)+wxy2_/pow(tantheta,2);
		double restrkz   = zTrack-zVertex;
		double pulltrkz  = (zTrack-zVertex)/TMath::Sqrt(dz2);

		h_probedxyRecoV_->Fill(dxyRecoV);
		h_probedzRecoV_->Fill(dzRecoV);
	
		h_probedzRefitV_->Fill(dxyFromMyVertex);
		h_probedxyRefitV_->Fill(dzFromMyVertex);

		h_probed0RefitV_->Fill(d0);
		h_probez0RefitV_->Fill(z0);

		h_probesignIP2DRefitV_->Fill(s_ip2dpv_corr);
		h_probed3DRefitV_->Fill(ip3d_corr);
		h_probereszRefitV_->Fill(restrkz);

		h_probeRecoVSigZ_->Fill(dzRecoV/dzsigmaRecoV);
		h_probeRecoVSigXY_->Fill(dxyRecoV/dxysigmaRecoV); 	
		h_probeRefitVSigZ_->Fill(dzFromMyVertex/dz_err);
		h_probeRefitVSigXY_->Fill(dxyFromMyVertex/s_ip2dpv_err); 
		h_probeRefitVSig3D_->Fill(ip3d_corr/ip3d_err);
		h_probeRefitVLogSig3D_->Fill(log10(ip3d_corr/ip3d_err));
		h_probeRefitVSigResZ_->Fill(pulltrkz);
		
		a_dxyVsPhi->Fill(trackphi,dxyFromMyVertex*cmToum);
		a_dzVsPhi->Fill(trackphi,z0*cmToum);  
		n_dxyVsPhi->Fill(trackphi,dxyFromMyVertex/s_ip2dpv_err); 
		n_dzVsPhi->Fill(trackphi,z0/z0_error);
  
		a_dxyVsEta->Fill(tracketa,dxyFromMyVertex*cmToum); 
		a_dzVsEta->Fill(tracketa,z0*cmToum);  
		n_dxyVsEta->Fill(tracketa,dxyFromMyVertex/s_ip2dpv_err); 
		n_dzVsEta->Fill(tracketa,z0/z0_error); 
 
		// filling the binned distributions
		for(int i=0; i<nBins_; i++){
		  
		  float phiF = (-TMath::Pi()+i*phiSect_)*(180/TMath::Pi());
		  float phiL = (-TMath::Pi()+(i+1)*phiSect_)*(180/TMath::Pi());
		  
		  float etaF=-2.5+i*etaSect_;
		  float etaL=-2.5+(i+1)*etaSect_;

		  if(tracketa >= etaF && tracketa < etaL ){

		    a_dxyEtaResiduals[i]->Fill(dxyFromMyVertex*cmToum);
		    a_dzEtaResiduals[i]->Fill(dzFromMyVertex*cmToum);   
		    n_dxyEtaResiduals[i]->Fill(dxyFromMyVertex/s_ip2dpv_err);
		    n_dzEtaResiduals[i]->Fill(dzFromMyVertex/dz_err);	    
		    a_IP2DEtaResiduals[i]->Fill(s_ip2dpv_corr*cmToum);
		    n_IP2DEtaResiduals[i]->Fill(s_ip2dpv_corr/s_ip2dpv_err);
		    a_reszEtaResiduals[i]->Fill(restrkz*cmToum);
		    n_reszEtaResiduals[i]->Fill(pulltrkz);
		    a_d3DEtaResiduals[i]->Fill(ip3d_corr*cmToum);   
		    n_d3DEtaResiduals[i]->Fill(ip3d_corr/ip3d_err);
		    a_IP3DEtaResiduals[i]->Fill(s_ip3dpv_corr*cmToum);
		    n_IP3DEtaResiduals[i]->Fill(s_ip3dpv_corr/s_ip3dpv_err);

		  }
		  	  
		  if(trackphi >= phiF && trackphi < phiL ){
		    a_dxyPhiResiduals[i]->Fill(dxyFromMyVertex*cmToum);
		    a_dzPhiResiduals[i]->Fill(dzFromMyVertex*cmToum); 
		    n_dxyPhiResiduals[i]->Fill(dxyFromMyVertex/s_ip2dpv_err);
		    n_dzPhiResiduals[i]->Fill(dzFromMyVertex/dz_err); 
		    a_IP2DPhiResiduals[i]->Fill(s_ip2dpv_corr*cmToum);
		    n_IP2DPhiResiduals[i]->Fill(s_ip2dpv_corr/s_ip2dpv_err); 
		    a_reszPhiResiduals[i]->Fill(restrkz*cmToum);
		    n_reszPhiResiduals[i]->Fill(pulltrkz);
		    a_d3DPhiResiduals[i]->Fill(ip3d_corr*cmToum);   
		    n_d3DPhiResiduals[i]->Fill(ip3d_corr/ip3d_err);
		    a_IP3DPhiResiduals[i]->Fill(s_ip3dpv_corr*cmToum);
		    n_IP3DPhiResiduals[i]->Fill(s_ip3dpv_corr/s_ip3dpv_err);

		    for(int j=0; j<nBins_; j++){

		      float etaJ=-2.5+j*etaSect_;
		      float etaK=-2.5+(j+1)*etaSect_;

		      if(tracketa >= etaJ && tracketa < etaK ){
			a_dxyResidualsMap[i][j]->Fill(dxyFromMyVertex*cmToum); 
			a_dzResidualsMap[i][j]->Fill(dzFromMyVertex*cmToum);   		
			n_dxyResidualsMap[i][j]->Fill(dxyFromMyVertex/s_ip2dpv_err); 
			n_dzResidualsMap[i][j]->Fill(dzFromMyVertex/dz_err);  
			a_d3DResidualsMap[i][j]->Fill(ip3d_corr*cmToum);   
			n_d3DResidualsMap[i][j]->Fill(ip3d_corr/ip3d_err); 

		      }
		    }
		  }		
		}
	      }
  	          
	      if(debug_){
		edm::LogInfo("PrimaryVertexValidation")<<" myVertex.x()= "<<myVertex.x()<<"\n"
						       <<" myVertex.y()= "<<myVertex.y()<<" \n"
						       <<" myVertex.z()= "<<myVertex.z()<<" \n"
						       <<" theTrack.dz(myVertex)= "<<theTrack.dz(myVertex)<<" \n"
						       <<" zPCA -myVertex.z() = "<<(theTrack.vertex().z() -myVertex.z());
		
	      }// ends if debug_
	    } // ends if the fitted vertex is Valid

	    delete theFitter;

	  }  catch ( cms::Exception& er ) {
	    LogTrace("PrimaryVertexValidation")<<"caught std::exception "<<er.what()<<std::endl;
	  }
		
	} //ends if theFinalTracks.size() > 2
	
	else {
	  if(debug_)
	    edm::LogInfo("PrimaryVertexValidation")<<"Not enough tracks to make a vertex.  Returns no vertex info";
	}
	  
	++nTracks_;  
	++nTracksPerClus_;

	if(debug_)
	  edm::LogInfo("PrimaryVertexValidation")<<"Track "<<i<<" : pT = "<<theTrack.pt();
	  
      }// for loop on tracks
  } // for loop on track clusters
  
  // Fill the TTree if needed

  if(storeNtuple_){  
    rootTree_->Fill();
  }

  
} 

// ------------ method called to discriminate 1D from 2D hits  ------------
bool PrimaryVertexValidation::isHit2D(const TrackingRecHit &hit) const
{
  if (hit.dimension() < 2) {
    return false; // some (muon...) stuff really has RecHit1D
  } else {
    const DetId detId(hit.geographicalId());
    if (detId.det() == DetId::Tracker) {
      if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) {
        return true; // pixel is always 2D
      } else { // should be SiStrip now
        if (dynamic_cast<const SiStripRecHit2D*>(&hit)) return false; // normal hit
        else if (dynamic_cast<const SiStripMatchedRecHit2D*>(&hit)) return true; // matched is 2D
        else if (dynamic_cast<const ProjectedSiStripRecHit2D*>(&hit)) return false; // crazy hit...
        else {
          edm::LogError("UnkownType") << "@SUB=AlignmentTrackSelector::isHit2D"
                                      << "Tracker hit not in pixel and neither SiStripRecHit2D nor "
                                      << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
          return false;
        }
      }
    } else { // not tracker??
      edm::LogWarning("DetectorMismatch") << "@SUB=AlignmentTrackSelector::isHit2D"
                                          << "Hit not in tracker with 'official' dimension >=2.";
      return true; // dimension() >= 2 so accept that...
    }
  }
  // never reached...
}

// ------------ method to check the presence of pixel hits  ------------
bool PrimaryVertexValidation::hasFirstLayerPixelHits(const reco::TransientTrack track)
{
  using namespace reco;
  const HitPattern& p = track.hitPattern();      
  for (int i=0; i<p.numberOfHits(HitPattern::TRACK_HITS); i++) {
    uint32_t pattern = p.getHitPattern(HitPattern::TRACK_HITS, i);   
    if (p.pixelBarrelHitFilter(pattern) || p.pixelEndcapHitFilter(pattern) ) {
      if (p.getLayer(pattern) == 1) {
	if (p.validHitFilter(pattern)) {
	  return true;
	}
      }
    }
  }
  return false;
} 

// ------------ method called once each job before begining the event loop  ------------
void PrimaryVertexValidation::beginJob()
{
  edm::LogInfo("PrimaryVertexValidation") 
    <<"######################################\n"
    <<"Begin Job \n" 
    <<"######################################";

  // Define TTree for output
  Nevt_    = 0;
  
  //  rootFile_ = new TFile(filename_.c_str(),"recreate");
  edm::Service<TFileService> fs;
  rootTree_ = fs->make<TTree>("tree","PV Validation tree");
 
  // Track Paramters 

  if(lightNtupleSwitch_){ 

    rootTree_->Branch("EventNumber",&EventNumber_,"EventNumber/i");
    rootTree_->Branch("RunNumber",&RunNumber_,"RunNumber/i");
    rootTree_->Branch("LuminosityBlockNumber",&LuminosityBlockNumber_,"LuminosityBlockNumber/i");
    rootTree_->Branch("nOfflineVertices",&nOfflineVertices_,"nOfflineVertices/I");
    rootTree_->Branch("nTracks",&nTracks_,"nTracks/I");
    rootTree_->Branch("phi",&phi_,"phi[nTracks]/D");
    rootTree_->Branch("eta",&eta_,"eta[nTracks]/D");
    rootTree_->Branch("pt",&pt_,"pt[nTracks]/D");
    rootTree_->Branch("dxyFromMyVertex",&dxyFromMyVertex_,"dxyFromMyVertex[nTracks]/D");
    rootTree_->Branch("dzFromMyVertex",&dzFromMyVertex_,"dzFromMyVertex[nTracks]/D"); 
    rootTree_->Branch("d3DFromMyVertex",&d3DFromMyVertex_,"d3DFromMyVertex[nTracks]/D"); 
    rootTree_->Branch("IPTsigFromMyVertex",&IPTsigFromMyVertex_,"IPTsigFromMyVertex_[nTracks]/D");    
    rootTree_->Branch("IPLsigFromMyVertex",&IPLsigFromMyVertex_,"IPLsigFromMyVertex_[nTracks]/D"); 
    rootTree_->Branch("IP3DsigFromMyVertex",&IP3DsigFromMyVertex_,"IP3DsigFromMyVertex_[nTracks]/D"); 
    rootTree_->Branch("hasRecVertex",&hasRecVertex_,"hasRecVertex[nTracks]/I");
    rootTree_->Branch("isGoodTrack",&isGoodTrack_,"isGoodTrack[nTracks]/I");
    rootTree_->Branch("isHighPurity",&isHighPurity_,"isHighPurity_[nTracks]/I");
    
  } else {
    
    rootTree_->Branch("nTracks",&nTracks_,"nTracks/I");
    rootTree_->Branch("nTracksPerClus",&nTracksPerClus_,"nTracksPerClus/I");
    rootTree_->Branch("nClus",&nClus_,"nClus/I");
    rootTree_->Branch("xOfflineVertex",&xOfflineVertex_,"xOfflineVertex/D");
    rootTree_->Branch("yOfflineVertex",&yOfflineVertex_,"yOfflineVertex/D");
    rootTree_->Branch("zOfflineVertex",&zOfflineVertex_,"zOfflineVertex/D");
    rootTree_->Branch("BSx0",&BSx0_,"BSx0/D");
    rootTree_->Branch("BSy0",&BSy0_,"BSy0/D");
    rootTree_->Branch("BSz0",&BSz0_,"BSz0/D");
    rootTree_->Branch("Beamsigmaz",&Beamsigmaz_,"Beamsigmaz/D");
    rootTree_->Branch("Beamdxdz",&Beamdxdz_,"Beamdxdz/D");
    rootTree_->Branch("BeamWidthX",&BeamWidthX_,"BeamWidthX/D");
    rootTree_->Branch("BeamWidthY",&BeamWidthY_,"BeamWidthY/D");
    rootTree_->Branch("pt",&pt_,"pt[nTracks]/D");
    rootTree_->Branch("p",&p_,"p[nTracks]/D");
    rootTree_->Branch("nhits",&nhits_,"nhits[nTracks]/I");
    rootTree_->Branch("nhits1D",&nhits1D_,"nhits1D[nTracks]/I");
    rootTree_->Branch("nhits2D",&nhits2D_,"nhits2D[nTracks]/I");
    rootTree_->Branch("nhitsBPIX",&nhitsBPIX_,"nhitsBPIX[nTracks]/I");
    rootTree_->Branch("nhitsFPIX",&nhitsFPIX_,"nhitsFPIX[nTracks]/I");
    rootTree_->Branch("nhitsTIB",&nhitsTIB_,"nhitsTIB[nTracks]/I");
    rootTree_->Branch("nhitsTID",&nhitsTID_,"nhitsTID[nTracks]/I");
    rootTree_->Branch("nhitsTOB",&nhitsTOB_,"nhitsTOB[nTracks]/I");
    rootTree_->Branch("nhitsTEC",&nhitsTEC_,"nhitsTEC[nTracks]/I");
    rootTree_->Branch("eta",&eta_,"eta[nTracks]/D");
    rootTree_->Branch("theta",&theta_,"theta[nTracks]/D");
    rootTree_->Branch("phi",&phi_,"phi[nTracks]/D");
    rootTree_->Branch("chi2",&chi2_,"chi2[nTracks]/D");
    rootTree_->Branch("chi2ndof",&chi2ndof_,"chi2ndof[nTracks]/D");
    rootTree_->Branch("charge",&charge_,"charge[nTracks]/I");
    rootTree_->Branch("qoverp",&qoverp_,"qoverp[nTracks]/D");
    rootTree_->Branch("dz",&dz_,"dz[nTracks]/D");
    rootTree_->Branch("dxy",&dxy_,"dxy[nTracks]/D");
    rootTree_->Branch("dzBs",&dzBs_,"dzBs[nTracks]/D");
    rootTree_->Branch("dxyBs",&dxyBs_,"dxyBs[nTracks]/D");
    rootTree_->Branch("xPCA",&xPCA_,"xPCA[nTracks]/D");
    rootTree_->Branch("yPCA",&yPCA_,"yPCA[nTracks]/D");
    rootTree_->Branch("zPCA",&zPCA_,"zPCA[nTracks]/D");
    rootTree_->Branch("xUnbiasedVertex",&xUnbiasedVertex_,"xUnbiasedVertex[nTracks]/D");
    rootTree_->Branch("yUnbiasedVertex",&yUnbiasedVertex_,"yUnbiasedVertex[nTracks]/D");
    rootTree_->Branch("zUnbiasedVertex",&zUnbiasedVertex_,"zUnbiasedVertex[nTracks]/D");
    rootTree_->Branch("chi2normUnbiasedVertex",&chi2normUnbiasedVertex_,"chi2normUnbiasedVertex[nTracks]/F");
    rootTree_->Branch("chi2UnbiasedVertex",&chi2UnbiasedVertex_,"chi2UnbiasedVertex[nTracks]/F");
    rootTree_->Branch("DOFUnbiasedVertex",&DOFUnbiasedVertex_," DOFUnbiasedVertex[nTracks]/F");
    rootTree_->Branch("chi2ProbUnbiasedVertex",&chi2ProbUnbiasedVertex_,"chi2ProbUnbiasedVertex[nTracks]/F");
    rootTree_->Branch("sumOfWeightsUnbiasedVertex",&sumOfWeightsUnbiasedVertex_,"sumOfWeightsUnbiasedVertex[nTracks]/F");
    rootTree_->Branch("tracksUsedForVertexing",&tracksUsedForVertexing_,"tracksUsedForVertexing[nTracks]/I");
    rootTree_->Branch("dxyFromMyVertex",&dxyFromMyVertex_,"dxyFromMyVertex[nTracks]/D");
    rootTree_->Branch("dzFromMyVertex",&dzFromMyVertex_,"dzFromMyVertex[nTracks]/D");
    rootTree_->Branch("dxyErrorFromMyVertex",&dxyErrorFromMyVertex_,"dxyErrorFromMyVertex_[nTracks]/D"); 
    rootTree_->Branch("dzErrorFromMyVertex",&dzErrorFromMyVertex_,"dzErrorFromMyVertex_[nTracks]/D");  
    rootTree_->Branch("IPTsigFromMyVertex",&IPTsigFromMyVertex_,"IPTsigFromMyVertex_[nTracks]/D");    
    rootTree_->Branch("IPLsigFromMyVertex",&IPLsigFromMyVertex_,"IPLsigFromMyVertex_[nTracks]/D"); 
    rootTree_->Branch("hasRecVertex",&hasRecVertex_,"hasRecVertex[nTracks]/I");
    rootTree_->Branch("isGoodTrack",&isGoodTrack_,"isGoodTrack[nTracks]/I");   
  }

  // event histograms
  TFileDirectory EventFeatures = fs->mkdir("EventFeatures");

  TH1F::SetDefaultSumw2(kTRUE);

  h_nTracks           = EventFeatures.make<TH1F>("h_nTracks","number of tracks per event;n_{tracks}/event;n_{events}",300,-0.5,299.5);	     
  h_nClus             = EventFeatures.make<TH1F>("h_nClus","number of track clusters;n_{clusters}/event;n_{events}",50,-0.5,49.5);	     
  h_nOfflineVertices  = EventFeatures.make<TH1F>("h_nOfflineVertices","number of offline reconstructed vertices;n_{vertices}/event;n_{events}",50,-0.5,49.5);  
  h_runNumber         = EventFeatures.make<TH1F>("h_runNumber","run number;run number;n_{events}",100000,150000.,250000.);	     
  h_xOfflineVertex    = EventFeatures.make<TH1F>("h_xOfflineVertex","x-coordinate of offline vertex;x_{vertex};n_{events}",100,-0.1,0.1);    
  h_yOfflineVertex    = EventFeatures.make<TH1F>("h_yOfflineVertex","y-coordinate of offline vertex;y_{vertex};n_{events}",100,-0.1,0.1);    
  h_zOfflineVertex    = EventFeatures.make<TH1F>("h_zOfflineVertex","z-coordinate of offline vertex;z_{vertex};n_{events}",100,-30.,30.);    
  h_xErrOfflineVertex = EventFeatures.make<TH1F>("h_xErrOfflineVertex","x-coordinate error of offline vertex;err_{x}^{vtx};n_{events}",100,0.,0.01); 
  h_yErrOfflineVertex = EventFeatures.make<TH1F>("h_yErrOfflineVertex","y-coordinate error of offline vertex;err_{y}^{vtx};n_{events}",100,0.,0.01); 
  h_zErrOfflineVertex = EventFeatures.make<TH1F>("h_zErrOfflineVertex","z-coordinate error of offline vertex;err_{z}^{vtx};n_{events}",100,0.,10.); 
  h_BSx0              = EventFeatures.make<TH1F>("h_BSx0","x-coordinate of reco beamspot;x^{BS}_{0};n_{events}",100,-0.1,0.1);    	     
  h_BSy0              = EventFeatures.make<TH1F>("h_BSy0","y-coordinate of reco beamspot;y^{BS}_{0};n_{events}",100,-0.1,0.1);    	     
  h_BSz0              = EventFeatures.make<TH1F>("h_BSz0","z-coordinate of reco beamspot;z^{BS}_{0};n_{events}",100,-1.,1.);    	     
  h_Beamsigmaz        = EventFeatures.make<TH1F>("h_Beamsigmaz","z-coordinate beam width;#sigma_{Z}^{beam};n_{events}",100,0.,1.);	     	     
  h_BeamWidthX        = EventFeatures.make<TH1F>("h_BeamWidthX","x-coordinate beam width;#sigma_{X}^{beam};n_{events}",100,0.,0.01);	     
  h_BeamWidthY        = EventFeatures.make<TH1F>("h_BeamWidthY","y-coordinate beam width;#sigma_{Y}^{beam};n_{events}",100,0.,0.01);        

  // probe track histograms
  TFileDirectory ProbeFeatures = fs->mkdir("ProbeTrackFeatures");

  h_probePt_         = ProbeFeatures.make<TH1F>("h_probePt","p_{T} of probe track;track p_{T} (GeV); tracks",100,0.,50.);   
  h_probeP_          = ProbeFeatures.make<TH1F>("h_probeP","momentum of probe track;track p (GeV); tracks",100,0.,100.);   
  h_probeEta_        = ProbeFeatures.make<TH1F>("h_probeEta","#eta of the probe track;track #eta;tracks",54,-2.7,2.7);  
  h_probePhi_        = ProbeFeatures.make<TH1F>("h_probePhi","#phi of probe track;track #phi (rad);tracks",100,-3.15,3.15);  

  h2_probeEtaPhi_    = ProbeFeatures.make<TH2F>("h2_probeEtaPhi","probe track #phi vs #eta;#eta of probe track;track #phi of probe track (rad); tracks",54,-2.7,2.7,100,-3.15,3.15);  
  h2_probeEtaPt_     = ProbeFeatures.make<TH2F>("h2_probeEtaPt","probe track p_{T} vs #eta;#eta of probe track;track p_{T} (GeV); tracks",54,-2.7,2.7,100,0.,50.);    

  h_probeChi2_       = ProbeFeatures.make<TH1F>("h_probeChi2","#chi^{2} of probe track;track #chi^{2}; tracks",100,0.,100.); 
  h_probeNormChi2_   = ProbeFeatures.make<TH1F>("h_probeNormChi2"," normalized #chi^{2} of probe track;track #chi^{2}/ndof; tracks",100,0.,10.);
  h_probeCharge_     = ProbeFeatures.make<TH1F>("h_probeCharge","charge of profe track;track charge Q;tracks",3,-1.5,1.5);
  h_probeQoverP_     = ProbeFeatures.make<TH1F>("h_probeQoverP","q/p of probe track; track Q/p (GeV^{-1});tracks",200,-1.,1.);
  h_probedzRecoV_    = ProbeFeatures.make<TH1F>("h_probedzRecoV","d_{z}(V_{offline}) of probe track;track d_{z}(V_{off}) (cm);tracks",200,-1.,1.);  
  h_probedxyRecoV_   = ProbeFeatures.make<TH1F>("h_probedxyRecoV","d_{xy}(V_{offline}) of probe track;track d_{xy}(V_{off}) (cm);tracks",200,-1.,1.);      
  h_probedzRefitV_   = ProbeFeatures.make<TH1F>("h_probedzRefitV","d_{z}(V_{refit}) of probe track;track d_{z}(V_{fit}) (cm);tracks",200,-1.,1.);
  h_probesignIP2DRefitV_ = ProbeFeatures.make<TH1F>("h_probesignIPRefitV","ip_{2D}(V_{refit}) of probe track;track ip_{2D}(V_{fit}) (cm);tracks",200,-1.,1.);
  h_probedxyRefitV_  = ProbeFeatures.make<TH1F>("h_probedxyRefitV","d_{xy}(V_{refit}) of probe track;track d_{xy}(V_{fit}) (cm);tracks",200,-1.,1.); 

  h_probez0RefitV_   = ProbeFeatures.make<TH1F>("h_probez0RefitV","z_{0}(V_{refit}) of probe track;track z_{0}(V_{fit}) (cm);tracks",200,-1.,1.);
  h_probed0RefitV_   = ProbeFeatures.make<TH1F>("h_probed0RefitV","d_{0}(V_{refit}) of probe track;track d_{0}(V_{fit}) (cm);tracks",200,-1.,1.);

  h_probed3DRefitV_  = ProbeFeatures.make<TH1F>("h_probed3DRefitV","d_{3D}(V_{refit}) of probe track;track d_{3D}(V_{fit}) (cm);tracks",200,0.,1.); 
  h_probereszRefitV_ = ProbeFeatures.make<TH1F>("h_probeReszRefitV","z_{track} -z_{V_{refit}};track res_{z}(V_{refit}) (cm);tracks",200,-1.,1.); 

  h_probeRecoVSigZ_  = ProbeFeatures.make<TH1F>("h_probeRecoVSigZ"  ,"Longitudinal DCA Significance (reco);d_{z}(V_{off})/#sigma_{dz};tracks",100,-8,8);
  h_probeRecoVSigXY_ = ProbeFeatures.make<TH1F>("h_probeRecoVSigXY" ,"Transverse DCA Significance (reco);d_{xy}(V_{off})/#sigma_{dxy};tracks",100,-8,8);
  h_probeRefitVSigZ_ = ProbeFeatures.make<TH1F>("h_probeRefitVSigZ" ,"Longitudinal DCA Significance (refit);d_{z}(V_{fit})/#sigma_{dz};tracks",100,-8,8);
  h_probeRefitVSigXY_= ProbeFeatures.make<TH1F>("h_probeRefitVSigXY","Transverse DCA Significance (refit);d_{xy}(V_{fit})/#sigma_{dxy};tracks",100,-8,8);
  h_probeRefitVSig3D_= ProbeFeatures.make<TH1F>("h_probeRefitVSig3D","3D DCA Significance (refit);d_{3D}/#sigma_{3D};tracks",100,0.,20.); 
  h_probeRefitVLogSig3D_ = ProbeFeatures.make<TH1F>("h_probeRefitVLogSig3D","log_{10}(3D DCA-Significance) (refit);log_{10}(d_{3D}/#sigma_{3D});tracks",100,-5.,4.); 
  h_probeRefitVSigResZ_ = ProbeFeatures.make<TH1F>("h_probeRefitVSigResZ" ,"Longitudinal residual significance (refit);(z_{track} -z_{V_{fit}})/#sigma_{res_{z}};tracks",100,-8,8);

  h_probeHits_       = ProbeFeatures.make<TH1F>("h_probeNRechits"    ,"N_{hits}     ;N_{hits}    ;tracks",40,-0.5,39.5);
  h_probeHits1D_     = ProbeFeatures.make<TH1F>("h_probeNRechits1D"  ,"N_{hits} 1D  ;N_{hits} 1D ;tracks",40,-0.5,39.5);
  h_probeHits2D_     = ProbeFeatures.make<TH1F>("h_probeNRechits2D"  ,"N_{hits} 2D  ;N_{hits} 2D ;tracks",40,-0.5,39.5);
  h_probeHitsInTIB_  = ProbeFeatures.make<TH1F>("h_probeNRechitsTIB" ,"N_{hits} TIB ;N_{hits} TIB;tracks",40,-0.5,39.5);
  h_probeHitsInTOB_  = ProbeFeatures.make<TH1F>("h_probeNRechitsTOB" ,"N_{hits} TOB ;N_{hits} TOB;tracks",40,-0.5,39.5);
  h_probeHitsInTID_  = ProbeFeatures.make<TH1F>("h_probeNRechitsTID" ,"N_{hits} TID ;N_{hits} TID;tracks",40,-0.5,39.5);
  h_probeHitsInTEC_  = ProbeFeatures.make<TH1F>("h_probeNRechitsTEC" ,"N_{hits} TEC ;N_{hits} TEC;tracks",40,-0.5,39.5);
  h_probeHitsInBPIX_ = ProbeFeatures.make<TH1F>("h_probeNRechitsBPIX","N_{hits} BPIX;N_{hits} BPIX;tracks",40,-0.5,39.5);
  h_probeHitsInFPIX_ = ProbeFeatures.make<TH1F>("h_probeNRechitsFPIX","N_{hits} FPIX;N_{hits} FPIX;tracks",40,-0.5,39.5);

  // refit vertex features
  TFileDirectory RefitVertexFeatures = fs->mkdir("RefitVertexFeatures");
  h_fitVtxNtracks_          = RefitVertexFeatures.make<TH1F>("h_fitVtxNtracks"  ,"N_{trks} used in vertex fit;N^{fit}_{tracks};vertices"        ,100,-0.5,99.5);
  h_fitVtxNdof_             = RefitVertexFeatures.make<TH1F>("h_fitVtxNdof"     ,"N_{DOF} of vertex fit;N_{DOF} of refit vertex;vertices"          ,100,-0.5,99.5);
  h_fitVtxChi2_             = RefitVertexFeatures.make<TH1F>("h_fitVtxChi2"     ,"#chi^{2} of vertex fit;vertex #chi^{2};vertices"            ,100,-0.5,99.5);
  h_fitVtxChi2ndf_          = RefitVertexFeatures.make<TH1F>("h_fitVtxChi2ndf"  ,"#chi^{2}/ndf of vertex fit;vertex #chi^{2}/ndf;vertices"    ,100,-0.5,9.5);
  h_fitVtxChi2Prob_         = RefitVertexFeatures.make<TH1F>("h_fitVtxChi2Prob" ,"Prob(#chi^{2},ndf) of vertex fit;Prob(#chi^{2},ndf);vertices",40,0.,1.);
  h_fitVtxTrackWeights_     = RefitVertexFeatures.make<TH1F>("h_fitVtxTrackWeights","track weights associated to track;track weights;tracks",40,0.,1.);
  h_fitVtxTrackAverageWeight_ = RefitVertexFeatures.make<TH1F>("h_fitVtxTrackAverageWeight_","average track weight per vertex;#LT track weight #GT;vertices",40,0.,1.);

  if(useTracksFromRecoVtx_) {

    TFileDirectory RecoVertexFeatures = fs->mkdir("RecoVertexFeatures");
    h_recoVtxNtracks_          = RecoVertexFeatures.make<TH1F>("h_recoVtxNtracks"  ,"N^{vtx}_{trks};N^{vtx}_{trks};vertices"        ,100,-0.5,99.5);
    h_recoVtxChi2ndf_          = RecoVertexFeatures.make<TH1F>("h_recoVtxChi2ndf"  ,"#chi^{2}/ndf vtx;#chi^{2}/ndf vtx;vertices"    ,10,-0.5,9.5);
    h_recoVtxChi2Prob_         = RecoVertexFeatures.make<TH1F>("h_recoVtxChi2Prob" ,"Prob(#chi^{2},ndf);Prob(#chi^{2},ndf);vertices",40,0.,1.);
    h_recoVtxSumPt_            = RecoVertexFeatures.make<TH1F>("h_recoVtxSumPt"    ,"Sum(p^{trks}_{T});Sum(p^{trks}_{T});vertices"  ,100,0.,200.);
    
  }


  TFileDirectory DA = fs->mkdir("DA");
  //DA.cd();
  hDA=bookVertexHistograms(DA);
  //for(std::map<std::string,TH1*>::const_iterator hist=hDA.begin(); hist!=hDA.end(); hist++){
  //hist->second->SetDirectory(DA);
  // DA.make<TH1F>(hist->second);
  // }

  // initialize the residuals histograms 

  float dxymax_phi = 2000; 
  float dzmax_phi  = 2000; 
  float dxymax_eta = 3000; 
  float dzmax_eta  = 3000;

  float d3Dmax_phi = hypot(dxymax_phi,dzmax_phi);
  float d3Dmax_eta = hypot(dxymax_eta,dzmax_eta);

  const int mybins_ = 500;

  ///////////////////////////////////////////////////////////////////
  //  
  // Unbiased track-to-vertex residuals
  // The vertex is refit without the probe track
  //
  ///////////////////////////////////////////////////////////////////

  TFileDirectory AbsTransPhiRes  = fs->mkdir("Abs_Transv_Phi_Residuals");
  TFileDirectory AbsTransEtaRes  = fs->mkdir("Abs_Transv_Eta_Residuals");
		 					  
  TFileDirectory AbsLongPhiRes   = fs->mkdir("Abs_Long_Phi_Residuals");
  TFileDirectory AbsLongEtaRes   = fs->mkdir("Abs_Long_Eta_Residuals");
		 		  
  TFileDirectory Abs3DPhiRes     = fs->mkdir("Abs_3D_Phi_Residuals");
  TFileDirectory Abs3DEtaRes     = fs->mkdir("Abs_3D_Eta_Residuals");

  TFileDirectory NormTransPhiRes = fs->mkdir("Norm_Transv_Phi_Residuals");
  TFileDirectory NormTransEtaRes = fs->mkdir("Norm_Transv_Eta_Residuals");
		 					  
  TFileDirectory NormLongPhiRes  = fs->mkdir("Norm_Long_Phi_Residuals");
  TFileDirectory NormLongEtaRes  = fs->mkdir("Norm_Long_Eta_Residuals");

  TFileDirectory Norm3DPhiRes    = fs->mkdir("Norm_3D_Phi_Residuals");
  TFileDirectory Norm3DEtaRes    = fs->mkdir("Norm_3D_Eta_Residuals");

  TFileDirectory AbsDoubleDiffRes   = fs->mkdir("Abs_DoubleDiffResiduals");
  TFileDirectory NormDoubleDiffRes  = fs->mkdir("Norm_DoubleDiffResiduals");

  for ( int i=0; i<nBins_; ++i ) {

    float phiF = (-TMath::Pi()+i*phiSect_)*(180/TMath::Pi());
    float phiL = (-TMath::Pi()+(i+1)*phiSect_)*(180/TMath::Pi());
    
    float etaF=-2.5+i*etaSect_;
    float etaL=-2.5+(i+1)*etaSect_;
    
    // dxy vs phi and eta
     
    a_dxyPhiResiduals[i] = AbsTransPhiRes.make<TH1F>(Form("histo_dxy_phi_plot%i",i),
						     Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{xy} [#mum];tracks",phiF,phiL),
						     mybins_,-dxymax_phi,dxymax_phi);
    
    a_dxyEtaResiduals[i] = AbsTransEtaRes.make<TH1F>(Form("histo_dxy_eta_plot%i",i),
						     Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{xy} [#mum];tracks",etaF,etaL),
						     mybins_,-dxymax_eta,dxymax_eta);

    // IP2D vs phi and eta

    a_IP2DPhiResiduals[i] = AbsTransPhiRes.make<TH1F>(Form("histo_IP2D_phi_plot%i",i),
						     Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;IP_{2D} [#mum];tracks",phiF,phiL),
						     mybins_,-dxymax_phi,dxymax_phi);
    
    a_IP2DEtaResiduals[i] = AbsTransEtaRes.make<TH1F>(Form("histo_IP2D_eta_plot%i",i),
						     Form("%.2f<#eta^{probe}_{tk}<%.2f;IP_{2D} [#mum];tracks",etaF,etaL),
						     mybins_,-dxymax_eta,dxymax_eta);

    // IP3D vs phi and eta

    a_IP3DPhiResiduals[i] = Abs3DPhiRes.make<TH1F>(Form("histo_IP3D_phi_plot%i",i),
						   Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;IP_{3D} [#mum];tracks",phiF,phiL),
						   mybins_,-dxymax_phi,dxymax_phi);
    
    a_IP3DEtaResiduals[i] = Abs3DEtaRes.make<TH1F>(Form("histo_IP3D_eta_plot%i",i),
						   Form("%.2f<#eta^{probe}_{tk}<%.2f;IP_{3D} [#mum];tracks",etaF,etaL),
						   mybins_,-dxymax_eta,dxymax_eta);

    // dz vs phi and eta

    a_dzPhiResiduals[i]  = AbsLongPhiRes.make<TH1F>(Form("histo_dz_phi_plot%i",i),
						    Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{z} [#mum];tracks",phiF,phiL),
						    mybins_,-dzmax_phi,dzmax_phi);
    
    a_dzEtaResiduals[i]  = AbsLongEtaRes.make<TH1F>(Form("histo_dz_eta_plot%i",i),
						    Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{z} [#mum];tracks",etaF,etaL),
						    mybins_,-dzmax_eta,dzmax_eta);


    // resz vs phi and eta

    a_reszPhiResiduals[i]  = AbsLongPhiRes.make<TH1F>(Form("histo_resz_phi_plot%i",i),
						    Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;z_{trk} - z_{vtx} [#mum];tracks",phiF,phiL),
						    mybins_,-dzmax_phi,dzmax_phi);
    
    a_reszEtaResiduals[i]  = AbsLongEtaRes.make<TH1F>(Form("histo_resz_eta_plot%i",i),
						    Form("%.2f<#eta^{probe}_{tk}<%.2f;z_{trk} - z_{vtx} [#mum];tracks",etaF,etaL),
						    mybins_,-dzmax_eta,dzmax_eta);

    // d3D vs phi and eta

    a_d3DPhiResiduals[i] = Abs3DPhiRes.make<TH1F>(Form("histo_d3D_phi_plot%i",i),
						  Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{3D} [#mum];tracks",phiF,phiL),
						  mybins_,0.,d3Dmax_phi);
    
    a_d3DEtaResiduals[i] = Abs3DEtaRes.make<TH1F>(Form("histo_d3D_eta_plot%i",i),
						  Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{3D} [#mum];tracks",etaF,etaL),
						  mybins_,0.,d3Dmax_eta);
    
    // normalized dxy vs eta and phi
   				
    n_dxyPhiResiduals[i] = NormTransPhiRes.make<TH1F>(Form("histo_norm_dxy_phi_plot%i",i),
						      Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}};tracks",phiF,phiL),
						      mybins_,-dxymax_phi/100.,dxymax_phi/100.);
    
    n_dxyEtaResiduals[i] = NormTransEtaRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i",i),
						      Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{xy}/#sigma_{d_{xy}};tracks",etaF,etaL),
						      mybins_,-dxymax_eta/100.,dxymax_eta/100.);
    
    // normalized IP2d vs eta and phi
    
    n_IP2DPhiResiduals[i] = NormTransPhiRes.make<TH1F>(Form("histo_norm_IP2D_phi_plot%i",i),
						       Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;IP_{2D}/#sigma_{IP_{2D}};tracks",phiF,phiL),
						       mybins_,-dxymax_phi/100.,dxymax_phi/100.);
    
    n_IP2DEtaResiduals[i] = NormTransEtaRes.make<TH1F>(Form("histo_norm_IP2D_eta_plot%i",i),
						       Form("%.2f<#eta^{probe}_{tk}<%.2f;IP_{2D}/#sigma_{IP_{2D}};tracks",etaF,etaL),
						       mybins_,-dxymax_eta/100.,dxymax_eta/100.);
    
    // normalized IP3d vs eta and phi
    
    n_IP3DPhiResiduals[i] = Norm3DPhiRes.make<TH1F>(Form("histo_norm_IP3D_phi_plot%i",i),
						    Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;IP_{3D}/#sigma_{IP_{3D}};tracks",phiF,phiL),
						    mybins_,-dxymax_phi/100.,dxymax_phi/100.);
    
    n_IP3DEtaResiduals[i] = Norm3DEtaRes.make<TH1F>(Form("histo_norm_IP3D_eta_plot%i",i),
						    Form("%.2f<#eta^{probe}_{tk}<%.2f;IP_{3D}/#sigma_{IP_{3D}};tracks",etaF,etaL),
						    mybins_,-dxymax_eta/100.,dxymax_eta/100.);

    // normalized dz vs phi and eta

    n_dzPhiResiduals[i]  = NormLongPhiRes.make<TH1F>(Form("histo_norm_dz_phi_plot%i",i),
						     Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}};tracks",phiF,phiL),
						     mybins_,-dzmax_phi/100.,dzmax_phi/100.);
    
    n_dzEtaResiduals[i]  = NormLongEtaRes.make<TH1F>(Form("histo_norm_dz_eta_plot%i",i),
						     Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{z}/#sigma_{d_{z}};tracks",etaF,etaL),
						     mybins_,-dzmax_eta/100.,dzmax_eta/100.);

    // pull of resz

    n_reszPhiResiduals[i]  = NormLongPhiRes.make<TH1F>(Form("histo_norm_resz_phi_plot%i",i),
						     Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;(z_{trk}-z_{vtx})/#sigma_{res_{z}};tracks",phiF,phiL),
						     mybins_,-dzmax_phi/100.,dzmax_phi/100.);
    
    n_reszEtaResiduals[i]  = NormLongEtaRes.make<TH1F>(Form("histo_norm_resz_eta_plot%i",i),
						     Form("%.2f<#eta^{probe}_{tk}<%.2f;(z_{trk}-z_{vtx})/#sigma_{res_{z}};tracks",etaF,etaL),
						     mybins_,-dzmax_eta/100.,dzmax_eta/100.);

    // normalized d3D vs phi and eta

    n_d3DPhiResiduals[i] = Norm3DPhiRes.make<TH1F>(Form("histo_norm_d3D_phi_plot%i",i),
						   Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{3D}/#sigma_{d_{3D}};tracks",phiF,phiL),
						   mybins_,0.,d3Dmax_phi/100.);
    
    n_d3DEtaResiduals[i] = Norm3DEtaRes.make<TH1F>(Form("histo_norm_d3D_eta_plot%i",i),
						   Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{3D}/#sigma_{d_{3D}};tracks",etaF,etaL),
						   mybins_,0.,d3Dmax_eta/100.);

    for ( int j=0; j<nBins_; ++j ) {
 
      a_dxyResidualsMap[i][j] = AbsDoubleDiffRes.make<TH1F>(Form("histo_dxy_eta_plot%i_phi_plot%i",i,j),
							    Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy};tracks",etaF,etaL,phiF,phiL),
							    mybins_,-dzmax_eta,dzmax_eta);
      
      a_dzResidualsMap[i][j]  = AbsDoubleDiffRes.make<TH1F>(Form("histo_dz_eta_plot%i_phi_plot%i",i,j),
							    Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z};tracks",etaF,etaL,phiF,phiL),
							    mybins_,-dzmax_eta,dzmax_eta);
      
      a_d3DResidualsMap[i][j] = AbsDoubleDiffRes.make<TH1F>(Form("histo_d3D_eta_plot%i_phi_plot%i",i,j),
							    Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{3D};tracks",etaF,etaL,phiF,phiL),
							    mybins_,0.,d3Dmax_eta);
      
      n_dxyResidualsMap[i][j] = NormDoubleDiffRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i_phi_plot%i",i,j),
							     Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}};tracks",etaF,etaL,phiF,phiL),
							     mybins_,-dzmax_eta/100,dzmax_eta/100);

      n_dzResidualsMap[i][j]  = NormDoubleDiffRes.make<TH1F>(Form("histo_norm_dz_eta_plot%i_phi_plot%i",i,j),
							     Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}};tracks",etaF,etaL,phiF,phiL),
							     mybins_,-dzmax_eta/100,dzmax_eta/100);

      n_d3DResidualsMap[i][j] = NormDoubleDiffRes.make<TH1F>(Form("histo_norm_d3D_eta_plot%i_phi_plot%i",i,j),
							     Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{3D}/#sigma_{d_{3D}};tracks",etaF,etaL,phiF,phiL),
							     mybins_,0.,d3Dmax_eta);

    }
  }

  // declaration of the directories
  
  TFileDirectory BiasVsParameter = fs->mkdir("BiasVsParameter");

  a_dxyVsPhi = BiasVsParameter.make<TH2F>("h2_dxy_vs_phi","d_{xy} vs track #phi;track #phi [rad];track d_{xy}(PV) [#mum]",
					  48,-TMath::Pi(),TMath::Pi(),mybins_,-dxymax_phi,dxymax_phi); 
 
  a_dzVsPhi  = BiasVsParameter.make<TH2F>("h2_dz_vs_phi","d_{z} vs track #phi;track #phi [rad];track d_{z}(PV) [#mum]",
					  48,-TMath::Pi(),TMath::Pi(),mybins_,-dzmax_phi,dzmax_phi);   
               
  n_dxyVsPhi = BiasVsParameter.make<TH2F>("h2_n_dxy_vs_phi","d_{xy}/#sigma_{d_{xy}} vs track #phi;track #phi [rad];track d_{xy}(PV)/#sigma_{d_{xy}}",
					  48,-TMath::Pi(),TMath::Pi(),mybins_,-dxymax_phi/100.,dxymax_phi/100.); 
  
  n_dzVsPhi  = BiasVsParameter.make<TH2F>("h2_n_dz_vs_phi","d_{z}/#sigma_{d_{z}} vs track #phi;track #phi [rad];track d_{z}(PV)/#sigma_{d_{z}}",
					  48,-TMath::Pi(),TMath::Pi(),mybins_,-dzmax_phi/100.,dzmax_phi/100.);   
               
  a_dxyVsEta = BiasVsParameter.make<TH2F>("h2_dxy_vs_eta","d_{xy} vs track #eta;track #eta;track d_{xy}(PV) [#mum]",
					  48,-2.5,2.5,mybins_,-dxymax_eta,dzmax_eta);
  
  a_dzVsEta  = BiasVsParameter.make<TH2F>("h2_dz_vs_eta","d_{z} vs track #eta;track #eta;track d_{z}(PV) [#mum]",
					  48,-2.5,2.5,mybins_,-dzmax_eta,dzmax_eta);   
               
  n_dxyVsEta = BiasVsParameter.make<TH2F>("h2_n_dxy_vs_eta","d_{xy}/#sigma_{d_{xy}} vs track #eta;track #eta;track d_{xy}(PV)/#sigma_{d_{xy}}",
					  48,-2.5,2.5,mybins_,-dxymax_eta/100.,dxymax_eta/100.);  

  n_dzVsEta  = BiasVsParameter.make<TH2F>("h2_n_dz_vs_eta","d_{z}/#sigma_{d_{z}} vs track #eta;track #eta;track d_{z}(PV)/#sigma_{d_{z}}",
					  48,-2.5,2.5,mybins_,-dzmax_eta/100.,dzmax_eta/100.);   

  TFileDirectory MeanTrendsDir   = fs->mkdir("MeanTrends");
  TFileDirectory WidthTrendsDir  = fs->mkdir("WidthTrends");
  TFileDirectory MedianTrendsDir = fs->mkdir("MedianTrends");
  TFileDirectory MADTrendsDir    = fs->mkdir("MADTrends");

  TFileDirectory Mean2DMapsDir   = fs->mkdir("MeanMaps");
  TFileDirectory Width2DMapsDir  = fs->mkdir("WidthMaps");

  Double_t highedge=nBins_-0.5;
  Double_t lowedge=-0.5;

  // means and widths from the fit

  a_dxyPhiMeanTrend  = MeanTrendsDir.make<TH1F> ("means_dxy_phi",
						 "#LT d_{xy} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy} #GT [#mum]",
						 nBins_,lowedge,highedge); 
  
  a_dxyPhiWidthTrend = WidthTrendsDir.make<TH1F>("widths_dxy_phi",
						 "#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{xy}} [#mum]",
						 nBins_,lowedge,highedge);
  
  a_dzPhiMeanTrend   = MeanTrendsDir.make<TH1F> ("means_dz_phi",
						 "#LT d_{z} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z} #GT [#mum]",
						 nBins_,lowedge,highedge); 

  a_dzPhiWidthTrend  = WidthTrendsDir.make<TH1F>("widths_dz_phi","#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{z}} [#mum]",
						 nBins_,lowedge,highedge);
  			  
  a_dxyEtaMeanTrend  = MeanTrendsDir.make<TH1F> ("means_dxy_eta",
						 "#LT d_{xy} #GT vs #eta sector;#eta (sector);#LT d_{xy} #GT [#mum]",
						 nBins_,lowedge,highedge);

  a_dxyEtaWidthTrend = WidthTrendsDir.make<TH1F>("widths_dxy_eta",
						 "#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{xy}} [#mum]",
						 nBins_,lowedge,highedge);
  
  a_dzEtaMeanTrend   = MeanTrendsDir.make<TH1F> ("means_dz_eta",
						 "#LT d_{z} #GT vs #eta sector;#eta (sector);#LT d_{z} #GT [#mum]"
						 ,nBins_,lowedge,highedge); 
  
  a_dzEtaWidthTrend  = WidthTrendsDir.make<TH1F>("widths_dz_eta",
						 "#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{z}} [#mum]",
						 nBins_,lowedge,highedge);
  			  
  n_dxyPhiMeanTrend  = MeanTrendsDir.make<TH1F> ("norm_means_dxy_phi",
						 "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy}/#sigma_{d_{xy}} #GT",
						 nBins_,lowedge,highedge);
  
  n_dxyPhiWidthTrend = WidthTrendsDir.make<TH1F>("norm_widths_dxy_phi",
						 "width(d_{xy}/#sigma_{d_{xy}}) vs #phi sector;#varphi (sector) [degrees]; width(d_{xy}/#sigma_{d_{xy}})",
						 nBins_,lowedge,highedge);
  
  n_dzPhiMeanTrend   = MeanTrendsDir.make<TH1F> ("norm_means_dz_phi",
						 "#LT d_{z}/#sigma_{d_{z}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z}/#sigma_{d_{z}} #GT",
						 nBins_,lowedge,highedge); 
  
  n_dzPhiWidthTrend  = WidthTrendsDir.make<TH1F>("norm_widths_dz_phi",
						 "width(d_{z}/#sigma_{d_{z}}) vs #phi sector;#varphi (sector) [degrees];width(d_{z}/#sigma_{d_{z}})",
						 nBins_,lowedge,highedge);
  			  								
  n_dxyEtaMeanTrend  = MeanTrendsDir.make<TH1F> ("norm_means_dxy_eta",
						 "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector;#eta (sector);#LT d_{xy}/#sigma_{d_{z}} #GT",
						 nBins_,lowedge,highedge);
  
  n_dxyEtaWidthTrend = WidthTrendsDir.make<TH1F>("norm_widths_dxy_eta",
						 "width(d_{xy}/#sigma_{d_{xy}}) vs #eta sector;#eta (sector);width(d_{xy}/#sigma_{d_{z}})",
						 nBins_,lowedge,highedge);

  n_dzEtaMeanTrend   = MeanTrendsDir.make<TH1F> ("norm_means_dz_eta",
						 "#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector;#eta (sector);#LT d_{z}/#sigma_{d_{z}} #GT",
						 nBins_,lowedge,highedge);  
  
  n_dzEtaWidthTrend  = WidthTrendsDir.make<TH1F>("norm_widths_dz_eta",
						 "width(d_{z}/#sigma_{d_{z}}) vs #eta sector;#eta (sector);width(d_{z}/#sigma_{d_{z}})",
						 nBins_,lowedge,highedge);                        
  
  // 2D maps

  a_dxyMeanMap       =  Mean2DMapsDir.make<TH2F>  ("means_dxy_map",
						   "#LT d_{xy} #GT map;#eta (sector);#varphi (sector) [degrees]",
						   nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  
  a_dzMeanMap        =  Mean2DMapsDir.make<TH2F>  ("means_dz_map",
						   "#LT d_{z} #GT map;#eta (sector);#varphi (sector) [degrees]",
						   nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  		     
  n_dxyMeanMap       =  Mean2DMapsDir.make<TH2F>  ("norm_means_dxy_map",
						   "#LT d_{xy}/#sigma_{d_{xy}} #GT map;#eta (sector);#varphi (sector) [degrees]",
						   nBins_,lowedge,highedge,nBins_,lowedge,highedge);

  n_dzMeanMap        =  Mean2DMapsDir.make<TH2F>  ("norm_means_dz_map",
						   "#LT d_{z}/#sigma_{d_{z}} #GT map;#eta (sector);#varphi (sector) [degrees]",
						   nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  		     
  a_dxyWidthMap      =  Width2DMapsDir.make<TH2F> ("widths_dxy_map",
						   "#sigma_{d_{xy}} map;#eta (sector);#varphi (sector) [degrees]",
						   nBins_,lowedge,highedge,nBins_,lowedge,highedge);

  a_dzWidthMap       =  Width2DMapsDir.make<TH2F> ("widths_dz_map",
						   "#sigma_{d_{z}} map;#eta (sector);#varphi (sector) [degrees]",
						   nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  		     
  n_dxyWidthMap      =  Width2DMapsDir.make<TH2F> ("norm_widths_dxy_map",
						   "width(d_{xy}/#sigma_{d_{xy}}) map;#eta (sector);#varphi (sector) [degrees]",
						   nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  
  n_dzWidthMap       =  Width2DMapsDir.make<TH2F> ("norm_widths_dz_map",
						   "width(d_{z}/#sigma_{d_{z}}) map;#eta (sector);#varphi (sector) [degrees]",
						   nBins_,lowedge,highedge,nBins_,lowedge,highedge);

  // medians and MADs

  a_dxyPhiMedianTrend = MedianTrendsDir.make<TH1F>("medians_dxy_phi",
						   "Median of d_{xy} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}) [#mum]",
						   nBins_,lowedge,highedge); 		
	     
  a_dxyPhiMADTrend    = MADTrendsDir.make<TH1F>   ("MADs_dxy_phi",
						   "Median absolute deviation of d_{xy} vs #phi sector;#varphi (sector) [degrees];MAD(d_{xy}) [#mum]",
						   nBins_,lowedge,highedge);
				    
  a_dzPhiMedianTrend  = MedianTrendsDir.make<TH1F>("medians_dz_phi",
						   "Median of d_{z} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}) [#mum]",
						   nBins_,lowedge,highedge); 
				    
  a_dzPhiMADTrend     = MADTrendsDir.make<TH1F>   ("MADs_dz_phi",
						   "Median absolute deviation of d_{z} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}) [#mum]",
						   nBins_,lowedge,highedge);				    
  
  a_dxyEtaMedianTrend = MedianTrendsDir.make<TH1F>("medians_dxy_eta",
						   "Median of d_{xy} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}) [#mum]",
						   nBins_,lowedge,highedge);		
			      
  a_dxyEtaMADTrend    = MADTrendsDir.make<TH1F>   ("MADs_dxy_eta",
						   "Median absolute deviation of d_{xy} vs #eta sector;#eta (sector);MAD(d_{xy}) [#mum]",
						   nBins_,lowedge,highedge);	
				    
  a_dzEtaMedianTrend  = MedianTrendsDir.make<TH1F>("medians_dz_eta",
						   "Median of d_{z} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}) [#mum]",
						   nBins_,lowedge,highedge); 	
				      
  a_dzEtaMADTrend     = MADTrendsDir.make<TH1F>   ("MADs_dz_eta",
						   "Median absolute deviation of d_{z} vs #eta sector;#eta (sector);MAD(d_{z}) [#mum]",
						   nBins_,lowedge,highedge);				          
  
  n_dxyPhiMedianTrend = MedianTrendsDir.make<TH1F>("norm_medians_dxy_phi",
						   "Median of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}/#sigma_{d_{xy}})",
						   nBins_,lowedge,highedge); 

  n_dxyPhiMADTrend    = MADTrendsDir.make<TH1F>   ("norm_MADs_dxy_phi",
						   "Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees]; MAD(d_{xy}/#sigma_{d_{xy}})",
						   nBins_,lowedge,highedge);   

  n_dzPhiMedianTrend  = MedianTrendsDir.make<TH1F>("norm_medians_dz_phi",
						   "Median of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}/#sigma_{d_{z}})",
						   nBins_,lowedge,highedge);  
    
  n_dzPhiMADTrend     = MADTrendsDir.make<TH1F>   ("norm_MADs_dz_phi",
						   "Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}/#sigma_{d_{z}})",
						   nBins_,lowedge,highedge);	    
  
  n_dxyEtaMedianTrend = MedianTrendsDir.make<TH1F>("norm_medians_dxy_eta",
						   "Median of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}/#sigma_{d_{z}})",
						   nBins_,lowedge,highedge);
		    
  n_dxyEtaMADTrend    = MADTrendsDir.make<TH1F>   ("norm_MADs_dxy_eta",
						   "Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);MAD(d_{xy}/#sigma_{d_{z}})",
						   nBins_,lowedge,highedge);
		    
  n_dzEtaMedianTrend  = MedianTrendsDir.make<TH1F>("norm_medians_dz_eta",
						   "Median of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}/#sigma_{d_{z}})",
						   nBins_,lowedge,highedge);  	
	    
  n_dzEtaMADTrend     = MADTrendsDir.make<TH1F>   ("norm_MADs_dz_eta",
						   "Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);MAD(d_{z}/#sigma_{d_{z}})",
						   nBins_,lowedge,highedge);                      


  ///////////////////////////////////////////////////////////////////
  //  
  // plots of biased residuals
  // The vertex includes the probe track
  //
  ///////////////////////////////////////////////////////////////////

  if (useTracksFromRecoVtx_){

    TFileDirectory AbsTransPhiBiasRes  = fs->mkdir("Abs_Transv_Phi_BiasResiduals");
    TFileDirectory AbsTransEtaBiasRes  = fs->mkdir("Abs_Transv_Eta_BiasResiduals");
    
    TFileDirectory AbsLongPhiBiasRes   = fs->mkdir("Abs_Long_Phi_BiasResiduals");
    TFileDirectory AbsLongEtaBiasRes   = fs->mkdir("Abs_Long_Eta_BiasResiduals");
    
    TFileDirectory NormTransPhiBiasRes = fs->mkdir("Norm_Transv_Phi_BiasResiduals");
    TFileDirectory NormTransEtaBiasRes = fs->mkdir("Norm_Transv_Eta_BiasResiduals");
    
    TFileDirectory NormLongPhiBiasRes  = fs->mkdir("Norm_Long_Phi_BiasResiduals");
    TFileDirectory NormLongEtaBiasRes  = fs->mkdir("Norm_Long_Eta_BiasResiduals");
    
    TFileDirectory AbsDoubleDiffBiasRes   = fs->mkdir("Abs_DoubleDiffBiasResiduals");
    TFileDirectory NormDoubleDiffBiasRes  = fs->mkdir("Norm_DoubleDiffBiasResiduals");
    
    for ( int i=0; i<nBins_; ++i ) {
      
      float phiF = (-TMath::Pi()+i*phiSect_)*(180/TMath::Pi());
      float phiL = (-TMath::Pi()+(i+1)*phiSect_)*(180/TMath::Pi());
      
      float etaF=-2.5+i*etaSect_;
      float etaL=-2.5+(i+1)*etaSect_;
      
      a_dxyPhiBiasResiduals[i] = AbsTransPhiBiasRes.make<TH1F>(Form("histo_dxy_phi_plot%i",i),
							       Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{xy} [#mum];tracks",phiF,phiL),
							       mybins_,-dxymax_phi,dxymax_phi);

      a_dxyEtaBiasResiduals[i] = AbsTransEtaBiasRes.make<TH1F>(Form("histo_dxy_eta_plot%i",i),
							       Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{xy} [#mum];tracks",etaF,etaL),
							       mybins_,-dxymax_eta,dxymax_eta);
      
      a_dzPhiBiasResiduals[i]  = AbsLongPhiBiasRes.make<TH1F>(Form("histo_dz_phi_plot%i",i),
							      Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f #circ;d_{z} [#mum];tracks",phiF,phiL),
							      mybins_,-dzmax_phi,dzmax_phi);

      a_dzEtaBiasResiduals[i]  = AbsLongEtaBiasRes.make<TH1F>(Form("histo_dz_eta_plot%i",i),
							      Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{z} [#mum];tracks",etaF,etaL),
							      mybins_,-dzmax_eta,dzmax_eta);
      
      n_dxyPhiBiasResiduals[i] = NormTransPhiBiasRes.make<TH1F>(Form("histo_norm_dxy_phi_plot%i",i),
								Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}};tracks",phiF,phiL),
								mybins_,-dxymax_phi/100.,dxymax_phi/100.);

      n_dxyEtaBiasResiduals[i] = NormTransEtaBiasRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i",i),
								Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{xy}/#sigma_{d_{xy}};tracks",etaF,etaL),
								mybins_,-dxymax_eta/100.,dxymax_eta/100.);
      
      n_dzPhiBiasResiduals[i]  = NormLongPhiBiasRes.make<TH1F>(Form("histo_norm_dz_phi_plot%i",i),
							       Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}};tracks",phiF,phiL),
							       mybins_,-dzmax_phi/100.,dzmax_phi/100.);

      n_dzEtaBiasResiduals[i]  = NormLongEtaBiasRes.make<TH1F>(Form("histo_norm_dz_eta_plot%i",i),
							       Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{z}/#sigma_{d_{z}};tracks",etaF,etaL),
							       mybins_,-dzmax_eta/100.,dzmax_eta/100.);
      
      for ( int j=0; j<nBins_; ++j ) {
	
	a_dxyBiasResidualsMap[i][j] = AbsDoubleDiffBiasRes.make<TH1F>(Form("histo_dxy_eta_plot%i_phi_plot%i",i,j),
								      Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy} [#mum];tracks",etaF,etaL,phiF,phiL),
								      mybins_,-dzmax_eta,dzmax_eta);
	
	a_dzBiasResidualsMap[i][j]  = AbsDoubleDiffBiasRes.make<TH1F>(Form("histo_dxy_eta_plot%i_phi_plot%i",i,j),
								      Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z} [#mum];tracks",etaF,etaL,phiF,phiL),
								      mybins_,-dzmax_eta,dzmax_eta);
	
	n_dxyBiasResidualsMap[i][j] = NormDoubleDiffBiasRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i_phi_plot%i",i,j),
								       Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}};tracks",etaF,etaL,phiF,phiL),
								       mybins_,-dzmax_eta/100,dzmax_eta/100);

	n_dzBiasResidualsMap[i][j]  = NormDoubleDiffBiasRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i_phi_plot%i",i,j),
								       Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}};tracks",etaF,etaL,phiF,phiL),
								       mybins_,-dzmax_eta/100,dzmax_eta/100);
      }
    }
    
    // declaration of the directories

    TFileDirectory MeanBiasTrendsDir   = fs->mkdir("MeanBiasTrends");
    TFileDirectory WidthBiasTrendsDir  = fs->mkdir("WidthBiasTrends");
    TFileDirectory MedianBiasTrendsDir = fs->mkdir("MedianBiasTrends");
    TFileDirectory MADBiasTrendsDir    = fs->mkdir("MADBiasTrends");
    
    TFileDirectory Mean2DBiasMapsDir   = fs->mkdir("MeanBiasMaps");
    TFileDirectory Width2DBiasMapsDir  = fs->mkdir("WidthBiasMaps");
    
    // means and widths from the fit
    
    a_dxyPhiMeanBiasTrend  = MeanBiasTrendsDir.make<TH1F> ("means_dxy_phi",
							   "#LT d_{xy} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy} #GT [#mum]",
							   nBins_,lowedge,highedge); 

    a_dxyPhiWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>("widths_dxy_phi",
							   "#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{xy}} [#mum]",
							   nBins_,lowedge,highedge);

    a_dzPhiMeanBiasTrend   = MeanBiasTrendsDir.make<TH1F> ("means_dz_phi",
							   "#LT d_{z} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z} #GT [#mum]",
							   nBins_,lowedge,highedge); 

    a_dzPhiWidthBiasTrend  = WidthBiasTrendsDir.make<TH1F>("widths_dz_phi",
							   "#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{z}} [#mum]",
							   nBins_,lowedge,highedge);
    
    a_dxyEtaMeanBiasTrend  = MeanBiasTrendsDir.make<TH1F> ("means_dxy_eta",
							   "#LT d_{xy} #GT vs #eta sector;#eta (sector);#LT d_{xy} #GT [#mum]",
							   nBins_,lowedge,highedge);

    a_dxyEtaWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>("widths_dxy_eta",
							   "#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{xy}} [#mum]",
							   nBins_,lowedge,highedge);

    a_dzEtaMeanBiasTrend   = MeanBiasTrendsDir.make<TH1F> ("means_dz_eta",
							   "#LT d_{z} #GT vs #eta sector;#eta (sector);#LT d_{z} #GT [#mum]",
							   nBins_,lowedge,highedge); 

    a_dzEtaWidthBiasTrend  = WidthBiasTrendsDir.make<TH1F>("widths_dz_eta",
							   "#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{z}} [#mum]",
							   nBins_,lowedge,highedge);
    
    n_dxyPhiMeanBiasTrend  = MeanBiasTrendsDir.make<TH1F> ("norm_means_dxy_phi",
							   "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy}/#sigma_{d_{xy}} #GT",
							   nBins_,lowedge,highedge);

    n_dxyPhiWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>("norm_widths_dxy_phi",
							   "width(d_{xy}/#sigma_{d_{xy}}) vs #phi sector;#varphi (sector) [degrees]; width(d_{xy}/#sigma_{d_{xy}})",
							   nBins_,lowedge,highedge);

    n_dzPhiMeanBiasTrend   = MeanBiasTrendsDir.make<TH1F> ("norm_means_dz_phi",
							   "#LT d_{z}/#sigma_{d_{z}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z}/#sigma_{d_{z}} #GT",
							   nBins_,lowedge,highedge); 

    n_dzPhiWidthBiasTrend  = WidthBiasTrendsDir.make<TH1F>("norm_widths_dz_phi",
							   "width(d_{z}/#sigma_{d_{z}}) vs #phi sector;#varphi (sector) [degrees];width(d_{z}/#sigma_{d_{z}})",
							   nBins_,lowedge,highedge);
    
    n_dxyEtaMeanBiasTrend  = MeanBiasTrendsDir.make<TH1F> ("norm_means_dxy_eta",
							   "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector;#eta (sector);#LT d_{xy}/#sigma_{d_{z}} #GT",
							   nBins_,lowedge,highedge);

    n_dxyEtaWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>("norm_widths_dxy_eta",
							   "width(d_{xy}/#sigma_{d_{xy}}) vs #eta sector;#eta (sector);width(d_{xy}/#sigma_{d_{z}})",
							   nBins_,lowedge,highedge);

    n_dzEtaMeanBiasTrend   = MeanBiasTrendsDir.make<TH1F> ("norm_means_dz_eta",
							   "#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector;#eta (sector);#LT d_{z}/#sigma_{d_{z}} #GT",
							   nBins_,lowedge,highedge);  

    n_dzEtaWidthBiasTrend  = WidthBiasTrendsDir.make<TH1F>("norm_widths_dz_eta",
							   "width(d_{z}/#sigma_{d_{z}}) vs #eta sector;#eta (sector);width(d_{z}/#sigma_{d_{z}})",
							   nBins_,lowedge,highedge);                        
    
    // 2D maps
    
    a_dxyMeanBiasMap       =  Mean2DBiasMapsDir.make<TH2F>  ("means_dxy_map",
							     "#LT d_{xy} #GT map;#eta (sector);#varphi (sector) [degrees]",
							     nBins_,lowedge,highedge,nBins_,lowedge,highedge);

    a_dzMeanBiasMap        =  Mean2DBiasMapsDir.make<TH2F>  ("means_dz_map",
							     "#LT d_{z} #GT map;#eta (sector);#varphi (sector) [degrees]",
							     nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    
    n_dxyMeanBiasMap       =  Mean2DBiasMapsDir.make<TH2F>  ("norm_means_dxy_map",
							     "#LT d_{xy}/#sigma_{d_{xy}} #GT map;#eta (sector);#varphi (sector) [degrees]",
							     nBins_,lowedge,highedge,nBins_,lowedge,highedge);

    n_dzMeanBiasMap        =  Mean2DBiasMapsDir.make<TH2F>  ("norm_means_dz_map",
							     "#LT d_{z}/#sigma_{d_{z}} #GT map;#eta (sector);#varphi (sector) [degrees]",
							     nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    
    a_dxyWidthBiasMap      =  Width2DBiasMapsDir.make<TH2F> ("widths_dxy_map",
							     "#sigma_{d_{xy}} map;#eta (sector);#varphi (sector) [degrees]",
							     nBins_,lowedge,highedge,nBins_,lowedge,highedge);

    a_dzWidthBiasMap       =  Width2DBiasMapsDir.make<TH2F> ("widths_dz_map",
							     "#sigma_{d_{z}} map;#eta (sector);#varphi (sector) [degrees]",
							     nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    
    n_dxyWidthBiasMap      =  Width2DBiasMapsDir.make<TH2F> ("norm_widths_dxy_map",
							     "width(d_{xy}/#sigma_{d_{xy}}) map;#eta (sector);#varphi (sector) [degrees]",
							     nBins_,lowedge,highedge,nBins_,lowedge,highedge);

    n_dzWidthBiasMap       =  Width2DBiasMapsDir.make<TH2F> ("norm_widths_dz_map",
							     "width(d_{z}/#sigma_{d_{z}}) map;#eta (sector);#varphi (sector) [degrees]",
							     nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    
    // medians and MADs
    
    a_dxyPhiMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>("medians_dxy_phi",
							     "Median of d_{xy} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}) [#mum]",
							     nBins_,lowedge,highedge); 	
		     
    a_dxyPhiMADBiasTrend    = MADBiasTrendsDir.make<TH1F>   ("MADs_dxy_phi",
							     "Median absolute deviation of d_{xy} vs #phi sector;#varphi (sector) [degrees];MAD(d_{xy}) [#mum]",
							     nBins_,lowedge,highedge);	
			    
    a_dzPhiMedianBiasTrend  = MedianBiasTrendsDir.make<TH1F>("medians_dz_phi",
							     "Median of d_{z} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}) [#mum]",
							     nBins_,lowedge,highedge); 		
		    
    a_dzPhiMADBiasTrend     = MADBiasTrendsDir.make<TH1F>   ("MADs_dz_phi",
							     "Median absolute deviation of d_{z} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}) [#mum]",
							     nBins_,lowedge,highedge);				    
    
    a_dxyEtaMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>("medians_dxy_eta",
							     "Median of d_{xy} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}) [#mum]",
							     nBins_,lowedge,highedge);	
				      
    a_dxyEtaMADBiasTrend    = MADBiasTrendsDir.make<TH1F>   ("MADs_dxy_eta",
							     "Median absolute deviation of d_{xy} vs #eta sector;#eta (sector);MAD(d_{xy}) [#mum]",
							     nBins_,lowedge,highedge);		
			    
    a_dzEtaMedianBiasTrend  = MedianBiasTrendsDir.make<TH1F>("medians_dz_eta",
							     "Median of d_{z} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}) [#mum]",
							     nBins_,lowedge,highedge); 	
				      
    a_dzEtaMADBiasTrend     = MADBiasTrendsDir.make<TH1F>   ("MADs_dz_eta",
							     "Median absolute deviation of d_{z} vs #eta sector;#eta (sector);MAD(d_{z}) [#mum]",
							     nBins_,lowedge,highedge);				          
    
    n_dxyPhiMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>("norm_medians_dxy_phi",
							     "Median of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}/#sigma_{d_{xy}})",
							     nBins_,lowedge,highedge); 

    n_dxyPhiMADBiasTrend    = MADBiasTrendsDir.make<TH1F>   ("norm_MADs_dxy_phi",
							     "Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees]; MAD(d_{xy}/#sigma_{d_{xy}})",
							     nBins_,lowedge,highedge); 
  
    n_dzPhiMedianBiasTrend  = MedianBiasTrendsDir.make<TH1F>("norm_medians_dz_phi",
							     "Median of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}/#sigma_{d_{z}})",
							     nBins_,lowedge,highedge); 
     
    n_dzPhiMADBiasTrend     = MADBiasTrendsDir.make<TH1F>   ("norm_MADs_dz_phi",
							     "Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}/#sigma_{d_{z}})",
							     nBins_,lowedge,highedge);	    
    
    n_dxyEtaMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>("norm_medians_dxy_eta",
							     "Median of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}/#sigma_{d_{z}})",
							     nBins_,lowedge,highedge);	
	    
    n_dxyEtaMADBiasTrend    = MADBiasTrendsDir.make<TH1F>   ("norm_MADs_dxy_eta",
							     "Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);MAD(d_{xy}/#sigma_{d_{z}})",
							     nBins_,lowedge,highedge);	
	    
    n_dzEtaMedianBiasTrend  = MedianBiasTrendsDir.make<TH1F>("norm_medians_dz_eta",
							     "Median of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}/#sigma_{d_{z}})",
							     nBins_,lowedge,highedge);  
		    
    n_dzEtaMADBiasTrend     = MADBiasTrendsDir.make<TH1F>   ("norm_MADs_dz_eta",
							     "Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);MAD(d_{z}/#sigma_{d_{z}})",
							     nBins_,lowedge,highedge);                      
    
  }
}
// ------------ method called once each job just after ending the event loop  ------------
void PrimaryVertexValidation::endJob() 
{

  edm::LogInfo("PrimaryVertexValidation")
    <<"######################################\n"
    <<"# PrimaryVertexValidation::endJob()\n" 
    <<"# Number of analyzed events: "<<Nevt_<<"\n"
    <<"######################################";

  if(useTracksFromRecoVtx_){

    fillTrendPlot(a_dxyPhiMeanBiasTrend ,a_dxyPhiBiasResiduals,"mean","phi");  
    fillTrendPlot(a_dxyPhiWidthBiasTrend,a_dxyPhiBiasResiduals,"width","phi");
    fillTrendPlot(a_dzPhiMeanBiasTrend  ,a_dzPhiBiasResiduals ,"mean","phi");   
    fillTrendPlot(a_dzPhiWidthBiasTrend ,a_dzPhiBiasResiduals ,"width","phi");  
    
    fillTrendPlot(a_dxyEtaMeanBiasTrend ,a_dxyEtaBiasResiduals,"mean","eta"); 
    fillTrendPlot(a_dxyEtaWidthBiasTrend,a_dxyEtaBiasResiduals,"width","eta");
    fillTrendPlot(a_dzEtaMeanBiasTrend  ,a_dzEtaBiasResiduals ,"mean","eta"); 
    fillTrendPlot(a_dzEtaWidthBiasTrend ,a_dzEtaBiasResiduals ,"width","eta");
    
    fillTrendPlot(n_dxyPhiMeanBiasTrend ,n_dxyPhiBiasResiduals,"mean","phi"); 
    fillTrendPlot(n_dxyPhiWidthBiasTrend,n_dxyPhiBiasResiduals,"width","phi");
    fillTrendPlot(n_dzPhiMeanBiasTrend  ,n_dzPhiBiasResiduals ,"mean","phi"); 
    fillTrendPlot(n_dzPhiWidthBiasTrend ,n_dzPhiBiasResiduals ,"width","phi");
    
    fillTrendPlot(n_dxyEtaMeanBiasTrend ,n_dxyEtaBiasResiduals,"mean","eta"); 
    fillTrendPlot(n_dxyEtaWidthBiasTrend,n_dxyEtaBiasResiduals,"width","eta");
    fillTrendPlot(n_dzEtaMeanBiasTrend  ,n_dzEtaBiasResiduals ,"mean","eta"); 
    fillTrendPlot(n_dzEtaWidthBiasTrend ,n_dzEtaBiasResiduals ,"width","eta");
    
    // medians and MADs	  
    
    fillTrendPlot(a_dxyPhiMedianBiasTrend,a_dxyPhiBiasResiduals,"median","phi");  
    fillTrendPlot(a_dxyPhiMADBiasTrend   ,a_dxyPhiBiasResiduals,"mad","phi"); 
    fillTrendPlot(a_dzPhiMedianBiasTrend ,a_dzPhiBiasResiduals ,"median","phi");  
    fillTrendPlot(a_dzPhiMADBiasTrend    ,a_dzPhiBiasResiduals ,"mad","phi"); 
    
    fillTrendPlot(a_dxyEtaMedianBiasTrend,a_dxyEtaBiasResiduals,"median","eta");  
    fillTrendPlot(a_dxyEtaMADBiasTrend   ,a_dxyEtaBiasResiduals,"mad","eta"); 
    fillTrendPlot(a_dzEtaMedianBiasTrend ,a_dzEtaBiasResiduals ,"median","eta");  
    fillTrendPlot(a_dzEtaMADBiasTrend    ,a_dzEtaBiasResiduals ,"mad","eta"); 
    
    fillTrendPlot(n_dxyPhiMedianBiasTrend,n_dxyPhiBiasResiduals,"median","phi");  
    fillTrendPlot(n_dxyPhiMADBiasTrend   ,n_dxyPhiBiasResiduals,"mad","phi"); 
    fillTrendPlot(n_dzPhiMedianBiasTrend ,n_dzPhiBiasResiduals ,"median","phi");  
    fillTrendPlot(n_dzPhiMADBiasTrend    ,n_dzPhiBiasResiduals ,"mad","phi"); 
    
    fillTrendPlot(n_dxyEtaMedianBiasTrend,n_dxyEtaBiasResiduals,"median","eta");  
    fillTrendPlot(n_dxyEtaMADBiasTrend   ,n_dxyEtaBiasResiduals,"mad","eta"); 
    fillTrendPlot(n_dzEtaMedianBiasTrend ,n_dzEtaBiasResiduals ,"median","eta");  
    fillTrendPlot(n_dzEtaMADBiasTrend    ,n_dzEtaBiasResiduals ,"mad","eta"); 
   
    // 2d Maps

    fillMap(a_dxyMeanBiasMap ,a_dxyBiasResidualsMap,"mean"); 
    fillMap(a_dxyWidthBiasMap,a_dxyBiasResidualsMap,"width");
    fillMap(a_dzMeanBiasMap  ,a_dzBiasResidualsMap,"mean"); 
    fillMap(a_dzWidthBiasMap ,a_dzBiasResidualsMap,"width");

    fillMap(n_dxyMeanBiasMap ,n_dxyBiasResidualsMap,"mean"); 
    fillMap(n_dxyWidthBiasMap,n_dxyBiasResidualsMap,"width");
    fillMap(n_dzMeanBiasMap  ,n_dzBiasResidualsMap,"mean"); 
    fillMap(n_dzWidthBiasMap ,n_dzBiasResidualsMap,"width");
   
  }

  fillTrendPlot(a_dxyPhiMeanTrend ,a_dxyPhiResiduals,"mean","phi");  
  fillTrendPlot(a_dxyPhiWidthTrend,a_dxyPhiResiduals,"width","phi");
  fillTrendPlot(a_dzPhiMeanTrend  ,a_dzPhiResiduals ,"mean","phi");   
  fillTrendPlot(a_dzPhiWidthTrend ,a_dzPhiResiduals ,"width","phi");  
  
  fillTrendPlot(a_dxyEtaMeanTrend ,a_dxyEtaResiduals,"mean","eta"); 
  fillTrendPlot(a_dxyEtaWidthTrend,a_dxyEtaResiduals,"width","eta");
  fillTrendPlot(a_dzEtaMeanTrend  ,a_dzEtaResiduals ,"mean","eta"); 
  fillTrendPlot(a_dzEtaWidthTrend ,a_dzEtaResiduals ,"width","eta");
  
  fillTrendPlot(n_dxyPhiMeanTrend ,n_dxyPhiResiduals,"mean","phi"); 
  fillTrendPlot(n_dxyPhiWidthTrend,n_dxyPhiResiduals,"width","phi");
  fillTrendPlot(n_dzPhiMeanTrend  ,n_dzPhiResiduals ,"mean","phi"); 
  fillTrendPlot(n_dzPhiWidthTrend ,n_dzPhiResiduals ,"width","phi");
  
  fillTrendPlot(n_dxyEtaMeanTrend ,n_dxyEtaResiduals,"mean","eta"); 
  fillTrendPlot(n_dxyEtaWidthTrend,n_dxyEtaResiduals,"width","eta");
  fillTrendPlot(n_dzEtaMeanTrend  ,n_dzEtaResiduals ,"mean","eta"); 
  fillTrendPlot(n_dzEtaWidthTrend ,n_dzEtaResiduals ,"width","eta");
    
  // medians and MADs	  
  
  fillTrendPlot(a_dxyPhiMedianTrend,a_dxyPhiResiduals,"median","phi");  
  fillTrendPlot(a_dxyPhiMADTrend   ,a_dxyPhiResiduals,"mad","phi"); 
  fillTrendPlot(a_dzPhiMedianTrend ,a_dzPhiResiduals ,"median","phi");  
  fillTrendPlot(a_dzPhiMADTrend    ,a_dzPhiResiduals ,"mad","phi"); 
  
  fillTrendPlot(a_dxyEtaMedianTrend,a_dxyEtaResiduals,"median","eta");  
  fillTrendPlot(a_dxyEtaMADTrend   ,a_dxyEtaResiduals,"mad","eta"); 
  fillTrendPlot(a_dzEtaMedianTrend ,a_dzEtaResiduals ,"median","eta");  
  fillTrendPlot(a_dzEtaMADTrend    ,a_dzEtaResiduals ,"mad","eta"); 
  
  fillTrendPlot(n_dxyPhiMedianTrend,n_dxyPhiResiduals,"median","phi");  
  fillTrendPlot(n_dxyPhiMADTrend   ,n_dxyPhiResiduals,"mad","phi"); 
  fillTrendPlot(n_dzPhiMedianTrend ,n_dzPhiResiduals ,"median","phi");  
  fillTrendPlot(n_dzPhiMADTrend    ,n_dzPhiResiduals ,"mad","phi"); 
  
  fillTrendPlot(n_dxyEtaMedianTrend,n_dxyEtaResiduals,"median","eta");  
  fillTrendPlot(n_dxyEtaMADTrend   ,n_dxyEtaResiduals,"mad","eta"); 
  fillTrendPlot(n_dzEtaMedianTrend ,n_dzEtaResiduals ,"median","eta");  
  fillTrendPlot(n_dzEtaMADTrend    ,n_dzEtaResiduals ,"mad","eta"); 
    
  // 2D Maps

  fillMap(a_dxyMeanMap ,a_dxyResidualsMap,"mean"); 
  fillMap(a_dxyWidthMap,a_dxyResidualsMap,"width");
  fillMap(a_dzMeanMap  ,a_dzResidualsMap,"mean"); 
  fillMap(a_dzWidthMap ,a_dzResidualsMap,"width");
  
  fillMap(n_dxyMeanMap ,n_dxyResidualsMap,"mean"); 
  fillMap(n_dxyWidthMap,n_dxyResidualsMap,"width");
  fillMap(n_dzMeanMap  ,n_dzResidualsMap,"mean"); 
  fillMap(n_dzWidthMap ,n_dzResidualsMap,"width");

}

//*************************************************************
void PrimaryVertexValidation::SetVarToZero() 
//*************************************************************
{
  nTracks_ = 0;
  nClus_ = 0;
  nOfflineVertices_=0;
  RunNumber_ =0;
  LuminosityBlockNumber_=0;
  xOfflineVertex_ =-999.;
  yOfflineVertex_ =-999.;
  zOfflineVertex_ =-999.;
  xErrOfflineVertex_=0.;
  yErrOfflineVertex_=0.;
  zErrOfflineVertex_=0.;
  BSx0_ = -999.;
  BSy0_ = -999.;
  BSz0_ = -999.;
  Beamsigmaz_=-999.;
  Beamdxdz_=-999.;   
  BeamWidthX_=-999.;
  BeamWidthY_=-999.;
  wxy2_=-999.;

  for ( int i=0; i<nMaxtracks_; ++i ) {
    
    pt_[i]        = 0;
    p_[i]         = 0;
    nhits_[i]     = 0;
    nhits1D_[i]   = 0;
    nhits2D_[i]   = 0;
    nhitsBPIX_[i] = 0;
    nhitsFPIX_[i] = 0;
    nhitsTIB_[i]  = 0;
    nhitsTID_[i]  = 0;
    nhitsTOB_[i]  = 0;
    nhitsTEC_[i]  = 0;
    isHighPurity_[i] = 0;
    eta_[i]       = 0;
    theta_[i]     = 0;
    phi_[i]       = 0;
    chi2_[i]      = 0;
    chi2ndof_[i]  = 0;
    charge_[i]    = 0;
    qoverp_[i]    = 0;
    dz_[i]        = 0;
    dxy_[i]       = 0;
    dzBs_[i]      = 0;
    dxyBs_[i]     = 0;
    xPCA_[i]      = 0;
    yPCA_[i]      = 0;
    zPCA_[i]      = 0;
    xUnbiasedVertex_[i] =0;    
    yUnbiasedVertex_[i] =0;
    zUnbiasedVertex_[i] =0;
    chi2normUnbiasedVertex_[i]=0;
    chi2UnbiasedVertex_[i]=0;
    chi2ProbUnbiasedVertex_[i]=0;
    DOFUnbiasedVertex_[i]=0;
    sumOfWeightsUnbiasedVertex_[i]=0;
    tracksUsedForVertexing_[i]=0;
    dxyFromMyVertex_[i]=0;
    dzFromMyVertex_[i]=0;
    d3DFromMyVertex_[i]=0;
    dxyErrorFromMyVertex_[i]=0; 
    dzErrorFromMyVertex_[i]=0;
    d3DErrorFromMyVertex_[i]=0;
    IPTsigFromMyVertex_[i]=0;   
    IPLsigFromMyVertex_[i]=0;  
    IP3DsigFromMyVertex_[i]=0;
    hasRecVertex_[i] = 0;
    isGoodTrack_[i]  = 0;
  } 
}

//*************************************************************
std::pair<Double_t,Double_t> PrimaryVertexValidation::getMedian(TH1F *histo)
//*************************************************************
{
  Double_t median = 999;
  int nbins = histo->GetNbinsX();

  //extract median from histogram
  double *x = new double[nbins];
  double *y = new double[nbins];
  for (int j = 0; j < nbins; j++) {
    x[j] = histo->GetBinCenter(j+1);
    y[j] = histo->GetBinContent(j+1);
  }
  median = TMath::Median(nbins, x, y);
  
  delete[] x; x = 0;
  delete[] y; y = 0;  

  std::pair<Double_t,Double_t> result;
  result = std::make_pair(median,median/TMath::Sqrt(histo->GetEntries()));

  return result;

}

//*************************************************************
std::pair<Double_t,Double_t> PrimaryVertexValidation::getMAD(TH1F *histo)
//*************************************************************
{

  int nbins = histo->GetNbinsX();
  Double_t median = getMedian(histo).first;
  Double_t x_lastBin = histo->GetBinLowEdge(nbins+1);
  const char *HistoName =histo->GetName();
  TString Finalname = Form("resMed%s",HistoName);
  TH1F *newHisto = new TH1F(Finalname,Finalname,nbins,0.,x_lastBin);
  Double_t *residuals = new Double_t[nbins];
  Double_t *weights = new Double_t[nbins];

  for (int j = 0; j < nbins; j++) {
    residuals[j] = TMath::Abs(median - histo->GetBinCenter(j+1));
    weights[j]=histo->GetBinContent(j+1);
    newHisto->Fill(residuals[j],weights[j]);
  }
  
  Double_t theMAD = (getMedian(newHisto).first)*1.4826;
  
  delete[] residuals; residuals=0;
  delete[] weights; weights=0;
  newHisto->Delete("");
  
  std::pair<Double_t,Double_t> result;
  result = std::make_pair(theMAD,theMAD/histo->GetEntries());

  return result;

}

//*************************************************************
std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > PrimaryVertexValidation::fitResiduals(TH1 *hist)
//*************************************************************
{
  //float fitResult(9999);
  //if (hist->GetEntries() < 20) return ;
  
  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();
  
  TF1 func("tmp", "gaus", mean - 1.5*sigma, mean + 1.5*sigma); 
  if (0 == hist->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
    mean  = func.GetParameter(1);
    sigma = func.GetParameter(2);
    // second fit: three sigma of first fit around mean of first fit
    func.SetRange(mean - 2*sigma, mean + 2*sigma);
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
    if (0 == hist->Fit(&func, "Q0LR")) {
      if (hist->GetFunction(func.GetName())) { // Take care that it is later on drawn:
	hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
      }
    }
  }

  float res_mean  = func.GetParameter(1);
  float res_width = func.GetParameter(2);
  
  float res_mean_err  = func.GetParError(1);
  float res_width_err = func.GetParError(2);

  std::pair<Double_t,Double_t> resultM;
  std::pair<Double_t,Double_t> resultW;

  resultM = std::make_pair(res_mean,res_mean_err);
  resultW = std::make_pair(res_width,res_width_err);

  std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t>  > result;
  
  result = std::make_pair(resultM,resultW);
  return result;
}

//*************************************************************
void PrimaryVertexValidation::fillTrendPlot(TH1F* trendPlot, TH1F* residualsPlot[100], TString fitPar_, TString var_)
//*************************************************************
{
   
  for ( int i=0; i<nBins_; ++i ) {
    
    char phipositionString[129];
    float phiInterval = phiSect_*(180/TMath::Pi());
    float phiposition = (-180+i*phiInterval)+(phiInterval/2);
    sprintf(phipositionString,"%.f",phiposition);
    
    char etapositionString[129];
    float etaposition = (-2.5+i*etaSect_)+(etaSect_/2);
    sprintf(etapositionString,"%.1f",etaposition);
    
    if(fitPar_=="mean"){
      float mean_      = fitResiduals(residualsPlot[i]).first.first;
      float meanErr_   = fitResiduals(residualsPlot[i]).first.second;
      trendPlot->SetBinContent(i+1,mean_);
      trendPlot->SetBinError(i+1,meanErr_);
    } else if (fitPar_=="width"){
      float width_     = fitResiduals(residualsPlot[i]).second.first;
      float widthErr_  = fitResiduals(residualsPlot[i]).second.second;
      trendPlot->SetBinContent(i+1,width_);
      trendPlot->SetBinError(i+1,widthErr_);
    } else if (fitPar_=="median"){
      float median_    = getMedian(residualsPlot[i]).first;
      float medianErr_ = getMedian(residualsPlot[i]).second;
      trendPlot->SetBinContent(i+1,median_);
      trendPlot->SetBinError(i+1,medianErr_);
    } else if (fitPar_=="mad"){
      float mad_       = getMAD(residualsPlot[i]).first; 
      float madErr_    = getMAD(residualsPlot[i]).second;
      trendPlot->SetBinContent(i+1,mad_);
      trendPlot->SetBinError(i+1,madErr_);
    } else {
      std::cout<<"PrimaryVertexValidation::fillTrendPlot() "<<fitPar_<<" unknown estimator!"<<std::endl;
    }

    if(var_=="eta"){
      trendPlot->GetXaxis()->SetBinLabel(i+1,etapositionString); 
    } else if(var_=="phi"){
      trendPlot->GetXaxis()->SetBinLabel(i+1,phipositionString); 
    } else {
      std::cout<<"PrimaryVertexValidation::fillTrendPlot() "<<var_<<" unknown track parameter!"<<std::endl;
    }
  }
}

//*************************************************************
void PrimaryVertexValidation::fillMap(TH2F* trendMap, TH1F* residualsMapPlot[100][100], TString fitPar_)
//*************************************************************
{
 
  for ( int i=0; i<nBins_; ++i ) {
    
    char phipositionString[129];
    float phiInterval = phiSect_*(180/TMath::Pi());
    float phiposition = (-180+i*phiInterval)+(phiInterval/2);
    sprintf(phipositionString,"%.f",phiposition);
    
    trendMap->GetYaxis()->SetBinLabel(i+1,phipositionString); 

    for ( int j=0; j<nBins_; ++j ) {

      char etapositionString[129];
      float etaposition = (-2.5+j*etaSect_)+(etaSect_/2);
      sprintf(etapositionString,"%.1f",etaposition);

      if(i==0) { trendMap->GetXaxis()->SetBinLabel(j+1,etapositionString); }

      if(fitPar_=="mean"){
	float mean_      = fitResiduals(residualsMapPlot[i][j]).first.first;
	float meanErr_   = fitResiduals(residualsMapPlot[i][j]).first.second;
	trendMap->SetBinContent(j+1,i+1,mean_);
	trendMap->SetBinError(j+1,i+1,meanErr_);
      } else if (fitPar_=="width"){
	float width_     = fitResiduals(residualsMapPlot[i][j]).second.first;
	float widthErr_  = fitResiduals(residualsMapPlot[i][j]).second.second;
	trendMap->SetBinContent(j+1,i+1,width_);
	trendMap->SetBinError(j+1,i+1,widthErr_);
      } else if (fitPar_=="median"){
	float median_    = getMedian(residualsMapPlot[i][j]).first;
	float medianErr_ = getMedian(residualsMapPlot[i][j]).second;
	trendMap->SetBinContent(j+1,i+1,median_);
	trendMap->SetBinError(j+1,i+1,medianErr_);
      } else if (fitPar_=="mad"){
	float mad_       = getMAD(residualsMapPlot[i][j]).first; 
	float madErr_    = getMAD(residualsMapPlot[i][j]).second;
	trendMap->SetBinContent(j+1,i+1,mad_);
	trendMap->SetBinError(j+1,i+1,madErr_);
      } else {
	std::cout<<"PrimaryVertexValidation::fillMap() "<<fitPar_<<" unknown estimator!"<<std::endl;
      }   
    } // closes loop on eta bins
  } // cloeses loop on phi bins
}

//*************************************************************
bool PrimaryVertexValidation::vtxSort( const reco::Vertex & a, const reco::Vertex & b )
//*************************************************************
{
  if( a.tracksSize() != b.tracksSize() )
    return a.tracksSize() > b.tracksSize() ? true : false ;
  else
    return a.chi2() < b.chi2() ? true : false ;
}

//*************************************************************
bool PrimaryVertexValidation::passesTrackCuts(const reco::Track & track, const reco::Vertex & vertex,std::string qualityString_, double dxyErrMax_,double dzErrMax_, double ptErrMax_)
//*************************************************************
{
 
   math::XYZPoint vtxPoint(0.0,0.0,0.0);
   double vzErr =0.0, vxErr=0.0, vyErr=0.0;
   vtxPoint=vertex.position();
   vzErr=vertex.zError();
   vxErr=vertex.xError();
   vyErr=vertex.yError();

   double dxy=0.0, dz=0.0, dxysigma=0.0, dzsigma=0.0;
   dxy = track.dxy(vtxPoint);
   dz = track.dz(vtxPoint);
   dxysigma = sqrt(track.d0Error()*track.d0Error()+vxErr*vyErr);
   dzsigma = sqrt(track.dzError()*track.dzError()+vzErr*vzErr);
 
   if(track.quality(reco::TrackBase::qualityByName(qualityString_)) != 1)return false;
   if(fabs(dxy/dxysigma) > dxyErrMax_) return false;
   if(fabs(dz/dzsigma) > dzErrMax_) return false;
   if(track.ptError() / track.pt() > ptErrMax_) return false;

   return true;
}


//*************************************************************
std::map<std::string, TH1*> PrimaryVertexValidation::bookVertexHistograms(TFileDirectory dir)
//*************************************************************
{

  TH1F::SetDefaultSumw2(kTRUE);

  std::map<std::string, TH1*> h;
  
  // histograms of track quality (Data and MC)
  std::string types[] = {"all","sel"};
  for(int t=0; t<2; t++){
    h["pseudorapidity_"+types[t]] =dir.make <TH1F>(("rapidity_"+types[t]).c_str(),"track pseudorapidity; track #eta; tracks",100,-3., 3.);
    h["z0_"+types[t]] = dir.make<TH1F>(("z0_"+types[t]).c_str(),"track z_{0};track z_{0} (cm);tracks",80,-40., 40.);
    h["phi_"+types[t]] = dir.make<TH1F>(("phi_"+types[t]).c_str(),"track #phi; track #phi;tracks",80,-TMath::Pi(), TMath::Pi());
    h["eta_"+types[t]] = dir.make<TH1F>(("eta_"+types[t]).c_str(),"track #eta; track #eta;tracks",80,-4., 4.);
    h["pt_"+types[t]] = dir.make<TH1F>(("pt_"+types[t]).c_str(),"track p_{T}; track p_{T} [GeV];tracks",100,0., 20.);
    h["p_"+types[t]] = dir.make<TH1F>(("p_"+types[t]).c_str(),"track p; track p [GeV];tracks",100,0., 20.);
    h["found_"+types[t]] = dir.make<TH1F>(("found_"+types[t]).c_str(),"n. found hits;n^{found}_{hits};tracks",30, 0., 30.);
    h["lost_"+types[t]] = dir.make<TH1F>(("lost_"+types[t]).c_str(),"n. lost hits;n^{lost}_{hits};tracks",20, 0., 20.);
    h["nchi2_"+types[t]] = dir.make<TH1F>(("nchi2_"+types[t]).c_str(),"normalized track #chi^{2};track #chi^{2}/ndf;tracks",100, 0., 20.);
    h["rstart_"+types[t]] = dir.make<TH1F>(("rstart_"+types[t]).c_str(),"track start radius; track innermost radius r (cm);tracks",100, 0., 20.);
    h["expectedInner_"+types[t]] = dir.make<TH1F>(("expectedInner_"+types[t]).c_str(),"n. expected inner hits;n^{expected}_{inner};tracks",10, 0., 10.);
    h["expectedOuter_"+types[t]] = dir.make<TH1F>(("expectedOuter_"+types[t]).c_str(),"n. expected outer hits;n^{expected}_{outer};tracks ",10, 0., 10.);
    h["logtresxy_"+types[t]] = dir.make<TH1F>(("logtresxy_"+types[t]).c_str(),"log10(track r-#phi resolution/#mum);log10(track r-#phi resolution/#mum);tracks",100, 0., 5.);
    h["logtresz_"+types[t]] = dir.make<TH1F>(("logtresz_"+types[t]).c_str(),"log10(track z resolution/#mum);log10(track z resolution/#mum);tracks",100, 0., 5.);
    h["tpullxy_"+types[t]] = dir.make<TH1F>(("tpullxy_"+types[t]).c_str(),"track r-#phi pull;pull_{r-#phi};tracks",100, -10., 10.);
    h["tpullz_"+types[t]] = dir.make<TH1F>(("tpullz_"+types[t]).c_str(),"track r-z pull;pull_{r-z};tracks",100, -50., 50.);
    h["tlogDCAxy_"+types[t]] = dir.make<TH1F>(("tlogDCAxy_"+types[t]).c_str(),"track log_{10}(DCA_{r-#phi});track log_{10}(DCA_{r-#phi});tracks",200, -5., 3.);
    h["tlogDCAz_"+types[t]] = dir.make<TH1F>(("tlogDCAz_"+types[t]).c_str(),"track log_{10}(DCA_{r-z});track log_{10}(DCA_{r-z});tracks",200, -5., 5.);
    h["lvseta_"+types[t]] = dir.make<TH2F>(("lvseta_"+types[t]).c_str(),"cluster length vs #eta;track #eta;cluster length",60,-3., 3., 20, 0., 20);
    h["lvstanlambda_"+types[t]] = dir.make<TH2F>(("lvstanlambda_"+types[t]).c_str(),"cluster length vs tan #lambda; tan#lambda;cluster length",60,-6., 6., 20, 0., 20);
    h["restrkz_"+types[t]] = dir.make<TH1F>(("restrkz_"+types[t]).c_str(),"z-residuals (track vs vertex);res_{z} (cm);tracks", 200, -5., 5.);
    h["restrkzvsphi_"+types[t]] = dir.make<TH2F>(("restrkzvsphi_"+types[t]).c_str(),"z-residuals (track - vertex) vs track #phi;track #phi;res_{z} (cm)", 12,-TMath::Pi(),TMath::Pi(),100, -0.5,0.5);
    h["restrkzvseta_"+types[t]] = dir.make<TH2F>(("restrkzvseta_"+types[t]).c_str(),"z-residuals (track - vertex) vs track #eta;track #eta;res_{z} (cm)", 12,-3.,3.,200, -0.5,0.5);
    h["pulltrkzvsphi_"+types[t]] = dir.make<TH2F>(("pulltrkzvsphi_"+types[t]).c_str(),"normalized z-residuals (track - vertex) vs track #phi;track #phi;res_{z}/#sigma_{res_{z}}", 12,-TMath::Pi(),TMath::Pi(),100, -5., 5.);
    h["pulltrkzvseta_"+types[t]] = dir.make<TH2F>(("pulltrkzvseta_"+types[t]).c_str(),"normalized z-residuals (track - vertex) vs track #eta;track #eta;res_{z}/#sigma_{res_{z}}", 12,-3.,3.,100, -5., 5.);
    h["pulltrkz_"+types[t]] = dir.make<TH1F>(("pulltrkz_"+types[t]).c_str(),"normalized z-residuals (track vs vertex);res_{z}/#sigma_{res_{z}};tracks", 100, -5., 5.);
    h["sigmatrkz0_"+types[t]] = dir.make<TH1F>(("sigmatrkz0_"+types[t]).c_str(),"z-resolution (excluding beam);#sigma^{trk}_{z_{0}} (cm);tracks", 100, 0., 5.);
    h["sigmatrkz_"+types[t]] = dir.make<TH1F>(("sigmatrkz_"+types[t]).c_str(),"z-resolution (including beam);#sigma^{trk}_{z} (cm);tracks", 100,0., 5.);
    h["nbarrelhits_"+types[t]] = dir.make<TH1F>(("nbarrelhits_"+types[t]).c_str(),"number of pixel barrel hits;n. hits Barrel Pixel;tracks", 10, 0., 10.);
    h["nbarrelLayers_"+types[t]] = dir.make<TH1F>(("nbarrelLayers_"+types[t]).c_str(),"number of pixel barrel layers;n. layers Barrel Pixel;tracks", 10, 0., 10.);
    h["nPxLayers_"+types[t]] = dir.make<TH1F>(("nPxLayers_"+types[t]).c_str(),"number of pixel layers (barrel+endcap);n. Pixel layers;tracks", 10, 0., 10.);
    h["nSiLayers_"+types[t]] = dir.make<TH1F>(("nSiLayers_"+types[t]).c_str(),"number of Tracker layers;n. Tracker layers;tracks", 20, 0., 20.);
    h["trackAlgo_"+types[t]] = dir.make<TH1F>(("trackAlgo_"+types[t]).c_str(),"track algorithm;track algo;tracks", 30, 0., 30.);
    h["trackQuality_"+types[t]] = dir.make<TH1F>(("trackQuality_"+types[t]).c_str(),"track quality;track quality;tracks", 7, -1., 6.);
  }

  return h;
  
}

//*************************************************************
void PrimaryVertexValidation::fillTrackHistos(std::map<std::string, TH1*> & h, const std::string & ttype, const reco::TransientTrack * tt,const reco::Vertex & v, const reco::BeamSpot & beamSpot, double fBfield_)
//*************************************************************
{

  using namespace reco;

  fill(h,"pseudorapidity_"+ttype,tt->track().eta());
  fill(h,"z0_"+ttype,tt->track().vz());
  fill(h,"phi_"+ttype,tt->track().phi());
  fill(h,"eta_"+ttype,tt->track().eta());
  fill(h,"pt_"+ttype,tt->track().pt());
  fill(h,"p_"+ttype,tt->track().p());
  fill(h,"found_"+ttype,tt->track().found());
  fill(h,"lost_"+ttype,tt->track().lost());
  fill(h,"nchi2_"+ttype,tt->track().normalizedChi2());
  fill(h,"rstart_"+ttype,(tt->track().innerPosition()).Rho());
  
  double d0Error=tt->track().d0Error();
  double d0=tt->track().dxy(beamSpot.position());
  double dz=tt->track().dz(beamSpot.position());
  if (d0Error>0){
    fill(h,"logtresxy_"+ttype,log(d0Error/0.0001)/log(10.));
    fill(h,"tpullxy_"+ttype,d0/d0Error);
    fill(h,"tlogDCAxy_"+ttype,log(fabs(d0/d0Error)));
    
  }
  //double z0=tt->track().vz();
  double dzError=tt->track().dzError();
  if(dzError>0){
    fill(h,"logtresz_"+ttype,log(dzError/0.0001)/log(10.));
    fill(h,"tpullz_"+ttype,dz/dzError);
    fill(h,"tlogDCAz_"+ttype,log(fabs(dz/dzError)));
  }
  
  //
  double wxy2_=pow(beamSpot.BeamWidthX(),2)+pow(beamSpot.BeamWidthY(),2);

  fill(h,"sigmatrkz_"+ttype,sqrt(pow(tt->track().dzError(),2)+wxy2_/pow(tan(tt->track().theta()),2)));
  fill(h,"sigmatrkz0_"+ttype,tt->track().dzError());
  
  // track vs vertex
  if( v.isValid()){ // && (v.ndof()<10.)) {
    // emulate clusterizer input
    //const TransientTrack & tt = theB_->build(&t); wrong !!!!
    //reco::TransientTrack tt = theB_->build(&t); 
    //ttt->track().setBeamSpot(beamSpot); // need the setBeamSpot !
    double z=(tt->stateAtBeamLine().trackStateAtPCA()).position().z();
    double tantheta=tan((tt->stateAtBeamLine().trackStateAtPCA()).momentum().theta());
    double dz2= pow(tt->track().dzError(),2)+wxy2_/pow(tantheta,2);
    
    fill(h,"restrkz_"+ttype,z-v.position().z());
    fill(h,"restrkzvsphi_"+ttype,tt->track().phi(), z-v.position().z());
    fill(h,"restrkzvseta_"+ttype,tt->track().eta(), z-v.position().z());
    fill(h,"pulltrkzvsphi_"+ttype,tt->track().phi(), (z-v.position().z())/sqrt(dz2));
    fill(h,"pulltrkzvseta_"+ttype,tt->track().eta(), (z-v.position().z())/sqrt(dz2));
    
    fill(h,"pulltrkz_"+ttype,(z-v.position().z())/sqrt(dz2));
    
    double x1=tt->track().vx()-beamSpot.x0(); double y1=tt->track().vy()-beamSpot.y0();
    
    double kappa=-0.002998*fBfield_*tt->track().qoverp()/cos(tt->track().theta());
    double D0=x1*sin(tt->track().phi())-y1*cos(tt->track().phi())-0.5*kappa*(x1*x1+y1*y1);
    double q=sqrt(1.-2.*kappa*D0);
    double s0=(x1*cos(tt->track().phi())+y1*sin(tt->track().phi()))/q;
    // double s1;
    if (fabs(kappa*s0)>0.001){
      //s1=asin(kappa*s0)/kappa;
    }else{
      //double ks02=(kappa*s0)*(kappa*s0);
      //s1=s0*(1.+ks02/6.+3./40.*ks02*ks02+5./112.*pow(ks02,3));
    }
    // sp.ddcap=-2.*D0/(1.+q);
    //double zdcap=tt->track().vz()-s1/tan(tt->track().theta());
    
  }
  //
  
  // collect some info on hits and clusters
  fill(h,"nbarrelLayers_"+ttype,tt->track().hitPattern().pixelBarrelLayersWithMeasurement());
  fill(h,"nPxLayers_"+ttype,tt->track().hitPattern().pixelLayersWithMeasurement());
  fill(h,"nSiLayers_"+ttype,tt->track().hitPattern().trackerLayersWithMeasurement());
  fill(h,"expectedInner_"+ttype,tt->track().hitPattern().numberOfHits(HitPattern::MISSING_INNER_HITS));
  fill(h,"expectedOuter_"+ttype,tt->track().hitPattern().numberOfHits(HitPattern::MISSING_OUTER_HITS));
  fill(h,"trackAlgo_"+ttype,tt->track().algo());
  fill(h,"trackQuality_"+ttype,tt->track().qualityMask());
  
  //
  int longesthit=0, nbarrel=0;
  for(trackingRecHit_iterator hit=tt->track().recHitsBegin(); hit!=tt->track().recHitsEnd(); hit++){
    if ((**hit).isValid() && (**hit).geographicalId().det() == DetId::Tracker ){
      bool barrel = DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
      //bool endcap = DetId::DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
      if (barrel){
	const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit*>( &(**hit));
	edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
	if (clust.isNonnull()) {
	  nbarrel++;
	  if (clust->sizeY()-longesthit>0) longesthit=clust->sizeY();
	  if (clust->sizeY()>20.){
	    fill(h,"lvseta_"+ttype,tt->track().eta(), 19.9);
	    fill(h,"lvstanlambda_"+ttype,tan(tt->track().lambda()), 19.9);
	  }else{
	    fill(h,"lvseta_"+ttype,tt->track().eta(), float(clust->sizeY()));
	    fill(h,"lvstanlambda_"+ttype,tan(tt->track().lambda()), float(clust->sizeY()));
	  }
	}
      }
    }
  }
  fill(h,"nbarrelhits_"+ttype,float(nbarrel));
  //------------------------------------------------------------------- 
}

//*************************************************************
void PrimaryVertexValidation::add(std::map<std::string, TH1*>& h, TH1* hist)
//*************************************************************
{ 
  h[hist->GetName()]=hist; 
  hist->StatOverflows(kTRUE);
}

//*************************************************************
void PrimaryVertexValidation::fill(std::map<std::string, TH1*>& h, std::string s, double x)
//*************************************************************
{
  // cout << "fill1 " << s << endl;
  if(h.count(s)==0){
    std::cout << "Trying to fill non-exiting Histogram named " << s << std::endl;
    return;
  }
  h[s]->Fill(x);
}

//*************************************************************
void PrimaryVertexValidation::fill(std::map<std::string, TH1*>& h, std::string s, double x, double y)
//*************************************************************
{
  // cout << "fill2 " << s << endl;
  if(h.count(s)==0){
    std::cout << "Trying to fill non-exiting Histogram named " << s << std::endl;
    return;
  }
  h[s]->Fill(x,y);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexValidation);
