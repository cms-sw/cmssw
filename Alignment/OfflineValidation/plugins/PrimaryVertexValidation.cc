// -*- C++ -*-
//
// Package:    PrimaryVertexValidation
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

// user include files
#include "Alignment/OfflineValidation/plugins/PrimaryVertexValidation.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h>
#include <SimDataFormats/TrackingHit/interface/PSimHit.h>
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TVector3.h"
#include "TFile.h"
#include "TROOT.h"
#include "TChain.h"
#include "TNtuple.h"
#include <TMatrixD.h>
#include <TVectorD.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <DataFormats/GeometrySurface/interface/Surface.h>
#include <DataFormats/GeometrySurface/interface/GloballyPositioned.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/PrimaryVertexProducer/interface/GapClusterizerInZ.h"
#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ.h"

 const int kBPIX = PixelSubdetector::PixelBarrel;
 const int kFPIX = PixelSubdetector::PixelEndcap;

// Constructor

PrimaryVertexValidation::PrimaryVertexValidation(const edm::ParameterSet& iConfig):
  storeNtuple_(iConfig.getParameter<bool>("storeNtuple")),
  lightNtupleSwitch_(iConfig.getParameter<bool>("isLightNtuple")),
  useTracksFromRecoVtx_(iConfig.getParameter<bool>("useTracksFromRecoVtx")),
  askFirstLayerHit_(iConfig.getParameter<bool>("askFirstLayerHit")),
  ptOfProbe_(iConfig.getUntrackedParameter<double>("probePt",0.)),
  etaOfProbe_(iConfig.getUntrackedParameter<double>("probeEta",2.4)),
  nBins_(iConfig.getUntrackedParameter<int>("numberOfBins",24)),
  debug_(iConfig.getParameter<bool>("Debug")),
  TrackCollectionTag_(iConfig.getParameter<edm::InputTag>("TrackCollectionTag"))
{
  
  // now do what ever initialization is needed
  // initialize phase space boundaries
  
  phipitch_ = (2*TMath::Pi())/nBins_;
  etapitch_ = 5./nBins_;

  // old version
  // theTrackClusterizer_ = new GapClusterizerInZ(iConfig.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkGapClusParameters"));

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
  using namespace IPTools;

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
  // Retrieve the Track information
  //=======================================================
  
  edm::Handle<reco::TrackCollection>  trackCollectionHandle;
  iEvent.getByLabel(TrackCollectionTag_, trackCollectionHandle);
  
  //=======================================================
  // Retrieve offline vartex information (only for reco)
  //=======================================================
 
  edm::Handle<reco::VertexCollection> vertices;
  try {
    iEvent.getByLabel("offlinePrimaryVertices", vertices);
  } catch (...) {
    if(debug_)
      cout << "No offlinePrimaryVertices found!" << endl;
  }
  if ( vertices.isValid() ) {
    xOfflineVertex_ = (*vertices)[0].x();
    yOfflineVertex_ = (*vertices)[0].y();
    zOfflineVertex_ = (*vertices)[0].z();
  }

  unsigned int vertexCollectionSize = vertices.product()->size();
  int nvvertex = 0;
  
  for (unsigned int i=0; i<vertexCollectionSize; i++) {
    const reco::Vertex& vertex = vertices->at(i);
    if (vertex.isValid()) nvvertex++;
  }

  nOfflineVertices_ = nvvertex;
 

  if ( vertices->size() && useTracksFromRecoVtx_ ) {
   
    double sumpt    = 0;
    size_t ntracks  = 0;
    double chi2ndf  = 0.; 
    double chi2prob = 0.;

    if (!vertices->at(0).isFake()) {
      
      reco::Vertex pv = vertices->at(0);
      
      ntracks  = pv.tracksSize();
      chi2ndf  = pv.normalizedChi2();
      chi2prob = TMath::Prob(pv.chi2(),(int)pv.ndof());
      
      h_recoVtxNtracks_->Fill(ntracks);   
      h_recoVtxChi2ndf_->Fill(chi2ndf);    
      h_recoVtxChi2Prob_->Fill(chi2prob);   
      
      for (reco::Vertex::trackRef_iterator itrk = pv.tracks_begin();itrk != pv.tracks_end(); ++itrk) {
	double pt = (**itrk).pt();
	sumpt += pt*pt;
	
	const math::XYZPoint myVertex(pv.position().x(),pv.position().y(),pv.position().z());
	
	double dxyRes = (**itrk).dxy(myVertex);
	double dzRes  = (**itrk).dz(myVertex);
	
	double dxy_err = (**itrk).dxyError();
	double dz_err  = (**itrk).dzError();
	
	float_t trackphi = ((**itrk).phi())*(180/TMath::Pi());
	float_t tracketa = (**itrk).eta();
	
	for(Int_t i=0; i<nBins_; i++){
	  
	  float phiF = (-TMath::Pi()+i*phipitch_)*(180/TMath::Pi());
	  float phiL = (-TMath::Pi()+(i+1)*phipitch_)*(180/TMath::Pi());
	  
	  float etaF=-2.5+i*etapitch_;
	  float etaL=-2.5+(i+1)*etapitch_;
	  
	  if(trackphi >= phiF && trackphi < phiL ){
	    
	    a_dxyPhiBiasResiduals[i]->Fill(dxyRes*cmToum);
	    a_dzPhiBiasResiduals[i]->Fill(dzRes*cmToum); 
	    n_dxyPhiBiasResiduals[i]->Fill((dxyRes)/dxy_err);
	    n_dzPhiBiasResiduals[i]->Fill((dzRes)/dz_err); 
	    
	    for(Int_t j=0; j<nBins_; j++){
	      
	      float etaJ=-2.5+j*etapitch_;
	      float etaK=-2.5+(j+1)*etapitch_;
	      
	      if(tracketa >= etaJ && tracketa < etaK ){
		
		a_dxyBiasResidualsMap[i][j]->Fill(dxyRes*cmToum); 
		a_dzBiasResidualsMap[i][j]->Fill(dzRes*cmToum);   
		
		n_dxyBiasResidualsMap[i][j]->Fill((dxyRes)/dxy_err); 
		n_dzBiasResidualsMap[i][j]->Fill((dzRes)/dz_err);  
		
	      }
	    }
	  }		
	  
	  if(tracketa >= etaF && tracketa < etaL ){
	    a_dxyEtaBiasResiduals[i]->Fill(dxyRes*cmToum);
	    a_dzEtaBiasResiduals[i]->Fill(dzRes*cmToum); 
	    n_dxyEtaBiasResiduals[i]->Fill((dxyRes)/dxy_err);
	    n_dzEtaBiasResiduals[i]->Fill((dzRes)/dz_err);
	  }
	}
      }
      
      h_recoVtxSumPt_->Fill(sumpt);   

    }
  }

  //=======================================================
  // Retrieve Beamspot information
  //=======================================================

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);
    
  if ( beamSpotHandle.isValid() )
    {
      beamSpot = *beamSpotHandle;
      BSx0_ = beamSpot.x0();
      BSy0_ = beamSpot.y0();
      BSz0_ = beamSpot.z0();
      Beamsigmaz_ = beamSpot.sigmaZ();    
      Beamdxdz_ = beamSpot.dxdz();	     
      BeamWidthX_ = beamSpot.BeamWidthX();
      BeamWidthY_ = beamSpot.BeamWidthY();
    } else
    {
      if(debug_)
	cout << "No BeamSpot found!" << endl;
    }
  
  if(debug_)
    std::cout<<"Beamspot x:"<<BSx0_<<" y:"<<BSy0_<<" z:"<<BSz0_<<std::endl; 
  
  //double sigmaz = beamSpot.sigmaZ();
  //double dxdz = beamSpot.dxdz();
  //double BeamWidth = beamSpot.BeamWidth();
  
  //=======================================================
  // Starts here ananlysis
  //=======================================================
  
  RunNumber_=iEvent.eventAuxiliary().run();
  LuminosityBlockNumber_=iEvent.eventAuxiliary().luminosityBlock();
  EventNumber_=iEvent.eventAuxiliary().id().event();
    
  if(debug_)
    std::cout<<"PrimaryVertexValidation::analyze() looping over "<<trackCollectionHandle->size()<< "tracks." <<std::endl;       

  //======================================================
  // Interface RECO tracks to vertex reconstruction
  //======================================================
 
  std::vector<reco::TransientTrack> t_tks;
  unsigned int k = 0;   
  for(reco::TrackCollection::const_iterator track = trackCollectionHandle->begin(); track!= trackCollectionHandle->end(); ++track, ++k){
  
    reco::TrackRef trackref(trackCollectionHandle,k);
    reco::TransientTrack theTTRef = reco::TransientTrack(trackref, &*theMGField, theTrackingGeometry );
    t_tks.push_back(theTTRef);
  
  }
  
  if(debug_) {cout << "PrimaryVertexValidation"
		  << "Found: " << t_tks.size() << " reconstructed tracks" << "\n";
  }
  
  //======================================================
  // select the tracks
  //======================================================

  std::vector<reco::TransientTrack> seltks = theTrackFilter_->select(t_tks);
    
  //======================================================
  // clusterize tracks in Z
  //======================================================

  vector< vector<reco::TransientTrack> > clusters = theTrackClusterizer_->clusterize(seltks);
  
  if (debug_){
    cout <<  " clustering returned  "<< clusters.size() << " clusters  from " << t_tks.size() << " selected tracks" <<endl;
  }
  
  nClus_=clusters.size();  

  //======================================================
  // Starts loop on clusters 
  //======================================================

  for (vector< vector<reco::TransientTrack> >::const_iterator iclus = clusters.begin(); iclus != clusters.end(); iclus++) {

    nTracksPerClus_=0;

    unsigned int i = 0;   
    for(vector<reco::TransientTrack>::const_iterator theTrack = iclus->begin(); theTrack!= iclus->end(); ++theTrack, ++i)
      {
	if ( nTracks_ >= nMaxtracks_ ) {
	  std::cout << " PrimaryVertexValidation::analyze() : Warning - Number of tracks: " << nTracks_ << " , greater than " << nMaxtracks_ << std::endl;
	  continue;
	}
	
	pt_[nTracks_]       = theTrack->track().pt();
	p_[nTracks_]        = theTrack->track().p();
	nhits_[nTracks_]    = theTrack->track().numberOfValidHits();
	eta_[nTracks_]      = theTrack->track().eta();
	theta_[nTracks_]    = theTrack->track().theta();
	phi_[nTracks_]      = theTrack->track().phi();
	chi2_[nTracks_]     = theTrack->track().chi2();
	chi2ndof_[nTracks_] = theTrack->track().normalizedChi2();
	charge_[nTracks_]   = theTrack->track().charge();
	qoverp_[nTracks_]   = theTrack->track().qoverp();
	dz_[nTracks_]       = theTrack->track().dz();
	dxy_[nTracks_]      = theTrack->track().dxy();
	
	reco::TrackBase::TrackQuality _trackQuality = reco::TrackBase::qualityByName("highPurity");
	isHighPurity_[nTracks_] = theTrack->track().quality(_trackQuality);
	
	math::XYZPoint point(beamSpot.x0(),beamSpot.y0(), beamSpot.z0());
	dxyBs_[nTracks_]    = theTrack->track().dxy(point);
	dzBs_[nTracks_]     = theTrack->track().dz(point);

	xPCA_[nTracks_]     = theTrack->track().vertex().x();
	yPCA_[nTracks_]     = theTrack->track().vertex().y();
	zPCA_[nTracks_]     = theTrack->track().vertex().z(); 
	
	//=======================================================
	// Retrieve rechit information
	//=======================================================  
	
	int nRecHit1D =0;
	int nRecHit2D =0;
	int nhitinTIB =0; 
	int nhitinTOB =0; 
	int nhitinTID =0; 
	int nhitinTEC =0; 
	int nhitinBPIX=0;
	int nhitinFPIX=0; 
	
	for (trackingRecHit_iterator iHit = theTrack->recHitsBegin(); iHit != theTrack->recHitsEnd(); ++iHit) {
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
	if(askFirstLayerHit_) pass = this->hasFirstLayerPixelHits((*theTrack));
	if (pass){
	  isGoodTrack_[nTracks_]=1;
	}
      
	//=======================================================
	// Fit unbiased vertex
	//=======================================================
	
	vector<reco::TransientTrack> theFinalTracks;
	theFinalTracks.clear();

	for(vector<reco::TransientTrack>::const_iterator tk = iclus->begin(); tk!= iclus->end(); ++tk){
	  
	  pass = this->hasFirstLayerPixelHits((*tk));
	  if (pass){
	    if( tk == theTrack ) continue;
	    else {
	      theFinalTracks.push_back((*tk));
	    }
	  }
	}
	
	if(theFinalTracks.size() > 2){
	    
	  if(debug_)
	    std::cout <<"PrimaryVertexValidation::analyze() :Transient Track Collection size: "<<theFinalTracks.size()<<std::endl;
	  
	  try{
	      
	    VertexFitter<5>* fitter = new AdaptiveVertexFitter;
	    TransientVertex theFittedVertex = fitter->vertex(theFinalTracks);
	    
	    if(theFittedVertex.isValid ()){
	      
	      if(theFittedVertex.hasTrackWeight()){
		for(size_t rtracks= 0; rtracks < theFinalTracks.size(); rtracks++){
		  sumOfWeightsUnbiasedVertex_[nTracks_] += theFittedVertex.trackWeight(theFinalTracks[rtracks]);
		}
	      }
	      
	      const math::XYZPoint theRecoVertex(xOfflineVertex_,yOfflineVertex_,zOfflineVertex_);
	      
	      const math::XYZPoint myVertex(theFittedVertex.position().x(),theFittedVertex.position().y(),theFittedVertex.position().z());
	      hasRecVertex_[nTracks_]    = 1;
	      xUnbiasedVertex_[nTracks_] = theFittedVertex.position().x();
	      yUnbiasedVertex_[nTracks_] = theFittedVertex.position().y();
	      zUnbiasedVertex_[nTracks_] = theFittedVertex.position().z();
	      
	      chi2normUnbiasedVertex_[nTracks_] = theFittedVertex.normalisedChiSquared();
	      chi2UnbiasedVertex_[nTracks_]     = theFittedVertex.totalChiSquared();
	      DOFUnbiasedVertex_[nTracks_]      = theFittedVertex.degreesOfFreedom();   
	      tracksUsedForVertexing_[nTracks_] = theFinalTracks.size();

	      h_fitVtxNtracks_->Fill(theFinalTracks.size());        
	      h_fitVtxChi2_->Fill(theFittedVertex.totalChiSquared());
	      h_fitVtxNdof_->Fill(theFittedVertex.degreesOfFreedom());
	      h_fitVtxChi2ndf_->Fill(theFittedVertex.normalisedChiSquared());        
	      h_fitVtxChi2Prob_->Fill(TMath::Prob(theFittedVertex.totalChiSquared(),(int)theFittedVertex.degreesOfFreedom()));       
	               
	      dszFromMyVertex_[nTracks_] = theTrack->track().dsz(myVertex);
	      dxyFromMyVertex_[nTracks_] = theTrack->track().dxy(myVertex);
	      dzFromMyVertex_[nTracks_]  = theTrack->track().dz(myVertex);
	      
	      double dz_err = hypot(theTrack->track().dzError(),theFittedVertex.positionError().czz());

	      // PV2D 

	      // std::pair<bool,Measurement1D> ip2dpv = absoluteTransverseImpactParameter(*theTrack,theFittedVertex);
	      // double ip2d_corr = ip2dpv.second.value();
	      // double ip2d_err  = ip2dpv.second.error();

	      std::pair<bool,Measurement1D> s_ip2dpv =
		signedTransverseImpactParameter(*theTrack,GlobalVector(theTrack->track().px(),
								       theTrack->track().py(),
								       theTrack->track().pz()),
						theFittedVertex); 
	      
	      // double s_ip2dpv_corr = s_ip2dpv.second.value();
	      double s_ip2dpv_err  = s_ip2dpv.second.error();
	      
	      // PV3D

	      // std::pair<bool,Measurement1D> ip3dpv = absoluteImpactParameter3D(*theTrack,theFittedVertex);
	      // double ip3d_corr = ip3dpv.second.value(); 
	      // double ip3d_err  = ip3dpv.second.error(); 

	      // std::pair<bool,Measurement1D> s_ip3dpv = 
	      //	signedImpactParameter3D(*theTrack,GlobalVector(theTrack->track().px(),
	      //						       theTrack->track().py(),
	      //						       theTrack->track().pz()),
	      //			theFittedVertex);
	      
	      // double s_ip3dpv_corr = s_ip3dpv.second.value();
	      // double s_ip3dpv_err  = s_ip3dpv.second.error();
	      
	      dxyErrorFromMyVertex_[nTracks_] = s_ip2dpv_err;
	      dzErrorFromMyVertex_[nTracks_]  = dz_err;
	      
	      IPTsigFromMyVertex_[nTracks_]   = (theTrack->track().dxy(myVertex))/s_ip2dpv_err;
	      IPLsigFromMyVertex_[nTracks_]   = (theTrack->track().dz(myVertex))/dz_err;
	      
	      // fill directly the histograms of residuals
	      
	      float_t trackphi = (theTrack->track().phi())*(180/TMath::Pi());
	      float_t tracketa = theTrack->track().eta();
	      float_t trackpt  = theTrack->track().pt();

	      // checks on the probe track quality
	      if(trackpt >= ptOfProbe_ && fabs(tracketa)<= etaOfProbe_){

		h_probePt_->Fill(theTrack->track().pt());
		h_probeEta_->Fill(theTrack->track().eta());
		h_probePhi_->Fill(theTrack->track().phi());
		h_probeChi2_->Fill(theTrack->track().chi2());
		h_probeNormChi2_->Fill(theTrack->track().normalizedChi2());
		h_probeCharge_->Fill(theTrack->track().charge());
		h_probeQoverP_->Fill(theTrack->track().qoverp());
		h_probedz_->Fill(theTrack->track().dz(theRecoVertex));
		h_probedxy_->Fill((theTrack->track().dxy(theRecoVertex)));

		h_probeHits_->Fill(theTrack->track().numberOfValidHits());       
		h_probeHits1D_->Fill(nRecHit1D);
		h_probeHits2D_->Fill(nRecHit2D);
		h_probeHitsInTIB_->Fill(nhitinBPIX);
		h_probeHitsInTOB_->Fill(nhitinFPIX);
		h_probeHitsInTID_->Fill(nhitinTIB);
		h_probeHitsInTEC_->Fill(nhitinTID);
		h_probeHitsInBPIX_->Fill(nhitinTOB);
		h_probeHitsInFPIX_->Fill(nhitinTEC);
		
		for(Int_t i=0; i<nBins_; i++){
		  
		  float phiF = (-TMath::Pi()+i*phipitch_)*(180/TMath::Pi());
		  float phiL = (-TMath::Pi()+(i+1)*phipitch_)*(180/TMath::Pi());
		  
		  float etaF=-2.5+i*etapitch_;
		  float etaL=-2.5+(i+1)*etapitch_;
		  	  
		  if(trackphi >= phiF && trackphi < phiL ){
		    a_dxyPhiResiduals[i]->Fill(theTrack->track().dxy(myVertex)*cmToum);
		    a_dzPhiResiduals[i]->Fill(theTrack->track().dz(myVertex)*cmToum); 
		    n_dxyPhiResiduals[i]->Fill((theTrack->track().dxy(myVertex))/s_ip2dpv_err);
		    n_dzPhiResiduals[i]->Fill((theTrack->track().dz(myVertex))/dz_err); 

		    for(Int_t j=0; j<nBins_; j++){

		      float etaJ=-2.5+j*etapitch_;
		      float etaK=-2.5+(j+1)*etapitch_;

		      if(tracketa >= etaJ && tracketa < etaK ){
			
			a_dxyResidualsMap[i][j]->Fill(theTrack->track().dxy(myVertex)*cmToum); 
			a_dzResidualsMap[i][j]->Fill(theTrack->track().dz(myVertex)*cmToum);   
			
			n_dxyResidualsMap[i][j]->Fill((theTrack->track().dxy(myVertex))/s_ip2dpv_err); 
			n_dzResidualsMap[i][j]->Fill((theTrack->track().dz(myVertex))/dz_err);  
			
		      }
		    }
		  }		
		  
		  if(tracketa >= etaF && tracketa < etaL ){
		    a_dxyEtaResiduals[i]->Fill(theTrack->track().dxy(myVertex)*cmToum);
		    a_dzEtaResiduals[i]->Fill(theTrack->track().dz(myVertex)*cmToum); 
		    n_dxyEtaResiduals[i]->Fill((theTrack->track().dxy(myVertex))/s_ip2dpv_err);
		    n_dzEtaResiduals[i]->Fill((theTrack->track().dz(myVertex))/dz_err);
		  }
		}
	      }
  	          
	      if(debug_){
		std::cout<<"PrimaryVertexValidation::analyze() : myVertex.x()= "<<myVertex.x()<<" myVertex.y()= "<<myVertex.y()<<" theFittedVertex.z()= "<<myVertex.z()<<std::endl;	  
		std::cout<<"PrimaryVertexValidation::analyze() : theTrack->track().dz(myVertex)= "<<theTrack->track().dz(myVertex)<<std::endl;
		std::cout<<"PrimaryVertexValidation::analyze() : zPCA -myVertex.z() = "<<(theTrack->track().vertex().z() -myVertex.z() )<<std::endl; 
	      }// ends if debug_
	    } // ends if the fitted vertex is Valid

	    delete fitter;

	  }  catch ( cms::Exception& er ) {
	    LogTrace("PrimaryVertexValidation::analyze RECO")<<"caught std::exception "<<er.what()<<std::endl;
	  }
		
	} //ends if theFinalTracks.size() > 2
	
	else {
	  if(debug_)
	      std::cout << "PrimaryVertexValidation::analyze() :Not enough tracks to make a vertex.  Returns no vertex info" << std::endl;
	}
	  
	++nTracks_;  
	++nTracksPerClus_;

	if(debug_)
	  cout<< "Track "<<i<<" : pT = "<<theTrack->track().pt()<<endl;
	
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
    const HitPattern &p = track.hitPattern();
    for (int i = 0; i < p.numberOfHits(HitPattern::TRACK_HITS); i++) {
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
  edm::LogInfo("beginJob") << "Begin Job" << std::endl;
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
    rootTree_->Branch("IPTsigFromMyVertex",&IPTsigFromMyVertex_,"IPTsigFromMyVertex_[nTracks]/D");    
    rootTree_->Branch("IPLsigFromMyVertex",&IPLsigFromMyVertex_,"IPLsigFromMyVertex_[nTracks]/D"); 
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
    rootTree_->Branch("sumOfWeightsUnbiasedVertex",&sumOfWeightsUnbiasedVertex_,"sumOfWeightsUnbiasedVertex[nTracks]/F");
    rootTree_->Branch("tracksUsedForVertexing",&tracksUsedForVertexing_,"tracksUsedForVertexing[nTracks]/I");
    rootTree_->Branch("dxyFromMyVertex",&dxyFromMyVertex_,"dxyFromMyVertex[nTracks]/D");
    rootTree_->Branch("dzFromMyVertex",&dzFromMyVertex_,"dzFromMyVertex[nTracks]/D");
    rootTree_->Branch("dszFromMyVertex",&dszFromMyVertex_,"dszFromMyVertex[nTracks]/D");
    rootTree_->Branch("dxyErrorFromMyVertex",&dxyErrorFromMyVertex_,"dxyErrorFromMyVertex_[nTracks]/D"); 
    rootTree_->Branch("dzErrorFromMyVertex",&dzErrorFromMyVertex_,"dzErrorFromMyVertex_[nTracks]/D");  
    rootTree_->Branch("IPTsigFromMyVertex",&IPTsigFromMyVertex_,"IPTsigFromMyVertex_[nTracks]/D");    
    rootTree_->Branch("IPLsigFromMyVertex",&IPLsigFromMyVertex_,"IPLsigFromMyVertex_[nTracks]/D"); 
    rootTree_->Branch("hasRecVertex",&hasRecVertex_,"hasRecVertex[nTracks]/I");
    rootTree_->Branch("isGoodTrack",&isGoodTrack_,"isGoodTrack[nTracks]/I");   
  }

  // probe track histograms

  TFileDirectory ProbeFeatures = fs->mkdir("ProbeTrackFeatures");

  h_probePt_         = ProbeFeatures.make<TH1F>("h_probePt","p_{T} of probe track;track p_{T} (GeV); tracks",100,0.,50.);   
  h_probeEta_        = ProbeFeatures.make<TH1F>("h_probeEta","#eta of probe track;track #eta; tracks",54,-2.7,2.7);  
  h_probePhi_        = ProbeFeatures.make<TH1F>("h_probePhi","#phi of probe track;track #phi [rad]; tracks",100,-3.15,3.15);  
  h_probeChi2_       = ProbeFeatures.make<TH1F>("h_probeChi2","#chi^{2} of probe track;track #chi^{2}; tracks",100,0.,100.); 
  h_probeNormChi2_   = ProbeFeatures.make<TH1F>("h_probeNormChi2"," normalized #chi^{2} of probe track;track #chi^{2}/ndof; tracks",100,0.,10.);
  h_probeCharge_     = ProbeFeatures.make<TH1F>("h_probeCharge","charge of profe track;track charge Q;tracks",3,-1.5,1.5);
  h_probeQoverP_     = ProbeFeatures.make<TH1F>("h_probeQoverP","q/p of probe track; track Q/p (GeV^{-1})",200,-1.,1.);
  h_probedz_         = ProbeFeatures.make<TH1F>("h_probedz","d_{z} of probe track;track d_{z} (cm);tracks",100,-5.,5.);  
  h_probedxy_        = ProbeFeatures.make<TH1F>("h_probedxy","d_{xy} of probe track;track d_{xy} (#mum);tracks",200,-1.,1.);      

  h_probeHits_       = ProbeFeatures.make<TH1F>("h_probeNRechits"    ,"N_{hits}     ;N_{hits}    ;tracks",40,-0.5,39.5);
  h_probeHits1D_     = ProbeFeatures.make<TH1F>("h_probeNRechits1D"  ,"N_{hits} 1D  ;N_{hits} 1D ;tracks",40,-0.5,39.5);
  h_probeHits2D_     = ProbeFeatures.make<TH1F>("h_probeNRechits2D"  ,"N_{hits} 2D  ;N_{hits} 2D ;tracks",40,-0.5,39.5);
  h_probeHitsInTIB_  = ProbeFeatures.make<TH1F>("h_probeNRechitsTIB" ,"N_{hits} TIB ;N_{hits} TIB;tracks",40,-0.5,39.5);
  h_probeHitsInTOB_  = ProbeFeatures.make<TH1F>("h_probeNRechitsTOB" ,"N_{hits} TOB ;N_{hits} TOB;tracks",40,-0.5,39.5);
  h_probeHitsInTID_  = ProbeFeatures.make<TH1F>("h_probeNRechitsTID" ,"N_{hits} TID ;N_{hits} TID;tracks",40,-0.5,39.5);
  h_probeHitsInTEC_  = ProbeFeatures.make<TH1F>("h_probeNRechitsTEC" ,"N_{hits} TEC ;N_{hits} TEC;tracks",40,-0.5,39.5);
  h_probeHitsInBPIX_ = ProbeFeatures.make<TH1F>("h_probeNRechitsBPIX","N_{hits} BPIX;N_{hits} BPIX;tracks",40,-0.5,39.5);
  h_probeHitsInFPIX_ = ProbeFeatures.make<TH1F>("h_probeNRechitsFPIX","N_{hits} FPIX;N_{hits} FPIX;tracks",40,-0.5,39.5);

  TFileDirectory RefitVertexFeatures = fs->mkdir("RefitVertexFeatures");
  h_fitVtxNtracks_          = RefitVertexFeatures.make<TH1F>("h_fitVtxNtracks"  ,"N^{vtx}_{trks};N^{vtx}_{trks};vertices"        ,100,-0.5,99.5);
  h_fitVtxNdof_             = RefitVertexFeatures.make<TH1F>("h_fitVtxNdof"     ,"N^{vtx}_{DOF};N^{vtx}_{DOF};vertices"          ,100,-0.5,99.5);
  h_fitVtxChi2_             = RefitVertexFeatures.make<TH1F>("h_fitVtxChi2"     ,"#chi^{2} vtx;#chi^{2} vtx;vertices"            ,100,-0.5,99.5);
  h_fitVtxChi2ndf_          = RefitVertexFeatures.make<TH1F>("h_fitVtxChi2ndf"  ,"#chi^{2}/ndf vtx;#chi^{2}/ndf vtx;vertices"    ,100,-0.5,9.5);
  h_fitVtxChi2Prob_         = RefitVertexFeatures.make<TH1F>("h_fitVtxChi2Prob" ,"Prob(#chi^{2},ndf);Prob(#chi^{2},ndf);vertices",40,0.,1.);

  if(useTracksFromRecoVtx_) {

    TFileDirectory RecoVertexFeatures = fs->mkdir("RecoVertexFeatures");
    h_recoVtxNtracks_          = RecoVertexFeatures.make<TH1F>("h_recoVtxNtracks"  ,"N^{vtx}_{trks};N^{vtx}_{trks};vertices"        ,100,-0.5,99.5);
    h_recoVtxChi2ndf_          = RecoVertexFeatures.make<TH1F>("h_recoVtxChi2ndf"  ,"#chi^{2}/ndf vtx;#chi^{2}/ndf vtx;vertices"    ,10,-0.5,9.5);
    h_recoVtxChi2Prob_         = RecoVertexFeatures.make<TH1F>("h_recoVtxChi2Prob" ,"Prob(#chi^{2},ndf);Prob(#chi^{2},ndf);vertices",40,0.,1.);
    h_recoVtxSumPt_            = RecoVertexFeatures.make<TH1F>("h_recoVtxSumPt"    ,"Sum(p^{trks}_{T});Sum(p^{trks}_{T});vertices"  ,100,0.,200.);
    
  }

  // initialize the residuals histograms 

  float dxymax_phi = 2000; 
  float dzmax_phi  = 2000; 
  float dxymax_eta = 3000; 
  float dzmax_eta  = 3000;

  const Int_t mybins_ = 500;

  ///////////////////////////////////////////////////////////////////
  //  
  // Usual plots from refitting the vertex
  // The vertex is refit without the probe track
  //
  ///////////////////////////////////////////////////////////////////

  TFileDirectory AbsTransPhiRes  = fs->mkdir("Abs_Transv_Phi_Residuals");
  TFileDirectory AbsTransEtaRes  = fs->mkdir("Abs_Transv_Eta_Residuals");
		 					  
  TFileDirectory AbsLongPhiRes   = fs->mkdir("Abs_Long_Phi_Residuals");
  TFileDirectory AbsLongEtaRes   = fs->mkdir("Abs_Long_Eta_Residuals");
		 		  
  TFileDirectory NormTransPhiRes = fs->mkdir("Norm_Transv_Phi_Residuals");
  TFileDirectory NormTransEtaRes = fs->mkdir("Norm_Transv_Eta_Residuals");
		 					  
  TFileDirectory NormLongPhiRes  = fs->mkdir("Norm_Long_Phi_Residuals");
  TFileDirectory NormLongEtaRes  = fs->mkdir("Norm_Long_Eta_Residuals");

  TFileDirectory AbsDoubleDiffRes   = fs->mkdir("Abs_DoubleDiffResiduals");
  TFileDirectory NormDoubleDiffRes  = fs->mkdir("Norm_DoubleDiffResiduals");

  for ( int i=0; i<nBins_; ++i ) {

    float phiF = (-TMath::Pi()+i*phipitch_)*(180/TMath::Pi());
    float phiL = (-TMath::Pi()+(i+1)*phipitch_)*(180/TMath::Pi());
    
    float etaF=-2.5+i*etapitch_;
    float etaL=-2.5+(i+1)*etapitch_;
    
    a_dxyPhiResiduals[i] = AbsTransPhiRes.make<TH1F>(Form("histo_dxy_phi_plot%i",i),Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{xy} [#mum];tracks",phiF,phiL),mybins_,-dxymax_phi,dxymax_phi);
    a_dxyEtaResiduals[i] = AbsTransEtaRes.make<TH1F>(Form("histo_dxy_eta_plot%i",i),Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{xy} [#mum];tracks",etaF,etaL),mybins_,-dxymax_eta,dxymax_eta);

    a_dzPhiResiduals[i]  = AbsLongPhiRes.make<TH1F>(Form("histo_dz_phi_plot%i",i),Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f #circ;d_{z} [#mum];tracks",phiF,phiL),mybins_,-dzmax_phi,dzmax_phi);
    a_dzEtaResiduals[i]  = AbsLongEtaRes.make<TH1F>(Form("histo_dz_eta_plot%i",i),Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{z} [#mum];tracks",etaF,etaL),mybins_,-dzmax_eta,dzmax_eta);
    			   				
    n_dxyPhiResiduals[i] = NormTransPhiRes.make<TH1F>(Form("histo_norm_dxy_phi_plot%i",i),Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}} [#mum];tracks",phiF,phiL),mybins_,-dxymax_phi/100.,dxymax_phi/100.);
    n_dxyEtaResiduals[i] = NormTransEtaRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i",i),Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{xy}/#sigma_{d_{xy}} [#mum];tracks",etaF,etaL),mybins_,-dxymax_eta/100.,dxymax_eta/100.);

    n_dzPhiResiduals[i]  = NormLongPhiRes.make<TH1F>(Form("histo_norm_dz_phi_plot%i",i),Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}} [#mum];tracks",phiF,phiL),mybins_,-dzmax_phi/100.,dzmax_phi/100.);
    n_dzEtaResiduals[i]  = NormLongEtaRes.make<TH1F>(Form("histo_norm_dz_eta_plot%i",i),Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{z}/#sigma_{d_{z}} [#mum];tracks",etaF,etaL),mybins_,-dzmax_eta/100.,dzmax_eta/100.);

    for ( int j=0; j<nBins_; ++j ) {
      
      a_dxyResidualsMap[i][j] = AbsDoubleDiffRes.make<TH1F>(Form("histo_dxy_eta_plot%i_phi_plot%i",i,j),Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy} [#mum];tracks",etaF,etaL,phiF,phiL),mybins_,-dzmax_eta,dzmax_eta);
      a_dzResidualsMap[i][j]  = AbsDoubleDiffRes.make<TH1F>(Form("histo_dxy_eta_plot%i_phi_plot%i",i,j),Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z} [#mum];tracks",etaF,etaL,phiF,phiL),mybins_,-dzmax_eta,dzmax_eta);
      				       
      n_dxyResidualsMap[i][j] = NormDoubleDiffRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i_phi_plot%i",i,j),Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}} [#mum];tracks",etaF,etaL,phiF,phiL),mybins_,-dzmax_eta/100,dzmax_eta/100);
      n_dzResidualsMap[i][j]  = NormDoubleDiffRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i_phi_plot%i",i,j),Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}} [#mum];tracks",etaF,etaL,phiF,phiL),mybins_,-dzmax_eta/100,dzmax_eta/100);
    }

  }

  // declaration of the directories

  TFileDirectory MeanTrendsDir   = fs->mkdir("MeanTrends");
  TFileDirectory WidthTrendsDir  = fs->mkdir("WidthTrends");
  TFileDirectory MedianTrendsDir = fs->mkdir("MedianTrends");
  TFileDirectory MADTrendsDir    = fs->mkdir("MADTrends");

  TFileDirectory Mean2DMapsDir   = fs->mkdir("MeanMaps");
  TFileDirectory Width2DMapsDir  = fs->mkdir("WidthMaps");

  Double_t highedge=nBins_-0.5;
  Double_t lowedge=-0.5;

  // means and widths from the fit

  a_dxyPhiMeanTrend  = MeanTrendsDir.make<TH1F> ("means_dxy_phi","#LT d_{xy} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy} #GT [#mum]",nBins_,lowedge,highedge); 
  a_dxyPhiWidthTrend = WidthTrendsDir.make<TH1F>("widths_dxy_phi","#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{xy}} [#mum]",nBins_,lowedge,highedge);
  a_dzPhiMeanTrend   = MeanTrendsDir.make<TH1F> ("means_dz_phi","#LT d_{z} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z} #GT [#mum]",nBins_,lowedge,highedge); 
  a_dzPhiWidthTrend  = WidthTrendsDir.make<TH1F>("widths_dz_phi","#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{z}} [#mum]",nBins_,lowedge,highedge);
  			  
  a_dxyEtaMeanTrend  = MeanTrendsDir.make<TH1F> ("means_dxy_eta","#LT d_{xy} #GT vs #eta sector;#eta (sector);#LT d_{xy} #GT [#mum]",nBins_,lowedge,highedge);
  a_dxyEtaWidthTrend = WidthTrendsDir.make<TH1F>("widths_dxy_eta","#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{xy}} [#mum]",nBins_,lowedge,highedge);
  a_dzEtaMeanTrend   = MeanTrendsDir.make<TH1F> ("means_dz_eta","#LT d_{z} #GT vs #eta sector;#eta (sector);#LT d_{z} #GT [#mum]",nBins_,lowedge,highedge); 
  a_dzEtaWidthTrend  = WidthTrendsDir.make<TH1F>("widths_dz_eta","#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{z}} [#mum]",nBins_,lowedge,highedge);
  			  
  n_dxyPhiMeanTrend  = MeanTrendsDir.make<TH1F> ("norm_means_dxy_phi","#LT d_{xy}/#sigma_{d_{xy}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy}/#sigma_{d_{xy}} #GT",nBins_,lowedge,highedge);
  n_dxyPhiWidthTrend = WidthTrendsDir.make<TH1F>("norm_widths_dxy_phi","width(d_{xy}/#sigma_{d_{xy}}) vs #phi sector;#varphi (sector) [degrees]; width(d_{xy}/#sigma_{d_{xy}})",nBins_,lowedge,highedge);
  n_dzPhiMeanTrend   = MeanTrendsDir.make<TH1F> ("norm_means_dz_phi","#LT d_{z}/#sigma_{d_{z}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z}/#sigma_{d_{z}} #GT",nBins_,lowedge,highedge); 
  n_dzPhiWidthTrend  = WidthTrendsDir.make<TH1F>("norm_widths_dz_phi","width(d_{z}/#sigma_{d_{z}}) vs #phi sector;#varphi (sector) [degrees];width(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);
  			  								
  n_dxyEtaMeanTrend  = MeanTrendsDir.make<TH1F> ("norm_means_dxy_eta","#LT d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector;#eta (sector);#LT d_{xy}/#sigma_{d_{z}} #GT",nBins_,lowedge,highedge);
  n_dxyEtaWidthTrend = WidthTrendsDir.make<TH1F>("norm_widths_dxy_eta","width(d_{xy}/#sigma_{d_{xy}}) vs #eta sector;#eta (sector);width(d_{xy}/#sigma_{d_{z}})",nBins_,lowedge,highedge);
  n_dzEtaMeanTrend   = MeanTrendsDir.make<TH1F> ("norm_means_dz_eta","#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector;#eta (sector);#LT d_{z}/#sigma_{d_{z}} #GT",nBins_,lowedge,highedge);  
  n_dzEtaWidthTrend  = WidthTrendsDir.make<TH1F>("norm_widths_dz_eta","width(d_{z}/#sigma_{d_{z}}) vs #eta sector;#eta (sector);width(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);                        
  
  // 2D maps

  a_dxyMeanMap       =  Mean2DMapsDir.make<TH2F>  ("means_dxy_map","#LT d_{xy} #GT map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  a_dzMeanMap        =  Mean2DMapsDir.make<TH2F>  ("means_dz_map","#LT d_{z} #GT map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  		     
  n_dxyMeanMap       =  Mean2DMapsDir.make<TH2F>  ("norm_means_dxy_map","#LT d_{xy}/#sigma_{d_{xy}} #GT map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  n_dzMeanMap        =  Mean2DMapsDir.make<TH2F>  ("norm_means_dz_map","#LT d_{z}/#sigma_{d_{z}} #GT map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  		     
  a_dxyWidthMap      =  Width2DMapsDir.make<TH2F> ("widths_dxy_map","#sigma_{d_{xy}} map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  a_dzWidthMap       =  Width2DMapsDir.make<TH2F> ("widths_dz_map","#sigma_{d_{z}} map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  		     
  n_dxyWidthMap      =  Width2DMapsDir.make<TH2F> ("norm_widths_dxy_map","width(d_{xy}/#sigma_{d_{xy}}) map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
  n_dzWidthMap       =  Width2DMapsDir.make<TH2F> ("norm_widths_dz_map","width(d_{z}/#sigma_{d_{z}}) map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);

  // medians and MADs

  a_dxyPhiMedianTrend = MedianTrendsDir.make<TH1F>("medians_dxy_phi","Median of d_{xy} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}) [#mum]",nBins_,lowedge,highedge); 			     
  a_dxyPhiMADTrend    = MADTrendsDir.make<TH1F>   ("MADs_dxy_phi","Median absolute deviation of d_{xy} vs #phi sector;#varphi (sector) [degrees];MAD(d_{xy}) [#mum]",nBins_,lowedge,highedge);				    
  a_dzPhiMedianTrend  = MedianTrendsDir.make<TH1F>("medians_dz_phi","Median of d_{z} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}) [#mum]",nBins_,lowedge,highedge); 				    
  a_dzPhiMADTrend     = MADTrendsDir.make<TH1F>   ("MADs_dz_phi","Median absolute deviation of d_{z} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}) [#mum]",nBins_,lowedge,highedge);				    
  
  a_dxyEtaMedianTrend = MedianTrendsDir.make<TH1F>("medians_dxy_eta","Median of d_{xy} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}) [#mum]",nBins_,lowedge,highedge);					      
  a_dxyEtaMADTrend    = MADTrendsDir.make<TH1F>   ("MADs_dxy_eta","Median absolute deviation of d_{xy} vs #eta sector;#eta (sector);MAD(d_{xy}) [#mum]",nBins_,lowedge,highedge);					    
  a_dzEtaMedianTrend  = MedianTrendsDir.make<TH1F>("medians_dz_eta","Median of d_{z} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}) [#mum]",nBins_,lowedge,highedge); 					      
  a_dzEtaMADTrend     = MADTrendsDir.make<TH1F>   ("MADs_dz_eta","Median absolute deviation of d_{z} vs #eta sector;#eta (sector);MAD(d_{z}) [#mum]",nBins_,lowedge,highedge);				          
  
  n_dxyPhiMedianTrend = MedianTrendsDir.make<TH1F>("norm_medians_dxy_phi","Median of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}/#sigma_{d_{xy}})",nBins_,lowedge,highedge); 
  n_dxyPhiMADTrend    = MADTrendsDir.make<TH1F>   ("norm_MADs_dxy_phi","Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees]; MAD(d_{xy}/#sigma_{d_{xy}})",nBins_,lowedge,highedge);   
  n_dzPhiMedianTrend  = MedianTrendsDir.make<TH1F>("norm_medians_dz_phi","Median of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);      
  n_dzPhiMADTrend     = MADTrendsDir.make<TH1F>   ("norm_MADs_dz_phi","Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);	    
  
  n_dxyEtaMedianTrend = MedianTrendsDir.make<TH1F>("norm_medians_dxy_eta","Median of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}/#sigma_{d_{z}})",nBins_,lowedge,highedge);		    
  n_dxyEtaMADTrend    = MADTrendsDir.make<TH1F>   ("norm_MADs_dxy_eta","Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);MAD(d_{xy}/#sigma_{d_{z}})",nBins_,lowedge,highedge);		    
  n_dzEtaMedianTrend  = MedianTrendsDir.make<TH1F>("norm_medians_dz_eta","Median of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);  		    
  n_dzEtaMADTrend     = MADTrendsDir.make<TH1F>   ("norm_MADs_dz_eta","Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);MAD(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);                      


  ///////////////////////////////////////////////////////////////////
  //  
  // plots of biased residuals
  // The vertex is refit without the probe track
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
      
      float phiF = (-TMath::Pi()+i*phipitch_)*(180/TMath::Pi());
      float phiL = (-TMath::Pi()+(i+1)*phipitch_)*(180/TMath::Pi());
      
      float etaF=-2.5+i*etapitch_;
      float etaL=-2.5+(i+1)*etapitch_;
      
      a_dxyPhiBiasResiduals[i] = AbsTransPhiBiasRes.make<TH1F>(Form("histo_dxy_phi_plot%i",i),Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{xy} [#mum];tracks",phiF,phiL),mybins_,-dxymax_phi,dxymax_phi);
      a_dxyEtaBiasResiduals[i] = AbsTransEtaBiasRes.make<TH1F>(Form("histo_dxy_eta_plot%i",i),Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{xy} [#mum];tracks",etaF,etaL),mybins_,-dxymax_eta,dxymax_eta);
      
      a_dzPhiBiasResiduals[i]  = AbsLongPhiBiasRes.make<TH1F>(Form("histo_dz_phi_plot%i",i),Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f #circ;d_{z} [#mum];tracks",phiF,phiL),mybins_,-dzmax_phi,dzmax_phi);
      a_dzEtaBiasResiduals[i]  = AbsLongEtaBiasRes.make<TH1F>(Form("histo_dz_eta_plot%i",i),Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{z} [#mum];tracks",etaF,etaL),mybins_,-dzmax_eta,dzmax_eta);
      
      n_dxyPhiBiasResiduals[i] = NormTransPhiBiasRes.make<TH1F>(Form("histo_norm_dxy_phi_plot%i",i),Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}} [#mum];tracks",phiF,phiL),mybins_,-dxymax_phi/100.,dxymax_phi/100.);
      n_dxyEtaBiasResiduals[i] = NormTransEtaBiasRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i",i),Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{xy}/#sigma_{d_{xy}} [#mum];tracks",etaF,etaL),mybins_,-dxymax_eta/100.,dxymax_eta/100.);
      
      n_dzPhiBiasResiduals[i]  = NormLongPhiBiasRes.make<TH1F>(Form("histo_norm_dz_phi_plot%i",i),Form("%.2f#circ<#varphi^{probe}_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}} [#mum];tracks",phiF,phiL),mybins_,-dzmax_phi/100.,dzmax_phi/100.);
      n_dzEtaBiasResiduals[i]  = NormLongEtaBiasRes.make<TH1F>(Form("histo_norm_dz_eta_plot%i",i),Form("%.2f<#eta^{probe}_{tk}<%.2f;d_{z}/#sigma_{d_{z}} [#mum];tracks",etaF,etaL),mybins_,-dzmax_eta/100.,dzmax_eta/100.);
      
      for ( int j=0; j<nBins_; ++j ) {
	
      a_dxyBiasResidualsMap[i][j] = AbsDoubleDiffBiasRes.make<TH1F>(Form("histo_dxy_eta_plot%i_phi_plot%i",i,j),Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy} [#mum];tracks",etaF,etaL,phiF,phiL),mybins_,-dzmax_eta,dzmax_eta);
      a_dzBiasResidualsMap[i][j]  = AbsDoubleDiffBiasRes.make<TH1F>(Form("histo_dxy_eta_plot%i_phi_plot%i",i,j),Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z} [#mum];tracks",etaF,etaL,phiF,phiL),mybins_,-dzmax_eta,dzmax_eta);
      
      n_dxyBiasResidualsMap[i][j] = NormDoubleDiffBiasRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i_phi_plot%i",i,j),Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}} [#mum];tracks",etaF,etaL,phiF,phiL),mybins_,-dzmax_eta/100,dzmax_eta/100);
      n_dzBiasResidualsMap[i][j]  = NormDoubleDiffBiasRes.make<TH1F>(Form("histo_norm_dxy_eta_plot%i_phi_plot%i",i,j),Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}} [#mum];tracks",etaF,etaL,phiF,phiL),mybins_,-dzmax_eta/100,dzmax_eta/100);
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
    
    a_dxyPhiMeanBiasTrend  = MeanBiasTrendsDir.make<TH1F> ("means_dxy_phi","#LT d_{xy} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy} #GT [#mum]",nBins_,lowedge,highedge); 
    a_dxyPhiWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>("widths_dxy_phi","#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{xy}} [#mum]",nBins_,lowedge,highedge);
    a_dzPhiMeanBiasTrend   = MeanBiasTrendsDir.make<TH1F> ("means_dz_phi","#LT d_{z} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z} #GT [#mum]",nBins_,lowedge,highedge); 
    a_dzPhiWidthBiasTrend  = WidthBiasTrendsDir.make<TH1F>("widths_dz_phi","#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{z}} [#mum]",nBins_,lowedge,highedge);
    
    a_dxyEtaMeanBiasTrend  = MeanBiasTrendsDir.make<TH1F> ("means_dxy_eta","#LT d_{xy} #GT vs #eta sector;#eta (sector);#LT d_{xy} #GT [#mum]",nBins_,lowedge,highedge);
    a_dxyEtaWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>("widths_dxy_eta","#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{xy}} [#mum]",nBins_,lowedge,highedge);
    a_dzEtaMeanBiasTrend   = MeanBiasTrendsDir.make<TH1F> ("means_dz_eta","#LT d_{z} #GT vs #eta sector;#eta (sector);#LT d_{z} #GT [#mum]",nBins_,lowedge,highedge); 
    a_dzEtaWidthBiasTrend  = WidthBiasTrendsDir.make<TH1F>("widths_dz_eta","#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{z}} [#mum]",nBins_,lowedge,highedge);
    
    n_dxyPhiMeanBiasTrend  = MeanBiasTrendsDir.make<TH1F> ("norm_means_dxy_phi","#LT d_{xy}/#sigma_{d_{xy}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy}/#sigma_{d_{xy}} #GT",nBins_,lowedge,highedge);
    n_dxyPhiWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>("norm_widths_dxy_phi","width(d_{xy}/#sigma_{d_{xy}}) vs #phi sector;#varphi (sector) [degrees]; width(d_{xy}/#sigma_{d_{xy}})",nBins_,lowedge,highedge);
    n_dzPhiMeanBiasTrend   = MeanBiasTrendsDir.make<TH1F> ("norm_means_dz_phi","#LT d_{z}/#sigma_{d_{z}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z}/#sigma_{d_{z}} #GT",nBins_,lowedge,highedge); 
    n_dzPhiWidthBiasTrend  = WidthBiasTrendsDir.make<TH1F>("norm_widths_dz_phi","width(d_{z}/#sigma_{d_{z}}) vs #phi sector;#varphi (sector) [degrees];width(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);
    
    n_dxyEtaMeanBiasTrend  = MeanBiasTrendsDir.make<TH1F> ("norm_means_dxy_eta","#LT d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector;#eta (sector);#LT d_{xy}/#sigma_{d_{z}} #GT",nBins_,lowedge,highedge);
    n_dxyEtaWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>("norm_widths_dxy_eta","width(d_{xy}/#sigma_{d_{xy}}) vs #eta sector;#eta (sector);width(d_{xy}/#sigma_{d_{z}})",nBins_,lowedge,highedge);
    n_dzEtaMeanBiasTrend   = MeanBiasTrendsDir.make<TH1F> ("norm_means_dz_eta","#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector;#eta (sector);#LT d_{z}/#sigma_{d_{z}} #GT",nBins_,lowedge,highedge);  
    n_dzEtaWidthBiasTrend  = WidthBiasTrendsDir.make<TH1F>("norm_widths_dz_eta","width(d_{z}/#sigma_{d_{z}}) vs #eta sector;#eta (sector);width(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);                        
    
    // 2D maps
    
    a_dxyMeanBiasMap       =  Mean2DBiasMapsDir.make<TH2F>  ("means_dxy_map","#LT d_{xy} #GT map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    a_dzMeanBiasMap        =  Mean2DBiasMapsDir.make<TH2F>  ("means_dz_map","#LT d_{z} #GT map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    
    n_dxyMeanBiasMap       =  Mean2DBiasMapsDir.make<TH2F>  ("norm_means_dxy_map","#LT d_{xy}/#sigma_{d_{xy}} #GT map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    n_dzMeanBiasMap        =  Mean2DBiasMapsDir.make<TH2F>  ("norm_means_dz_map","#LT d_{z}/#sigma_{d_{z}} #GT map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    
    a_dxyWidthBiasMap      =  Width2DBiasMapsDir.make<TH2F> ("widths_dxy_map","#sigma_{d_{xy}} map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    a_dzWidthBiasMap       =  Width2DBiasMapsDir.make<TH2F> ("widths_dz_map","#sigma_{d_{z}} map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    
    n_dxyWidthBiasMap      =  Width2DBiasMapsDir.make<TH2F> ("norm_widths_dxy_map","width(d_{xy}/#sigma_{d_{xy}}) map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    n_dzWidthBiasMap       =  Width2DBiasMapsDir.make<TH2F> ("norm_widths_dz_map","width(d_{z}/#sigma_{d_{z}}) map;#eta (sector);#varphi (sector) [degrees]",nBins_,lowedge,highedge,nBins_,lowedge,highedge);
    
    // medians and MADs
    
    a_dxyPhiMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>("medians_dxy_phi","Median of d_{xy} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}) [#mum]",nBins_,lowedge,highedge); 			     
    a_dxyPhiMADBiasTrend    = MADBiasTrendsDir.make<TH1F>   ("MADs_dxy_phi","Median absolute deviation of d_{xy} vs #phi sector;#varphi (sector) [degrees];MAD(d_{xy}) [#mum]",nBins_,lowedge,highedge);				    
    a_dzPhiMedianBiasTrend  = MedianBiasTrendsDir.make<TH1F>("medians_dz_phi","Median of d_{z} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}) [#mum]",nBins_,lowedge,highedge); 				    
    a_dzPhiMADBiasTrend     = MADBiasTrendsDir.make<TH1F>   ("MADs_dz_phi","Median absolute deviation of d_{z} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}) [#mum]",nBins_,lowedge,highedge);				    
    
    a_dxyEtaMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>("medians_dxy_eta","Median of d_{xy} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}) [#mum]",nBins_,lowedge,highedge);					      
    a_dxyEtaMADBiasTrend    = MADBiasTrendsDir.make<TH1F>   ("MADs_dxy_eta","Median absolute deviation of d_{xy} vs #eta sector;#eta (sector);MAD(d_{xy}) [#mum]",nBins_,lowedge,highedge);					    
    a_dzEtaMedianBiasTrend  = MedianBiasTrendsDir.make<TH1F>("medians_dz_eta","Median of d_{z} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}) [#mum]",nBins_,lowedge,highedge); 					      
    a_dzEtaMADBiasTrend     = MADBiasTrendsDir.make<TH1F>   ("MADs_dz_eta","Median absolute deviation of d_{z} vs #eta sector;#eta (sector);MAD(d_{z}) [#mum]",nBins_,lowedge,highedge);				          
    
    n_dxyPhiMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>("norm_medians_dxy_phi","Median of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}/#sigma_{d_{xy}})",nBins_,lowedge,highedge); 
    n_dxyPhiMADBiasTrend    = MADBiasTrendsDir.make<TH1F>   ("norm_MADs_dxy_phi","Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees]; MAD(d_{xy}/#sigma_{d_{xy}})",nBins_,lowedge,highedge);   
    n_dzPhiMedianBiasTrend  = MedianBiasTrendsDir.make<TH1F>("norm_medians_dz_phi","Median of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);      
    n_dzPhiMADBiasTrend     = MADBiasTrendsDir.make<TH1F>   ("norm_MADs_dz_phi","Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);	    
    
    n_dxyEtaMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>("norm_medians_dxy_eta","Median of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}/#sigma_{d_{z}})",nBins_,lowedge,highedge);		    
    n_dxyEtaMADBiasTrend    = MADBiasTrendsDir.make<TH1F>   ("norm_MADs_dxy_eta","Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);MAD(d_{xy}/#sigma_{d_{z}})",nBins_,lowedge,highedge);		    
    n_dzEtaMedianBiasTrend  = MedianBiasTrendsDir.make<TH1F>("norm_medians_dz_eta","Median of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);  		    
    n_dzEtaMADBiasTrend     = MADBiasTrendsDir.make<TH1F>   ("norm_MADs_dz_eta","Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);MAD(d_{z}/#sigma_{d_{z}})",nBins_,lowedge,highedge);                      
    
  }
}
// ------------ method called once each job just after ending the event loop  ------------
void PrimaryVertexValidation::endJob() 
{

  std::cout<<"######################################"<<std::endl;
  std::cout<<"Number of analyzed events: "<<Nevt_<<std::endl;
  std::cout<<"######################################"<<std::endl;

  if(useTracksFromRecoVtx_){

    FillTrendPlot(a_dxyPhiMeanBiasTrend ,a_dxyPhiBiasResiduals,"mean","phi");  
    FillTrendPlot(a_dxyPhiWidthBiasTrend,a_dxyPhiBiasResiduals,"width","phi");
    FillTrendPlot(a_dzPhiMeanBiasTrend  ,a_dzPhiBiasResiduals ,"mean","phi");   
    FillTrendPlot(a_dzPhiWidthBiasTrend ,a_dzPhiBiasResiduals ,"width","phi");  
    
    FillTrendPlot(a_dxyEtaMeanBiasTrend ,a_dxyEtaBiasResiduals,"mean","eta"); 
    FillTrendPlot(a_dxyEtaWidthBiasTrend,a_dxyEtaBiasResiduals,"width","eta");
    FillTrendPlot(a_dzEtaMeanBiasTrend  ,a_dzEtaBiasResiduals ,"mean","eta"); 
    FillTrendPlot(a_dzEtaWidthBiasTrend ,a_dzEtaBiasResiduals ,"width","eta");
    
    FillTrendPlot(n_dxyPhiMeanBiasTrend ,n_dxyPhiBiasResiduals,"mean","phi"); 
    FillTrendPlot(n_dxyPhiWidthBiasTrend,n_dxyPhiBiasResiduals,"width","phi");
    FillTrendPlot(n_dzPhiMeanBiasTrend  ,n_dzPhiBiasResiduals ,"mean","phi"); 
    FillTrendPlot(n_dzPhiWidthBiasTrend ,n_dzPhiBiasResiduals ,"width","phi");
    
    FillTrendPlot(n_dxyEtaMeanBiasTrend ,n_dxyEtaBiasResiduals,"mean","eta"); 
    FillTrendPlot(n_dxyEtaWidthBiasTrend,n_dxyEtaBiasResiduals,"width","eta");
    FillTrendPlot(n_dzEtaMeanBiasTrend  ,n_dzEtaBiasResiduals ,"mean","eta"); 
    FillTrendPlot(n_dzEtaWidthBiasTrend ,n_dzEtaBiasResiduals ,"width","eta");
    
    // medians and MADs	  
    
    FillTrendPlot(a_dxyPhiMedianBiasTrend,a_dxyPhiBiasResiduals,"median","phi");  
    FillTrendPlot(a_dxyPhiMADBiasTrend   ,a_dxyPhiBiasResiduals,"mad","phi"); 
    FillTrendPlot(a_dzPhiMedianBiasTrend ,a_dzPhiBiasResiduals ,"median","phi");  
    FillTrendPlot(a_dzPhiMADBiasTrend    ,a_dzPhiBiasResiduals ,"mad","phi"); 
    
    FillTrendPlot(a_dxyEtaMedianBiasTrend,a_dxyEtaBiasResiduals,"median","eta");  
    FillTrendPlot(a_dxyEtaMADBiasTrend   ,a_dxyEtaBiasResiduals,"mad","eta"); 
    FillTrendPlot(a_dzEtaMedianBiasTrend ,a_dzEtaBiasResiduals ,"median","eta");  
    FillTrendPlot(a_dzEtaMADBiasTrend    ,a_dzEtaBiasResiduals ,"mad","eta"); 
    
    FillTrendPlot(n_dxyPhiMedianBiasTrend,n_dxyPhiBiasResiduals,"median","phi");  
    FillTrendPlot(n_dxyPhiMADBiasTrend   ,n_dxyPhiBiasResiduals,"mad","phi"); 
    FillTrendPlot(n_dzPhiMedianBiasTrend ,n_dzPhiBiasResiduals ,"median","phi");  
    FillTrendPlot(n_dzPhiMADBiasTrend    ,n_dzPhiBiasResiduals ,"mad","phi"); 
    
    FillTrendPlot(n_dxyEtaMedianBiasTrend,n_dxyEtaBiasResiduals,"median","eta");  
    FillTrendPlot(n_dxyEtaMADBiasTrend   ,n_dxyEtaBiasResiduals,"mad","eta"); 
    FillTrendPlot(n_dzEtaMedianBiasTrend ,n_dzEtaBiasResiduals ,"median","eta");  
    FillTrendPlot(n_dzEtaMADBiasTrend    ,n_dzEtaBiasResiduals ,"mad","eta"); 
    
  }

  FillTrendPlot(a_dxyPhiMeanTrend ,a_dxyPhiResiduals,"mean","phi");  
  FillTrendPlot(a_dxyPhiWidthTrend,a_dxyPhiResiduals,"width","phi");
  FillTrendPlot(a_dzPhiMeanTrend  ,a_dzPhiResiduals ,"mean","phi");   
  FillTrendPlot(a_dzPhiWidthTrend ,a_dzPhiResiduals ,"width","phi");  
  
  FillTrendPlot(a_dxyEtaMeanTrend ,a_dxyEtaResiduals,"mean","eta"); 
  FillTrendPlot(a_dxyEtaWidthTrend,a_dxyEtaResiduals,"width","eta");
  FillTrendPlot(a_dzEtaMeanTrend  ,a_dzEtaResiduals ,"mean","eta"); 
  FillTrendPlot(a_dzEtaWidthTrend ,a_dzEtaResiduals ,"width","eta");
  
  FillTrendPlot(n_dxyPhiMeanTrend ,n_dxyPhiResiduals,"mean","phi"); 
  FillTrendPlot(n_dxyPhiWidthTrend,n_dxyPhiResiduals,"width","phi");
  FillTrendPlot(n_dzPhiMeanTrend  ,n_dzPhiResiduals ,"mean","phi"); 
  FillTrendPlot(n_dzPhiWidthTrend ,n_dzPhiResiduals ,"width","phi");
  
  FillTrendPlot(n_dxyEtaMeanTrend ,n_dxyEtaResiduals,"mean","eta"); 
  FillTrendPlot(n_dxyEtaWidthTrend,n_dxyEtaResiduals,"width","eta");
  FillTrendPlot(n_dzEtaMeanTrend  ,n_dzEtaResiduals ,"mean","eta"); 
  FillTrendPlot(n_dzEtaWidthTrend ,n_dzEtaResiduals ,"width","eta");
    
  // medians and MADs	  
  
  FillTrendPlot(a_dxyPhiMedianTrend,a_dxyPhiResiduals,"median","phi");  
  FillTrendPlot(a_dxyPhiMADTrend   ,a_dxyPhiResiduals,"mad","phi"); 
  FillTrendPlot(a_dzPhiMedianTrend ,a_dzPhiResiduals ,"median","phi");  
  FillTrendPlot(a_dzPhiMADTrend    ,a_dzPhiResiduals ,"mad","phi"); 
  
  FillTrendPlot(a_dxyEtaMedianTrend,a_dxyEtaResiduals,"median","eta");  
  FillTrendPlot(a_dxyEtaMADTrend   ,a_dxyEtaResiduals,"mad","eta"); 
  FillTrendPlot(a_dzEtaMedianTrend ,a_dzEtaResiduals ,"median","eta");  
  FillTrendPlot(a_dzEtaMADTrend    ,a_dzEtaResiduals ,"mad","eta"); 
  
  FillTrendPlot(n_dxyPhiMedianTrend,n_dxyPhiResiduals,"median","phi");  
  FillTrendPlot(n_dxyPhiMADTrend   ,n_dxyPhiResiduals,"mad","phi"); 
  FillTrendPlot(n_dzPhiMedianTrend ,n_dzPhiResiduals ,"median","phi");  
  FillTrendPlot(n_dzPhiMADTrend    ,n_dzPhiResiduals ,"mad","phi"); 
  
  FillTrendPlot(n_dxyEtaMedianTrend,n_dxyEtaResiduals,"median","eta");  
  FillTrendPlot(n_dxyEtaMADTrend   ,n_dxyEtaResiduals,"mad","eta"); 
  FillTrendPlot(n_dzEtaMedianTrend ,n_dzEtaResiduals ,"median","eta");  
  FillTrendPlot(n_dzEtaMADTrend    ,n_dzEtaResiduals ,"mad","eta"); 
    
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
  BSx0_ = -999.;
  BSy0_ = -999.;
  BSz0_ = -999.;
  Beamsigmaz_=-999.;
  Beamdxdz_=-999.;   
  BeamWidthX_=-999.;
  BeamWidthY_=-999.;

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
    DOFUnbiasedVertex_[i]=0;
    sumOfWeightsUnbiasedVertex_[i]=0;
    tracksUsedForVertexing_[i]=0;
    dxyFromMyVertex_[i]=0;
    dzFromMyVertex_[i]=0;
    dszFromMyVertex_[i]=0;
    dxyErrorFromMyVertex_[i]=0; 
    dzErrorFromMyVertex_[i]=0; 
    IPTsigFromMyVertex_[i]=0;   
    IPLsigFromMyVertex_[i]=0;    
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
  delete [] y; y = 0;  

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
void PrimaryVertexValidation::FillTrendPlot(TH1F* trendPlot, TH1F* residualsPlot[100], const TString& fitPar_, const TString& var_)
//*************************************************************
{
 
  float phiInterval = (360.)/nBins_;
  float etaInterval = 5./nBins_;
  
  for ( int i=0; i<nBins_; ++i ) {
    
    char phipositionString[129];
    float phiposition = (-180+i*phiInterval)+(phiInterval/2);
    sprintf(phipositionString,"%.f",phiposition);
    
    char etapositionString[129];
    float etaposition = (-2.5+i*etaInterval)+(etaInterval/2);
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
      std::cout<<"PrimaryVertexValidation::FillTrendPlot() "<<fitPar_<<" unknown estimator!"<<std::endl;
    }

    if(var_=="eta"){
      trendPlot->GetXaxis()->SetBinLabel(i+1,phipositionString); 
    } else if(var_=="phi"){
      trendPlot->GetXaxis()->SetBinLabel(i+1,etapositionString); 
    } else {
      std::cout<<"PrimaryVertexValidation::FillTrendPlot() "<<var_<<" unknown track parameter!"<<std::endl;
    }
  }
}



//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexValidation);
