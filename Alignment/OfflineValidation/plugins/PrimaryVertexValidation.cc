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

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TROOT.h"
#include "TChain.h"
#include "TNtuple.h"
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

 const int kBPIX = PixelSubdetector::PixelBarrel;
 const int kFPIX = PixelSubdetector::PixelEndcap;

// Constructor

PrimaryVertexValidation::PrimaryVertexValidation(const edm::ParameterSet& iConfig)
  : theConfig(iConfig), 
    Nevt_(0),
    theTrackFilter_(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters")),
    rootFile_(0),
    rootTree_(0)
{
  //now do what ever initialization is needed
  debug_    = iConfig.getParameter<bool>       ("Debug");  
  TrackCollectionTag_      = iConfig.getParameter<edm::InputTag>("TrackCollectionTag");  
  filename_ = iConfig.getParameter<std::string>("OutputFileName");

  SetVarToZero();
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

  /*
  double OfflineVertexX = 0.;
  double OfflineVertexY = 0.;
  double OfflineVertexZ = 0.;
  edm::Handle<reco::VertexCollection> vertices;
  try {
    iEvent.getByLabel("offlinePrimaryVertices", vertices);
  } catch (...) {
    std::cout << "No offlinePrimaryVertices found!" << std::endl;
  }
  if ( vertices.isValid() ) {
    OfflineVertexX = (*vertices)[0].x();
    OfflineVertexY = (*vertices)[0].y();
    OfflineVertexZ = (*vertices)[0].z();
  }
  */
  
  //=======================================================
  // Retrieve Beamspot information (only for reco)
  //=======================================================

  /*
    BeamSpot beamSpot;
    edm::Handle<BeamSpot> beamSpotHandle;
    iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);
    
    if ( beamSpotHandle.isValid() )
    {
    beamSpot = *beamSpotHandle;
    
    } else
    {
    edm::LogInfo("PrimaryVertexValidation")
    << "No beam spot available from EventSetup \n";
    }
    
    double BSx0 = beamSpot.x0();
    double BSy0 = beamSpot.y0();
    double BSz0 = beamSpot.z0();
    
    if(debug_)
    std::cout<<"Beamspot x:"<<BSx0<<" y:"<<BSy0<<" z:"<<BSz0<std::<std::endl; 
    
    //double sigmaz = beamSpot.sigmaZ();
    //double dxdz = beamSpot.dxdz();
    //double BeamWidth = beamSpot.BeamWidth();
    */
  
  //=======================================================
  // Starts here ananlysis
  //=======================================================
  
  if(debug_)
    std::cout<<"PrimaryVertexValidation::analyze() looping over "<<trackCollectionHandle->size()<< "tracks." <<std::endl;       
  
  unsigned int i = 0;   
  for(reco::TrackCollection::const_iterator track = trackCollectionHandle->begin(); track!= trackCollectionHandle->end(); ++track, ++i)
    {
      if ( nTracks_ >= nMaxtracks_ ) {
	std::cout << " PrimaryVertexValidation::analyze() : Warning - Number of tracks: " << nTracks_ << " , greater than " << nMaxtracks_ << std::endl;
	continue;
      }

      pt_[nTracks_]       = track->pt();
      p_[nTracks_]        = track->p();
      nhits_[nTracks_]    = track->numberOfValidHits();
      eta_[nTracks_]      = track->eta();
      phi_[nTracks_]      = track->phi();
      chi2_[nTracks_]     = track->chi2();
      chi2ndof_[nTracks_] = track->normalizedChi2();
      charge_[nTracks_]   = track->charge();
      qoverp_[nTracks_]   = track->qoverp();
      dz_[nTracks_]       = track->dz();
      dxy_[nTracks_]      = track->dxy();
      xPCA_[nTracks_]     = track->vertex().x();
      yPCA_[nTracks_]     = track->vertex().y();
      zPCA_[nTracks_]     = track->vertex().z(); 
    
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
      
      for (trackingRecHit_iterator iHit = track->recHitsBegin(); iHit != track->recHitsEnd(); ++iHit) {
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

      nhits1D_[nTracks_]     =nRecHit1D;
      nhits2D_[nTracks_]     =nRecHit2D;
      nhitsBPIX_[nTracks_]   =nhitinBPIX;
      nhitsFPIX_[nTracks_]   =nhitinFPIX;
      nhitsTIB_[nTracks_]    =nhitinTIB;
      nhitsTID_[nTracks_]    =nhitinTID;
      nhitsTOB_[nTracks_]    =nhitinTOB;
      nhitsTEC_[nTracks_]    =nhitinTEC;

      //=======================================================
      // Good tracks for vertexing selection
      //=======================================================  

      reco::TrackRef trackref(trackCollectionHandle,i);
      bool hasTheProbeFirstPixelLayerHit = false;
      hasTheProbeFirstPixelLayerHit = this->hasFirstLayerPixelHits(trackref);
      reco::TransientTrack theTTRef = reco::TransientTrack(trackref, &*theMGField, theTrackingGeometry );
      if (theTrackFilter_(theTTRef)&&hasTheProbeFirstPixelLayerHit){
	isGoodTrack_[nTracks_]=1;
      }
      
      //=======================================================
      // Fit unbiased vertex
      //=======================================================  
      
      std::vector<reco::TransientTrack> transientTracks;
      for(size_t j = 0; j < trackCollectionHandle->size(); j++)
	{
	  reco::TrackRef tk(trackCollectionHandle, j);
	  if( tk == trackref ) continue;
	  bool hasTheTagFirstPixelLayerHit = false;
	  hasTheTagFirstPixelLayerHit = this->hasFirstLayerPixelHits(tk);
	  reco::TransientTrack theTT = reco::TransientTrack(tk, &*theMGField, theTrackingGeometry );
	  if (theTrackFilter_(theTT)&&hasTheTagFirstPixelLayerHit){
	    transientTracks.push_back(theTT);
	  }
	}
      
      if(transientTracks.size() > 2){
	
	if(debug_)
	  std::cout <<"PrimaryVertexValidation::analyze() :Transient Track Collection size: "<<transientTracks.size()<<std::endl;
	
	try{

	  VertexFitter<5>* fitter = new AdaptiveVertexFitter;
	  TransientVertex theFittedVertex = fitter->vertex(transientTracks);
	  
	  if(theFittedVertex.isValid ()){
	    
	    if(theFittedVertex.hasTrackWeight()){
	      for(size_t rtracks= 0; rtracks < transientTracks.size(); rtracks++){
		sumOfWeightsUnbiasedVertex_[nTracks_] += theFittedVertex.trackWeight(transientTracks[rtracks]);
	      }
	    }

	    const math::XYZPoint myVertex(theFittedVertex.position().x(),theFittedVertex.position().y(),theFittedVertex.position().z());
	    hasRecVertex_[nTracks_]    = 1;
	    xUnbiasedVertex_[nTracks_] = theFittedVertex.position().x();
	    yUnbiasedVertex_[nTracks_] = theFittedVertex.position().y();
	    zUnbiasedVertex_[nTracks_] = theFittedVertex.position().z();
	    chi2normUnbiasedVertex_[nTracks_] = theFittedVertex.normalisedChiSquared();
	    chi2UnbiasedVertex_[nTracks_] = theFittedVertex.totalChiSquared();
	    DOFUnbiasedVertex_[nTracks_] = theFittedVertex.degreesOfFreedom();   
	    tracksUsedForVertexing_[nTracks_] = transientTracks.size();
	    dxyFromMyVertex_[nTracks_] = track->dxy(myVertex);
	    dzFromMyVertex_[nTracks_]  = track->dz(myVertex);
	    dszFromMyVertex_[nTracks_] = track->dsz(myVertex);
	    
	    if(debug_){
	      std::cout<<"PrimaryVertexValidation::analyze() :myVertex.x()= "<<myVertex.x()<<" myVertex.y()= "<<myVertex.y()<<" theFittedVertex.z()= "<<myVertex.z()<<std::endl;	  
	      std::cout<<"PrimaryVertexValidation::analyze() : track->dz(myVertex)= "<<track->dz(myVertex)<<std::endl;
	      std::cout<<"PrimaryVertexValidation::analyze() : zPCA -myVertex.z() = "<<(track->vertex().z() -myVertex.z() )<<std::endl; 
	    }// ends if debug_
	  } // ends if the fitted vertex is Valid
	  
	}  catch ( cms::Exception& er ) {
	  LogTrace("PrimaryVertexValidation::analyze RECO")<<"caught std::exception "<<er.what()<<std::endl;
	}
      } //ends if transientTracks.size() > 2
      
      else {
	std::cout << "PrimaryVertexValidation::analyze() :Not enough tracks to make a vertex.  Returns no vertex info" << std::endl;
      }

      ++nTracks_;  
	
      if(debug_)
	std::cout<< "Track "<<i<<" : pT = "<<track->pt()<<std::endl;
      
    }// for loop on tracks
  
  rootTree_->Fill();
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
bool PrimaryVertexValidation::hasFirstLayerPixelHits(const reco::TrackRef track)
{
  bool accepted = false;
  // hit pattern of the track
  const reco::HitPattern& p = track->hitPattern();      
  for (int i=0; i<p.numberOfHits(); i++) {
    uint32_t pattern = p.getHitPattern(i);   
    if (p.pixelBarrelHitFilter(pattern) || p.pixelEndcapHitFilter(pattern) ) {
      if (p.getLayer(pattern) == 1) {
	if (p.validHitFilter(pattern)) {
	  accepted = true;
	}
      }
    }
  }
  return accepted;
} 

// ------------ method called once each job before begining the event loop  ------------
void PrimaryVertexValidation::beginJob()
{
  edm::LogInfo("beginJob") << "Begin Job" << std::endl;
  // Define TTree for output

  Nevt_    = 0;
  
  rootFile_ = new TFile(filename_.c_str(),"recreate");
  rootTree_ = new TTree("tree","PV Validation tree");
  
  // Track Paramters 
  rootTree_->Branch("nTracks",&nTracks_,"nTracks/I");
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
  rootTree_->Branch("phi",&phi_,"phi[nTracks]/D");
  rootTree_->Branch("chi2",&chi2_,"chi2[nTracks]/D");
  rootTree_->Branch("chi2ndof",&chi2ndof_,"chi2ndof[nTracks]/D");
  rootTree_->Branch("charge",&charge_,"charge[nTracks]/I");
  rootTree_->Branch("qoverp",&qoverp_,"qoverp[nTracks]/D");
  rootTree_->Branch("dz",&dz_,"dz[nTracks]/D");
  rootTree_->Branch("dxy",&dxy_,"dxy[nTracks]/D");
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
  rootTree_->Branch("hasRecVertex",&hasRecVertex_,"hasRecVertex[nTracks]/I");
  rootTree_->Branch("isGoodTrack",&isGoodTrack_,"isGoodTrack[nTracks]/I");
}

// ------------ method called once each job just after ending the event loop  ------------
void PrimaryVertexValidation::endJob() 
{

  std::cout<<"######################################"<<std::endl;
  std::cout<<"Number of analyzed events: "<<Nevt_<<std::endl;
  std::cout<<"######################################"<<std::endl;
  
   if ( rootFile_ ) {
     rootFile_->Write();
     rootFile_->Close();
   }
}

void PrimaryVertexValidation::SetVarToZero() {
  
  nTracks_ = 0;
  for ( int i=0; i<nMaxtracks_; ++i ) {
    pt_[i]        = 0;
    p_[i]         = 0;
    nhits_[i]     = 0;
    nhits1D_[i]   = 0;
    nhits2D_[i]   = 0;
    nhitsBPIX_[i]  = 0;
    nhitsFPIX_[i]  = 0;
    nhitsTIB_[i]   = 0;
    nhitsTID_[i]   = 0;
    nhitsTOB_[i]   = 0;
    nhitsTEC_[i]   = 0;
    eta_[i]       = 0;
    phi_[i]       = 0;
    chi2_[i]      = 0;
    chi2ndof_[i]  = 0;
    charge_[i]    = 0;
    qoverp_[i]    = 0;
    dz_[i]        = 0;
    dxy_[i]       = 0;
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
    hasRecVertex_[i] = 0;
    isGoodTrack_[i]  = 0;
  } 
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexValidation);
