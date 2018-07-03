// -*- C++ -*-
//
// Package:    LhcTrackAnalyzer
// Class:      LhcTrackAnalyzer
// 
/**\class LhcTrackAnalyzer LhcTrackAnalyzer.cc MySub/LhcTrackAnalyzer/src/LhcTrackAnalyzer.cc

Originally written by M.Musich
Expanded by A. Bonato

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//

// updated to 25/2/2009 5.30 pm

//
//

// system include files
#include <memory>


// user include files
#include "Alignment/HIPAlignmentAlgorithm/interface/LhcTrackAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TROOT.h"
#include "TChain.h"
#include "TNtuple.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <DataFormats/GeometrySurface/interface/Surface.h>
#include <DataFormats/GeometrySurface/interface/GloballyPositioned.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Constructor

LhcTrackAnalyzer::LhcTrackAnalyzer(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  debug_    = iConfig.getParameter<bool>       ("Debug");  
  TrackCollectionTag_      = iConfig.getParameter<edm::InputTag>("TrackCollectionTag");  
  PVtxCollectionTag_      = iConfig.getParameter<edm::InputTag>("PVtxCollectionTag");  
  filename_ = iConfig.getParameter<std::string>("OutputFileName");
}
   
// Destructor
LhcTrackAnalyzer::~LhcTrackAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
LhcTrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  using namespace std;

  //=======================================================
  // Initialize Root-tuple variables
  //=======================================================

  SetVarToZero();
  
  //=======================================================
  // Retrieve the Track information
  //=======================================================
  
  Handle< TrackCollection>  trackCollectionHandle;
  iEvent.getByLabel(TrackCollectionTag_, trackCollectionHandle);
  Handle<VertexCollection>  vertexCollectionHandle;
  iEvent.getByLabel(PVtxCollectionTag_, vertexCollectionHandle);
  for(VertexCollection::const_iterator vtx = vertexCollectionHandle->begin();vtx!=vertexCollectionHandle->end(); ++vtx)
    {
      if(vtx==vertexCollectionHandle->begin()){
	if(vtx->isFake())goodvtx_=false;
	else goodvtx_=true;
      }
      else break;
    }



  goodbx_=true;
  // int bx = iEvent.bunchCrossing();
  //if (bx==51 || bx==2724) goodbx_=true;
  

 
  run_=iEvent.id().run();
  event_=iEvent.id().event();
 
  if(debug_)
    cout<<"LhcTrackAnalyzer::analyze() looping over "<< trackCollectionHandle->size()<< "tracks." << endl;    
  
  // unsigned int i = 0;   
  for(TrackCollection::const_iterator track = trackCollectionHandle->begin(); track!= trackCollectionHandle->end(); ++track)
    {
      if ( nTracks_ >= nMaxtracks_ ) {
	std::cout << " LhcTrackAnalyzer::analyze() : Warning - Run "<< run_<<" Event "<< event_<<"\tNumber of tracks: " <<  trackCollectionHandle->size()<< " , greater than " << nMaxtracks_ << std::endl;
	  continue;
	 
	}
	  pt_[nTracks_]       = track->pt();
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
	  validhits_[nTracks_][0]=track->numberOfValidHits();
	  validhits_[nTracks_][1]=track->hitPattern().numberOfValidPixelBarrelHits();
	  validhits_[nTracks_][2]=track->hitPattern().numberOfValidPixelEndcapHits();
	  validhits_[nTracks_][3]=track->hitPattern().numberOfValidStripTIBHits();
	  validhits_[nTracks_][4]=track->hitPattern().numberOfValidStripTIDHits();
	  validhits_[nTracks_][5]=track->hitPattern().numberOfValidStripTOBHits();
	  validhits_[nTracks_][6]=track->hitPattern().numberOfValidStripTECHits();



	  int myalgo=-88;
	  if(track->algo()==reco::TrackBase::undefAlgorithm)myalgo=0;
	  if(track->algo()==reco::TrackBase::ctf)myalgo=1;
	  if(track->algo()==reco::TrackBase::initialStep)myalgo=4;
	  if(track->algo()==reco::TrackBase::lowPtTripletStep)myalgo=5;
	  if(track->algo()==reco::TrackBase::pixelPairStep)myalgo=6;
	  if(track->algo()==reco::TrackBase::detachedTripletStep)myalgo=7;
	  if(track->algo()==reco::TrackBase::mixedTripletStep)myalgo=8;
	  if(track->algo()==reco::TrackBase::pixelLessStep)myalgo=9;
	  if(track->algo()==reco::TrackBase::tobTecStep)myalgo=10;
	  if(track->algo()==reco::TrackBase::jetCoreRegionalStep)myalgo=11;
    // This class is pending the migration to Phase1 tracks
	  if(track->algo() == reco::TrackBase::highPtTripletStep ||
	     track->algo() == reco::TrackBase::lowPtQuadStep ||
	     track->algo() == reco::TrackBase::detachedQuadStep) {
	  throw cms::Exception("Not implemented") << "LhcTrackAnalyzer does not yet support phase1 tracks, encountered one from algo " << reco::TrackBase::algoName(track->algo());
	  	}
	  trkAlgo_[nTracks_]  = myalgo;

	  int myquality=-99;
	  if(track->quality(reco::TrackBase::undefQuality))myquality=-1;
	  if(track->quality(reco::TrackBase::loose))myquality=0;
	  if(track->quality(reco::TrackBase::tight))myquality=1;
	  if(track->quality(reco::TrackBase::highPurity))myquality=2;
	  //if(track->quality(reco::TrackBase::confirmed))myquality=3;
	  // if(track->quality(reco::TrackBase::goodIterative))myquality=4;
	  // if(track->quality(reco::TrackBase::qualitySize))myquality=5;	  
	  trkQuality_[nTracks_]= myquality;

	  if(track->quality(reco::TrackBase::highPurity))isHighPurity_[nTracks_]=1;
	  else isHighPurity_[nTracks_]=0;
	  nTracks_++;


	

     
    }//end loop on tracks

  for(int d=0;d<nTracks_;++d){
    if(abs(trkQuality_[d])>5)cout<<"MYQUALITY!!! " <<trkQuality_[d] <<" at track # "<<d<<"/"<< nTracks_<<endl;
  }

 


  rootTree_->Fill();
} 


// ------------ method called once each job before begining the event loop  ------------
void LhcTrackAnalyzer::beginJob()
{
  edm::LogInfo("beginJob") << "Begin Job" << std::endl;
  // Define TTree for output
  rootFile_ = new TFile(filename_.c_str(),"recreate");
  rootTree_ = new TTree("tree","Lhc Track tree");
  
  // Track Paramters 
  rootTree_->Branch("run",&run_,"run/I");
  rootTree_->Branch("event",&event_,"event/I");
  rootTree_->Branch("goodbx",&goodbx_,"goodbx/O");
  rootTree_->Branch("goodvtx",&goodvtx_,"goodvtx/O");
  rootTree_->Branch("nTracks",&nTracks_,"nTracks/I");
  rootTree_->Branch("pt",&pt_,"pt[nTracks]/D");
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
 rootTree_->Branch("isHighPurity",&isHighPurity_,"isHighPurity[nTracks]/I");
  rootTree_->Branch("trkQuality",&trkQuality_,"trkQuality[nTracks]/I");
  rootTree_->Branch("trkAlgo",&trkAlgo_,"trkAlgo[nTracks]/I");
  rootTree_->Branch("nValidHits",&validhits_,"nValidHits[nTracks][7]/I");
 
  
}

// ------------ method called once each job just after ending the event loop  ------------
void LhcTrackAnalyzer::endJob() 
{
   if ( rootFile_ ) {
     rootFile_->Write();
     rootFile_->Close();
   }

   

}

void LhcTrackAnalyzer::SetVarToZero() {
  run_=-1;
  event_=-99;
  nTracks_ = 0;
  for ( int i=0; i<nMaxtracks_; ++i ) {
    pt_[i]        = 0;
    eta_[i]       = 0;
    phi_[i]       = 0;
    chi2_[i]      = 0;
    chi2ndof_[i]  = 0;
    charge_[i]    = 0;
    qoverp_[i]    = 0;
    dz_[i]        = 0;
    dxy_[i]       = 0;
    xPCA_[i]        = 0;
    yPCA_[i]        = 0;
    zPCA_[i]        = 0;
    trkQuality_[i]  = 0;
    trkAlgo_[i]     = -1;
    isHighPurity_[i]=-3;
    for(int j=0;j<7;j++){
      validhits_[nTracks_][j]=-1*j;
    }
  }


}

//define this as a plug-in
DEFINE_FWK_MODULE(LhcTrackAnalyzer);
