// -*- C++ -*-
//
//
// Producer for a L1TkEGTauParticle
// 

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkEGTauParticle.h"
#include "L1Trigger/Phase2L1Taus/interface/L1TkEGTauEtComparator.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h" 

// for L1Tracks:
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <string>
#include "TMath.h"

//#define DEBUG

using namespace l1t ;
//
// class declaration
//

class L1TkEGTauParticleProducer : public edm::EDProducer {
public:
  
  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;
  typedef edm::Ptr< L1TTTrackType > L1TTTrackRefPtr;
  typedef std::vector< L1TTTrackRefPtr > L1TTTrackRefPtr_Collection;
  typedef edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > L1TTStubRef;
  typedef edm::Ref< EGammaBxCollection > EGammaRef ;
  typedef std::vector< EGammaRef > EGammaVectorRef ;

  explicit L1TkEGTauParticleProducer(const edm::ParameterSet&);
  ~L1TkEGTauParticleProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  class TrackPtComparator{
    
    unsigned int nFitParams_;
  public:
    TrackPtComparator(unsigned int nFitParams){ nFitParams_ = nFitParams;}
    bool operator() (const L1TTTrackRefPtr trackA, L1TTTrackRefPtr trackB ) const {
      return ( trackA->getMomentum(nFitParams_).perp() > trackB->getMomentum(nFitParams_).perp() );
    }
  };

  struct EtComparator{ 
    bool operator() (const EGammaRef objA, const EGammaRef objB) const { return ( objA->et() > objB->et() ); }
  };

  float CorrectedEta(float eta, float zTrack);
  
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void GetShrinkingConeSizes(float tk_pt,
			     const float shrinkCone_Constant,
			     const float sigCone_dRCutoff,
  			     float &sigCone_dRMin,
  			     float &sigCone_dRMax,
  			     float &isoCone_dRMin,
  			     float &isoCone_dRMax,
  			     const bool isoCone_useCone);
  
  float CalculateVtxIso(std::vector< L1TTTrackRefPtr > allTracks,
			std::vector< unsigned int > clustTracksIndx,
			bool useIsoCone=false); 

  // float CalculateRelIso(std::vector< L1TTTrackRefPtr > allTracks,
  // 			std::vector< unsigned int > clustTracksIndx,
  //                    const float deltaZ0_max, 
  // 			bool useIsoCone=false); 
  
  // ----------member data ---------------------------

  // Label of the objects which are created (e.g. "TkEG")
  std::string label;
 
  // L1 Tracks
  unsigned int cfg_tk_nFitParams;       // Number of Fit Parameters: 4 or 5 ? (pT, eta, phi, z0, d0)
  float cfg_tk_minPt;                   // Min pT applied on all L1TTTracks [GeV]
  float cfg_tk_minEta;                  // Min |eta| applied on all L1TTTracks [unitless]
  float cfg_tk_maxEta;                  // Max |eta| applied on all L1TTTracks [unitless]
  float cfg_tk_maxChiSq;                // Max chi squared for L1TTTracks [unitless]
  unsigned int cfg_tk_minStubs;         // Min number of stubs per L1TTTrack [unitless]   

  // L1 EGammas 
  float cfg_eg_minEt;                   // Min eT applied on all L1EGammaCrystalClusters [GeV]
  float cfg_eg_minEta;                  // Min |eta| applied on all L1EGammaCrystalClusters [GeV]
  float cfg_eg_maxEta;                  // Max |eta| applied on all L1EGammaCrystalClusters [GeV]

  // Seed-tracks parameters
  float cfg_seedtk_minPt;               // Min pT of L1TkEG seed L1TTTracks [GeV]
  float cfg_seedtk_maxEta;              // Max |eta| of L1TkEG seed L1TTTracks [unitless]
  float cfg_seedtk_maxChiSq;            // Max chi squared of L1TkEG seed L1TTTracks [unitless]
  unsigned int cfg_seedtk_minStubs;     // Min number of stubs of L1TkEG seed L1TTTracks [unitless]   
  float cfg_seedtk_maxDeltaR;           // Max opening of the cone in which the TkEG seed track is the leading one in pT [unitless]

  // Shrinking Cone parameters
  float cfg_shrinkCone_Constant;        // Constant which is used for defining the opening of the signal cone : sigCone_dRMax = (cfg_shrinkCone_Constant)/(pT of the TkEG seed track) [GeV]
  float cfg_sigCone_cutoffDeltaR;       // Cutoff value for the maximum dR of the shrinking signal cone [unitless]
  bool  cfg_isoCone_useCone;            // Usage of isolation cone (true) or isolation annulus (false)
  float cfg_sigCone_dRMin;              // Min dR of signal cone [unitless]
  float cfg_isoCone_dRMax;              // Max dR of isolation cone/annulus [unitless]
  float sigCone_dRMax;                  // Max dR of signal cone [unitless]
  float isoCone_dRMin;                  // Min dR of isolation cone/annulus (= max dR of signal cone) [unitless]
  
  // Tracks & EGs clustering parameters
  float cfg_maxDeltaZ_trks;             // Max POCAz difference between TkEG seed track and additional L1TkEG signal-cone L1TTTracks [cm]
  float cfg_maxInvMass_trks;            // Max Invariant Mass of the Track Cluster (including the L1TkEG seed L1TTTrack) [GeV/c^2]
  float cfg_maxInvMass_EGs;             // Max Invariant Mass of the Track+EG Cluster (including the L1TkEG seed L1TTTrack) [GeV/c^2]

  // Isolation parameters
  bool  cfg_useVtxIso;                 // Usage of vertex isolation on L1TkEG candidates (no tracks in the isolation cone coming from the same vertex with the seed track)
  float cfg_vtxIso_WP;                 // Working point of vertex isolation (no isolation cone track with |dz0| <= cfg_vtxIso_WP)

  const edm::EDGetTokenT< EGammaBxCollection > egToken;
  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
  
} ;


//
// constructors and destructor
//
L1TkEGTauParticleProducer::L1TkEGTauParticleProducer(const edm::ParameterSet& iConfig) :
  egToken(consumes< EGammaBxCollection >(iConfig.getParameter<edm::InputTag>("L1EGammaInputTag"))),
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
  {
  
  label = iConfig.getParameter<std::string>("label");  // label of the collection produced
  
  // L1 Tracks  
  cfg_tk_nFitParams = (unsigned int)iConfig.getParameter<unsigned int>("tk_nFitParams");
  cfg_tk_minPt      = (float)iConfig.getParameter<double>("tk_minPt");
  cfg_tk_minEta     = (float)iConfig.getParameter<double>("tk_minEta");
  cfg_tk_maxEta     = (float)iConfig.getParameter<double>("tk_maxEta");
  cfg_tk_maxChiSq   = (float)iConfig.getParameter<double>("tk_maxChiSq");
  cfg_tk_minStubs   = (unsigned int)iConfig.getParameter<unsigned int>("tk_minStubs");

  // L1 EGammas
  cfg_eg_minEt  = (float)iConfig.getParameter<double>("eg_minEt");
  cfg_eg_minEta = (float)iConfig.getParameter<double>("eg_minEta");
  cfg_eg_maxEta = (float)iConfig.getParameter<double>("eg_maxEta");

  // Seed-tracks parameters
  cfg_seedtk_minPt      = (float)iConfig.getParameter<double>("seedtk_minPt");
  cfg_seedtk_maxEta     = (float)iConfig.getParameter<double>("seedtk_maxEta");
  cfg_seedtk_maxChiSq   = (float)iConfig.getParameter<double>("seedtk_maxChiSq");
  cfg_seedtk_minStubs   = (unsigned int)iConfig.getParameter<unsigned int>("seedtk_minStubs");
  cfg_seedtk_maxDeltaR  = (float)iConfig.getParameter<double>("seedtk_maxDeltaR");

  // Shrinking Cone parameters
  cfg_shrinkCone_Constant  = (float)iConfig.getParameter<double>("shrinkCone_Constant");
  cfg_sigCone_cutoffDeltaR = (float)iConfig.getParameter<double>("sigCone_cutoffDeltaR");
  cfg_isoCone_useCone      = (bool)iConfig.getParameter<bool>("isoCone_useCone");
  cfg_sigCone_dRMin            = (float)iConfig.getParameter<double>("sigCone_dRMin");
  cfg_isoCone_dRMax            = (float)iConfig.getParameter<double>("isoCone_dRMax");
  
  // Tracks & EGs clustering parameters
  cfg_maxDeltaZ_trks  = (float)iConfig.getParameter<double>("maxDeltaZ_trks");
  cfg_maxInvMass_trks = (float)iConfig.getParameter<double>("maxInvMass_trks");
  cfg_maxInvMass_EGs  = (float)iConfig.getParameter<double>("maxInvMass_EGs");
   
  // Isolation parameters
  cfg_useVtxIso = (bool)iConfig.getParameter<bool>("useVtxIso");
  cfg_vtxIso_WP = (float)iConfig.getParameter<double>("vtxIso_WP");

  produces<L1TkEGTauParticleCollection>(label);
}

L1TkEGTauParticleProducer::~L1TkEGTauParticleProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkEGTauParticleProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  using namespace edm;
  
  std::unique_ptr<L1TkEGTauParticleCollection> result(new L1TkEGTauParticleCollection);

  // Constants 
  const float pionMass  = 0.13957018;
    
  // the L1Tracks
  edm::Handle<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trkIter;
  
  if (!L1TTTrackHandle.isValid() ) {
    LogError("L1TkEGTauParticleProducer")
      << "\nWarning: L1TTTrackCollectionType not found in the event. Exit."
      << std::endl;
    return;
  }
  
  // the L1EGamma objects
  edm::Handle<EGammaBxCollection> eGammaHandle;
  iEvent.getByToken(egToken, eGammaHandle);  
  EGammaBxCollection eGammaCollection = (*eGammaHandle.product());
  EGammaBxCollection::const_iterator egIter;
  
  if( !eGammaHandle.isValid() ) {
    LogError("L1TkEGTauParticleProducer")
      << "\nWarning: L1EmParticleCollection not found in the event. Exit."
      << std::endl;
    return;
  }
  
  ///////////////////////////////////////////////////////////////
  //  Select Tracks and EGs passing the quality criteria
  ///////////////////////////////////////////////////////////////
  
#ifdef DEBUG
  std::cout<<"\n--- Select all tracks passing the quality criteria"<<std::endl;
#endif

  std::vector< L1TTTrackRefPtr > SelTTTrackPtrs;
  unsigned int track_counter = 0;

  // For-loop: All the L1TTTracks
  for (trkIter = L1TTTrackHandle->begin(); trkIter != L1TTTrackHandle->end(); ++trkIter) {
    
    /// Make a pointer to the L1TTTracks
    L1TTTrackRefPtr track_RefPtr( L1TTTrackHandle, track_counter++ );

    // Apply quality criteria on the L1TTTracks
    float Pt   = trkIter->getMomentum(cfg_tk_nFitParams).perp();
    float Eta  = trkIter->getMomentum(cfg_tk_nFitParams).eta();
    float Chi2 = trkIter->getChi2(cfg_tk_nFitParams);
    std::vector< L1TTStubRef > Stubs = trkIter-> getStubRefs();
    unsigned int NStubs              = Stubs.size();

    if ( Pt < cfg_tk_minPt ) continue;
    if ( fabs(Eta) < cfg_tk_minEta ) continue;
    if ( fabs(Eta) > cfg_tk_maxEta ) continue;
    if ( Chi2 > cfg_tk_maxChiSq ) continue;
    if ( NStubs < cfg_tk_minStubs ) continue;

    //std::cout << track_RefPtr->getMomentum(cfg_tk_nFitParams).perp()<<"    "<< Pt << std::endl;
	
    SelTTTrackPtrs.push_back(track_RefPtr);

  }// End-loop: All the L1TTTracks

  // Sort by pT all selected L1TTTracks
  std::sort( SelTTTrackPtrs.begin(), SelTTTrackPtrs.end(), TrackPtComparator(cfg_tk_nFitParams) );
  
  // ---------------------------------

#ifdef DEBUG
  std::cout<<"\n--- Select all EGs passing the quality criteria"<<std::endl;
#endif

  std::vector< EGammaRef > SelEGsPtrs;
  unsigned int ieg = 0;

  // For-loop: All the L1EGs
  for (egIter = eGammaCollection.begin(0); egIter != eGammaCollection.end(0);  ++egIter) {
    edm::Ref< EGammaBxCollection > EGammaRef( eGammaHandle, ieg++ );
    
    float Et = egIter -> et();
    float Eta = egIter -> eta();
    
    if (Et < cfg_eg_minEt) continue;
    if (fabs(Eta) < cfg_eg_minEta) continue;
    if (fabs(Eta) > cfg_eg_maxEta) continue;
    
    SelEGsPtrs.push_back(EGammaRef);

  }// End-loop: All the L1EGs
  
  // Sort by ET all selected L1EGs
  std::sort( SelEGsPtrs.begin(), SelEGsPtrs.end(), EtComparator() ); 

  ///////////////////////////////////////////////////////////////
  //  Tracks + EG Algorithm
  ///////////////////////////////////////////////////////////////

#ifdef DEBUG
  std::cout << "\n\t=== Tracks + EG Algorithm" <<std::endl;
#endif
  
  std::vector< L1TTTrackRefPtr > TrackCluster; 
  std::vector< unsigned int > TrackClusterIndx;
  std::vector< EGammaRef > EGcluster;
  std::vector< unsigned int > EGclusterIndx;

  // For-loop: All the selected L1TTTracks
  for ( unsigned int i=0; i < SelTTTrackPtrs.size(); i++ ){

    // Clear clusters vectors
    TrackCluster.clear();
    TrackClusterIndx.clear();
    EGcluster.clear();
    EGclusterIndx.clear();

    L1TTTrackRefPtr iTrk = SelTTTrackPtrs.at(i);
    float iPt   = iTrk->getMomentum(cfg_tk_nFitParams).perp();
    float iEta  = iTrk->getMomentum(cfg_tk_nFitParams).eta();
    float iPhi  = iTrk->getMomentum(cfg_tk_nFitParams).phi();
    float iChi2 = iTrk->getChi2(cfg_tk_nFitParams);
    float iz0   = iTrk->getPOCA(cfg_tk_nFitParams).z();
    std::vector< L1TTStubRef > iStubs = iTrk-> getStubRefs();
    unsigned int iNStubs              = iStubs.size();

    // Apply seed track cuts
    if ( iPt   < cfg_seedtk_minPt ) continue;
    if ( fabs(iEta)  > cfg_seedtk_maxEta ) continue;
    if ( iChi2 > cfg_seedtk_maxChiSq ) continue;
    if ( iNStubs < cfg_seedtk_minStubs ) continue;
    
    // Check that there are no close tracks (in terms of deltaR) with higher Pt
    bool highPtNeighbourFound = false;
    for (unsigned int j=0; !highPtNeighbourFound &&  j < SelTTTrackPtrs.size(); j++) {
      L1TTTrackRefPtr jTrk = SelTTTrackPtrs.at(j);
      float jPt   = jTrk->getMomentum(cfg_tk_nFitParams).perp();
      float jEta  = jTrk->getMomentum(cfg_tk_nFitParams).eta();
      float jPhi  = jTrk->getMomentum(cfg_tk_nFitParams).phi();
            
      float deltaR = reco::deltaR(iEta, iPhi, jEta, jPhi);
      if (deltaR < cfg_seedtk_maxDeltaR && jPt > iPt) highPtNeighbourFound = true;
      
    }
    
    // If not, build a tau-candidate with seed track the leading one
    // Start: No highPtNeighbourFound 
    if (!highPtNeighbourFound) {

      // Build a tau candidate  
      GetShrinkingConeSizes(iPt, cfg_shrinkCone_Constant, cfg_sigCone_cutoffDeltaR, cfg_sigCone_dRMin, sigCone_dRMax, isoCone_dRMin, cfg_isoCone_dRMax, cfg_isoCone_useCone);

#ifdef DEBUG
      std::cout<<"Shrinking cone for tau-seed with pT = "<< iPt <<" GeV: sigCone_dRMin = "<< cfg_sigCone_dRMin <<"  , sigCone_dRMax = "<< sigCone_dRMax <<"  , isoCone_dRMin = "<< isoCone_dRMin <<"  , isoCone_dRMax = "<< cfg_isoCone_dRMax <<std::endl;
#endif 
      
      TrackCluster.push_back(iTrk);
      TrackClusterIndx.push_back(i);

      // Tracks Clustering
      for (unsigned int j=0; j < SelTTTrackPtrs.size(); j++) {

	if (i == j) continue;
	
	L1TTTrackRefPtr jTrk = SelTTTrackPtrs.at(j);
	float jEta  = jTrk->getMomentum(cfg_tk_nFitParams).eta();
	float jPhi  = jTrk->getMomentum(cfg_tk_nFitParams).phi();
	float jz0   = jTrk->getPOCA(cfg_tk_nFitParams).z();

	// Apply dz0 and dR criteria for track clustering 
	float deltaz0 = fabs(iz0-jz0);
	if (deltaz0 > cfg_maxDeltaZ_trks) continue;
	float deltaR  = reco::deltaR(iEta, iPhi, jEta, jPhi);
	if (deltaR > sigCone_dRMax) continue;

	TrackCluster.push_back(jTrk);
	TrackClusterIndx.push_back(j);

      }// Tracks Clustering

      // EGs Clustering
      for ( unsigned int j=0; j < SelEGsPtrs.size(); j++ ){

	EGammaRef jEG = SelEGsPtrs.at(j);
	float jEta = jEG->eta();
	float jPhi = jEG->phi();
	
	// Apply dR criteria for EG clustering
	float deltaR = reco::deltaR(iEta, iPhi, CorrectedEta(jEta, iz0), jPhi);
	if (deltaR > sigCone_dRMax) continue;
	
	EGcluster.push_back(jEG);
	EGclusterIndx.push_back(j);

      }// EGs Clustering
      
      // Calculate total p4 of the tau candidate 
      math::XYZTLorentzVector p4_total, p4_trks, p4_egs, p4_tmp;
      
      //TLorentzVector p4, p4tmp;

      for (unsigned int j=0; j < TrackCluster.size(); j++) {
	L1TTTrackRefPtr jTrk = TrackCluster.at(j);
	double px = jTrk->getMomentum(cfg_tk_nFitParams).x();
	double py = jTrk->getMomentum(cfg_tk_nFitParams).y();
	double pz = jTrk->getMomentum(cfg_tk_nFitParams).z();
	double e = sqrt(px*px+py*py+pz*pz+pionMass*pionMass);
	p4_tmp.SetCoordinates(px,py,pz,e);
	p4_trks += p4_tmp;
		
	// std::cout<<"Px =  "<<px<<"   "<<p4tmp.Px()<<std::endl;
	// std::cout<<"Py =  "<<py<<"   "<<p4tmp.Py()<<std::endl;
	// std::cout<<"Pz =  "<<pz<<"   "<<p4tmp.Pz()<<std::endl;
	// std::cout<<"E  =  "<<e<<"   "<<p4tmp.E()<<std::endl;
	// std::cout<<"Mass = "<<pionMass<<"   "<<p4tmp.M()<<std::endl;
      }
      
      for (unsigned int j=0; j < EGcluster.size(); j++) {
	EGammaRef jEG = SelEGsPtrs.at(j);
	float jEt  = jEG->et();
	float jEta = jEG->eta();
	float jPhi = jEG->phi();
	float jPt  = sqrt( (jEt/(sin(2*atan(exp(-jEta)))))*(jEt/(sin(2*atan(exp(-jEta))))) - pionMass*pionMass )*sin(2*atan(exp(-jEta)));

	Double_t px=jPt*cos(jPhi);
	Double_t py=jPt*sin(jPhi);
	Double_t theta=2*atan(exp(-jEta));
	Double_t pz=jPt/tan(theta);
	Double_t p=sqrt(jPt*jPt+pz*pz);
	Double_t e=sqrt(pionMass*pionMass+p*p);

	p4_tmp.SetCoordinates(px,py,pz,e);
	p4_egs += p4_tmp;
	
	// std::cout<<"Pt =  "<<jPt<<"   "<<p4tmp.Pt()<<std::endl;
	// std::cout<<"Px =  "<<px<<"   "<<p4tmp.Px()<<std::endl;
	// std::cout<<"Py =  "<<py<<"   "<<p4tmp.Py()<<std::endl;
	// std::cout<<"Pz =  "<<pz<<"   "<<p4tmp.Pz()<<std::endl;
	// std::cout<<"E  =  "<<e<<"   "<<p4tmp.E()<<std::endl;
	// std::cout<<"Mass = "<<pionMass<<"   "<<p4tmp.M()<<std::endl;
      }
   
      // Calculate Isolation
      float vtxIso = CalculateVtxIso(SelTTTrackPtrs, TrackClusterIndx, cfg_isoCone_useCone);
      
      // Build the tau candidate
      p4_total = p4_trks + p4_egs;
      L1TkEGTauParticle trkEG(p4_total, TrackCluster, EGcluster, vtxIso);

      // Apply Mass cut
      if (p4_trks.M() > cfg_maxInvMass_trks) continue;
      if (p4_egs.M() > cfg_maxInvMass_EGs) continue;
      
      // Apply Isolation
      if (cfg_useVtxIso) {
	if ( vtxIso > cfg_vtxIso_WP ) result -> push_back( trkEG );
      }

    }// End: No highPtNeighbourFound
    
  }// End-loop: All the L1TTTracks
  
  // Sort the TkEG candidates by eT before saving to the event 
  sort( result->begin(), result->end(), L1TkEGTau::EtComparator() );

  iEvent.put(std::move(result), label );
  
}


// --------------------------------------------------------------------------------------
void L1TkEGTauParticleProducer::GetShrinkingConeSizes(float tk_pt,
						   const float shrinkCone_Constant,
						   const float sigCone_dRCutoff,
						   float &sigCone_dRMin,
						   float &sigCone_dRMax,
						   float &isoCone_dRMin,
						   float &isoCone_dRMax,
						   const bool isoCone_useCone)
{
  // Signal cone
  double signalCone_min = sigCone_dRMin;
  double signalCone_max = (shrinkCone_Constant)/(tk_pt);
  if (signalCone_max > sigCone_dRCutoff) signalCone_max = sigCone_dRCutoff;
 
  // Isolation cone/annulus
  double isoCone_min;
  if (isoCone_useCone) isoCone_min = 0.0;
  else isoCone_min = signalCone_max;
  double isoCone_max = isoCone_dRMax;
      
  // Assign signal and isolation cone sizes
  sigCone_dRMin = signalCone_min;
  sigCone_dRMax = signalCone_max;  
  isoCone_dRMin = isoCone_min;
  isoCone_dRMax = isoCone_max;

  return;
}

// --------------------------------------------------------------------------------------

float L1TkEGTauParticleProducer::CalculateVtxIso(std::vector< L1TTTrackRefPtr > allTracks,
					      std::vector< unsigned int > clustTracksIndx,
					      bool useIsoCone) {

  // Initializations
  float  mindZ0 = 999.9;
  float dz0, deltaR;
  
  // Seed track properties
  L1TTTrackRefPtr seedTrack = allTracks.at(clustTracksIndx.at(0));
  float seedEta  = seedTrack->getMomentum(cfg_tk_nFitParams).eta();
  float seedPhi  = seedTrack->getMomentum(cfg_tk_nFitParams).phi();
  float seedz0   = seedTrack->getPOCA(cfg_tk_nFitParams).z();
  
  // For-loop: All the Tracks
  for (unsigned int i=0; i < allTracks.size(); i++) {

    L1TTTrackRefPtr iTrk = allTracks.at(i);
    float iEta  = iTrk->getMomentum(cfg_tk_nFitParams).eta();
    float iPhi  = iTrk->getMomentum(cfg_tk_nFitParams).phi();
    float iz0   = iTrk->getPOCA(cfg_tk_nFitParams).z();
        
    if (useIsoCone) {
      // Check if the track is clustered in the tau candidate
      bool clustered = false;
      
      for (unsigned int j=0; j < clustTracksIndx.size(); j++) {
	if (i == clustTracksIndx.at(j)) clustered = true;
      }
      if (clustered) continue;	
    }
    
    // Check if the track is in the iso-cone
    deltaR = reco:: deltaR(seedEta, seedPhi, iEta, iPhi);
    
    if (deltaR > isoCone_dRMin && deltaR < cfg_isoCone_dRMax) {
      // Calculate mindz0
      dz0 = fabs(iz0 - seedz0);
      if (dz0 < mindZ0) mindZ0 = dz0;
    }
    
  } // End-loop: All the Tracks
  
  float vtxIso = mindZ0;

  return vtxIso;
  
  
  
}  

// --------------------------------------------------------------------------------------

float L1TkEGTauParticleProducer::CorrectedEta(float eta, float zTrack)  {

  // Correct the eta of the L1EG object once we know the zTrack
  // (normaly we use the zvertex but since we care about the dR of the EG and the track it is not needed)
 
  
  bool IsBarrel = ( fabs(eta) < 1.479 );
  float REcal = 129. ;
  float ZEcal = 315.4 ;
  
  float theta = 2. * TMath::ATan( TMath::Exp( - eta ) );
  if (theta < 0) theta = theta + TMath::Pi();
  float tantheta = TMath::Tan( theta );
  
  float delta;
  if (IsBarrel) {
    delta = REcal / tantheta ;
  }
  else {
    if (theta > 0) delta =  ZEcal;
    if (theta < 0) delta = -ZEcal;
  }
  
  float tanthetaprime = delta * tantheta / (delta - zTrack );
  
  float thetaprime = TMath::ATan( tanthetaprime );
  if (thetaprime < 0) thetaprime = thetaprime + TMath::Pi();
  
  float etaprime = -TMath::Log( TMath::Tan( thetaprime / 2.) );
  return etaprime;
  
}

// ------------ method called once each job just before starting event loop  ------------
void
L1TkEGTauParticleProducer::beginJob()
{
  // std::cout << "L1TkEGTauParticleProducer::beginJob() " << std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkEGTauParticleProducer::endJob() {
  // std::cout << "L1TkEGTauParticleProducer::endJob() " << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkEGTauParticleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkEGTauParticleProducer);



