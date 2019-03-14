// -*- C++ -*-
//
//
// producer for a L1CaloTkTauParticle
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

#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1TrackTrigger/interface/L1CaloTkTauParticle.h"
#include "L1Trigger/Phase2L1Taus/interface/L1CaloTkTauEtComparator.h"

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

class L1CaloTkTauParticleProducer : public edm::EDProducer {
public:
  
  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;
  typedef edm::Ptr< L1TTTrackType > L1TTTrackRefPtr;
  typedef std::vector< L1TTTrackRefPtr > L1TTTrackRefPtr_Collection;
  typedef edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > L1TTStubRef;

  explicit L1CaloTkTauParticleProducer(const edm::ParameterSet&);
  ~L1CaloTkTauParticleProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions); //TODO: what is this?

  class TrackPtComparator{
    
    unsigned int nFitParams_;
  public:
    TrackPtComparator(unsigned int nFitParams){ nFitParams_ = nFitParams;}
    bool operator() (const L1TTTrackRefPtr trackA, L1TTTrackRefPtr trackB ) const {
      return ( trackA->getMomentum(nFitParams_).perp() > trackB->getMomentum(nFitParams_).perp() );
    }
  };

  float CorrectedEta(float eta, float zv); //TODO: double check
  float CalibrateCaloTau(float Et, float Eta);
  float findClosest(float arr[], int n, float target);
  float getClosest(float val1, float val2, float target);

  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void GetShrinkingConeSizes(float pt, //TODO double check
			     const float cfg_shrinkCone_Constant,
			     const float cfg_sigCone_dRCutoff,
  			     float &cfg_sigCone_dRMin,
  			     float &sigCone_dRMax,
  			     float &isoCone_dRMin,
  			     float &cfg_isoCone_dRMax,
  			     const bool cfg_isoCone_useCone);
  
  // ----------member data ---------------------------
  
  // General
  std::string label;                    // Label of the objects which are created

  // L1 tracks
  unsigned int cfg_tk_nFitParams;       // Number of Fit Parameters: 4 or 5 ? (pT, eta, phi, z0, d0)
  float cfg_tk_minPt;                   // Min pT applied on all L1TTTracks [GeV]
  float cfg_tk_minEta;                  // Min |eta| applied on all L1TTTracks [unitless]
  float cfg_tk_maxEta;                  // Max |eta| applied on all L1TTTracks [unitless]
  float cfg_tk_maxChiSq;                // Max chi squared for L1TTTracks [unitless]
  unsigned int cfg_tk_minStubs;         // Min number of stubs per L1TTTrack [unitless]   

  // Seed tracks
  float cfg_seedTk_minPt;               // Min pT of L1TkEG seed L1TTTracks [GeV]
  float cfg_seedTk_minEta;              // Min |eta| of L1TkEG seed L1TTTracks [unitless]
  float cfg_seedTk_maxEta;              // Max |eta| of L1TkEG seed L1TTTracks [unitless]
  float cfg_seedTk_maxChiSq;            // Max chi squared of L1TkEG seed L1TTTracks [unitless]
  float cfg_seedTk_minStubs;            // Min number of stubs of L1TkEG seed L1TTTracks [unitless]
  bool  cfg_seedTk_useMaxDeltaR;        // Require the seed track to be the leading track within the cone of cfg_seedTk_maxDeltaR
  float cfg_seedTk_maxDeltaR;           // Cone in which the seed track is required to be the leading one in pT [unitless]

  // Signal cone and clustering parameters
  float cfg_shrinkCone_Constant;        // Constant which is used for defining the opening of the signal cone : sigCone_dRMax = (cfg_shrinkCone_Constant)/(pT of the seed track) [GeV]    
  float cfg_sigCone_cutoffDeltaR;       // Cutoff value for the maximum dR of the shrinking signal cone [unitless]    
  float cfg_sigCone_dRMin;              // Min dR of signal cone [unitless]    
  float cfg_sigConeTks_maxDeltaZ;       // Max POCAz difference between the seed track and additional signal-cone L1TTTracks [cm]       
  float cfg_sigConeTks_maxInvMass;      // Max Invariant Mass of the Track Cluster (including the L1TkEG seed L1TTTrack) [GeV/c^2]        

  // Isolation cone parameters
  bool cfg_isoCone_useCone;             // Usage of isolation cone (true) or isolation annulus (false)
  float cfg_isoCone_dRMax;              // Max dR of isolation cone/annulus [unitless]

  // Matching parameters
  float cfg_matching_maxDeltaR;         // Matching cone around the CaloTau, inside which the seed track with the smallest dR (or highest pT, see the next parameter) is chosen
  bool cfg_matchHighestPt;              // Match track with highest pT (inside the mathcing cone) instead of the one with the smallest dR


  // CaloTau parameters
  float cfg_caloTauEtMin;               // Minimum (non-calibrated) Et for CaloTaus considered by the algorithm
  bool cfg_calibrateCaloTaus;           // Calibrate the Et values of CaloTaus (true) or use the default values (false)

  // Isolation parameters
  bool  cfg_useVtxIso;                  // Usage of vertex isolation on (no tracks in the isolation cone coming from the same vertex with the seed track) 
  float cfg_vtxIso_WP;                  // Working point of vertex isolation (no isolation cone track with |dz0| <= cfg_vtxIso_WP)

  // Experimental features currently not in use
//  bool cfg_useRelIso;
//  float cfg_relIso_WP;
  float cfg_relIso_maxDeltaZ;
//  bool cfg_useJetWidth;
//  float cfg_jetWidth_WP;

  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
  std::vector< edm::EDGetTokenT<l1t::TauBxCollection> > tauTokens;

} ;


//
// constructors and destructor
//
L1CaloTkTauParticleProducer::L1CaloTkTauParticleProducer(const edm::ParameterSet& iConfig) :
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
  {

  // CaloTaus
  const auto& taus = iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("L1CaloTauInputTag");
  for (const auto& tau: taus) {
    tauTokens.push_back(consumes<l1t::TauBxCollection>(tau));
  }

  // General
  label = iConfig.getParameter<std::string>("label");  // label of the collection produced

  // L1 tracks
  cfg_tk_nFitParams         = (unsigned int)iConfig.getParameter<unsigned int>("tk_nFitParams");
  cfg_tk_minPt              = (float)iConfig.getParameter<double>("tk_minPt");
  cfg_tk_minEta             = (float)iConfig.getParameter<double>("tk_minEta");
  cfg_tk_maxEta             = (float)iConfig.getParameter<double>("tk_maxEta");
  cfg_tk_maxChiSq           = (float)iConfig.getParameter<double>("tk_maxChiSq");
  cfg_tk_minStubs           = (unsigned int)iConfig.getParameter<unsigned int>("tk_minStubs");

  // Seed tracks
  cfg_seedTk_minPt          = (float)iConfig.getParameter<double>("seedTk_minPt");
  cfg_seedTk_minEta         = (float)iConfig.getParameter<double>("seedTk_minEta");
  cfg_seedTk_maxEta         = (float)iConfig.getParameter<double>("seedTk_maxEta");
  cfg_seedTk_maxChiSq       = (float)iConfig.getParameter<double>("seedTk_maxChiSq");
  cfg_seedTk_minStubs       = (unsigned int)iConfig.getParameter<unsigned int>("seedTk_minStubs");
  cfg_seedTk_useMaxDeltaR   = (bool)iConfig.getParameter<bool>("seedTk_useMaxDeltaR");
  cfg_seedTk_maxDeltaR      = (float)iConfig.getParameter<double>("seedTk_maxDeltaR");

  // Matching parameters
  cfg_matching_maxDeltaR    = (float)iConfig.getParameter<double>("matching_maxDeltaR");
  cfg_matchHighestPt        = (bool)iConfig.getParameter<bool>("matchHighestPt");

  // Signal cone and clustering parameters
  cfg_shrinkCone_Constant   = (float)iConfig.getParameter<double>("shrinkCone_Constant");
  cfg_sigCone_cutoffDeltaR  = (float)iConfig.getParameter<double>("sigCone_cutoffDeltaR");
  cfg_sigCone_dRMin         = (float)iConfig.getParameter<double>("sigCone_dRMin");
  cfg_sigConeTks_maxDeltaZ  = (float)iConfig.getParameter<double>("sigConeTks_maxDeltaZ");
  cfg_sigConeTks_maxInvMass = (float)iConfig.getParameter<double>("sigConeTks_maxInvMass");

  // Isolation cone parameters
  cfg_isoCone_useCone       = (bool)iConfig.getParameter<bool>("isoCone_useCone");
  cfg_isoCone_dRMax         = (float)iConfig.getParameter<double>("isoCone_dRMax");

  // CaloTaus
  cfg_caloTauEtMin          = (float)iConfig.getParameter<double>("caloTauEtMin");
  cfg_calibrateCaloTaus     = (bool)iConfig.getParameter<bool>("calibrateCaloTaus");

  // Isolation
  cfg_useVtxIso             = (bool)iConfig.getParameter<bool>("useVtxIso");
  cfg_vtxIso_WP             = (float)iConfig.getParameter<double>("vtxIso_WP");
//  cfg_jetWidth_WP           = (float)iConfig.getParameter<double>("jetWidth_WP");
//  cfg_relIso_WP             = (float)iConfig.getParameter<double>("relIso_WP");
  cfg_relIso_maxDeltaZ      = (float)iConfig.getParameter<double>("relIso_maxDeltaZ");
  
  produces<L1CaloTkTauParticleCollection>(label);

}

L1CaloTkTauParticleProducer::~L1CaloTkTauParticleProducer() {}

// ------------ method called to produce the data  ------------
void L1CaloTkTauParticleProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  
  std::unique_ptr<L1CaloTkTauParticleCollection> L1CaloTkTauCandidates(new L1CaloTkTauParticleCollection);

  // Constants 
  const float pionMass  = 0.13957018;

  // Construct a vector of CaloTaus  
  std::vector<Tau> caloTaus;
  for (auto & tautoken: tauTokens){
    edm::Handle<l1t::TauBxCollection> tau;
    iEvent.getByToken(tautoken,  tau);
    if (tau.isValid()){ 
      for (int ibx = tau->getFirstBX(); ibx <= tau->getLastBX(); ++ibx) {
        for (l1t::TauBxCollection::const_iterator it=tau->begin(ibx); it!=tau->end(ibx); it++){
          if (it->pt() > 0){
	        caloTaus.push_back(*it);
          }
        }
      } // end of if(tau.isValid())
    } else {
      edm::LogWarning("MissingProduct") << "Warning: Valid tau not found!" << std::endl;
    }
  }

  // Get the L1Tracks
  edm::Handle<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trkIter;  
  if (!L1TTTrackHandle.isValid() ) {
    LogError("L1CaloTkTauParticleProducer")
      << "\nWarning: L1TTTrackCollectionType not found in the event. Exit."
      << std::endl;
    return;
  }
  
  
#ifdef DEBUG
  std::cout<<"\n--- Select all tracks passing the quality criteria"<<std::endl;
#endif

  // Build a track collection using only tracks passing the quality criteria
  std::vector< L1TTTrackRefPtr > SeedTTTrackPtrs;
  std::vector< L1TTTrackRefPtr > SigConeTTTrackPtrs;
  unsigned int cfg_seedTk_counter = 0;

  // Loop over all tracks
  for (trkIter = L1TTTrackHandle->begin(); trkIter != L1TTTrackHandle->end(); ++trkIter) {
    // Make a pointer
    L1TTTrackRefPtr track_RefPtr(L1TTTrackHandle, cfg_seedTk_counter++);
    // Retrieve track information
    float Pt   = trkIter->getMomentum(cfg_tk_nFitParams).perp();
    float Eta  = trkIter->getMomentum(cfg_tk_nFitParams).eta();
    float Chi2 = trkIter->getChi2(cfg_tk_nFitParams);
    std::vector< L1TTStubRef > Stubs = trkIter-> getStubRefs();
    unsigned int NStubs              = Stubs.size();
    // Select seed tracks
    if ( Pt   > cfg_seedTk_minPt  &&
         fabs(Eta)  > cfg_seedTk_minEta && 
         fabs(Eta)  < cfg_seedTk_maxEta && 
         Chi2 < cfg_seedTk_maxChiSq &&
         NStubs > cfg_seedTk_minStubs )  
            SeedTTTrackPtrs.push_back(track_RefPtr);
    // Select signal cone tracks
    if ( Pt   > cfg_tk_minPt  &&
         fabs(Eta)  > cfg_tk_minEta && 
         fabs(Eta)  < cfg_tk_maxEta && 
         Chi2 < cfg_tk_maxChiSq &&
         NStubs > cfg_tk_minStubs )  
            SigConeTTTrackPtrs.push_back(track_RefPtr);
  }// End of the track loop

  /// Sort all track collections by pT
  std::sort( SeedTTTrackPtrs.begin(), SeedTTTrackPtrs.end(), TrackPtComparator(cfg_tk_nFitParams) );
  std::sort( SigConeTTTrackPtrs.begin(), SigConeTTTrackPtrs.end(), TrackPtComparator(cfg_tk_nFitParams) );

  ///////////////////////////////////////////////////////////////
  //  CaloTkTau Algorithm
  ///////////////////////////////////////////////////////////////
  
  bool foundMatchTrk;
  double matchTk_dR;  
  double matchTk_pT;

  // Loop over calo taus
  for (std::vector<Tau>::iterator caloTauIter = caloTaus.begin(); caloTauIter != caloTaus.end(); ++caloTauIter) {
          
      // Calibrate calo taus if needed
      float caloEt = caloTauIter->et();
      float caloEta = caloTauIter->eta();
      if (caloEt < cfg_caloTauEtMin) continue; 
      if(cfg_calibrateCaloTaus) {
          caloEt = CalibrateCaloTau(caloEt, caloEta);
      }

      // Find a matching track
      foundMatchTrk = false;
      matchTk_dR = 999.9;
      matchTk_pT = 0.0;
      // Loop over seed tracks
      L1TTTrackRefPtr matchedTrack;
      for ( unsigned int i=0; i < SeedTTTrackPtrs.size(); i++ ){
//        std::cout << "Checking track #" << i << std::endl;
        L1TTTrackRefPtr iTrk = SeedTTTrackPtrs.at(i);
        double dR = reco::deltaR(iTrk->getMomentum(cfg_tk_nFitParams).eta(), iTrk->getMomentum(cfg_tk_nFitParams).phi(), caloTauIter->eta(), caloTauIter->phi());
        double iTrkPt = iTrk->getMomentum(cfg_tk_nFitParams).perp();
        // Pick the highest-pT track inside the cone...
        if (cfg_matchHighestPt && iTrkPt > matchTk_pT && dR < cfg_matching_maxDeltaR){
          matchTk_pT = iTrkPt;
          foundMatchTrk = true;
          matchedTrack = iTrk;
        } 
        // ..or pick the track with the smallest dR w.r.t. CaloTau
        if (!cfg_matchHighestPt && dR < matchTk_dR && dR < cfg_matching_maxDeltaR) {
          matchTk_dR = dR;
          foundMatchTrk = true;
          matchedTrack = iTrk;
        }
      } 

      // Proceed only if a matching track is found
      if (!foundMatchTrk) continue;

      // Define the signal and isolation cones
      float signalCone_dRmin = 0.0; // unchanged by GetShrinkingConeSizes function
      float signalCone_dRmax = -999.0;
      float isolationCone_dRmin = -999.0;
      float isolationCone_dRmax = cfg_isoCone_dRMax; // unchanged by GetShrinkingConeSizes function
      GetShrinkingConeSizes(caloEt, cfg_shrinkCone_Constant, cfg_sigCone_cutoffDeltaR, // input parameters
                            signalCone_dRmin, signalCone_dRmax,isolationCone_dRmin, isolationCone_dRmax, // these values are updated by this function
                            cfg_isoCone_useCone); // input parameter (if useCone is selected, isolationCone_dRmin is set to 0)

      // Initialize track lists
      std::vector< L1TTTrackRefPtr > sigConeTks;
      std::vector< L1TTTrackRefPtr > isoConeTks;
      
      // Initialize isolation variables
      float vtxIso = 999.0;
      float isoConePtSum = 0.0;
      float dRtimesPtSum = 0.0;

      // Add matching track as the first one in sigConeTks
      sigConeTks.push_back(matchedTrack);
      math::XYZTLorentzVector sigTks_p4; // track-based four-momentum
      double px = matchedTrack->getMomentum().x();
      double py = matchedTrack->getMomentum().y();
	  double pz = matchedTrack->getMomentum().z();
	  double e  = sqrt(px*px+py*py+pz*pz+pionMass*pionMass);
	  sigTks_p4.SetCoordinates(px,py,pz,e);
	  
      // Loop over tracks to select signal and isolation cone tracks
      bool isSignalTk;
      for ( unsigned int i=0; i < SigConeTTTrackPtrs.size(); i++ ){
        L1TTTrackRefPtr iTrk = SigConeTTTrackPtrs.at(i);
        // Skip the matched track (already added)
	    if( iTrk == matchedTrack ) continue;
        isSignalTk = false;
        // Calculate dR and dPOCAz
        double dR = reco::deltaR(iTrk->getMomentum(cfg_tk_nFitParams).eta(), iTrk->getMomentum(cfg_tk_nFitParams).phi(), 
                         matchedTrack->getMomentum(cfg_tk_nFitParams).eta(), matchedTrack->getMomentum(cfg_tk_nFitParams).phi());     
        double dPOCAz = fabs(matchedTrack->getPOCA(cfg_tk_nFitParams).z() - iTrk->getPOCA(cfg_tk_nFitParams).z());
        // Pick signal cone tracks
        if (dR >= signalCone_dRmin && dR < signalCone_dRmax && dPOCAz < cfg_sigConeTks_maxDeltaZ){
          // Add tracks (and sum four-momenta) up to tau invariant mass
          math::XYZTLorentzVector p4tmp;
          px = iTrk->getMomentum().x();
	      py = iTrk->getMomentum().y();
	      pz = iTrk->getMomentum().z();
	      e  = sqrt(px*px+py*py+pz*pz+pionMass*pionMass);
	      p4tmp.SetCoordinates(px,py,pz,e);
	      sigTks_p4 += p4tmp;
          if (sigTks_p4.M() > cfg_sigConeTks_maxInvMass) {
              sigTks_p4 -= p4tmp;
          }
          else {
//           std::cout << "Adding track with dR = " << dR << " and pT = " << sqrt(px*px+py*py) << "and E = " << e << std::endl;
           sigConeTks.push_back(iTrk);
           isSignalTk = true;               
          }
        } // end of signal cone checks
        // Pick isolation cone tracks
        if (dR > isolationCone_dRmin && dR < isolationCone_dRmax && !isSignalTk){
            if (dPOCAz < vtxIso){
                vtxIso = dPOCAz;
            }
            // Calculations for other isolation criteria (relative isolation, jet width)
            if (dPOCAz < cfg_relIso_maxDeltaZ){
                double tkPt = iTrk->getMomentum(cfg_tk_nFitParams).perp();
                isoConePtSum += tkPt;
                dRtimesPtSum += dR*tkPt;
                isoConeTks.push_back(iTrk);
            }
        }
        
      } // end of loop over tracks
               
      // Sanity check:
//      std::cout << "Signal cone now contains " << sigConeTks.size() << " tracks; signalCone_dRmin = " 
//      << signalCone_dRmin << "; signalCone_dRmax = " << signalCone_dRmax << "; total mass = " << sigTks_p4.M() << std::endl;
//      std::cout << "Isolation cone now contains " << isoConeTks.size() << " tracks; isolationCone_dRmin = " 
//      << isolationCone_dRmin << "; isolationCone_dRmax = " << isolationCone_dRmax << std::endl;


      // Loop over seed tracks and check for higher-pT tracks inside the isolation cone
      bool bIsLdgTrack = true;
      for ( unsigned int i=0; i < SeedTTTrackPtrs.size(); i++ ){
        L1TTTrackRefPtr iTrk = SeedTTTrackPtrs.at(i);
        double dR = reco::deltaR(iTrk->getMomentum(cfg_tk_nFitParams).eta(), iTrk->getMomentum(cfg_tk_nFitParams).phi(), 
                         matchedTrack->getMomentum(cfg_tk_nFitParams).eta(), matchedTrack->getMomentum(cfg_tk_nFitParams).phi());
	    // Skip tracks outside the cone defined by  cfg_seedTk_maxDeltaR
	    if (dR > cfg_seedTk_maxDeltaR) continue;
	    // Compare pT
        if (iTrk->getMomentum(cfg_tk_nFitParams).perp() > matchedTrack->getMomentum(cfg_tk_nFitParams).perp()){
//            std::cout << "Seed track not the leading track! Reject candidate!" << std::endl;
		    bIsLdgTrack = false;
		    break;
	    }
      }	// end of loop over tracks

	  // Proceed only if there is no higher pT track within signal or isolation cones
	  if (cfg_seedTk_useMaxDeltaR && !bIsLdgTrack) continue;      

      // Jet width
//      double jetWidth = 0;
//      if (dRtimesPtSum > 0.0 && isoConePtSum > 0.0)
//          jetWidth = dRtimesPtSum / isoConePtSum;
//      bool bPassJetWidth = (jetWidth  < cfg_jetWidth_WP);
//      if (!bPassJetWidth) continue;

      // Relative isolation
//      double relIso = isoConePtSum / matchedTrack->getMomentum(cfg_tk_nFitParams).perp();
//      bool bPassRelIso = (relIso < cfg_relIso_WP); // orthogonal to VtxIso
//      if (!bPassRelIso) continue;
      
      // Vertex isolation      
      bool bPassVtxIso = (vtxIso > cfg_vtxIso_WP); // orthogonal to RelIso
      if (cfg_useVtxIso && !bPassVtxIso) continue;
	  
      const math::XYZTLorentzVector p4 = sigTks_p4;
      Tau finalTau = *caloTauIter;
      L1CaloTkTauParticle caloTauCandidate(p4, sigConeTks, finalTau, vtxIso, caloEt);
      L1CaloTkTauCandidates -> push_back( caloTauCandidate );
      
  } // end of loop over calo taus

  // Sort the final CaloTkTau candidates
  std::sort( L1CaloTkTauCandidates->begin(), L1CaloTkTauCandidates->end(), L1CaloTkTau::EtComparator() );

  // Add final CaloTkTau candidates to the event
  iEvent.put(std::move(L1CaloTkTauCandidates), label );
  
}




// --------------------------------------------------------------------------------------
void L1CaloTkTauParticleProducer::GetShrinkingConeSizes(float pt,
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
  double signalCone_max = (shrinkCone_Constant)/(pt);
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

float L1CaloTkTauParticleProducer::CorrectedEta(float eta, float zv)  {

  // Correct the eta of the L1EG object once we know the zvertex
  
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
  
  float tanthetaprime = delta * tantheta / (delta - zv );
  
  float thetaprime = TMath::ATan( tanthetaprime );
  if (thetaprime < 0) thetaprime = thetaprime + TMath::Pi();
  
  float etaprime = -TMath::Log( TMath::Tan( thetaprime / 2.) );
  return etaprime;
  
}

// --------------------------------------------------------------------------------------

float L1CaloTkTauParticleProducer::CalibrateCaloTau(float Et, float Eta) {

    // Calibration scale factors as a function of Et and eta
    float eta_values [30] = {-1.2615 , -1.1745 , -1.0875 , -1.0005 , -0.9135 , -0.8265 , -0.7395 , -0.6525 , -0.5655 , -0.4785 , -0.3915 , -0.3045 , -0.2175 , -0.1305 , -0.0435 , 0.0435 , 0.1305 , 0.2175 , 0.3045 , 0.3915 , 0.4785 , 0.5655 , 0.6525 , 0.7395 , 0.8265 , 0.9135 , 1.0005 , 1.0875 , 1.1745 , 1.2615};

    float et20to40 [30] = {1.08888888889 , 1.08214285714 , 1.09558823529 , 1.09705882353 , 1.08928571429 , 1.17321428571 , 1.13863636364 , 0.951388888889 , 1.04134615385 , 1.05202702703 , 1.0737804878 , 1.07928571429 , 1.03722222222 , 1.027 , 1.0625 , 1.04659090909 , 1.00663265306 , 1.08520408163 , 1.05982142857 , 1.068 , 1.09705882353 , 1.04294871795 , 1.10714285714 , 1.135 , 1.07243589744 , 1.09527027027 , 1.10227272727 , 1.045 , 1.125 , 1.17833333333};
    float et40to60 [30] = {1.18958333333 , 1.30038461538 , 1.20746753247 , 1.23157894737 , 1.28042168675 , 1.29902597403 , 1.209 , 1.20691489362 , 1.24527027027 , 1.22334710744 , 1.20441176471 , 1.18962264151 , 1.16615044248 , 1.19963768116 , 1.20445205479 , 1.19322033898 , 1.18793103448 , 1.19318181818 , 1.21378504673 , 1.1619047619 , 1.30176767677 , 1.179 , 1.25026315789 , 1.27069892473 , 1.24722222222 , 1.24368686869 , 1.25069444444 , 1.34326923077 , 1.29326923077 , 1.12934782609};
    float et60to80 [30] = {1.58534482759 , 1.40650684932 , 1.49871794872 , 1.44539473684 , 1.43042168675 , 1.41547619048 , 1.39267241379 , 1.31238317757 , 1.31113138686 , 1.32147887324 , 1.20506993007 , 1.28493589744 , 1.275 , 1.26075949367 , 1.23529411765 , 1.22561728395 , 1.2175170068 , 1.27432885906 , 1.32 , 1.32578125 , 1.31092592593 , 1.34016393443 , 1.41803278689 , 1.36583333333 , 1.37180851064 , 1.42040229885 , 1.42175324675 , 1.45856164384 , 1.32023809524 , 1.4262195122};
    float et80to100 [30] = {1.47269230769 , 1.66231343284 , 1.553 , 1.5643258427 , 1.44587378641 , 1.50247252747 , 1.42363636364 , 1.38863636364 , 1.33968531469 , 1.36578947368 , 1.32950819672 , 1.30347222222 , 1.24542253521 , 1.21581632653 , 1.28823529412 , 1.25534188034 , 1.2147810219 , 1.32261904762 , 1.26145833333 , 1.34029850746 , 1.32115384615 , 1.36694915254 , 1.3975 , 1.41733870968 , 1.43305084746 , 1.56658415842 , 1.57279411765 , 1.55601265823 , 1.6765625 , 1.63642857143};
    float et100to150 [30] = {1.78409090909 , 1.72857142857 , 1.63770718232 , 1.61 , 1.54431818182 , 1.49327411168 , 1.50070850202 , 1.40084541063 , 1.39107929515 , 1.3392578125 , 1.34017509728 , 1.29774590164 , 1.28189655172 , 1.29989539749 , 1.299 , 1.2872 , 1.27120253165 , 1.31080246914 , 1.27810077519 , 1.26927966102 , 1.35357142857 , 1.3712962963 , 1.42927350427 , 1.48027522936 , 1.58156108597 , 1.52722222222 , 1.58534482759 , 1.67808988764 , 1.69887096774 , 1.7722972973};
    float et150to200 [30] = {1.72346938776 , 1.71826923077 , 1.64004065041 , 1.57863636364 , 1.51637931034 , 1.53665048544 , 1.40175438596 , 1.34458333333 , 1.2822519084 , 1.35403225806 , 1.3206043956 , 1.23417431193 , 1.2626146789 , 1.2048245614 , 1.22244897959 , 1.21875 , 1.26811926606 , 1.28713592233 , 1.25291666667 , 1.25432692308 , 1.26517857143 , 1.29241071429 , 1.41261904762 , 1.44426605505 , 1.50809352518 , 1.53815789474 , 1.6356741573 , 1.60185950413 , 1.72682481752 , 1.71026315789};
    float etGE200 [30] = {1.4875 , 1.56132478632 , 1.43695652174 , 1.37755102041 , 1.42888349515 , 1.29547619048 , 1.2260989011 , 1.2244047619 , 1.23230337079 , 1.16793478261 , 1.15030864198 , 1.13611111111 , 1.04772727273 , 1.12837078652 , 1.1130952381 , 1.09785714286 , 1.18611111111 , 1.20917721519 , 1.08095238095 , 1.168125 , 1.25625 , 1.20192307692 , 1.1356741573 , 1.31777108434 , 1.35804347826 , 1.31432038835 , 1.42342105263 , 1.5325 , 1.49905660377 , 1.53264705882};

    float dummy_eta = 0;

    // Apply calibration
    if(Et >= 20 && Et < 40) {
        dummy_eta = findClosest(eta_values,30,Eta);
	    for (int i=0; i<30; i++) {
		    if (dummy_eta == eta_values[i]) return (1/et20to40[i])*Et;
        }
	}
    else if(Et >= 40 && Et < 60) {
	    dummy_eta = findClosest(eta_values,30,Eta);
	    for (int i=0; i<30; i++) {
		    if (dummy_eta == eta_values[i]) return (1/et40to60[i])*Et;
	    }
	}
    else if(Et >= 60 && Et < 80) {
	    dummy_eta = findClosest(eta_values,30,Eta);
	    for (int i=0; i<30; i++) {
            if (dummy_eta == eta_values[i]) return (1/et60to80[i])*Et;
	    }
	}
    else if(Et >= 80 && Et < 100) {
        dummy_eta = findClosest(eta_values,30,Eta);
        for (int i=0; i<30; i++) {
	        if (dummy_eta == eta_values[i]) return (1/et80to100[i])*Et;
	    }
	}
    else if(Et >= 100 && Et < 150) {
	    dummy_eta = findClosest(eta_values,30,Eta);
	    for (int i=0; i<30; i++) {
            if (dummy_eta == eta_values[i]) return (1/et100to150[i])*Et;
	    }
	}
    else if(Et >= 150 && Et < 200) {
	    dummy_eta = findClosest(eta_values,30,Eta);
	    for (int i=0; i<30; i++) {
            if (dummy_eta == eta_values[i]) return (1/et150to200[i])*Et;
	    }
	}
    else if(Et >= 200) {
        dummy_eta = findClosest(eta_values,30,Eta);
	    for (int i=0; i<30; i++) {
	        if (dummy_eta == eta_values[i]) return (1/etGE200[i])*Et;
	    }
	}

    // If calibration cannot be applied, just return the original Et value
//    std::cout << "WARNING! Could not calibrate this calo tau with Et = " << Et << std::endl;
    return Et;

}

// --------------------------------------------------------------------------------------

float L1CaloTkTauParticleProducer::findClosest(float arr[], int n, float target) { 

    // Corner cases 
    if (target <= arr[0]) 
        return arr[0]; 
    if (target >= arr[n - 1]) 
        return arr[n - 1]; 
  
    // Doing binary search 
    int i = 0, j = n, mid = 0; 
    while (i < j) { 
        mid = (i + j) / 2; 
  
        if (arr[mid] == target) 
            return arr[mid]; 
  
        /* If target is less than array element, 
            then search in left */
        if (target < arr[mid]) { 
  
            // If target is greater than previous 
            // to mid, return closest of two 
            if (mid > 0 && target > arr[mid - 1]) 
                return getClosest(arr[mid - 1], 
                                  arr[mid], target); 
  
            /* Repeat for left half */
            j = mid; 
        } 
  
        // If target is greater than mid 
        else { 
            if (mid < n - 1 && target < arr[mid + 1]) 
                return getClosest(arr[mid], 
                                  arr[mid + 1], target); 
            // update i 
            i = mid + 1;  
        } 
    } 
  
    // Only single element left after search 
    return arr[mid]; 
} 

float L1CaloTkTauParticleProducer::getClosest(float val1, float val2, float target) 
{ 
    if (target - val1 >= val2 - target) 
        return val2; 
    else
        return val1; 
}

// ------------ method called once each job just before starting event loop  ------------
void
L1CaloTkTauParticleProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1CaloTkTauParticleProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1CaloTkTauParticleProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1CaloTkTauParticleProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1CaloTkTauParticleProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1CaloTkTauParticleProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1CaloTkTauParticleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1CaloTkTauParticleProducer);



