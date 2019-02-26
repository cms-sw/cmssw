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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1TrackTrigger/interface/L1CaloTkTauParticle.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1Upgrade.h"

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
  
  float DeltaPhi(float phi1, float phi2) ; //TODO: double check
  float deltaR(float eta1, float eta2, float phi1, float phi2) ; //TODO: double check
  float CorrectedEta(float eta, float zv); //TODO: double check

  float CalibrateCaloTau(float Et, float Eta);
  float findClosest(float arr[], int n, float target);
  float getClosest(float val1, float val2, float target);
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void GetShrinkingConeSizes(float pt, //TODO double check
			     const float shrinkCone_Constant,
			     const float sigCone_dRCutoff,
  			     float &sigCone_dRMin,
  			     float &sigCone_dRMax,
  			     float &isoCone_dRMin,
  			     float &isoCone_dRMax,
  			     const bool isoCone_useCone);
  
  float CalculateVtxIso(std::vector< L1TTTrackRefPtr > allTracks, // TODO double check
			std::vector< unsigned int > clustTracksIndx,
			bool useIsoCone=false); 

  // float CalculateRelIso(std::vector< L1TTTrackRefPtr > allTracks, //TODO implement
  // 			std::vector< unsigned int > clustTracksIndx,
  //                    const float deltaZ0_max, 
  // 			bool useIsoCone=false); 
  
  // ----------member data ---------------------------
  
  std::string label;
 
//  const edm::EDGetTokenT< EGammaBxCollection > egToken; //TODO: replace by tau token
  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
  std::vector< edm::EDGetTokenT<l1t::TauBxCollection> > tauTokens;

// ----- Imported from CaloTk::InitVars_

  unsigned int tk_nFitParams;

  // Matching tracks
  float seedTk_minPt;      
  float seedTk_minEta;
  float seedTk_maxEta;
  float seedTk_maxChiSq;
  float seedTk_minStubs;

  // Signal cone tracks
  float sigConeTks_minPt;
  float sigConeTks_minEta;
  float sigConeTks_maxEta;
  float sigConeTks_maxChiSq;
  unsigned int sigConeTks_minStubs;
  float sigConeTks_dPOCAz;
  float sigConeTks_maxInvMass;
 
  // Isolation cone tracks
  float isoConeTks_minPt;
  float isoConeTks_minEta;
  float isoConeTks_maxEta;
  float isoConeTks_maxChiSq;
  unsigned int isoConeTks_minStubs;

  // Signal cone parameters
  float shrinkCone_Constant;
  float sigCone_dRMin;
  float sigCone_dRMax;
  float sigCone_cutoffDeltaR;

  // Isolation cone
  float isoCone_dRMin;
  float isoCone_dRMax;
  bool isoCone_useCone;

  // CaloTaus
  bool calibrateCaloTaus;

  // Tau object
  float tau_jetWidth;
  float tau_vtxIsoWP;
  float tau_relIsoWP;
  float tau_relIsodZ0;

// ----------------------
  
} ;


//
// constructors and destructor
//
L1CaloTkTauParticleProducer::L1CaloTkTauParticleProducer(const edm::ParameterSet& iConfig) :
  //egToken(consumes< EGammaBxCollection >(iConfig.getParameter<edm::InputTag>("L1EGammaInputTag"))),
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
  {
  const auto& taus = iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("L1CaloTauInputTag");
  for (const auto& tau: taus) {
    tauTokens.push_back(consumes<l1t::TauBxCollection>(tau));
  }

  produces<L1CaloTkTauParticleCollection>(label);

  // Common to all tracks
  tk_nFitParams         = (unsigned int)iConfig.getParameter<unsigned int>("tk_nFitParams");

  // Seed tracks
  seedTk_minPt          = (float)iConfig.getParameter<double>("seedTk_minPt");
  seedTk_minEta         = (float)iConfig.getParameter<double>("seedTk_minEta");
  seedTk_maxEta         = (float)iConfig.getParameter<double>("seedTk_maxEta");
  seedTk_maxChiSq       = (float)iConfig.getParameter<double>("seedTk_maxChiSq");
  seedTk_minStubs       = (unsigned int)iConfig.getParameter<unsigned int>("seedTk_minStubs");

  // Signal cone tracks
  sigConeTks_minPt      = (float)iConfig.getParameter<double>("sigConeTks_minPt");
  sigConeTks_minEta     = (float)iConfig.getParameter<double>("sigConeTks_minEta");
  sigConeTks_maxEta     = (float)iConfig.getParameter<double>("sigConeTks_maxEta");
  sigConeTks_maxChiSq   = (float)iConfig.getParameter<double>("sigConeTks_maxChiSq");
  sigConeTks_minStubs   = (unsigned int)iConfig.getParameter<unsigned int>("sigConeTks_minStubs");
  // Signal cone tracks: extra properties //TODO: where are these applied?
  sigConeTks_dPOCAz     = (float)iConfig.getParameter<double>("sigConeTks_dPOCAz");
  sigConeTks_maxInvMass = (float)iConfig.getParameter<double>("sigConeTks_maxInvMass");
 
  // Isolation cone tracks
/*  isoConeTks_minPt     = (float)iConfig.getParameter<double>("isoConeTks_minPt");
  isoConeTks_minEta    = (float)iConfig.getParameter<double>("isoConeTks_minEta");
  isoConeTks_maxEta    = (float)iConfig.getParameter<double>("isoConeTks_maxEta");
  isoConeTks_maxChiSq  = (float)iConfig.getParameter<double>("isoConeTks_maxChiSq");
  isoConeTks_minStubs  = (unsigned int)iConfig.getParameter<unsigned int>("isoConeTks_minStubs"); */

  // Signal cone parameters
  shrinkCone_Constant  = (float)iConfig.getParameter<double>("shrinkCone_Constant");
  sigCone_dRMin        = (float)iConfig.getParameter<double>("sigCone_dRMin");
  sigCone_dRMax        = (float)iConfig.getParameter<double>("sigCone_dRMax");
  sigCone_cutoffDeltaR = (float)iConfig.getParameter<double>("sigCone_cutoffDeltaR");

  // Isolation cone
  isoCone_dRMin        = (float)iConfig.getParameter<double>("isoCone_dRMin");
  isoCone_dRMax        = (float)iConfig.getParameter<double>("isoCone_dRMax");
  isoCone_useCone      = (bool)iConfig.getParameter<bool>("isoCone_useCone");

  // CaloTaus
  calibrateCaloTaus    = (bool)iConfig.getParameter<bool>("calibrateCaloTaus");

  // Tau object
  tau_jetWidth         = (float)iConfig.getParameter<double>("tau_jetWidth");
  tau_vtxIsoWP         = (float)iConfig.getParameter<double>("tau_vtxIsoWP");
  tau_relIsoWP         = (float)iConfig.getParameter<double>("tau_relIsoWP");
  tau_relIsodZ0        = (float)iConfig.getParameter<double>("tau_relIsodZ0");
  
}

L1CaloTkTauParticleProducer::~L1CaloTkTauParticleProducer() {
}

// ------------ method called to produce the data  ------------
void L1CaloTkTauParticleProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  
  std::unique_ptr<L1CaloTkTauParticleCollection> L1CaloTkTauCandidates(new L1CaloTkTauParticleCollection);

  std::vector<Tau> caloTaus;
  // Construct a vector of CaloTaus  
  for (auto & tautoken: tauTokens){
    edm::Handle<l1t::TauBxCollection> tau;
    iEvent.getByToken(tautoken,  tau);
    if (tau.isValid()){ 
//      std::cout << "WE HAVE A GOOD TAU!" << std::endl;  
      for (int ibx = tau->getFirstBX(); ibx <= tau->getLastBX(); ++ibx) {
        for (l1t::TauBxCollection::const_iterator it=tau->begin(ibx); it!=tau->end(ibx); it++){
          if (it->pt() > 0){
	        caloTaus.push_back(*it);
          }
        }
      } // end of if(tau.isValid())
    } else {
      edm::LogWarning("MissingProduct") << "L1Upgrade Tau not found. Branch will not be filled" << std::endl;
    }
  }

  // SANITY CHECK: Loop over CaloTaus and print Et, Eta, Phi
//  for (auto tauIter = caloTaus.begin(); tauIter != caloTaus.end(); ++tauIter) {
//    std::cout << "Tau pT   = " << tauIter->pt() << std::endl;
//  }
  // Possible functions: et(), eta(), phi(), hwPt(), hwEta(), hwPhi(), hwIso() ibx, towerIPhi(), towerIEta(), rawEt(), isoEt(), nTT(), hasEM(), isMerged(), hwQual()

  // Access the L1Tracks
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

  // Build track collections (vectors of pointers) from tracks passing the quality criteria
  std::vector< L1TTTrackRefPtr > SeedTTTrackPtrs;
  std::vector< L1TTTrackRefPtr > SigConeTTTrackPtrs;
  std::vector< L1TTTrackRefPtr > IsoConeTTTrackPtrs;
  unsigned int seedTk_counter = 0;

  // Loop over all tracks
  for (trkIter = L1TTTrackHandle->begin(); trkIter != L1TTTrackHandle->end(); ++trkIter) {
    // Construct a pointer
    L1TTTrackRefPtr track_RefPtr(L1TTTrackHandle, seedTk_counter++);
    // Retrieve track information
    float Pt   = trkIter->getMomentum(tk_nFitParams).perp();
    float Eta  = trkIter->getMomentum(tk_nFitParams).eta();
    float Chi2 = trkIter->getChi2(tk_nFitParams);
    std::vector< L1TTStubRef > Stubs = trkIter-> getStubRefs();
    unsigned int NStubs              = Stubs.size();
    // Seed tracks
    if ( Pt   > seedTk_minPt  &&
         fabs(Eta)  > seedTk_minEta && 
         fabs(Eta)  < seedTk_maxEta &&
         Chi2 < seedTk_maxChiSq &&
         NStubs > seedTk_minStubs )  
            SeedTTTrackPtrs.push_back(track_RefPtr);
    // Signal cone tracks
    if ( Pt   > sigConeTks_minPt  &&
         fabs(Eta)  > sigConeTks_minEta && 
         fabs(Eta)  < sigConeTks_maxEta &&
         Chi2 < sigConeTks_maxChiSq &&
         NStubs > sigConeTks_minStubs )  
            SigConeTTTrackPtrs.push_back(track_RefPtr);
/*    // Isolation cone tracks
    if ( Pt   > isoConeTks_minPt  &&
         fabs(Eta)  > isoConeTks_minEta && 
         fabs(Eta)  < isoConeTks_maxEta &&
         Chi2 < isoConeTks_maxChiSq &&
         NStubs > isoConeTks_minStubs )  
            IsoConeTTTrackPtrs.push_back(track_RefPtr); */
  }// End of the track loop

  /// Sort all track collections by pT
  std::sort( SeedTTTrackPtrs.begin(), SeedTTTrackPtrs.end() );
  std::sort( SigConeTTTrackPtrs.begin(), SigConeTTTrackPtrs.end() );
//  std::sort( IsoConeTTTrackPtrs.begin(), IsoConeTTTrackPtrs.end() );

  // SANITY CHECK:
//  std::cout << "Vector sizes: " << SeedTTTrackPtrs.size() << "; " << SigConeTTTrackPtrs.size() << "; " << IsoConeTTTrackPtrs.size() << std::endl;

  // **** CALOTK ALGORITHM *** //
  
  // Loop over calo taus
  bool foundMatchTrk;
  double matchTk_dR = 999.9;  
  for (std::vector<Tau>::iterator caloTauIter = caloTaus.begin(); caloTauIter != caloTaus.end(); ++caloTauIter) {
    
      // Calibrate calo taus if needed
      float caloEt = caloTauIter->et();
      float caloEta = caloTauIter->eta();
      if(calibrateCaloTaus) {
          caloEt = (CalibrateCaloTau(caloEt, caloEta));
      }

      // Find a matching track
      foundMatchTrk = false;
      // Loop over seed tracks
      L1TTTrackRefPtr matchedTrack;
      for ( unsigned int i=0; i < SeedTTTrackPtrs.size(); i++ ){
//        std::cout << "Checking track #" << i << std::endl;
        L1TTTrackRefPtr iTrk = SeedTTTrackPtrs.at(i);
        double dR = reco::deltaR(iTrk->getMomentum(tk_nFitParams).eta(), iTrk->getMomentum(tk_nFitParams).phi(), caloTauIter->eta(), caloTauIter->phi());
        if (dR < 0.1 && dR < matchTk_dR) { //TODO: add 0.1 (matchingDR as input parameter)
          foundMatchTrk = true;
          matchedTrack = iTrk;
//          if (foundMatchTrk) std::cout << "IT'S A MATCH!!! dR = " << dR << std::endl; 
        }
      } 

      // Proceed only if a matching track is found
      if (!foundMatchTrk) continue;

      // Define the signal and isolation cones
      float signalCone_dRmin = 0.0; // unchanged by GetShrinkingConeSizes function
      float signalCone_dRmax = -999.0;
      float isolationCone_dRmin = -999.0;
      float isolationCone_dRmax = isoCone_dRMax; // unchanged by GetShrinkingConeSizes function
      GetShrinkingConeSizes(caloEt, shrinkCone_Constant, sigCone_cutoffDeltaR, // input parameters
                            signalCone_dRmin, signalCone_dRmax,isolationCone_dRmin, isolationCone_dRmax, // these values are updated by this function
                            isoCone_useCone); // input parameter (if useCone is selected, isolationCone_dRmin = 0)

      // Initialize track lists
      std::vector< L1TTTrackRefPtr > sigConeTks;
      std::vector< L1TTTrackRefPtr > isoConeTks;
      // Initialize isolation variables
      float vtxIso = 999.0;
      float isoConePtSum = 0.0;
      float dRtimesPtSum = 0.0;
      math::XYZTLorentzVector sigTks_p4; // track-based four-momentum, FIXME: currently not used for anything?!
      // Loop over tracks
      for ( unsigned int i=0; i < SigConeTTTrackPtrs.size(); i++ ){
        L1TTTrackRefPtr iTrk = SigConeTTTrackPtrs.at(i);
        // Skip the matching track //FIXME: in CaloTk.C code the matching track is first skipped and them added - why?
//      if( tk->index() == L1TkTau.GetMatchingTk().index() ) continue;
         // Calculate dR and dPOCAz
        double dR = reco::deltaR(iTrk->getMomentum(tk_nFitParams).eta(), iTrk->getMomentum(tk_nFitParams).phi(), 
                         matchedTrack->getMomentum(tk_nFitParams).eta(), matchedTrack->getMomentum(tk_nFitParams).phi());     
        double dPOCAz = abs(matchedTrack->getPOCA(tk_nFitParams).z() - iTrk->getPOCA(tk_nFitParams).z());
        // Pick signal cone tracks
        if (dR >= signalCone_dRmin && dR < signalCone_dRmax && dPOCAz < sigConeTks_dPOCAz){
           // Add tracks (and sum four-momenta) up to tau invariant mass
           math::XYZTLorentzVector p4tmp;
           double px = iTrk->getMomentum().x();
	       double py = iTrk->getMomentum().y();
	       double pz = iTrk->getMomentum().z();
	       double e  = sqrt(px*px+py*py+pz*pz+0.14*0.14); // assume pion mass 0.14 GeV
	       p4tmp.SetCoordinates(px,py,pz,e);
	       sigTks_p4 += p4tmp;
           if (sigTks_p4.M() > sigConeTks_maxInvMass) {
               sigTks_p4 -= p4tmp;
	           continue;
           }
           else {
            sigConeTks.push_back(iTrk);               
           }
         }
      
        // Pick isolation cone tracks
        if (dR > isolationCone_dRmin && dR < isolationCone_dRmax){ //FIXME: should we check that the same track is not inside signal and isolation cones?
            if (dPOCAz < vtxIso){
                vtxIso = dPOCAz;
            }
            if (dPOCAz < tau_relIsodZ0){ //FIXME: should this be inverted?
                double tkPt = iTrk->getMomentum(tk_nFitParams).perp();
                isoConePtSum += tkPt;
                dRtimesPtSum += dR*tkPt;
                isoConeTks.push_back(iTrk);
            }
        }
      } // end of loop over tracks
         
      // Sanity check:
//      std::cout << "Signal cone now contains " << sigConeTks.size() << " tracks; signalCone_dRmin = " 
//      << signalCone_dRmin << "; signalCone_dRmax = " << signalCone_dRmax << std::endl;
//      std::cout << "Isolation cone now contains " << isoConeTks.size() << " tracks; isolationCone_dRmin = " 
//      << isolationCone_dRmin << "; isolationCone_dRmax = " << isolationCone_dRmax << std::endl;


      // Loop over seed tracks and check for higher-pT tracks inside the isolation cone
      bool bIsLdgTrack = true;
      for ( unsigned int i=0; i < SeedTTTrackPtrs.size(); i++ ){
        L1TTTrackRefPtr iTrk = SeedTTTrackPtrs.at(i);
        double dR = reco::deltaR(iTrk->getMomentum(tk_nFitParams).eta(), iTrk->getMomentum(tk_nFitParams).phi(), 
                         matchedTrack->getMomentum(tk_nFitParams).eta(), matchedTrack->getMomentum(tk_nFitParams).phi());
	    // Skip tracks outside the isolation cone
	    if (dR > isolationCone_dRmax) continue;  // FIXME: use a smaller cone instead of isoCone?
	    // Compare pT
        if (iTrk->getMomentum(tk_nFitParams).perp() > matchedTrack->getMomentum(tk_nFitParams).perp()){
//            std::cout << "Seed track not the leading track! Reject candidate!" << std::endl;
		    bIsLdgTrack = false;
		    break;
	    }
      }	// end of loop over tracks

	  // Proceed only if there is no higher pT track within signal or isolation cones
	  if (!bIsLdgTrack) continue;      

//      std::cout << "Passed all cuts before the isolation criteria!" << std::endl;

      // ISOLATION OF THE TAU CANDIDATE

      // Jet width
      double jetWidth = 0;
      if (dRtimesPtSum > 0.0 && isoConePtSum > 0.0)
          jetWidth = dRtimesPtSum / isoConePtSum;
      bool bPassJetWidth = (jetWidth  < tau_jetWidth);
      if (!bPassJetWidth) continue;

      // Relative isolation
      double relIso = isoConePtSum / matchedTrack->getMomentum(tk_nFitParams).perp();
	  bool bPassRelIso = (relIso < tau_relIsoWP); // orthogonal to VtxIso
	  if (!bPassRelIso) continue;

      // Vertex isolation      
      bool bPassVtxIso = (vtxIso > tau_vtxIsoWP); // orthogonal to RelIso
	  if (!bPassVtxIso) continue;

//      std::cout << "Passed all isolation criteria with jetWidth = " << jetWidth << ", relIso = " << relIso << " and vtxIso = " << vtxIso << std::endl;

      const math::XYZTLorentzVector p4 = sigTks_p4;
      const Tau finalTau = *caloTauIter;
      L1CaloTkTauParticle caloTauCandidate(p4, sigConeTks, finalTau, vtxIso);
      L1CaloTkTauCandidates -> push_back( caloTauCandidate );
      
  } // end of loop over calo taus

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

float L1CaloTkTauParticleProducer::CalculateVtxIso(std::vector< L1TTTrackRefPtr > allTracks,
					      std::vector< unsigned int > clustTracksIndx,
					      bool useIsoCone) {

  // Initializations
  float  mindZ0 = 999.9;
  float dz0, deltaR;
  
  // Seed track properties
  L1TTTrackRefPtr seedTrack = allTracks.at(clustTracksIndx.at(0));
  float seedEta  = seedTrack->getMomentum(tk_nFitParams).eta();
  float seedPhi  = seedTrack->getMomentum(tk_nFitParams).phi();
  float seedz0   = seedTrack->getPOCA(tk_nFitParams).z();
  
  // For-loop: All the Tracks
  for (unsigned int i=0; i < allTracks.size(); i++) {

    L1TTTrackRefPtr iTrk = allTracks.at(i);
    float iEta  = iTrk->getMomentum(tk_nFitParams).eta();
    float iPhi  = iTrk->getMomentum(tk_nFitParams).phi();
    float iz0   = iTrk->getPOCA(tk_nFitParams).z();
        
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
    
    if (deltaR > isoCone_dRMin && deltaR < isoCone_dRMax) {
      // Calculate mindz0
      dz0 = fabs(iz0 - seedz0);
      if (dz0 < mindZ0) mindZ0 = dz0;
    }
    
  } // End-loop: All the Tracks
  
  float vtxIso = mindZ0;

  return vtxIso;
  
  
  
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

float L1CaloTkTauParticleProducer::DeltaPhi(float phi1, float phi2) {
  // dPhi between 0 and Pi
  float dphi = phi1 - phi2;
  if (dphi < 0) dphi = dphi + 2.*TMath::Pi();
  if (dphi > TMath::Pi()) dphi = 2.*TMath::Pi() - dphi;
  return dphi;
}

// --------------------------------------------------------------------------------------

float L1CaloTkTauParticleProducer::deltaR(float eta1, float eta2, float phi1, float phi2) {
  float deta = eta1 - eta2;
  float dphi = DeltaPhi(phi1, phi2);
  float DR= sqrt( deta*deta + dphi*dphi );
  return DR;
}

// --------------------------------------------------------------------------------------

float L1CaloTkTauParticleProducer::CalibrateCaloTau(float Et, float Eta) {

    // Calibration values as a function of eta
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
void
L1CaloTkTauParticleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1CaloTkTauParticleProducer);



