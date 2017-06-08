// -*- C++ -*-
//
//
// dummy producer for a L1TkEmParticle
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

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"

#include "DataFormats/Math/interface/LorentzVector.h"


// for L1Tracks:
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <string>
#include "TMath.h"

using namespace l1t ;
//
// class declaration
//

class L1TkEmParticleProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
   typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;

      explicit L1TkEmParticleProducer(const edm::ParameterSet&);
      ~L1TkEmParticleProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      float DeltaPhi(float phi1, float phi2) ;
      float deltaR(float eta1, float eta2, float phi1, float phi2) ;
      float CorrectedEta(float eta, float zv);


   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      //virtual void beginRun(edm::Run&, edm::EventSetup const&);
      //virtual void endRun(edm::Run&, edm::EventSetup const&);
      //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------

	std::string label;

	float ETmin; 	// min ET in GeV of L1EG objects

	float ZMAX;		// |z_track| < ZMAX in cm
	float CHI2MAX;		
	float DRmin;
	float DRmax;
	float PTMINTRA;
	bool PrimaryVtxConstrain;	// use the primary vertex (default = false)
        //bool DeltaZConstrain;	// use z = z of the leading track within DR < DRmax;
	float DeltaZMax;	// | z_track - z_primaryvtx | < DeltaZMax in cm. 
				// Used only when PrimaryVtxConstrain = True.
	float IsoCut;
	bool RelativeIsolation;

        const edm::EDGetTokenT< EGammaBxCollection > egToken;
        const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
        const edm::EDGetTokenT<L1TkPrimaryVertexCollection> vertexToken;

} ;


//
// constructors and destructor
//
L1TkEmParticleProducer::L1TkEmParticleProducer(const edm::ParameterSet& iConfig) :
  egToken(consumes< EGammaBxCollection >(iConfig.getParameter<edm::InputTag>("L1EGammaInputTag"))),
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
  vertexToken(consumes< L1TkPrimaryVertexCollection >(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))) 
  {
  
  label = iConfig.getParameter<std::string>("label");  // label of the collection produced
  // e.g. EG or IsoEG if all objects are kept
  // EGIsoTrk or IsoEGIsoTrk if only the EG or IsoEG
  // objects that pass a cut RelIso < IsoCut are written
  // in the new collection.
  
  
  ETmin = (float)iConfig.getParameter<double>("ETmin");
  
   // parameters for the calculation of the isolation :
   ZMAX = (float)iConfig.getParameter<double>("ZMAX");
   CHI2MAX = (float)iConfig.getParameter<double>("CHI2MAX");
   PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");
   DRmin = (float)iConfig.getParameter<double>("DRmin");
   DRmax = (float)iConfig.getParameter<double>("DRmax");
   PrimaryVtxConstrain = iConfig.getParameter<bool>("PrimaryVtxConstrain");
   //DeltaZConstrain = iConfig.getParameter<bool>("DeltaZConstrain");
   DeltaZMax = (float)iConfig.getParameter<double>("DeltaZMax");
	// cut applied on the isolation (if this number is <= 0, no cut is applied)
   IsoCut = (float)iConfig.getParameter<double>("IsoCut");
   RelativeIsolation = iConfig.getParameter<bool>("RelativeIsolation");

   produces<L1TkEmParticleCollection>(label);
}

L1TkEmParticleProducer::~L1TkEmParticleProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkEmParticleProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
   using namespace edm;
  
   std::unique_ptr<L1TkEmParticleCollection> result(new L1TkEmParticleCollection);
  
   // the L1EGamma objects
   edm::Handle<EGammaBxCollection> eGammaHandle;
   iEvent.getByToken(egToken, eGammaHandle);  
   EGammaBxCollection eGammaCollection = (*eGammaHandle.product());
   EGammaBxCollection::const_iterator egIter;
   
   // the L1Tracks
   edm::Handle<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > L1TTTrackHandle;
   iEvent.getByToken(trackToken, L1TTTrackHandle);
   L1TTTrackCollectionType::const_iterator trackIter;
   
   // the primary vertex (used only if PrimaryVtxConstrain = true)
   float zvtxL1tk = -999;
   if (PrimaryVtxConstrain) {
     edm::Handle<L1TkPrimaryVertexCollection> L1VertexHandle;
     iEvent.getByToken(vertexToken, L1VertexHandle);
     if (!L1VertexHandle.isValid() ) {
       LogWarning("L1TkEmParticleProducer")
	 << "\nWarning: L1TkPrimaryVertexCollection not found in the event. Won't use any PrimaryVertex constraint."
	 << std::endl;
       PrimaryVtxConstrain = false;
     }
     else {
       std::vector<L1TkPrimaryVertex>::const_iterator vtxIter = L1VertexHandle->begin();
       // by convention, the first vertex in the collection is the one that should
       // be used by default
       zvtxL1tk = vtxIter -> getZvertex();
     }
   }
   
  if (!L1TTTrackHandle.isValid() ) {
    LogError("L1TkEmParticleProducer")
      << "\nWarning: L1TTTrackCollectionType not found in the event. Exit."
      << std::endl;
    return;
  }
  
  // Now loop over the L1EGamma objects
  
  if( !eGammaHandle.isValid() )
    {
      LogError("L1TkEmParticleProducer")
	<< "\nWarning: L1EmParticleCollection not found in the event. Exit."
	<< std::endl;
      return;
    }
  
  int ieg = 0;
  for (egIter = eGammaCollection.begin(0); egIter != eGammaCollection.end(0);  ++egIter) // considering BX = only
    {
      edm::Ref< EGammaBxCollection > EGammaRef( eGammaHandle, ieg );
      ieg ++; 
      
      
      float eta = egIter -> eta();
      if (PrimaryVtxConstrain) {
	// The eta of the L1EG object is seen from (0,0,0).
	// if PrimaryVtxConstrain = true, use the zvtxL1tk to correct the eta(L1EG)
	// that is used in the calculation of DeltaR.
	eta = CorrectedEta( (float)eta, zvtxL1tk);
      }
      float phi = egIter -> phi();
      float et = egIter -> et();
      
      if (et < ETmin) continue;
      
      // calculate the isolation of the L1EG object with
      // respect to L1Tracks.
      
      float trkisol = -999;
      float sumPt = 0;
      
      //std::cout << " here an EG w et = " << et << std::endl;
      
      //float z_leadingTrack = -999;
      //float Pt_leadingTrack = -999;
      
      /*
	if (DeltaZConstrain) {
	// first loop over the tracks to find the leading one in DR < DRmax
	for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
	float Pt = trackIter->getMomentum().perp();
	float Eta = trackIter->getMomentum().eta();
	float Phi = trackIter->getMomentum().phi();
	float z  = trackIter->getPOCA().z();
	if (fabs(z) > ZMAX) continue;
	if (Pt < PTMINTRA) continue;
	float chi2 = trackIter->getChi2();
	if (chi2 > CHI2MAX) continue;
	float dr = deltaR(Eta, eta, Phi,phi);
	if (dr < DRmax) {
	if (Pt > Pt_leadingTrack) {
	Pt_leadingTrack = Pt;
	z_leadingTrack = z;
	}
	}
	} // end loop over the tracks
	} // endif DeltaZConstrain
      */
      
      for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
	
	float Pt = trackIter->getMomentum().perp();
	float Eta = trackIter->getMomentum().eta();
	float Phi = trackIter->getMomentum().phi();
	float z  = trackIter->getPOCA().z();
	if (fabs(z) > ZMAX) continue;
	if (Pt < PTMINTRA) continue;
	float chi2 = trackIter->getChi2();
	if (chi2 > CHI2MAX) continue;
	
	if (PrimaryVtxConstrain) {
	  if ( zvtxL1tk > -999 && fabs( z - zvtxL1tk) >= DeltaZMax) continue;
	}
	
	/*
	  if (DeltaZConstrain) {
	  if ( fabs( z - z_leadingTrack) >= DeltaZMax) continue;
	  }
	*/
	
	float dr = deltaR(Eta, eta, Phi,phi);
	if (dr < DRmax && dr >= DRmin)  {
	  //std::cout << " a track in the cone, z Pt = " << z << " " << Pt << std::endl;
	  sumPt += Pt;
	}
	
      }  // end loop over tracks
      
      if (RelativeIsolation) {
	if (et > 0) trkisol = sumPt / et;	// relative isolation
      }
      else {	// absolute isolation
	trkisol = sumPt ;
      }
      
      const math::XYZTLorentzVector P4 = egIter -> p4() ;
      L1TkEmParticle trkEm(  P4,
			     EGammaRef,
			     trkisol );
      
      if (IsoCut <= 0) {
	// write the L1TkEm particle to the collection, 
	// irrespective of its relative isolation
	result -> push_back( trkEm );
      }
      else {
	// the object is written to the collection only
	// if it passes the isolation cut
	if (trkisol <= IsoCut) result -> push_back( trkEm );
      }
      
      
    }  // end loop over EGamma objects
  
  iEvent.put(std::move(result), label );
  
}

// --------------------------------------------------------------------------------------

float L1TkEmParticleProducer::CorrectedEta(float eta, float zv)  {

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

float L1TkEmParticleProducer::DeltaPhi(float phi1, float phi2) {
// dPhi between 0 and Pi
   float dphi = phi1 - phi2;
   if (dphi < 0) dphi = dphi + 2.*TMath::Pi();
   if (dphi > TMath::Pi()) dphi = 2.*TMath::Pi() - dphi;
  return dphi;
}

// --------------------------------------------------------------------------------------

float L1TkEmParticleProducer::deltaR(float eta1, float eta2, float phi1, float phi2) {
    float deta = eta1 - eta2;
    float dphi = DeltaPhi(phi1, phi2);
    float DR= sqrt( deta*deta + dphi*dphi );
    return DR;
}


// ------------ method called once each job just before starting event loop  ------------
void
L1TkEmParticleProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkEmParticleProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkEmParticleProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkEmParticleProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkEmParticleProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkEmParticleProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkEmParticleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkEmParticleProducer);



