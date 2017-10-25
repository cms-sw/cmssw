/*
 *  FWSimTrackProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *
 */

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "FWCore/Common/interface/EventBase.h"

#include "TEveTrack.h"
#include "TParticle.h"
#include "TDatabasePDG.h"

class FWSimTrackProxyBuilder : public FWProxyBuilderBase
{
public:
   FWSimTrackProxyBuilder( void ) {} 
   ~FWSimTrackProxyBuilder( void ) override {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWSimTrackProxyBuilder( const FWSimTrackProxyBuilder& ) = delete;
   // Disable default assignment operator
   const FWSimTrackProxyBuilder& operator=( const FWSimTrackProxyBuilder& ) = delete;

   using FWProxyBuilderBase::build;
   void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* ) override;

   void addParticlesToPdgDataBase( void );
};

void
FWSimTrackProxyBuilder::addParticlesToPdgDataBase( void )
{
   static Bool_t bAdded = kFALSE;
   // Check if already called
   if(bAdded)return;
   bAdded = true;
   
   TDatabasePDG *pdgDB = TDatabasePDG::Instance();
   const Int_t kspe=50000000;
   
   // PDG nuclear states are 10-digit numbers
   // 10LZZZAAAI e.g. deuteron is 
   // 1000010020
   const Int_t kion=1000000000;
   
   /*
    const Double_t kAu2Gev=0.9314943228;
    */
   
   const Double_t khSlash = 1.0545726663e-27;
   const Double_t kErg2Gev = 1/1.6021773349e-3;
   const Double_t khShGev = khSlash*kErg2Gev;
   const Double_t kYear2Sec = 3600*24*365.25;
   
   //
   // Bottom mesons
   // mass and life-time from PDG
   //
   pdgDB->AddParticle("Upsilon(3S)","Upsilon(3S)",10.3552,kTRUE,
                      0,1,"Bottonium",200553);
   
   // QCD diffractive states
   pdgDB->AddParticle("rho_diff0","rho_diff0",0,kTRUE,
                      0,0,"QCD diffr. state",9900110);
   pdgDB->AddParticle("pi_diffr+","pi_diffr+",0,kTRUE,
                      0,1,"QCD diffr. state",9900210);
   pdgDB->AddParticle("omega_di","omega_di",0,kTRUE,
                      0,0,"QCD diffr. state",9900220);
   pdgDB->AddParticle("phi_diff","phi_diff",0,kTRUE,
                      0,0,"QCD diffr. state",9900330);
   pdgDB->AddParticle("J/psi_di","J/psi_di",0,kTRUE,
                      0,0,"QCD diffr. state",9900440);
   pdgDB->AddParticle("n_diffr0","n_diffr0",0,kTRUE,
                      0,0,"QCD diffr. state",9902110);
   pdgDB->AddParticle("p_diffr+","p_diffr+",0,kTRUE,
                      0,1,"QCD diffr. state",9902210);
   
   // From Herwig
   pdgDB->AddParticle("PSID    ", " ", 3.7699, kFALSE, 0.0, 0, "meson",   30443);
   
   pdgDB->AddParticle("A_00    ", " ", 0.9960, kFALSE, 0.0, 0, "meson",  9000111); 
   pdgDB->AddParticle("A_0+    ", " ", 0.9960, kFALSE, 0.0,+3, "meson",  9000211);  
   pdgDB->AddParticle("A_0-    ", " ", 0.9960, kFALSE, 0.0,-3, "meson", -9000211);  
   
   pdgDB->AddParticle("F0P0    ", " ", 0.9960, kFALSE, 0.0, 0, "meson",  9010221); 
   
   pdgDB->AddParticle("KDL_2+  ", " ", 1.773,  kFALSE, 0.0,+3, "meson",   10325); 
   pdgDB->AddParticle("KDL_2-  ", " ", 1.773,  kFALSE, 0.0,-3, "meson",  -10325); 
   
   pdgDB->AddParticle("KDL_20  ", " ", 1.773,  kFALSE, 0.0, 0, "meson",   10315); 
   pdgDB->AddParticle("KDL_2BR0", " ", 1.773,  kFALSE, 0.0, 0, "meson",  -10315); 
   
   pdgDB->AddParticle("PI_2+   ", " ", 1.670,  kFALSE, 0.0,+3, "meson",   10215);
   pdgDB->AddParticle("PI_2-   ", " ", 1.670,  kFALSE, 0.0,-3, "meson",  -10215);
   pdgDB->AddParticle("PI_20   ", " ", 1.670,  kFALSE, 0.0, 0, "meson",   10115);
   
   
   pdgDB->AddParticle("KD*+    ", " ", 1.717,  kFALSE, 0.0,+3, "meson",   30323); 
   pdgDB->AddParticle("KD*-    ", " ", 1.717,  kFALSE, 0.0,-3, "meson",  -30323); 
   
   pdgDB->AddParticle("KD*0    ", " ", 1.717,  kFALSE, 0.0, 0, "meson",   30313); 
   pdgDB->AddParticle("KDBR*0  ", " ", 1.717,  kFALSE, 0.0, 0, "meson",  -30313); 
   
   pdgDB->AddParticle("RHOD+   ", " ", 1.700,  kFALSE, 0.0,+3, "meson",   30213); 
   pdgDB->AddParticle("RHOD-   ", " ", 1.700,  kFALSE, 0.0,-3, "meson",  -30213); 
   pdgDB->AddParticle("RHOD0   ", " ", 1.700,  kFALSE, 0.0, 0, "meson",   30113); 
   
   pdgDB->AddParticle("ETA_2(L)", " ", 1.632,  kFALSE, 0.0, 0, "meson",   10225); 
   pdgDB->AddParticle("ETA_2(H)", " ", 1.854,  kFALSE, 0.0, 0, "meson",   10335); 
   pdgDB->AddParticle("OMEGA(H)", " ", 1.649,  kFALSE, 0.0, 0, "meson",   30223);
   
   
   pdgDB->AddParticle("KDH_2+  ", " ", 1.816,  kFALSE, 0.0,+3, "meson",   20325);
   pdgDB->AddParticle("KDH_2-  ", " ", 1.816,  kFALSE, 0.0,-3, "meson",  -20325);
   
   pdgDB->AddParticle("KDH_20  ", " ", 1.816,  kFALSE, 0.0, 0, "meson",   20315);
   pdgDB->AddParticle("KDH_2BR0", " ", 1.816,  kFALSE, 0.0, 0, "meson",  -20315);
   
   
   pdgDB->AddParticle("KD_3+   ", " ", 1.773,  kFALSE, 0.0,+3, "meson",     327);
   pdgDB->AddParticle("KD_3-   ", " ", 1.773,  kFALSE, 0.0,-3, "meson",    -327);
   
   pdgDB->AddParticle("KD_30   ", " ", 1.773,  kFALSE, 0.0, 0, "meson",     317);
   pdgDB->AddParticle("KD_3BR0 ", " ", 1.773,  kFALSE, 0.0, 0, "meson",    -317);
   
   pdgDB->AddParticle("RHO_3+  ", " ", 1.691,  kFALSE, 0.0,+3, "meson",     217);
   pdgDB->AddParticle("RHO_3-  ", " ", 1.691,  kFALSE, 0.0,-3, "meson",    -217);
   pdgDB->AddParticle("RHO_30  ", " ", 1.691,  kFALSE, 0.0, 0, "meson",     117);
   pdgDB->AddParticle("OMEGA_3 ", " ", 1.667,  kFALSE, 0.0, 0, "meson",     227);
   pdgDB->AddParticle("PHI_3   ", " ", 1.854,  kFALSE, 0.0, 0, "meson",     337);
   
   pdgDB->AddParticle("CHI2P_B0", " ", 10.232, kFALSE, 0.0, 0, "meson", 110551);
   pdgDB->AddParticle("CHI2P_B1", " ", 10.255, kFALSE, 0.0, 0, "meson", 120553);
   pdgDB->AddParticle("CHI2P_B2", " ", 10.269, kFALSE, 0.0, 0, "meson", 100555);
   pdgDB->AddParticle("UPSLON4S", " ", 10.580, kFALSE, 0.0, 0, "meson", 300553);
   
   
   // IONS
   //
   // Done by default now from Pythia6 table
   // Needed for other generators
   // So check if already defined
   
   
   Int_t ionCode = kion+10020;
   if(!pdgDB->GetParticle(ionCode)){
      pdgDB->AddParticle("Deuteron","Deuteron", 1.875613, kTRUE,
                         0,3,"Ion",ionCode);
   }
   pdgDB->AddAntiParticle("AntiDeuteron", - ionCode);
   
   ionCode = kion+10030;
   if(!pdgDB->GetParticle(ionCode)){
      pdgDB->AddParticle("Triton","Triton", 2.80925, kFALSE,
                         khShGev/(12.33*kYear2Sec),3,"Ion",ionCode);
   }
   pdgDB->AddAntiParticle("AntiTriton", - ionCode);
   
   ionCode = kion+20030;
   if(!pdgDB->GetParticle(ionCode)){
      pdgDB->AddParticle("HE3","HE3", 2.80923,kFALSE,
                         0,6,"Ion",ionCode);
   }
   pdgDB->AddAntiParticle("AntiHE3", - ionCode);
   
   ionCode = kion+20040;
   if(!pdgDB->GetParticle(ionCode)){
      pdgDB->AddParticle("Alpha","Alpha", 3.727417, kTRUE,
                         khShGev/(12.33*kYear2Sec), 6, "Ion", ionCode);
   }
   pdgDB->AddAntiParticle("AntiAlpha", - ionCode);
   
   // Special particles
   // 
   pdgDB->AddParticle("Cherenkov","Cherenkov",0,kFALSE,
                      0,0,"Special",kspe+50);
   pdgDB->AddParticle("FeedbackPhoton","FeedbackPhoton",0,kFALSE,
                      0,0,"Special",kspe+51);
   pdgDB->AddParticle("Lambda1520","Lambda1520",1.5195,kFALSE,
                      0.0156,0,"Resonance",3124);
   pdgDB->AddAntiParticle("Lambda1520bar",-3124);   
}

//______________________________________________________________________________

void
FWSimTrackProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const edm::SimTrackContainer* collection = nullptr;
   iItem->get( collection );

   if( nullptr == collection )
   {
      return;
   }
   addParticlesToPdgDataBase();

   TEveTrackPropagator* propagator = context().getTrackPropagator();
   
   
   edm::Handle<edm::SimVertexContainer> hitColl;
   const edm::EventBase *event = item()->getEvent();
   event->getByLabel( edm::InputTag( "g4SimHits" ), hitColl );
   
   int i = 0;
   for( std::vector<SimTrack>::const_iterator it = collection->begin(), end = collection->end(); it != end; ++it )
   {
      const SimTrack& iData = (*it);
      double vx = 0.0;
      double vy = 0.0;
      double vz = 0.0;
      double vt = 0.0;
      if(! iData.noVertex() && ( hitColl.isValid() && !hitColl->empty()))
      {
         int vInd = iData.vertIndex();
         vx = hitColl->at(vInd).position().x();
         vy = hitColl->at(vInd).position().y();
         vz = hitColl->at(vInd).position().z();
         vt = hitColl->at(vInd).position().t();
      }
      
     TParticle* particle = new TParticle;
     particle->SetPdgCode( iData.type());
     particle->SetMomentum( iData.momentum().px(), iData.momentum().py(), iData.momentum().pz(), iData.momentum().e());
     particle->SetProductionVertex( vx, vy, vz, vt );
  
     TEveTrack* track = new TEveTrack( particle, ++i, propagator );
     if( iData.charge() == 0 )
     {
       track->SetLineStyle( 7 );
     }
     track->AddPathMark( TEvePathMark( TEvePathMark::kReference, 
                                       TEveVector( iData.trackerSurfacePosition().x(), iData.trackerSurfacePosition().y(), iData.trackerSurfacePosition().z()),
                                       TEveVector( iData.trackerSurfaceMomentum().px(), iData.trackerSurfaceMomentum().py(), iData.trackerSurfaceMomentum().pz())));
     track->MakeTrack();
     setupAddElement( track, product );
   }
}

REGISTER_FWPROXYBUILDER( FWSimTrackProxyBuilder, edm::SimTrackContainer, "SimTracks", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
