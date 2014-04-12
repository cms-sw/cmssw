/*
 *  \author Jean-Roch Vlimant
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/MultiParticleInConeGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

MultiParticleInConeGunProducer::MultiParticleInConeGunProducer(const ParameterSet& pset) : 
   BaseFlatGunProducer(pset)
{


   ParameterSet defpset ;
   ParameterSet pgun_params = 
      pset.getParameter<ParameterSet>("PGunParameters") ;
  
   fMinPt = pgun_params.getParameter<double>("MinPt");
   fMaxPt = pgun_params.getParameter<double>("MaxPt");

   fInConeIds = pgun_params.getParameter< vector<int> >("InConeID");
   fMinDeltaR = pgun_params.getParameter<double>("MinDeltaR");
   fMaxDeltaR = pgun_params.getParameter<double>("MaxDeltaR");
   fMinMomRatio = pgun_params.getParameter<double>("MinMomRatio");
   fMaxMomRatio = pgun_params.getParameter<double>("MaxMomRatio");

   fInConeMinEta = pgun_params.getParameter<double>("InConeMinEta");
   fInConeMaxEta = pgun_params.getParameter<double>("InConeMaxEta");
   fInConeMinPhi = pgun_params.getParameter<double>("InConeMinPhi");
   fInConeMaxPhi = pgun_params.getParameter<double>("InConeMaxPhi");
   fInConeMaxTry = pgun_params.getParameter<unsigned int>("InConeMaxTry");
   
   produces<HepMCProduct>();
   produces<GenEventInfoProduct>();
}

MultiParticleInConeGunProducer::~MultiParticleInConeGunProducer()
{
   // no need to cleanup GenEvent memory - done in HepMCProduct
}

void MultiParticleInConeGunProducer::produce(Event &e, const EventSetup& es) 
{
   edm::Service<edm::RandomNumberGenerator> rng;
   CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

   if ( fVerbosity > 0 )
   {
      cout << " MultiParticleInConeGunProducer : Begin New Event Generation" << endl ; 
   }
   // event loop (well, another step in it...)
          
   // no need to clean up GenEvent memory - done in HepMCProduct
   // 
   
   // here re-create fEvt (memory)
   //
   fEvt = new HepMC::GenEvent() ;
   
   // now actualy, cook up the event from PDGTable and gun parameters
   //
   // 1st, primary vertex
   //
   //HepMC::GenVertex* Vtx = new HepMC::GenVertex(CLHEP::HepLorentzVector(0.,0.,0.));
   HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(0.,0.,0.));

   // loop over particles
   //
   int barcode = 1 ;
   for (unsigned int ip=0; ip<fPartIDs.size(); ++ip)
   {

       double pt     = CLHEP::RandFlat::shoot(engine, fMinPt, fMaxPt) ;
       double eta    = CLHEP::RandFlat::shoot(engine, fMinEta, fMaxEta) ;
       double phi    = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi) ;
       int PartID = fPartIDs[ip] ;
       const HepPDT::ParticleData* 
          PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
       double mass   = PData->mass().value() ;
       double theta  = 2.*atan(exp(-eta)) ;
       double mom    = pt/sin(theta) ;
       double px     = pt*cos(phi) ;
       double py     = pt*sin(phi) ;
       double pz     = mom*cos(theta) ;
       double energy2= mom*mom + mass*mass ;
       double energy = sqrt(energy2) ; 

       HepMC::FourVector p(px,py,pz,energy) ;
       HepMC::GenParticle* Part = 
           new HepMC::GenParticle(p,PartID,1);
       Part->suggest_barcode( barcode ) ;
       barcode++ ;
       Vtx->add_particle_out(Part);

       if ( fAddAntiParticle ){}

       // now add the particles in the cone
       for (unsigned iPic=0; iPic!=fInConeIds.size();iPic++){
	 unsigned int nTry=0;
	 while(true){
	   //shoot flat Deltar
	   double dR = CLHEP::RandFlat::shoot(engine, fMinDeltaR, fMaxDeltaR);
	   //shoot flat eta/phi mixing
	   double alpha = CLHEP::RandFlat::shoot(engine, -3.14159265358979323846, 3.14159265358979323846);
	   double dEta = dR*cos(alpha);
	   double dPhi = dR*sin(alpha);
	   
	   /*
	   //shoot Energy of associated particle	 
	   double energyIc = CLHEP::RandFlat::shoot(engine, fMinEInCone, fMaxEInCone);
	   if (mom2Ic>0){ momIC = sqrt(mom2Ic);}
	   */
	   //	 get kinematics
	   double etaIc = eta+dEta;
	   double phiIc = phi+dPhi;
	   //put it back in -Pi:Pi if necessary. multiple time might be necessary if dR > 3
	   const unsigned int maxL=100;
	   unsigned int iL=0;
	   while(iL++<maxL){
	     if (phiIc > 3.14159265358979323846) phiIc-=2*3.14159265358979323846;
	     else if(phiIc <-3.14159265358979323846) phiIc+=2*3.14159265358979323846;
	     
	     if (abs(phiIc)<3.14159265358979323846) break;
	   }
	     

	   //allow to skip it if you did not run out of possible drawing
	   if (nTry++<=fInConeMaxTry){
	     //draw another one if this one is not in acceptance
	     if (etaIc<fInConeMinEta || etaIc > fInConeMaxEta) continue;
	     if (phiIc<fInConeMinPhi || phiIc > fInConeMaxPhi) continue;
	   }
	   else{
	     if ( fVerbosity > 0 )
	       {
		 cout << " MultiParticleInConeGunProducer : could not produce a particle "
		      <<  fInConeIds[iPic]<<" in cone "
		      <<  fMinDeltaR<<" to "<<fMaxDeltaR<<" within the "<<fInConeMaxTry<<" allowed tries."<<endl;
	       }
	     /*	     edm::LogError("MultiParticleInConeGunProducer")<< " MultiParticleInConeGunProducer : could not produce a particle "<<
	       fInConeIds[iPic]<<" in cone "<<
	       fMinDeltaR<<" to "<<fMaxDeltaR<<" within the "<<fInConeMaxTry<<" allowed tries.";*/
	   }
	   int PartIDIc=fInConeIds[iPic];
	   const HepPDT::ParticleData* 
	     PDataIc = fPDGTable->particle(HepPDT::ParticleID(abs(PartIDIc)));
	   
	   //shoot momentum ratio
	   double momR = CLHEP::RandFlat::shoot(engine, fMinMomRatio, fMaxMomRatio);
	   double massIc= PDataIc->mass().value() ;
	   double momIc = momR * mom;
	   double energyIc = sqrt(momIc*momIc + massIc*massIc);

	   double thetaIc = 2.*atan(exp(-etaIc)) ;
	   double pxIc = momIc*sin(thetaIc)*cos(phiIc);
	   double pyIc = momIc*sin(thetaIc)*sin(phiIc);
	   double pzIc = momIc*cos(thetaIc);

	   HepMC::FourVector pIc(pxIc,pyIc,pzIc,energyIc) ;
	   HepMC::GenParticle* PartIc = new HepMC::GenParticle(pIc, PartIDIc, 1);
	   PartIc->suggest_barcode( barcode ) ;
	   barcode++ ;
	   Vtx->add_particle_out(PartIc);
	   break;
	 }//try many times while not in acceptance
       }//loop over the particle Ids in the cone
   }

   fEvt->add_vertex(Vtx) ;
   fEvt->set_event_number(e.id().event()) ;
   fEvt->set_signal_process_id(20) ; 
        
   if ( fVerbosity > 0 )
   {
      fEvt->print() ;  
   }

   auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
   BProduct->addHepMCData( fEvt );
   e.put(BProduct);

   auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
   e.put(genEventInfo);

   if ( fVerbosity > 0 )
     {
       cout << " MultiParticleInConeGunProducer : Event Generation Done " << endl;
     }
}

