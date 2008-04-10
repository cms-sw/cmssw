/*
 *  $Date: 2007/10/17 15:58:25 $
 *  $Revision: 1.1 $
 *  \author Jean-Roch Vlimant
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/MultiParticleInConeGunProducer.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// #include "FWCore/Utilities/interface/Exception.h"

// #include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

MultiParticleInConeGunProducer::MultiParticleInConeGunProducer(const ParameterSet& pset) : 
   BaseFlatGunProducer(pset)
{


   ParameterSet defpset ;
   ParameterSet pgun_params = 
      pset.getUntrackedParameter<ParameterSet>("PGunParameters",defpset) ;
  
   fMinPt = pgun_params.getUntrackedParameter<double>("MinPt",0.99);
   fMaxPt = pgun_params.getUntrackedParameter<double>("MaxPt",1.01);

   fInConeIds = pgun_params.getUntrackedParameter< vector<int> >("InConeID",vector<int>());
   fMinDeltaR = pgun_params.getUntrackedParameter<double>("MinDeltaR",0.0);
   fMaxDeltaR = pgun_params.getUntrackedParameter<double>("MaxDeltaR",0.1);
   fMinMomRatio = pgun_params.getUntrackedParameter<double>("MinMomRatio",0.99);
   fMaxMomRatio = pgun_params.getUntrackedParameter<double>("MaxMomRatio",1.01);

   fInConeMinEta = pgun_params.getUntrackedParameter<double>("InConeMinEta",-5.5);
   fInConeMaxEta = pgun_params.getUntrackedParameter<double>("InConeMaxEta",5.5);
   fInConeMinPhi = pgun_params.getUntrackedParameter<double>("InConeMinPhi",-3.14159265358979323846);
   fInConeMaxPhi = pgun_params.getUntrackedParameter<double>("InConeMaxPhi",3.14159265358979323846);
   fInConeMaxTry = pgun_params.getUntrackedParameter<uint>("InConeMaxTry",100);
   
   produces<HepMCProduct>();
}

MultiParticleInConeGunProducer::~MultiParticleInConeGunProducer()
{
   // no need to cleanup GenEvent memory - done in HepMCProduct
}

void MultiParticleInConeGunProducer::produce(Event &e, const EventSetup& es) 
{

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

       double pt     = fRandomGenerator->fire(fMinPt, fMaxPt) ;
       double eta    = fRandomGenerator->fire(fMinEta, fMaxEta) ;
       double phi    = fRandomGenerator->fire(fMinPhi, fMaxPhi) ;
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
	 uint nTry=0;
	 while(true){
	   //shoot flat Deltar
	   double dR = fRandomGenerator->fire(fMinDeltaR, fMaxDeltaR);
	   //shoot flat eta/phi mixing
	   double alpha = fRandomGenerator->fire(-3.14159265358979323846, 3.14159265358979323846);
	   double dEta = dR*cos(alpha);
	   double dPhi = dR*sin(alpha);
	   
	   /*
	   //shoot Energy of associated particle	 
	   double energyIc = fRandomGenerator->fire(fMinEInCone, fMaxEInCone);
	   if (mom2Ic>0){ momIC = sqrt(mom2Ic);}
	   */
	   //	 get kinematics
	   double etaIc = eta+dEta;
	   double phiIc = phi+dPhi;
	   //put it back in -Pi:Pi if necessary. multiple time might be necessary if dR > 3
	   const uint maxL=100;
	   uint iL=0;
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
	   double momR = fRandomGenerator->fire(fMinMomRatio, fMaxMomRatio);
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
   
   if ( fVerbosity > 0 )
     {
       cout << " MultiParticleInConeGunProducer : Event Generation Done " << endl;
     }
}

