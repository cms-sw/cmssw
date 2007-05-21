#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
//
//
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include <algorithm>


PhotonMCTruthFinder::PhotonMCTruthFinder( ) {
 std::cout << " PhotonMCTruthFinder CTOR " << std::endl;

 
}

std::vector<PhotonMCTruth> PhotonMCTruthFinder::find(std::vector<SimTrack> theSimTracks, std::vector<SimVertex> theSimVertices ) {
  std::cout << "  PhotonMCTruthFinder::find " << std::endl;

  std::vector<PhotonMCTruth> result;

  const float pi = 3.141592653592;
  const float twopi=2*pi;

// Local variables  
  const int SINGLE=1;
  const int DOUBLE=2;
  const int PYTHIA=3;
  const int ELECTRON_FLAV=1;
  const int PIZERO_FLAV=2;
  const int PHOTON_FLAV=3;
  
  int ievtype=0;
  int ievflav=0;
  
  
  std::vector<SimTrack*> photonTracks;
  std::vector<SimTrack*> pizeroTracks;
  std::vector<const SimTrack *> trkFromConversion;
  SimVertex primVtx;   
  std::vector<int> convInd;


  
  fill(theSimTracks,  theSimVertices);

  
  int iPV=-1;   
  int partType1=0;
  int partType2=0;
  std::vector<SimTrack>::iterator iFirstSimTk = theSimTracks.begin();
  if (  !(*iFirstSimTk).noVertex() ) {
    iPV =  (*iFirstSimTk).vertIndex();
    
    int vtxId =   (*iFirstSimTk).vertIndex();
    primVtx = theSimVertices[vtxId];
    
    
    
    partType1 = (*iFirstSimTk).type();
    std::cout <<  " Primary vertex id " << iPV << " first track type " << (*iFirstSimTk).type() << std::endl;  
  } else {
    std::cout << " First track has no vertex " << std::endl;
    
  }
  
  // Look at a second track
  iFirstSimTk++;
  if ( iFirstSimTk!=  theSimTracks.end() ) {
    
    if (  (*iFirstSimTk).vertIndex() == iPV) {
      partType2 = (*iFirstSimTk).type();  
      std::cout <<  " second track type " << (*iFirstSimTk).type() << " vertex " <<  (*iFirstSimTk).vertIndex() << std::endl;  
      
    } else {
      std::cout << " Only one kine track from Primary Vertex " << std::endl;
    }
  }
  
  std::cout << " Loop over all particles " << std::endl;
  
  int npv=0;
  int iPho=0;
  int iPizero=0;
  //   theSimTracks.reset();
  for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
    if (  (*iSimTk).noVertex() ) continue;


    int vertexId = (*iSimTk).vertIndex();
    SimVertex vertex = theSimVertices[vertexId];
 
    std::cout << " Particle type " <<  (*iSimTk).type() << " Sim Track ID " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  " vertex position " << vertex.position() << std::endl;  
    if ( (*iSimTk).vertIndex() == iPV ) {
      npv++;
      if ( (*iSimTk).type() == 22) {
	std::cout << " Found a primary photon with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  std::endl; 
	convInd.push_back(0);
        
	photonTracks.push_back( &(*iSimTk) );
	
	
	CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();

      } 
      
    }
    
  } 
  
  std::cout << " There are " << npv << " particles originating in the PV " << std::endl;
  
   if(npv > 4) {
     ievtype = PYTHIA;
   } else if(npv == 1) {
     if( abs(partType1) == 11 ) {
       ievtype = SINGLE; ievflav = ELECTRON_FLAV;
     } else if(partType1 == 111) {
       ievtype = SINGLE; ievflav = PIZERO_FLAV;
     } else if(partType1 == 22) {
       ievtype = SINGLE; ievflav = PHOTON_FLAV;
     }
   } else if(npv == 2) {
     if (  abs(partType1) == 11 && abs(partType2) == 11 ) {
       ievtype = DOUBLE; ievflav = ELECTRON_FLAV;
     } else if(partType1 == 111 && partType2 == 111)   {
       ievtype = DOUBLE; ievflav = PIZERO_FLAV;
     } else if(partType1 == 22 && partType2 == 22)   {
       ievtype = DOUBLE; ievflav = PHOTON_FLAV;
     }
   }      
   
   //////  Look into converted photons  

   int isAconversion=0;   
   if(ievflav == PHOTON_FLAV) {

     std::cout << " It's a primary PHOTON event with " << photonTracks.size() << " photons " << std::endl;

     int nConv=0;
     int iConv=0;
     iPho=0;
     for (std::vector<SimTrack*>::iterator iPhoTk = photonTracks.begin(); iPhoTk != photonTracks.end(); ++iPhoTk){
       std::cout << " Looping on the primary gamma looking for conversions " << (*iPhoTk)->momentum() << " photon track ID " << (*iPhoTk)->trackId() << std::endl;
       trkFromConversion.clear();           


       for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
	 if (  (*iSimTk).noVertex() )                    continue;
	 if ( (*iSimTk).vertIndex() == iPV )             continue; 
         if ( abs((*iSimTk).type()) != 11  )             continue;

	 int vertexId = (*iSimTk).vertIndex();
	 SimVertex vertex = theSimVertices[vertexId];
         int motherId=-1;
	

         std::cout << " Secondary from photons particle type " << (*iSimTk).type() << " trackId " <<  (*iSimTk).trackId() << " vertex ID " << vertexId << std::endl;
         if ( vertex.parentIndex()  ) {

	   unsigned  motherGeantId = vertex.parentIndex(); 
	   std::map<unsigned, unsigned >::iterator association = geantToIndex_.find( motherGeantId );
	   if(association != geantToIndex_.end() )
	     motherId = association->second;
	   
	   
	   int motherType = motherId == -1 ? 0 : theSimTracks[motherId].type();
	   
	   std::cout << " Parent to this vertex   motherId " << motherId << " mother type " <<  motherType << " Sim track ID " <<  theSimTracks[motherId].trackId() << std::endl;


	   if ( theSimTracks[motherId].trackId() == (*iPhoTk)->trackId() ) {
	     std::cout << " Found the Mother Photon " << std::endl;
             /// store this electron since it's from a converted photon
	     trkFromConversion.push_back(&(*iSimTk ) );
	   }
 
           


	 } else {
	   std::cout << " This vertex has no parent tracks " <<  std::endl;
	 }
	 
	 
       } // End of loop over the SimTracks      
       
       if ( trkFromConversion.size() > 0 ) {
         isAconversion=1;
	 std::cout  << " CONVERTED photon " <<   "\n";    
	 nConv++;
	 convInd[iPho]=nConv;         

	 int convVtxId =  trkFromConversion[0]->vertIndex();
	 SimVertex convVtx = theSimVertices[convVtxId];
	 CLHEP::HepLorentzVector vtxPosition = convVtx.position();

	 CLHEP::HepLorentzVector momentum = (*iPhoTk)->momentum();
	 float sign= vtxPosition.y()/fabs(vtxPosition.y() );

         if ( nConv <= 10) {         
	   std::cout  << " MC conversion vertex R " << vtxPosition.perp() << " R " << vtxPosition.z() << "\n";
	   
	   if ( trkFromConversion.size() > 1) {
	     idTrk1_[iConv]= trkFromConversion[0]->trackId();
	     idTrk2_[iConv]= trkFromConversion[1]->trackId();
	    
	   } else {
	     idTrk1_[iConv]=trkFromConversion[0]->trackId();
	     idTrk2_[iConv]=-1;
	   }
	   
	 }
	 
         iConv++;
	   
         result.push_back( PhotonMCTruth(isAconversion, (*iPhoTk)->momentum(), vtxPosition,   primVtx.position(), trkFromConversion ));

       } else {
         isAconversion=0;
	 std::cout  << " UNCONVERTED photon " <<   "\n";    
	 CLHEP::HepLorentzVector vtxPosition(0.,0.,0.,0.);
	 result.push_back( PhotonMCTruth(isAconversion, (*iPhoTk)->momentum(),  vtxPosition,   primVtx.position(), trkFromConversion ));
	
       }
       

       iPho++;   

     } // End loop over the primary photons
     
     
   }   // Event with one or two photons 





   return result;

}

void PhotonMCTruthFinder::fill(std::vector<SimTrack>& simTracks, std::vector<SimVertex>& simVertices ) {
  std::cout << "  PhotonMCTruthFinder::fill " << std::endl;



 // Watch out there ! A SimVertex is in mm (stupid), 
 
  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  // Empty event, do nothin'
  if ( nVtx == 0 ) return;

  // create a map associating geant particle id and position in the 
  // event SimTrack vector
  for( unsigned it=0; it<nTks; ++it ) {
    geantToIndex_[ simTracks[it].trackId() ] = it;
    std::cout << " PhotonMCTruthFinder::fill it " << it << " simTracks[it].trackId() " <<  simTracks[it].trackId() << std::endl;
 
  }  




}
