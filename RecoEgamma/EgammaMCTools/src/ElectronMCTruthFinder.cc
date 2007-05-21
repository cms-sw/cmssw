#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruthFinder.h"


#include <algorithm>                                                          

ElectronMCTruthFinder::ElectronMCTruthFinder() {

}

std::vector<ElectronMCTruth> ElectronMCTruthFinder::find(std::vector<SimTrack> theSimTracks, std::vector<SimVertex> theSimVertices ) {
  std::cout << "  ElectronMCTruthFinder::find " << std::endl;

  std::vector<ElectronMCTruth> result;

  // Local variables  
  const int SINGLE=1;
  const int DOUBLE=2;
  const int PYTHIA=3;
  const int ELECTRON_FLAV=1;
  const int PIZERO_FLAV=2;
  const int PHOTON_FLAV=3;
  
  int ievtype=0;
  int ievflav=0;
  
  std::vector<SimTrack> electronTracks;
  SimVertex primVtx;   
  
    
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
  
  HepLorentzVector primVtxPos= primVtx.position();           

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
  int iElec=0;
   for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
    if (  (*iSimTk).noVertex() ) continue;

    int vertexId = (*iSimTk).vertIndex();
    SimVertex vertex = theSimVertices[vertexId];
 
    std::cout << " Particle type " <<  (*iSimTk).type() << " Sim Track ID " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  " vertex position " << vertex.position() << std::endl;  
    if ( (*iSimTk).vertIndex() == iPV ) {
      npv++;
      if ( fabs((*iSimTk).type() ) == 11) {

	std::cout << " Found a primary electron with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  std::endl;
	
	electronTracks.push_back( *iSimTk );

	CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();

	iElec++;
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


   if (ievflav == ELECTRON_FLAV) {
     std::cout << " It's a primary ELECTRON event with " << electronTracks.size() << " electrons " << std::endl;
   }


   std::vector<Hep3Vector> bremPos;  
   std::vector<HepLorentzVector> pBrem;
   std::vector<float> xBrem;


   for (std::vector<SimTrack>::iterator iEleTk = electronTracks.begin(); iEleTk != electronTracks.end(); ++iEleTk){
     std::cout << " Looping on the primary electron pt  " << (*iEleTk).momentum().perp() << " electron track ID " << (*iEleTk).trackId() << std::endl;
    
     SimTrack trLast =(*iEleTk); 
     int eleId = (*iEleTk).trackId();
     float remainingEnergy =trLast.momentum().e();
     HepLorentzVector motherMomentum=(*iEleTk).momentum();
     HepLorentzVector primEleMom=(*iEleTk).momentum();

     bremPos.clear();
     pBrem.clear();
     xBrem.clear();     

     for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
     if (  (*iSimTk).noVertex() )                    continue;
     if ( (*iSimTk).vertIndex() == iPV )             continue;
     std::cout << " (*iEleTk)->trackId() " << (*iEleTk).trackId() << " (*iEleTk)->vertIndex() "<< (*iEleTk).vertIndex()  << " (*iSimTk).vertIndex() "  <<  (*iSimTk).vertIndex() << " (*iSimTk).type() " <<   (*iSimTk).type() << std::endl;

     int vertexId1 = (*iSimTk).vertIndex();
     SimVertex vertex1 = theSimVertices[vertexId1];
     int vertexId2 = trLast.vertIndex();
     SimVertex vertex2 = theSimVertices[vertexId2];


     int motherId=-1;
   
     if(  (  vertexId1 ==  vertexId2 ) && ( (*iSimTk).type() == (*iEleTk).type() ) && trLast.type() == 22   ) {
       std::cout << " Here a e/gamma brem vertex " << std::endl;
       std::cout << " Secondary from electron:  particle1  type " << (*iSimTk).type() << " trackId " <<  (*iSimTk).trackId() << " vertex ID " << vertexId1 << " vertex position " << vertex1.position().perp() << std::endl;
       std::cout << " Secondary from electron:  particle2  type " << trLast.type() << " trackId " <<  trLast.trackId() << " vertex ID " << vertexId2 << " vertex position " << vertex2.position().perp() << std::endl;
       
       std::cout << " Electron pt " << (*iSimTk).momentum().perp() << " photon pt " <<  trLast.momentum().perp() << " Mother electron pt " <<  motherMomentum.perp() << std::endl;
       std::cout << " eleId " << eleId << std::endl;
       float eLoss = remainingEnergy - ( (*iSimTk).momentum() + trLast.momentum()).e();
       std::cout << " eLoss " << eLoss << std::endl;              

       if ( vertex1.parentIndex()  ) {
       
	 unsigned  motherGeantId = vertex1.parentIndex(); 
	 std::map<unsigned, unsigned >::iterator association = geantToIndex_.find( motherGeantId );
	 if(association != geantToIndex_.end() )
	   motherId = association->second;
	 
	 int motherType = motherId == -1 ? 0 : theSimTracks[motherId].type();
	 std::cout << " Parent to this vertex   motherId " << motherId << " mother type " <<  motherType << " Sim track ID " <<  theSimTracks[motherId].trackId() << std::endl; 
	 if ( theSimTracks[motherId].trackId() == eleId ) {

	   std::cout << " Found the Mother Electron " << std::endl;
	   eleId= (*iSimTk).trackId();
	   remainingEnergy =   (*iSimTk).momentum().e();
           motherMomentum = (*iSimTk).momentum();

           
           pBrem.push_back(trLast.momentum());
	   bremPos.push_back(vertex1.position());
	   xBrem.push_back(eLoss);

	 }
       
	 
	 
	 
       } else {
	 std::cout << " This vertex has no parent tracks " <<  std::endl;
       }
       
     }
     trLast=(*iSimTk);

     } // End loop over all SimTracks 
     std::cout << " Going to build the ElectronMCTruth: pBrem size " << pBrem.size() << std::endl;
     /// here fill the electron

   

     result.push_back ( ElectronMCTruth( primEleMom, bremPos, pBrem, xBrem,  primVtxPos,  (*iEleTk)  )  ) ;
    
   } // End loop over primary electrons 
   
   
   return result;
}



void ElectronMCTruthFinder::fill(std::vector<SimTrack>& simTracks, 
                                 std::vector<SimVertex>& simVertices ) {
  std::cout << "  ElectronMCTruthFinder::fill " << std::endl;

  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  // Empty event, do nothin'
  if ( nVtx == 0 ) return;

  // create a map associating geant particle id and position in the 
  // event SimTrack vector
  for( unsigned it=0; it<nTks; ++it ) {
    geantToIndex_[ simTracks[it].trackId() ] = it;
    std::cout << " ElectronMCTruthFinder::fill it " << it << " simTracks[it].trackId() " <<  simTracks[it].trackId() << std::endl;
 
  }  


}
