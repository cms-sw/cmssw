#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruthFinder.h"


#include <algorithm>                                                          

ElectronMCTruthFinder::ElectronMCTruthFinder() {
  
  
}

ElectronMCTruthFinder::~ElectronMCTruthFinder() 
{
  
}


std::vector<ElectronMCTruth> ElectronMCTruthFinder::find(const std::vector<SimTrack>& theSimTracks, const std::vector<SimVertex>& theSimVertices ) {
  
  std::vector<ElectronMCTruth> result;
  std::vector<SimTrack> electronTracks;
  SimVertex primVtx;   
  
  fill(theSimTracks,  theSimVertices);
  
  int iPV=-1;   
  std::vector<SimTrack>::const_iterator iFirstSimTk = theSimTracks.begin();
  if (  !(*iFirstSimTk).noVertex() ) {
    iPV =  (*iFirstSimTk).vertIndex();
    
    int vtxId =   (*iFirstSimTk).vertIndex();
    primVtx = theSimVertices[vtxId];
  }
  

  math::XYZTLorentzVectorD primVtxPos(primVtx.position().x(),
                                      primVtx.position().y(),
                                      primVtx.position().z(),
                                      primVtx.position().e());
  
  // Look at a second track
  iFirstSimTk++;
  int npv=0;
  for (std::vector<SimTrack>::const_iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
    if (  (*iSimTk).noVertex() ) continue;
    if ( (*iSimTk).vertIndex() == iPV ) {
      npv++;
      if ( std::abs((*iSimTk).type() ) == 11) {
	electronTracks.push_back( *iSimTk );
      }
    }
  }
    

  /// Now store the electron truth 
  std::vector<CLHEP::Hep3Vector> bremPos;  
  std::vector<CLHEP::HepLorentzVector> pBrem;
  std::vector<float> xBrem;
  
  for (std::vector<SimTrack>::iterator iEleTk = electronTracks.begin(); iEleTk != electronTracks.end(); ++iEleTk){
    
    int hasBrem=0; 
    float totalBrem=0.;    
    
    SimTrack trLast =(*iEleTk); 
    unsigned int eleId = (*iEleTk).trackId();
    float remainingEnergy =trLast.momentum().e();
    math::XYZTLorentzVectorD motherMomentum((*iEleTk).momentum().x(),
                                            (*iEleTk).momentum().y(),
                                            (*iEleTk).momentum().z(),
                                            (*iEleTk).momentum().e());
    math::XYZTLorentzVectorD primEleMom(motherMomentum);
    int eleVtxIndex= (*iEleTk).vertIndex();
    
    bremPos.clear();
    pBrem.clear();
    xBrem.clear();     

   
    for (std::vector<SimTrack>::const_iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
	
      if (  (*iSimTk).noVertex() )                    continue;
      if ( (*iSimTk).vertIndex() == iPV )             continue;

      int vertexId1 = (*iSimTk).vertIndex();
      SimVertex vertex1 = theSimVertices[vertexId1];
      int vertexId2 = trLast.vertIndex();
      
      int motherId=-1;
      
      if(  (  vertexId1 ==  vertexId2 ) && ( (*iSimTk).type() == (*iEleTk).type() ) && trLast.type() == 22   ) {

	float eLoss = remainingEnergy - ( (*iSimTk).momentum() + trLast.momentum()).e();
        hasBrem=1;

	if ( vertex1.parentIndex()  ) {
	  
	  unsigned  motherGeantId = vertex1.parentIndex(); 
	  std::map<unsigned, unsigned >::iterator association = geantToIndex_.find( motherGeantId );
	  if(association != geantToIndex_.end() )
	    motherId = association->second;
	  
	  if ( theSimTracks[motherId].trackId() == eleId ) {

	    eleId= (*iSimTk).trackId();
	    remainingEnergy =   (*iSimTk).momentum().e();
	    motherMomentum = (*iSimTk).momentum();
	    pBrem.push_back( CLHEP::HepLorentzVector(trLast.momentum().px(),trLast.momentum().py(),
	                                      trLast.momentum().pz(),trLast.momentum().e()) );
	    bremPos.push_back( CLHEP::HepLorentzVector(vertex1.position().x(),vertex1.position().y(),
	                                        vertex1.position().z(),vertex1.position().t()) );
            totalBrem+=eLoss;
	    
	  }
	}
	
      }
      trLast=(*iSimTk);
    } // End loop over all SimTracks 

    /// here fill the electron
    xBrem.push_back(totalBrem);
    CLHEP::HepLorentzVector tmpEleMom(primEleMom.px(),primEleMom.py(),
                               primEleMom.pz(),primEleMom.e() ) ;
    CLHEP::HepLorentzVector tmpVtxPos(primVtxPos.x(),primVtxPos.y(),primVtxPos.z(),primVtxPos.t());
    result.push_back ( ElectronMCTruth( tmpEleMom, eleVtxIndex, hasBrem,  bremPos, pBrem, xBrem,  tmpVtxPos,(*iEleTk)  )  ) ;

    
  } // End loop over primary electrons 
  
    
   
   return result;
}



void ElectronMCTruthFinder::fill(const std::vector<SimTrack>& simTracks, 
                                 const std::vector<SimVertex>& simVertices ) {

  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  // Empty event, do nothin'
  if ( nVtx == 0 ) return;

  // create a map associating geant particle id and position in the 
  // event SimTrack vector
  for( unsigned it=0; it<nTks; ++it ) {
    geantToIndex_[ simTracks[it].trackId() ] = it;
  }  


}
