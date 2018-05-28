#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
//
//
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include <algorithm>


PhotonMCTruthFinder::PhotonMCTruthFinder( ) {
  
}

std::vector<PhotonMCTruth> PhotonMCTruthFinder::find(const std::vector<SimTrack>& theSimTracks, const std::vector<SimVertex>& theSimVertices ) {
  std::vector<PhotonMCTruth> result;
  
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
  
  std::vector<SimTrack> trkFromConversion;
  std::vector<ElectronMCTruth> electronsFromConversions;
  SimVertex primVtx;   
  
  fill(theSimTracks,  theSimVertices);
 
  if (    !theSimTracks.empty() ) {
    
    int iPV=-1;   
    int partType1=0;
    int partType2=0;
    std::vector<SimTrack>::const_iterator iFirstSimTk = theSimTracks.begin();
    if (  !(*iFirstSimTk).noVertex() ) {
      iPV =  (*iFirstSimTk).vertIndex();
      
      int vtxId =   (*iFirstSimTk).vertIndex();
      primVtx = theSimVertices[vtxId];
      partType1 = (*iFirstSimTk).type();
    }
    
    math::XYZTLorentzVectorD primVtxPos(primVtx.position().x(),
					primVtx.position().y(),
					primVtx.position().z(),
					primVtx.position().e());           
    
    // Look at a second track
    iFirstSimTk++;
    if ( iFirstSimTk!=  theSimTracks.end() ) {
      if (  (*iFirstSimTk).vertIndex() == iPV) {
	partType2 = (*iFirstSimTk).type();  
      }
    }
    
    int npv=0;
    for (std::vector<SimTrack>::const_iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
      if (  (*iSimTk).noVertex() ) continue;
      if ( (*iSimTk).vertIndex() == iPV ) {
	npv++;
	if ( (*iSimTk).type() == 22) {
	  photonTracks.push_back(&(const_cast<SimTrack&>(*iSimTk)));
	} 
      }
    } 
    
    
    if(npv >= 3) {
      ievtype = PYTHIA;
    } else if(npv == 1) {
      if( std::abs(partType1) == 11 ) {
	ievtype = SINGLE; ievflav = ELECTRON_FLAV;
      } else if(partType1 == 111) {
	ievtype = SINGLE; ievflav = PIZERO_FLAV;
      } else if(partType1 == 22) {
	ievtype = SINGLE; ievflav = PHOTON_FLAV;
      }
    } else if(npv == 2) {
      if (  std::abs(partType1) == 11 && std::abs(partType2) == 11 ) {
	ievtype = DOUBLE; ievflav = ELECTRON_FLAV;
      } else if(partType1 == 111 && partType2 == 111)   {
	ievtype = DOUBLE; ievflav = PIZERO_FLAV;
      } else if(partType1 == 22 && partType2 == 22)   {
	ievtype = DOUBLE; ievflav = PHOTON_FLAV;
      }
    }      
    
    //////  Look into converted photons  
    int isAconversion=0;   
    int phoMotherType=-1;
    int phoMotherVtxIndex=-1;
    int phoMotherId=-1;
    if( ievflav == PHOTON_FLAV || ievflav== PIZERO_FLAV || ievtype == PYTHIA ) {
      for (std::vector<SimTrack>::const_iterator iPhoTk = theSimTracks.begin(); iPhoTk != theSimTracks.end(); ++iPhoTk){
	
	trkFromConversion.clear();           
	electronsFromConversions.clear();
	
	if ( (*iPhoTk).type() != 22 ) continue;
	int photonVertexIndex= (*iPhoTk).vertIndex();
	int phoTrkId= (*iPhoTk).trackId();
	
	// check who is his mother
	SimVertex vertex = theSimVertices[ photonVertexIndex];
	phoMotherId=-1;
	if ( vertex.parentIndex() != -1 ) {
	  
	  unsigned  motherGeantId = vertex.parentIndex(); 
	  std::map<unsigned, unsigned >::iterator association = geantToIndex_.find( motherGeantId );
	  if(association != geantToIndex_.end() )
	    phoMotherId = association->second;
	  phoMotherType = phoMotherId == -1 ? 0 : theSimTracks[phoMotherId].type();
	  
	}

	
	for (std::vector<SimTrack>::const_iterator iEleTk = theSimTracks.begin(); iEleTk != theSimTracks.end(); ++iEleTk){
	  if (  (*iEleTk).noVertex() )                    continue;
	  if ( (*iEleTk).vertIndex() == iPV )             continue; 
	  if ( std::abs((*iEleTk).type()) != 11  )             continue;
	  
	  int vertexId = (*iEleTk).vertIndex();
	  SimVertex vertex = theSimVertices[vertexId];
	  int motherId=-1;
	  
	  if ( vertex.parentIndex() != -1 ) {
	    
	    unsigned  motherGeantId = vertex.parentIndex(); 
	    std::map<unsigned, unsigned >::iterator association = geantToIndex_.find( motherGeantId );
	    if(association != geantToIndex_.end() )
	      motherId = association->second;
	    
	    
	    std::vector<CLHEP::Hep3Vector> bremPos;  
	    std::vector<CLHEP::HepLorentzVector> pBrem;
	    std::vector<float> xBrem;
	    
	    if ( theSimTracks[motherId].trackId() == (*iPhoTk).trackId() ) {
	      /// find truth about this electron and store it since it's from a converted photon
	      
	      trkFromConversion.push_back( (*iEleTk ) );
	      SimTrack trLast =(*iEleTk); 
	      float remainingEnergy =trLast.momentum().e();
	      math::XYZTLorentzVectorD primEleMom((*iEleTk).momentum().x(),
						  (*iEleTk).momentum().y(),
						  (*iEleTk).momentum().z(),
						  (*iEleTk).momentum().e());  
	      math::XYZTLorentzVectorD motherMomentum(primEleMom);  
	      unsigned int eleId = (*iEleTk).trackId();     
	      int eleVtxIndex= (*iEleTk).vertIndex();
	      int hasBrem=0;           
	      
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
		  if ( vertex1.parentIndex() != -1  ) {
		    
		    unsigned  motherGeantId = vertex1.parentIndex(); 
		    std::map<unsigned, unsigned >::iterator association = geantToIndex_.find( motherGeantId );
		    if(association != geantToIndex_.end() )
		      motherId = association->second;
		    
		    if (theSimTracks[motherId].trackId() == eleId ) {
		      
		      eleId= (*iSimTk).trackId();
		      remainingEnergy =   (*iSimTk).momentum().e();
		      motherMomentum = (*iSimTk).momentum();
		      pBrem.push_back( CLHEP::HepLorentzVector(trLast.momentum().px(),
							       trLast.momentum().py(),
							       trLast.momentum().pz(),
							       trLast.momentum().e()) );
		      bremPos.push_back( CLHEP::HepLorentzVector(vertex1.position().x(),
								 vertex1.position().y(),
								 vertex1.position().z(),
								 vertex1.position().t()) );
		      xBrem.push_back(eLoss);
		      
		    }
		  }
		  
		}
		trLast=(*iSimTk);
		
	      } // End loop over all SimTracks 

	      /// here fill the electron
	      CLHEP::HepLorentzVector tmpEleMom(primEleMom.px(),
						primEleMom.py(),
						primEleMom.pz(),
						primEleMom.e() );
	      CLHEP::HepLorentzVector tmpVtxPos(primVtxPos.x(),
						primVtxPos.y(),
						primVtxPos.z(),
						primVtxPos.t() );
	      electronsFromConversions.push_back ( ElectronMCTruth( tmpEleMom, eleVtxIndex, hasBrem, bremPos, pBrem, 
								    xBrem,  tmpVtxPos , const_cast<SimTrack&>(*iEleTk)  )  ) ;
	    }   //// Electron from conversion found
	  }
	} // End of loop over the SimTracks      
	
	math::XYZTLorentzVectorD motherVtxPosition(0.,0.,0.,0.);
	CLHEP::HepLorentzVector phoMotherMom(0.,0.,0.,0.);
	CLHEP::HepLorentzVector phoMotherVtx(0.,0.,0.,0.); 
	
	if ( phoMotherId >= 0) {
	  
	  phoMotherVtxIndex = theSimTracks[phoMotherId].vertIndex();
	  SimVertex motherVtx = theSimVertices[ phoMotherVtxIndex];
	  motherVtxPosition =math::XYZTLorentzVectorD (motherVtx.position().x(),
						       motherVtx.position().y(),
						       motherVtx.position().z(),
						       motherVtx.position().e());
	  
	  phoMotherMom.setPx( theSimTracks[phoMotherId].momentum().x());
	  phoMotherMom.setPy( theSimTracks[phoMotherId].momentum().y());
	  phoMotherMom.setPz( theSimTracks[phoMotherId].momentum().z() );
	  phoMotherMom.setE( theSimTracks[phoMotherId].momentum().t());
	  
	  phoMotherVtx.setX ( motherVtxPosition.x());
	  phoMotherVtx.setY ( motherVtxPosition.y());
	  phoMotherVtx.setZ ( motherVtxPosition.z());
	  phoMotherVtx.setT ( motherVtxPosition.t());
	  
	}
	
	
	if ( !electronsFromConversions.empty() ) {
	  isAconversion=1;
	  int convVtxId =electronsFromConversions[0].vertexInd();
	  SimVertex convVtx = theSimVertices[convVtxId];
	  math::XYZTLorentzVectorD vtxPosition(convVtx.position().x(),
					       convVtx.position().y(),
					       convVtx.position().z(),
					       convVtx.position().e());
	  
	  CLHEP::HepLorentzVector tmpPhoMom( (*iPhoTk).momentum().px(), 
					     (*iPhoTk).momentum().py(),
					     (*iPhoTk).momentum().pz(), 
					     (*iPhoTk).momentum().e() ) ;

	  CLHEP::HepLorentzVector tmpVertex( vtxPosition.x(), 
					     vtxPosition.y(), 
					     vtxPosition.z(), vtxPosition.t() );

	  CLHEP::HepLorentzVector tmpPrimVtx( primVtxPos.x(), 
					      primVtxPos.y(), 
					      primVtxPos.z(), 
					      primVtxPos.t() ) ;
	  
	  result.push_back( PhotonMCTruth(isAconversion, tmpPhoMom, photonVertexIndex, phoTrkId, phoMotherType,phoMotherMom, phoMotherVtx, tmpVertex,  
					  tmpPrimVtx, electronsFromConversions ));
	  
	} else {
	  isAconversion=0;
	  CLHEP::HepLorentzVector vtxPosition(0.,0.,0.,0.);
	  CLHEP::HepLorentzVector tmpPhoMom( (*iPhoTk).momentum().px(), 
					     (*iPhoTk).momentum().py(),
					     (*iPhoTk).momentum().pz(), 
					     (*iPhoTk).momentum().e() ) ;

	  CLHEP::HepLorentzVector tmpPrimVtx( primVtxPos.x(), 
					      primVtxPos.y(), 
					      primVtxPos.z(), 
					      primVtxPos.t() ) ;
	  result.push_back( PhotonMCTruth(isAconversion, tmpPhoMom,  photonVertexIndex, phoTrkId, phoMotherType, phoMotherMom, phoMotherVtx, vtxPosition,   
					  tmpPrimVtx, electronsFromConversions ));
	  
	}
      } // End loop over the primary photons
    }   // Event with one or two photons 
  }
  
  return result;
  
}

void PhotonMCTruthFinder::fill(const std::vector<SimTrack>& simTracks, const std::vector<SimVertex>& simVertices ) {
  
  // Watch out there ! A SimVertex is in mm 
  
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
