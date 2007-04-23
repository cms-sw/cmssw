#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruthFinder.h"

// #include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
// #include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
// #include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
// #include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include <algorithm>                                                          

ElectronMCTruthFinder::ElectronMCTruthFinder() {

  for (int i = 0; i < 10; ++i) {
    mcElecEnergy_[i]=-99.;
    mcElecEt_[i]=-99.;
    mcElecPt_[i]=-99.;
    mcElecEta_[i]=-99.;
    mcElecPhi_[i]=-99.;
  }
}

std::vector<ElectronMCTruth> ElectronMCTruthFinder::find(std::vector<SimTrack> theSimTracks, std::vector<SimVertex> theSimVertices ) {
  std::cout << "  ElectronMCTruthFinder::find " << std::endl;

  std::vector<ElectronMCTruth> result;

  // const float pi = 3.141592653592;
  // const float twopi=2*pi;

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
  std::vector<SimTrack*> electronTracks;
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
  int iElec=0;
  int iPizero=0;
  //   theSimTracks.reset();
  for (std::vector<SimTrack>::iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk){
    if (  (*iSimTk).noVertex() ) continue;

    int vertexId = (*iSimTk).vertIndex();
    SimVertex vertex = theSimVertices[vertexId];
 
    // std::cout << " Particle type " <<  (*iSimTk).type() << " Sim Track ID " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  " vertex position " << vertex.position() << std::endl;  
    if ( (*iSimTk).vertIndex() == iPV ) {
      npv++;
      if ( (*iSimTk).type() == 22) {
	std::cout << " Found a primary photon with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  std::endl; 
	convInd.push_back(0);
        
	photonTracks.push_back( &(*iSimTk) );
	
	CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();
	//h_MCphoPt_->Fill(  momentum.perp());
	// h_MCphoEta_->Fill( momentum.pseudoRapidity());
	// h_MCphoE_->Fill( momentum.rho());
	
         if ( iPho < 10) {
           
	   // mcPhoEnergy_[iPho]= momentum.e(); 
	   // mcPhoPt_[iPho]= momentum.perp(); 
	   // mcPhoEt_[iPho]= momentum.et(); 
	   // mcPhoEta_[iPho]= momentum.pseudoRapidity();
	   // mcPhoPhi_[iPho]= momentum.phi();
	 }

	 iPho++;

      } else if ( (*iSimTk).type() == 11 || (*iSimTk).type()==-11 ) {
	std::cout << " Found a primary electron with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  std::endl;
	
	electronTracks.push_back( &(*iSimTk) );

	CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();

	/* if (iElec < 10) {
	  mcElecEnergy_[iElec] = momentum.e();
	  mcElecPt_[iElec] = momentum.perp();
	  mcElecEt_[iElec] = momentum.et();
	  mcElecEta_[iElec] = momentum.pseudoRapidity();
	  mcElecPhi_[iElec] = momentum.phi();
	} */

	iElec++;
        
        // std::cout << "        iElec++;" << std::endl;
	
      } else if ( (*iSimTk).type() == 111 ) {
	std::cout << " Found a primary pi0 with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  std::endl;	 
	
	pizeroTracks.push_back( &(*iSimTk) );
	
	CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();
	
	if ( iPizero < 10) {
	  
	  // mcPizEnergy_[iPizero]= momentum.e(); 
	  // mcPizPt_[iPizero]= momentum.perp(); 
	  // mcPizEt_[iPizero]= momentum.et(); 
	  // mcPizEta_[iPizero]= momentum.pseudoRapidity();
	  // mcPizPhi_[iPizero]= momentum.phi();
	   
	}
	
	iPizero++;
	
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

   if (ievflav == PHOTON_FLAV) {
     std::cout << " It's a primary PHOTON event with " << photonTracks.size() << " photons " << std::endl;
   }

   if (ievflav == ELECTRON_FLAV) {
     std::cout << " It's a primary ELECTRON event with " << electronTracks.size() << " electrons " << std::endl;
   }

   if (ievflav == PIZERO_FLAV) {
     std::cout << " It's a primary PIZERO event with " << pizeroTracks.size() << " pizeros " << std::endl;
   }

   std::cout << "Loop over vertices and find brem events:" << std::endl;

   int jPartType, kPartType;
   int jVtxId, kVtxId;
   int nBrems = 0;

   CLHEP::HepLorentzVector jMomentum, kMomentum, vtxPosition;

   SimVertex bremVtx;

   float jE, kE, phoFrac, r, z, eGamma, eElectron;

   ElectronMCTruth brem;
   
   for (std::vector<SimTrack>::iterator jSimTk = theSimTracks.begin(); jSimTk != theSimTracks.end(); ++jSimTk) {
     
     jPartType = (*jSimTk).type();
     jVtxId = (*jSimTk).vertIndex(); 
     
     if (abs(jPartType) == 11 || jPartType == 22) {
       for (std::vector<SimTrack>::iterator kSimTk = jSimTk; kSimTk != theSimTracks.end(); ++kSimTk) {
     
	 kPartType = (*kSimTk).type();
	 kVtxId = (*kSimTk).vertIndex();

         // std::cout << jVtxId << " " << kVtxId << " " << jPartType << " " << kPartType << std::endl;
	 
	 if (jVtxId == kVtxId) {
           if ((abs(jPartType) == 11 && kPartType == 22) || (jPartType == 22 && abs(kPartType) == 11)) {

	     jMomentum = (*jSimTk).momentum();
	     kMomentum = (*kSimTk).momentum();

	     bremVtx = theSimVertices[jVtxId];
	     vtxPosition = bremVtx.position();

	     r = vtxPosition.perp();
	     z = vtxPosition.z();

	     jE = jMomentum.e();
	     kE = kMomentum.e();

	     if (jPartType == 22) {
	       phoFrac = jE / (jE + kE);
	       eGamma = jE;
	       eElectron = kE;
	     } else {
	       phoFrac = kE / (jE + kE);
	       eGamma = kE;
	       eElectron = jE;
	     }
	     
	     brem.SetBrem(r, z, phoFrac, eGamma, eElectron);
	     result.push_back(brem);
	     
	     nBrems++;
	     std::cout << nBrems << " brem " << jPartType << " " << (*jSimTk).trackId() << " " << kPartType << " " << (*kSimTk).trackId() << " " << phoFrac << std::endl;
	   }
	 } 
       }       
     }	     
   }
          
   return result;
}



void ElectronMCTruthFinder::fill(std::vector<SimTrack>& simTracks, 
                                 std::vector<SimVertex>& simVertices ) {
  std::cout << "  ElectronMCTruthFinder::fill " << std::endl;

 // Watch out there ! A SimVertex is in mm (stupid), 
 
  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  // Empty event, do nothin'
  if ( nVtx == 0 ) return;

  // create a map associating geant particle id and position in the 
  // event SimTrack vector
  for( unsigned it=0; it<nTks; ++it ) {
    // geantToIndex_[ simTracks[it].trackId() ] = it;

    // std::cout << "geantToIndex_[ simTracks[it].trackId() ] = it;" << std::endl;
    
    // std::cout << " ElectronMCTruthFinder::fill it " << it << " simTracks[it].trackId() " <<  simTracks[it].trackId() << std::endl;
 
  }  

  // std::cout << "  ::fill done." << std::endl;

}
