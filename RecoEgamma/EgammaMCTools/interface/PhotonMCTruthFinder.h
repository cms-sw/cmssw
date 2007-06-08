#ifndef PhotonMCTruthFinder_h
#define PhotonMCTruthFinder_h
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"


#include <vector>
#include <map>
#include <iostream>

/** \class PhotonMCTruthFinder
 *   
 *        
 *  $Date: $
 *  $Revision: $
 *  \author N. Marinelli  Notre Dame
 *
 */


class PhotonMCTruth;
class PhotonMCTruthFinder {
public:

 PhotonMCTruthFinder(); 
 virtual ~PhotonMCTruthFinder() {  std::cout << " PhotonMCTruthFinder DTOR" << std::endl;}

 
 std::vector<PhotonMCTruth> find( std::vector<SimTrack> simTracks, std::vector<SimVertex> simVertices);  
     

 private:


 
 void fill( std::vector<SimTrack>& theSimTracks, std::vector<SimVertex>& theSimVertices);  
 std::map<unsigned, unsigned> geantToIndex_;

 float mcPhoEnergy_[10];
 float mcPhoEt_[10];
 float mcPhoPt_[10];
 float mcPhoEta_[10];
 float mcPhoPhi_[10];
 float mcConvR_[10];
 float mcConvZ_[10];
 int   idTrk1_[10];
 int   idTrk2_[10];
 
 float mcPizEnergy_[10];
 float mcPizEt_[10];
 float mcPizPt_[10];
 float mcPizEta_[10];
 float mcPizPhi_[10];




};


#endif

