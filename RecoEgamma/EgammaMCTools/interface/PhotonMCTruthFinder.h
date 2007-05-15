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
 *  $Date: 2007/04/13 12:27:55 $
 *  $Revision: 1.1 $
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


 int   idTrk1_[10];
 int   idTrk2_[10];
 


};


#endif

