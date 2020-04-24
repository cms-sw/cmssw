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
 *  \author N. Marinelli  Notre Dame
 *
 */


class PhotonMCTruth;
class PhotonMCTruthFinder {
public:

 PhotonMCTruthFinder(); 
 virtual ~PhotonMCTruthFinder() { }

 
 std::vector<PhotonMCTruth> find( const std::vector<SimTrack>& simTracks, const std::vector<SimVertex>& simVertices);  

 void clear() {geantToIndex_.clear();}
     

 private:


 
 void fill( const std::vector<SimTrack>& theSimTracks, const std::vector<SimVertex>& theSimVertices);  
 std::map<unsigned, unsigned> geantToIndex_;


 

};


#endif

