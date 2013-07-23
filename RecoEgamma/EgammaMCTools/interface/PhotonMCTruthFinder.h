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
 *  $Date: 2007/06/08 10:49:31 $
 *  $Revision: 1.3 $
 *  \author N. Marinelli  Notre Dame
 *
 */


class PhotonMCTruth;
class PhotonMCTruthFinder {
public:

 PhotonMCTruthFinder(); 
 virtual ~PhotonMCTruthFinder() {  std::cout << " PhotonMCTruthFinder DTOR" << std::endl;}

 
 std::vector<PhotonMCTruth> find( const std::vector<SimTrack>& simTracks, const std::vector<SimVertex>& simVertices);  

     

 private:


 
 void fill( const std::vector<SimTrack>& theSimTracks, const std::vector<SimVertex>& theSimVertices);  
 std::map<unsigned, unsigned> geantToIndex_;


 

};


#endif

