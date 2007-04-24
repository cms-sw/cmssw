#ifndef PizeroMCTruthFinder_h
#define PizeroMCTruthFinder_h

#include "SimDataFormats/Track/interface/SimTrack.h"                          
#include "SimDataFormats/Track/interface/SimTrackContainer.h"                 
#include "SimDataFormats/Vertex/interface/SimVertex.h"                        
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"               

#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruth.h"

class PizeroMCTruthFinder {

  public:

    PizeroMCTruthFinder();

    virtual ~PizeroMCTruthFinder() {}

    std::vector<PizeroMCTruth> find(std::vector<SimTrack> simTracks, std::vector<SimVertex> simVertices);

  private:

    void fill(std::vector<SimTrack>& theSimTracks, std::vector<SimVertex>& theSimVertices);

};


#endif
