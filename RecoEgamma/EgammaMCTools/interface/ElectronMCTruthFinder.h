#ifndef ElectronMCTruthFinder_h
#define ElectronMCTruthFinder_h

#include "SimDataFormats/Track/interface/SimTrack.h"                          
#include "SimDataFormats/Track/interface/SimTrackContainer.h"                 
#include "SimDataFormats/Vertex/interface/SimVertex.h"                        
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"               

#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"

class PhotonMCTruthFinder;
class ElectronMCTruthFinder {

  public:

    ElectronMCTruthFinder();

    virtual ~ElectronMCTruthFinder(); 


    std::vector<ElectronMCTruth> find(std::vector<SimTrack> simTracks, std::vector<SimVertex> simVertices);

  private:

    void fill(std::vector<SimTrack>& theSimTracks, std::vector<SimVertex>& theSimVertices);

    std::map<unsigned, unsigned> geantToIndex_;
    PhotonMCTruthFinder* thePhotonMCTruthFinder_;
};

#endif
