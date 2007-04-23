#ifndef ElectronMCTruthFinder_h
#define ElectronMCTruthFinder_h

#include "SimDataFormats/Track/interface/SimTrack.h"                          
#include "SimDataFormats/Track/interface/SimTrackContainer.h"                 
#include "SimDataFormats/Vertex/interface/SimVertex.h"                        
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"               

#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"

// class ElectronMCTruth;
class ElectronMCTruthFinder {

  public:

    ElectronMCTruthFinder();

    virtual ~ElectronMCTruthFinder() {std::cout << "~ElectronMCTruthFinder" << std::endl;}

    std::vector<ElectronMCTruth> find(std::vector<SimTrack> simTracks, std::vector<SimVertex> simVertices);

  private:

    void fill(std::vector<SimTrack>& theSimTracks, std::vector<SimVertex>& theSimVertices);

    std::map<unsigned, unsigned> geantToIndex_;

    float mcElecEnergy_[10];
    float mcElecEt_[10];
    float mcElecPt_[10];
    float mcElecEta_[10];
    float mcElecPhi_[10];
};

#endif
