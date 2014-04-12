#ifndef PizeroMCTruthFinder_h
#define PizeroMCTruthFinder_h

#include "SimDataFormats/Track/interface/SimTrack.h"                          
#include "SimDataFormats/Track/interface/SimTrackContainer.h"                 
#include "SimDataFormats/Vertex/interface/SimVertex.h"                        
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"               

#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"


class PhotonMCTruthFinder;
class ElectronMCTruthFinder;
class PizeroMCTruthFinder {

  public:

    PizeroMCTruthFinder();

    virtual ~PizeroMCTruthFinder(); 

    std::vector<PizeroMCTruth> find(const std::vector<SimTrack>& simTracks, const std::vector<SimVertex>& simVertices);

  private:

    void fill(const std::vector<SimTrack>& theSimTracks, const std::vector<SimVertex>& theSimVertices);

    std::map<unsigned, unsigned> geantToIndex_;
    PhotonMCTruthFinder* thePhotonMCTruthFinder_;
    ElectronMCTruthFinder* theElectronMCTruthFinder_;

};

#endif
