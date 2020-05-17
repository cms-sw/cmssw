#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruthFinder.h"

#include <algorithm>

PizeroMCTruthFinder::PizeroMCTruthFinder() {
  thePhotonMCTruthFinder_ = new PhotonMCTruthFinder();
  theElectronMCTruthFinder_ = new ElectronMCTruthFinder();
}

PizeroMCTruthFinder::~PizeroMCTruthFinder() {
  delete thePhotonMCTruthFinder_;
  delete theElectronMCTruthFinder_;
  std::cout << "~PizeroMCTruthFinder" << std::endl;
}

std::vector<PizeroMCTruth> PizeroMCTruthFinder::find(const std::vector<SimTrack>& theSimTracks,
                                                     const std::vector<SimVertex>& theSimVertices) {
  std::cout << "  PizeroMCTruthFinder::find " << std::endl;

  std::vector<PhotonMCTruth> mcPhotons = thePhotonMCTruthFinder_->find(theSimTracks, theSimVertices);

  std::vector<PizeroMCTruth> result;
  std::vector<PhotonMCTruth> photonsFromPizero;
  std::vector<ElectronMCTruth> electronsFromPizero;

  // Local variables
  //const int SINGLE=1;
  //const int DOUBLE=2;
  //const int PYTHIA=3;
  //const int ELECTRON_FLAV=1;
  //const int PIZERO_FLAV=2;
  //const int PHOTON_FLAV=3;

  //int ievtype=0;
  //int ievflav=0;

  std::vector<SimTrack> pizeroTracks;
  SimVertex primVtx;

  fill(theSimTracks, theSimVertices);

  int iPV = -1;
  //int partType1=0;
  //int partType2=0;
  auto iFirstSimTk = theSimTracks.begin();
  if (!(*iFirstSimTk).noVertex()) {
    iPV = (*iFirstSimTk).vertIndex();

    int vtxId = (*iFirstSimTk).vertIndex();
    primVtx = theSimVertices[vtxId];

    //partType1 = (*iFirstSimTk).type();
    std::cout << " Primary vertex id " << iPV << " first track type " << (*iFirstSimTk).type() << std::endl;
  } else {
    std::cout << " First track has no vertex " << std::endl;
  }

  // CLHEP::HepLorentzVector primVtxPos= primVtx.position();
  math::XYZTLorentzVectorD primVtxPos(
      primVtx.position().x(), primVtx.position().y(), primVtx.position().z(), primVtx.position().e());

  // Look at a second track
  iFirstSimTk++;
  if (iFirstSimTk != theSimTracks.end()) {
    if ((*iFirstSimTk).vertIndex() == iPV) {
      //partType2 = (*iFirstSimTk).type();
      std::cout << " second track type " << (*iFirstSimTk).type() << " vertex " << (*iFirstSimTk).vertIndex()
                << std::endl;

    } else {
      std::cout << " Only one kine track from Primary Vertex " << std::endl;
    }
  }

  int npv = 0;

  for (const auto& theSimTrack : theSimTracks) {
    if (theSimTrack.noVertex())
      continue;

    int vertexId = theSimTrack.vertIndex();
    SimVertex vertex = theSimVertices[vertexId];

    std::cout << " Particle type " << theSimTrack.type() << " Sim Track ID " << theSimTrack.trackId() << " momentum "
              << theSimTrack.momentum() << " vertex position " << vertex.position() << std::endl;

    if (theSimTrack.vertIndex() == iPV) {
      npv++;
      if (std::abs(theSimTrack.type()) == 111) {
        std::cout << " Found a primary pizero with ID  " << theSimTrack.trackId() << " momentum "
                  << theSimTrack.momentum() << std::endl;

        pizeroTracks.push_back(theSimTrack);

        // CLHEP::HepLorentzVector momentum = (*iSimTk).momentum();
        math::XYZTLorentzVectorD momentum(theSimTrack.momentum().x(),
                                          theSimTrack.momentum().y(),
                                          theSimTrack.momentum().z(),
                                          theSimTrack.momentum().e());
      }
    }
  }

  std::cout << " There are " << npv << " particles originating in the PV " << std::endl;

  //if(npv > 4) {
  //  ievtype = PYTHIA;
  //} else if(npv == 1) {
  //  if( std::abs(partType1) == 11 ) {
  //    ievtype = SINGLE; ievflav = ELECTRON_FLAV;
  //  } else if(partType1 == 111) {
  //    ievtype = SINGLE; ievflav = PIZERO_FLAV;
  //  } else if(partType1 == 22) {
  //    ievtype = SINGLE; ievflav = PHOTON_FLAV;
  //  }
  //} else if(npv == 2) {
  //  if (  std::abs(partType1) == 11 && std::abs(partType2) == 11 ) {
  //    ievtype = DOUBLE; ievflav = ELECTRON_FLAV;
  //  } else if(partType1 == 111 && partType2 == 111)   {
  //    ievtype = DOUBLE; ievflav = PIZERO_FLAV;
  //  } else if(partType1 == 22 && partType2 == 22)   {
  //    ievtype = DOUBLE; ievflav = PHOTON_FLAV;
  //  }
  //}

  for (auto& pizeroTrack : pizeroTracks) {
    std::cout << " Looping on the primary pizero pt  " << sqrt(pizeroTrack.momentum().perp2()) << " pizero track ID "
              << pizeroTrack.trackId() << std::endl;

    photonsFromPizero.clear();
    std::cout << " mcPhotons.size " << mcPhotons.size() << std::endl;
    for (auto& mcPhoton : mcPhotons) {
      int phoVtxIndex = mcPhoton.vertexInd();
      SimVertex phoVtx = theSimVertices[phoVtxIndex];
      unsigned int phoParentInd = phoVtx.parentIndex();
      std::cout << " photon parent vertex index " << phoParentInd << std::endl;

      if (phoParentInd == pizeroTrack.trackId()) {
        std::cout << "Matched Photon ID " << mcPhoton.trackId() << "  vtx " << phoParentInd << " with pizero "
                  << pizeroTrack.trackId() << std::endl;
        photonsFromPizero.push_back(mcPhoton);
      }
    }
    std::cout << " Photon matching the pizero vertex " << photonsFromPizero.size() << std::endl;

    // build pizero MC thruth
    CLHEP::HepLorentzVector tmpMom(pizeroTrack.momentum().px(),
                                   pizeroTrack.momentum().py(),
                                   pizeroTrack.momentum().pz(),
                                   pizeroTrack.momentum().e());
    CLHEP::HepLorentzVector tmpPos(
        primVtx.position().x(), primVtx.position().y(), primVtx.position().z(), primVtx.position().t());
    result.emplace_back(tmpMom, photonsFromPizero, tmpPos);

  }  // end loop over primary pizeros

  std::cout << " Pizero size " << result.size() << std::endl;

  return result;
}

void PizeroMCTruthFinder::fill(const std::vector<SimTrack>& simTracks, const std::vector<SimVertex>& simVertices) {
  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  // Empty event, do nothin'
  if (nVtx == 0)
    return;

  // create a map associating geant particle id and position in the
  // event SimTrack vector
  for (unsigned it = 0; it < nTks; ++it) {
    geantToIndex_[simTracks[it].trackId()] = it;
    std::cout << " PizeroMCTruthFinder::fill it " << it << " simTracks[it].trackId() " << simTracks[it].trackId()
              << std::endl;
  }
}
