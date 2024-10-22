#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruthFinder.h"

#include <algorithm>

ElectronMCTruthFinder::ElectronMCTruthFinder() {}

ElectronMCTruthFinder::~ElectronMCTruthFinder() {}

std::vector<ElectronMCTruth> ElectronMCTruthFinder::find(const std::vector<SimTrack>& theSimTracks,
                                                         const std::vector<SimVertex>& theSimVertices) {
  //std::cout << "  ElectronMCTruthFinder::find " << std::endl;

  std::vector<ElectronMCTruth> result;

  // Local variables
  //const int SINGLE=1;
  //const int DOUBLE=2;
  //const int PYTHIA=3;
  //const int ELECTRON_FLAV=1;
  //const int PIZERO_FLAV=2;
  //const int PHOTON_FLAV=3;

  //int ievtype=0;
  //int ievflav=0;

  std::vector<SimTrack> electronTracks;
  SimVertex primVtx;

  fill(theSimTracks, theSimVertices);

  int iPV = -1;
  //int partType1=0;
  //int partType2=0;
  std::vector<SimTrack>::const_iterator iFirstSimTk = theSimTracks.begin();
  if (!(*iFirstSimTk).noVertex()) {
    iPV = (*iFirstSimTk).vertIndex();

    int vtxId = (*iFirstSimTk).vertIndex();
    primVtx = theSimVertices[vtxId];

    //partType1 = (*iFirstSimTk).type();

    //std::cout <<  " Primary vertex id " << iPV << " first track type " << (*iFirstSimTk).type() << std::endl;
  } else {
    //std::cout << " First track has no vertex " << std::endl;
  }

  // CLHEP::HepLorentzVector primVtxPos= primVtx.position();
  math::XYZTLorentzVectorD primVtxPos(
      primVtx.position().x(), primVtx.position().y(), primVtx.position().z(), primVtx.position().e());

  // Look at a second track
  iFirstSimTk++;
  //if ( iFirstSimTk!=  theSimTracks.end() ) {
  //
  //  if (  (*iFirstSimTk).vertIndex() == iPV) {
  //    partType2 = (*iFirstSimTk).type();
  //
  //    //std::cout <<  " second track type " << (*iFirstSimTk).type() << " vertex " <<  (*iFirstSimTk).vertIndex() << std::endl;
  //
  //  } else {
  //    //std::cout << " Only one kine track from Primary Vertex " << std::endl;
  //  }
  //}

  //std::cout << " Loop over all particles " << std::endl;

  for (std::vector<SimTrack>::const_iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk) {
    if ((*iSimTk).noVertex())
      continue;

    //int vertexId = (*iSimTk).vertIndex();
    //SimVertex vertex = theSimVertices[vertexId];

    //std::cout << " Particle type " <<  (*iSimTk).type() << " Sim Track ID " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  " vertex position " << vertex.position() << " vertex ID " << vertexId  << std::endl;
    if ((*iSimTk).vertIndex() == iPV) {
      if (std::abs((*iSimTk).type()) == 11) {
        //std::cout << " Found a primary electron with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  std::endl;
        electronTracks.push_back(*iSimTk);
      }
    }
  }

  /// Now store the electron truth
  std::vector<CLHEP::Hep3Vector> bremPos;
  std::vector<CLHEP::HepLorentzVector> pBrem;
  std::vector<float> xBrem;

  for (std::vector<SimTrack>::iterator iEleTk = electronTracks.begin(); iEleTk != electronTracks.end(); ++iEleTk) {
    //std::cout << " Looping on the primary electron pt  " << std::sqrt((*iEleTk).momentum().perp2()) << " electron track ID " << (*iEleTk).trackId() << std::endl;

    SimTrack trLast = (*iEleTk);
    unsigned int eleId = (*iEleTk).trackId();
    float remainingEnergy = trLast.momentum().e();
    //    CLHEP::HepLorentzVector motherMomentum = (*iEleTk).momentum();
    //    CLHEP::HepLorentzVector primEleMom = (*iEleTk).momentum();
    math::XYZTLorentzVectorD motherMomentum(
        (*iEleTk).momentum().x(), (*iEleTk).momentum().y(), (*iEleTk).momentum().z(), (*iEleTk).momentum().e());
    math::XYZTLorentzVectorD primEleMom(motherMomentum);
    int eleVtxIndex = (*iEleTk).vertIndex();

    bremPos.clear();
    pBrem.clear();
    xBrem.clear();

    for (std::vector<SimTrack>::const_iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk) {
      if ((*iSimTk).noVertex())
        continue;
      if ((*iSimTk).vertIndex() == iPV)
        continue;

      //std::cout << " (*iEleTk)->trackId() " << (*iEleTk).trackId() << " (*iEleTk)->vertIndex() "<< (*iEleTk).vertIndex()  << " (*iSimTk).vertIndex() "  <<  (*iSimTk).vertIndex() << " (*iSimTk).type() " <<   (*iSimTk).type() << " (*iSimTk).trackId() " << (*iSimTk).trackId() << std::endl;

      int vertexId1 = (*iSimTk).vertIndex();
      const SimVertex& vertex1 = theSimVertices[vertexId1];
      int vertexId2 = trLast.vertIndex();
      //SimVertex vertex2 = theSimVertices[vertexId2];

      int motherId = -1;

      if ((vertexId1 == vertexId2) && ((*iSimTk).type() == (*iEleTk).type()) && trLast.type() == 22) {
        //std::cout << " Here a e/gamma brem vertex " << std::endl;

        //std::cout << " Secondary from electron:  particle1  type " << (*iSimTk).type() << " trackId " << (*iSimTk).trackId() << " vertex ID " << vertexId1 << " vertex position " << std::sqrt(vertex1.position().perp2()) << " parent index "<< vertex1.parentIndex() << std::endl;

        //std::cout << " Secondary from electron:  particle2  type " << trLast.type() << " trackId " <<  trLast.trackId()<< " vertex ID " << vertexId2 << " vertex position " << std::sqrt(vertex2.position().perp2()) << " parent index " << vertex2.parentIndex() << std::endl;

        //std::cout << " Electron pt " << std::sqrt((*iSimTk).momentum().perp2()) << " photon pt " <<  std::sqrt(trLast.momentum().perp2()) << "Mother electron pt " <<  sqrt(motherMomentum.perp2()) << std::endl;
        //std::cout << " eleId " << eleId << std::endl;
        float eLoss = remainingEnergy - ((*iSimTk).momentum() + trLast.momentum()).e();
        //std::cout << " eLoss " << eLoss << std::endl;

        if (vertex1.parentIndex()) {
          unsigned motherGeantId = vertex1.parentIndex();
          std::map<unsigned, unsigned>::iterator association = geantToIndex_.find(motherGeantId);
          if (association != geantToIndex_.end())
            motherId = association->second;

          //int motherType = motherId == -1 ? 0 : theSimTracks[motherId].type();
          //std::cout << " Parent to this vertex   motherId " << motherId << " mother type " <<  motherType << " Sim track ID " <<  theSimTracks[motherId].trackId() << std::endl;
          if (theSimTracks[motherId].trackId() == eleId) {
            //std::cout << "  ***** Found the Initial Mother Electron ****   theSimTracks[motherId].trackId() " <<  theSimTracks[motherId].trackId() << " eleId " <<  eleId << std::endl;
            eleId = (*iSimTk).trackId();
            remainingEnergy = (*iSimTk).momentum().e();
            motherMomentum = (*iSimTk).momentum();

            pBrem.push_back(CLHEP::HepLorentzVector(
                trLast.momentum().px(), trLast.momentum().py(), trLast.momentum().pz(), trLast.momentum().e()));
            bremPos.push_back(CLHEP::HepLorentzVector(
                vertex1.position().x(), vertex1.position().y(), vertex1.position().z(), vertex1.position().t()));
            xBrem.push_back(eLoss);
          }

        } else {
          //std::cout << " This vertex has no parent tracks " <<  std::endl;
        }
      }
      trLast = (*iSimTk);

    }  // End loop over all SimTracks
    //std::cout << " Going to build the ElectronMCTruth: pBrem size " << pBrem.size() << std::endl;
    /// here fill the electron
    CLHEP::HepLorentzVector tmpEleMom(primEleMom.px(), primEleMom.py(), primEleMom.pz(), primEleMom.e());
    CLHEP::HepLorentzVector tmpVtxPos(primVtxPos.x(), primVtxPos.y(), primVtxPos.z(), primVtxPos.t());
    result.push_back(ElectronMCTruth(tmpEleMom, eleVtxIndex, bremPos, pBrem, xBrem, tmpVtxPos, (*iEleTk)));

  }  // End loop over primary electrons

  return result;
}

void ElectronMCTruthFinder::fill(const std::vector<SimTrack>& simTracks, const std::vector<SimVertex>& simVertices) {
  //std::cout << "  ElectronMCTruthFinder::fill " << std::endl;

  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  // Empty event, do nothin'
  if (nVtx == 0)
    return;

  // create a map associating geant particle id and position in the
  // event SimTrack vector
  for (unsigned it = 0; it < nTks; ++it) {
    geantToIndex_[simTracks[it].trackId()] = it;
    //std::cout << " ElectronMCTruthFinder::fill it " << it << " simTracks[it].trackId() " <<  simTracks[it].trackId() << std::endl;
  }
}
