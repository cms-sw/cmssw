#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
//
//
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include <algorithm>

PhotonMCTruthFinder::PhotonMCTruthFinder() {
  //std::cout << " PhotonMCTruthFinder CTOR " << std::endl;
}

std::vector<PhotonMCTruth> PhotonMCTruthFinder::find(const std::vector<SimTrack>& theSimTracks,
                                                     const std::vector<SimVertex>& theSimVertices) {
  //  std::cout << "  PhotonMCTruthFinder::find " << std::endl;

  std::vector<PhotonMCTruth> result;

  //const float pi = 3.141592653592;
  //const float twopi=2*pi;

  // Local variables
  const int SINGLE = 1;
  const int DOUBLE = 2;
  const int PYTHIA = 3;
  const int ELECTRON_FLAV = 1;
  const int PIZERO_FLAV = 2;
  const int PHOTON_FLAV = 3;

  int ievtype = 0;
  int ievflav = 0;

  std::vector<SimTrack*> photonTracks;

  std::vector<SimTrack> trkFromConversion;
  std::vector<ElectronMCTruth> electronsFromConversions;
  SimVertex primVtx;

  fill(theSimTracks, theSimVertices);

  //  std::cout << " After fill " << theSimTracks.size() << " " << theSimVertices.size() << std::endl;
  if (!theSimTracks.empty()) {
    int iPV = -1;
    int partType1 = 0;
    int partType2 = 0;
    std::vector<SimTrack>::const_iterator iFirstSimTk = theSimTracks.begin();
    if (!(*iFirstSimTk).noVertex()) {
      iPV = (*iFirstSimTk).vertIndex();

      int vtxId = (*iFirstSimTk).vertIndex();
      primVtx = theSimVertices[vtxId];

      partType1 = (*iFirstSimTk).type();
      //    std::cout <<  " Primary vertex id " << iPV << " first track type " << (*iFirstSimTk).type() << std::endl;
    } else {
      //std::cout << " First track has no vertex " << std::endl;
    }

    math::XYZTLorentzVectorD primVtxPos(
        primVtx.position().x(), primVtx.position().y(), primVtx.position().z(), primVtx.position().e());

    // Look at a second track
    iFirstSimTk++;
    if (iFirstSimTk != theSimTracks.end()) {
      if ((*iFirstSimTk).vertIndex() == iPV) {
        partType2 = (*iFirstSimTk).type();
        //      std::cout <<  " second track type " << (*iFirstSimTk).type() << " vertex " <<  (*iFirstSimTk).vertIndex() << std::endl;

      } else {
        // std::cout << " Only one kine track from Primary Vertex " << std::endl;
      }
    }

    //std::cout << " Loop over all particles " << std::endl;

    int npv = 0;
    //int iPho=0;
    //int iPizero=0;
    //   theSimTracks.reset();
    for (std::vector<SimTrack>::const_iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end(); ++iSimTk) {
      if ((*iSimTk).noVertex())
        continue;

      //int vertexId = (*iSimTk).vertIndex();
      //SimVertex vertex = theSimVertices[vertexId];

      //    std::cout << " Particle type " <<  (*iSimTk).type() << " Sim Track ID " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  " vertex position " << vertex.position() << " vertex index " << (*iSimTk).vertIndex() << std::endl;
      if ((*iSimTk).vertIndex() == iPV) {
        npv++;
        if ((*iSimTk).type() == 22) {
          //	std::cout << " Found a primary photon with ID  " << (*iSimTk).trackId() << " momentum " << (*iSimTk).momentum() <<  std::endl;

          photonTracks.push_back(&(const_cast<SimTrack&>(*iSimTk)));
        }
      }
    }

    //  std::cout << " There are " << npv << " particles originating in the PV " << std::endl;

    if (npv >= 3) {
      ievtype = PYTHIA;
    } else if (npv == 1) {
      if (std::abs(partType1) == 11) {
        ievtype = SINGLE;
        ievflav = ELECTRON_FLAV;
      } else if (partType1 == 111) {
        ievtype = SINGLE;
        ievflav = PIZERO_FLAV;
      } else if (partType1 == 22) {
        ievtype = SINGLE;
        ievflav = PHOTON_FLAV;
      }
    } else if (npv == 2) {
      if (std::abs(partType1) == 11 && std::abs(partType2) == 11) {
        ievtype = DOUBLE;
        ievflav = ELECTRON_FLAV;
      } else if (partType1 == 111 && partType2 == 111) {
        ievtype = DOUBLE;
        ievflav = PIZERO_FLAV;
      } else if (partType1 == 22 && partType2 == 22) {
        ievtype = DOUBLE;
        ievflav = PHOTON_FLAV;
      }
    }

    //////  Look into converted photons

    int isAconversion = 0;
    int phoMotherType = -1;
    int phoMotherVtxIndex = -1;
    int phoMotherId = -1;
    if (ievflav == PHOTON_FLAV || ievflav == PIZERO_FLAV || ievtype == PYTHIA) {
      //     std::cout << " It's a primary PHOTON or PIZERO or PYTHIA event with " << photonTracks.size() << " photons ievtype " << ievtype << " ievflav " << ievflav<<  std::endl;

      //    for (std::vector<SimTrack*>::iterator iPhoTk = photonTracks.begin(); iPhoTk != photonTracks.end(); ++iPhoTk){
      //      std::cout << " All gamma found from PV " << (*iPhoTk)->momentum() << " photon track ID " << (*iPhoTk)->trackId() << " vertex index " << (*iPhoTk)->vertIndex() << std::endl;
      //  }

      for (std::vector<SimTrack>::const_iterator iPhoTk = theSimTracks.begin(); iPhoTk != theSimTracks.end();
           ++iPhoTk) {
        trkFromConversion.clear();
        electronsFromConversions.clear();

        if ((*iPhoTk).type() != 22)
          continue;
        int photonVertexIndex = (*iPhoTk).vertIndex();
        int phoTrkId = (*iPhoTk).trackId();
        //std::cout << " Looping on gamma looking for conversions " << (*iPhoTk).momentum() << " photon track ID " << (*iPhoTk).trackId() << std::endl;

        // check who is his mother
        const SimVertex& vertex = theSimVertices[photonVertexIndex];
        phoMotherId = -1;
        if (vertex.parentIndex() != -1) {
          unsigned motherGeantId = vertex.parentIndex();
          std::map<unsigned, unsigned>::iterator association = geantToIndex_.find(motherGeantId);
          if (association != geantToIndex_.end())
            phoMotherId = association->second;
          phoMotherType = phoMotherId == -1 ? 0 : theSimTracks[phoMotherId].type();

          if (phoMotherType == 111 || phoMotherType == 221 || phoMotherType == 331) {
            //std::cout << " Parent to this vertex   motherId " << phoMotherId << " mother type " <<  phoMotherType << " Sim track ID " <<  theSimTracks[phoMotherId].trackId() << std::endl;
            //std::cout << " Son of a pizero or eta " << phoMotherType << std::endl;
          }
        }

        for (std::vector<SimTrack>::const_iterator iEleTk = theSimTracks.begin(); iEleTk != theSimTracks.end();
             ++iEleTk) {
          if ((*iEleTk).noVertex())
            continue;
          if ((*iEleTk).vertIndex() == iPV)
            continue;
          if (std::abs((*iEleTk).type()) != 11)
            continue;

          int vertexId = (*iEleTk).vertIndex();
          const SimVertex& vertex = theSimVertices[vertexId];
          int motherId = -1;

          //std::cout << " Secondary from photons particle type " << (*iEleTk).type() << " trackId " <<  (*iEleTk).trackId() << " vertex ID " << vertexId << std::endl;
          if (vertex.parentIndex() != -1) {
            unsigned motherGeantId = vertex.parentIndex();
            std::map<unsigned, unsigned>::iterator association = geantToIndex_.find(motherGeantId);
            if (association != geantToIndex_.end())
              motherId = association->second;

            //int motherType = motherId == -1 ? 0 : theSimTracks[motherId].type();

            //std::cout << " Parent to this vertex   motherId " << motherId << " mother type " <<  motherType << " Sim track ID " <<  theSimTracks[motherId].trackId() << std::endl;

            std::vector<CLHEP::Hep3Vector> bremPos;
            std::vector<CLHEP::HepLorentzVector> pBrem;
            std::vector<float> xBrem;

            if (theSimTracks[motherId].trackId() == (*iPhoTk).trackId()) {
              //std::cout << " Found the Mother Photon " << std::endl;
              /// find truth about this electron and store it since it's from a converted photon

              trkFromConversion.push_back((*iEleTk));
              SimTrack trLast = (*iEleTk);
              float remainingEnergy = trLast.momentum().e();
              //HepLorentzVector primEleMom=(*iEleTk).momentum();
              //HepLorentzVector motherMomentum=(*iEleTk).momentum();
              math::XYZTLorentzVectorD primEleMom((*iEleTk).momentum().x(),
                                                  (*iEleTk).momentum().y(),
                                                  (*iEleTk).momentum().z(),
                                                  (*iEleTk).momentum().e());
              math::XYZTLorentzVectorD motherMomentum(primEleMom);
              unsigned int eleId = (*iEleTk).trackId();
              int eleVtxIndex = (*iEleTk).vertIndex();

              bremPos.clear();
              pBrem.clear();
              xBrem.clear();

              for (std::vector<SimTrack>::const_iterator iSimTk = theSimTracks.begin(); iSimTk != theSimTracks.end();
                   ++iSimTk) {
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

                  //std::cout << " Secondary from electron:  particle1  type " << (*iSimTk).type() << " trackId " <<  (*iSimTk).trackId() << " vertex ID " << vertexId1 << " vertex position " << sqrt(vertex1.position().perp2()) << " parent index "<< vertex1.parentIndex() << std::endl;

                  //std::cout << " Secondary from electron:  particle2  type " << trLast.type() << " trackId " <<  trLast.trackId() << " vertex ID " << vertexId2 << " vertex position " << sqrt(vertex2.position().perp2()) << " parent index " << vertex2.parentIndex() << std::endl;

                  //std::cout << " Electron pt " << sqrt((*iSimTk).momentum().perp2()) << " photon pt " << sqrt(trLast.momentum().perp2()) << " Mother electron pt " <<  sqrt(motherMomentum.perp2()) << std::endl;
                  //std::cout << " eleId " << eleId << std::endl;
                  float eLoss = remainingEnergy - ((*iSimTk).momentum() + trLast.momentum()).e();
                  //std::cout << " eLoss " << eLoss << std::endl;

                  if (vertex1.parentIndex() != -1) {
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

                      pBrem.push_back(CLHEP::HepLorentzVector(trLast.momentum().px(),
                                                              trLast.momentum().py(),
                                                              trLast.momentum().pz(),
                                                              trLast.momentum().e()));
                      bremPos.push_back(CLHEP::HepLorentzVector(vertex1.position().x(),
                                                                vertex1.position().y(),
                                                                vertex1.position().z(),
                                                                vertex1.position().t()));
                      xBrem.push_back(eLoss);
                    }

                  } else {
                    //std::cout << " This vertex has no parent tracks " <<  std::endl;
                  }
                }
                trLast = (*iSimTk);

              }  // End loop over all SimTracks
              //std::cout << " Going to build the ElectronMCTruth for this electron from converted photon: pBrem size " << pBrem.size() << std::endl;
              /// here fill the electron

              CLHEP::HepLorentzVector tmpEleMom(primEleMom.px(), primEleMom.py(), primEleMom.pz(), primEleMom.e());
              CLHEP::HepLorentzVector tmpVtxPos(primVtxPos.x(), primVtxPos.y(), primVtxPos.z(), primVtxPos.t());
              electronsFromConversions.push_back(ElectronMCTruth(
                  tmpEleMom, eleVtxIndex, bremPos, pBrem, xBrem, tmpVtxPos, const_cast<SimTrack&>(*iEleTk)));
            }  //// Electron from conversion found

          } else {
            //std::cout << " This vertex has no parent tracks " <<  std::endl;
          }

        }  // End of loop over the SimTracks

        //std::cout << " DEBUG trkFromConversion.size() " << trkFromConversion.size() << " electronsFromConversions.size() " << electronsFromConversions.size() << std::endl;

        math::XYZTLorentzVectorD motherVtxPosition(0., 0., 0., 0.);
        CLHEP::HepLorentzVector phoMotherMom(0., 0., 0., 0.);
        CLHEP::HepLorentzVector phoMotherVtx(0., 0., 0., 0.);

        if (phoMotherId >= 0) {
          phoMotherVtxIndex = theSimTracks[phoMotherId].vertIndex();
          const SimVertex& motherVtx = theSimVertices[phoMotherVtxIndex];
          motherVtxPosition = math::XYZTLorentzVectorD(
              motherVtx.position().x(), motherVtx.position().y(), motherVtx.position().z(), motherVtx.position().e());

          phoMotherMom.setPx(theSimTracks[phoMotherId].momentum().x());
          phoMotherMom.setPy(theSimTracks[phoMotherId].momentum().y());
          phoMotherMom.setPz(theSimTracks[phoMotherId].momentum().z());
          phoMotherMom.setE(theSimTracks[phoMotherId].momentum().t());
          // std::cout << " PhotonMCTruthFinder mother " << phoMotherId << " type " << phoMotherType << " Momentum" <<  phoMotherMom.et() << std::endl;

          phoMotherVtx.setX(motherVtxPosition.x());
          phoMotherVtx.setY(motherVtxPosition.y());
          phoMotherVtx.setZ(motherVtxPosition.z());
          phoMotherVtx.setT(motherVtxPosition.t());
        }

        if (!electronsFromConversions.empty()) {
          // if ( trkFromConversion.size() > 0 ) {
          isAconversion = 1;
          //std::cout  << " CONVERTED photon " <<   "\n";

          //int convVtxId =  trkFromConversion[0].vertIndex();
          int convVtxId = electronsFromConversions[0].vertexInd();
          const SimVertex& convVtx = theSimVertices[convVtxId];
          // CLHEP::HepLorentzVector vtxPosition = convVtx.position();
          math::XYZTLorentzVectorD vtxPosition(
              convVtx.position().x(), convVtx.position().y(), convVtx.position().z(), convVtx.position().e());

          //result.push_back( PhotonMCTruth(isAconversion, (*iPhoTk).momentum(), photonVertexIndex, phoTrkId, vtxPosition,   primVtx.position(), trkFromConversion ));
          CLHEP::HepLorentzVector tmpPhoMom((*iPhoTk).momentum().px(),
                                            (*iPhoTk).momentum().py(),
                                            (*iPhoTk).momentum().pz(),
                                            (*iPhoTk).momentum().e());
          CLHEP::HepLorentzVector tmpVertex(vtxPosition.x(), vtxPosition.y(), vtxPosition.z(), vtxPosition.t());
          CLHEP::HepLorentzVector tmpPrimVtx(primVtxPos.x(), primVtxPos.y(), primVtxPos.z(), primVtxPos.t());

          result.push_back(PhotonMCTruth(isAconversion,
                                         tmpPhoMom,
                                         photonVertexIndex,
                                         phoTrkId,
                                         phoMotherType,
                                         phoMotherMom,
                                         phoMotherVtx,
                                         tmpVertex,
                                         tmpPrimVtx,
                                         electronsFromConversions));

        } else {
          isAconversion = 0;
          //std::cout  << " UNCONVERTED photon " <<   "\n";
          CLHEP::HepLorentzVector vtxPosition(0., 0., 0., 0.);
          CLHEP::HepLorentzVector tmpPhoMom((*iPhoTk).momentum().px(),
                                            (*iPhoTk).momentum().py(),
                                            (*iPhoTk).momentum().pz(),
                                            (*iPhoTk).momentum().e());
          CLHEP::HepLorentzVector tmpPrimVtx(primVtxPos.x(), primVtxPos.y(), primVtxPos.z(), primVtxPos.t());
          //	 result.push_back( PhotonMCTruth(isAconversion, (*iPhoTk).momentum(),  photonVertexIndex, phoTrkId, vtxPosition,   primVtx.position(), trkFromConversion ));
          result.push_back(PhotonMCTruth(isAconversion,
                                         tmpPhoMom,
                                         photonVertexIndex,
                                         phoTrkId,
                                         phoMotherType,
                                         phoMotherMom,
                                         phoMotherVtx,
                                         vtxPosition,
                                         tmpPrimVtx,
                                         electronsFromConversions));
        }

      }  // End loop over the primary photons

    }  // Event with one or two photons

    //std::cout << "  PhotonMCTruthFinder photon size " << result.size() << std::endl;
  }

  return result;
}

void PhotonMCTruthFinder::fill(const std::vector<SimTrack>& simTracks, const std::vector<SimVertex>& simVertices) {
  //  std::cout << "  PhotonMCTruthFinder::fill " << std::endl;

  // Watch out there ! A SimVertex is in mm (stupid),

  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  //  std::cout << "  PhotonMCTruthFinder::fill " << nVtx << " " << nTks << std::endl;

  // Empty event, do nothin'
  if (nVtx == 0)
    return;

  // create a map associating geant particle id and position in the
  // event SimTrack vector
  for (unsigned it = 0; it < nTks; ++it) {
    geantToIndex_[simTracks[it].trackId()] = it;
    //    std::cout << " PhotonMCTruthFinder::fill it " << it << " simTracks[it].trackId() " <<  simTracks[it].trackId() << std::endl;
  }
}
