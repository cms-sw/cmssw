/** \file
 *
 *  $Date: 2008/04/25 21:23:15 $
 *  $Revision: 1.21 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/DTGeometry/interface/DTLayer.h>
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
// Muon  components
#include "Alignment/MuonAlignment/interface/AlignableDTBarrel.h"
#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"
#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCEndcap.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCRing.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableGEMEndcap.h"
#include "Alignment/MuonAlignment/interface/AlignableGEMStation.h"
#include "Alignment/MuonAlignment/interface/AlignableGEMRing.h"
#include "Alignment/MuonAlignment/interface/AlignableGEMSuperChamber.h"

#include <iostream>

//--------------------------------------------------------------------------------------------------
AlignableMuon::AlignableMuon(const DTGeometry* dtGeometry,
                             const CSCGeometry* cscGeometry,
                             const GEMGeometry* gemGeometry)
    : AlignableComposite(0, align::AlignableMuon),  // cannot yet set id, use 0
      alignableObjectId_(nullptr, dtGeometry, cscGeometry, gemGeometry) {
  // Build the muon barrel
  buildDTBarrel(dtGeometry);

  // Build the muon end caps
  buildCSCEndcap(cscGeometry);
  buildGEMEndcap(gemGeometry);

  // Set links to mothers recursively
  recursiveSetMothers(this);

  // now can set id as for all composites: id of first component
  theId = this->components()[0]->id();

  edm::LogInfo("AlignableMuon") << "Constructing alignable muon objects DONE";
}

//--------------------------------------------------------------------------------------------------
AlignableMuon::~AlignableMuon() {
  for (align::Alignables::iterator iter = theMuonComponents.begin(); iter != theMuonComponents.end(); iter++) {
    delete *iter;
  }
}

//------------------------------------------------------------------------------
void AlignableMuon::update(const DTGeometry* dtGeometry,
                           const CSCGeometry* cscGeometry,
                           const GEMGeometry* gemGeometry) {
  // update the muon barrel
  buildDTBarrel(dtGeometry, /* update = */ true);

  // update the muon end caps
  buildCSCEndcap(cscGeometry, /* update = */ true);
  buildGEMEndcap(gemGeometry, /* update = */ true);

  edm::LogInfo("Alignment") << "@SUB=AlignableMuon::update"
                            << "Updating alignable muon objects DONE";
}

//------------------------------------------------------------------------------
void AlignableMuon::buildDTBarrel(const DTGeometry* pDT, bool update) {
  LogDebug("Position") << "Constructing AlignableDTBarrel";

  // Temporary container for chambers in a given station and stations in a given wheel
  std::vector<AlignableDTChamber*> tmpDTChambersInStation;
  std::vector<AlignableDTStation*> tmpDTStationsInWheel;

  // Loop over wheels ( -2..2 )
  for (int iwh = -2; iwh < 3; iwh++) {
    // Loop over stations ( 1..4 )
    for (int ist = 1; ist < 5; ist++) {
      // Loop over geom DT Chambers
      int iChamber{0};
      std::vector<const GeomDet*>
          theSLs;  // FIXME: What is this vector meant to be for? Probably redundant since super layers are handled inside of the AlignableDTChamber.
      for (const auto& det : pDT->chambers()) {
        // Get the chamber ID
        DTChamberId chamberId = det->id();

        // Get wheel,station and sector of the chamber
        int wh = chamberId.wheel();
        int st = chamberId.station();
        //int se = chamberId.sector();

        // Select the chambers in a given wheel in a given station
        if (iwh == wh && ist == st) {
          if (update) {
            // Update the alignable DT chamber
            theDTBarrel.back()->wheel(iwh + 2).station(ist - 1).chamber(iChamber).update(det);
          } else {
            // Create the alignable DT chamber
            AlignableDTChamber* tmpDTChamber = new AlignableDTChamber(det);

            // Store the DT chambers in a given DT Station and Wheel
            tmpDTChambersInStation.push_back(tmpDTChamber);
          }

          ++iChamber;
          // End chamber selection
        }

        // End loop over chambers
      }

      if (!update) {
        // Store the DT chambers
        theDTChambers.insert(theDTChambers.end(), tmpDTChambersInStation.begin(), tmpDTChambersInStation.end());

        // Create the alignable DT station with chambers in a given station and wheel
        AlignableDTStation* tmpDTStation = new AlignableDTStation(tmpDTChambersInStation);

        // Store the DT stations in a given wheel
        tmpDTStationsInWheel.push_back(tmpDTStation);

        // Clear the temporary vector of chambers in a station
        tmpDTChambersInStation.clear();
      }
      // End loop over stations
    }

    if (!update) {
      // Store The DT stations
      theDTStations.insert(theDTStations.end(), tmpDTStationsInWheel.begin(), tmpDTStationsInWheel.end());

      // Create the alignable DT wheel
      AlignableDTWheel* tmpWheel = new AlignableDTWheel(tmpDTStationsInWheel);

      // Store the DT wheels
      theDTWheels.push_back(tmpWheel);

      // Clear temporary vector of stations in a wheel
      tmpDTStationsInWheel.clear();
    }

    // End loop over wheels
  }

  if (!update) {
    // Create the alignable Muon Barrel
    AlignableDTBarrel* tmpDTBarrel = new AlignableDTBarrel(theDTWheels);

    // Store the barrel
    theDTBarrel.push_back(tmpDTBarrel);

    // Store the barrel in the muon
    theMuonComponents.push_back(tmpDTBarrel);
  }
}

//------------------------------------------------------------------------------
void AlignableMuon::buildCSCEndcap(const CSCGeometry* pCSC, bool update) {
  LogDebug("Position") << "Constructing AlignableCSCBarrel";

  // Temporary container for stations in a given endcap
  std::vector<AlignableCSCStation*> tmpCSCStationsInEndcap;

  // Loop over endcaps ( 1..2 )
  for (int iec = 1; iec < 3; iec++) {
    // Temporary container for rings in a given station
    std::vector<AlignableCSCRing*> tmpCSCRingsInStation;

    // Loop over stations ( 1..4 )
    for (int ist = 1; ist < 5; ist++) {
      // Temporary container for chambers in a given ring
      std::vector<AlignableCSCChamber*> tmpCSCChambersInRing;

      // Loop over rings ( 1..4 )
      for (int iri = 1; iri < 5; iri++) {
        // Loop over geom CSC Chambers
        int iChamber{0};
        const CSCGeometry::ChamberContainer& vc = pCSC->chambers();
        for (const auto& det : vc) {
          // Get the CSCDet ID
          CSCDetId cscId = det->id();

          // Get chamber, station, ring, layer and endcap labels of the CSC chamber
          int ec = cscId.endcap();
          int st = cscId.station();
          int ri = cscId.ring();
          //int ch = cscId.chamber();

          // Select the chambers in a given endcap, station, and ring
          if (iec == ec && ist == st && iri == ri) {
            if (update) {
              // Update the alignable CSC chamber
              theCSCEndcaps[iec - 1]->station(ist - 1).ring(iri - 1).chamber(iChamber).update(det);
            } else {
              AlignableCSCChamber* tmpCSCChamber = new AlignableCSCChamber(det);

              // Store the alignable CSC chambers
              tmpCSCChambersInRing.push_back(tmpCSCChamber);
            }

            ++iChamber;
            // End If chamber selection
          }

          // End loop over geom CSC chambers
        }

        if (!update) {
          // Not all stations have 4 rings: only add the rings that exist (have chambers associated with them)
          if (!tmpCSCChambersInRing.empty()) {
            // Store the alignable CSC chambers
            theCSCChambers.insert(theCSCChambers.end(), tmpCSCChambersInRing.begin(), tmpCSCChambersInRing.end());

            // Create the alignable CSC ring with chambers in a given ring
            AlignableCSCRing* tmpCSCRing = new AlignableCSCRing(tmpCSCChambersInRing);

            // Store the CSC rings in a given station
            tmpCSCRingsInStation.push_back(tmpCSCRing);

            // Clear the temporary vector of chambers in ring
            tmpCSCChambersInRing.clear();

            // End if this ring exists
          }
        }

        // End loop over rings
      }

      if (!update) {
        // Create the alignable CSC station with rings in a given station
        AlignableCSCStation* tmpCSCStation = new AlignableCSCStation(tmpCSCRingsInStation);

        // Store the alignable CSC rings
        theCSCRings.insert(theCSCRings.end(), tmpCSCRingsInStation.begin(), tmpCSCRingsInStation.end());

        // Store the CSC stations in a given endcap
        tmpCSCStationsInEndcap.push_back(tmpCSCStation);

        // Clear the temporary vector of rings in station
        tmpCSCRingsInStation.clear();
      }

      // End loop over stations
    }

    if (!update) {
      // Create the alignable CSC endcap
      AlignableCSCEndcap* tmpEndcap = new AlignableCSCEndcap(tmpCSCStationsInEndcap);

      // Store the alignable CSC stations
      theCSCStations.insert(theCSCStations.end(), tmpCSCStationsInEndcap.begin(), tmpCSCStationsInEndcap.end());

      // Store the alignable CSC endcaps
      theCSCEndcaps.push_back(tmpEndcap);

      // Clear the temporary vector of stations in endcap
      tmpCSCStationsInEndcap.clear();
    }

    // End loop over endcaps
  }

  if (!update) {
    // Store the encaps in the muon components
    theMuonComponents.insert(theMuonComponents.end(), theCSCEndcaps.begin(), theCSCEndcaps.end());
  }
}

//--------------------------------------------------------------------------------------------------
void AlignableMuon::buildGEMEndcap(const GEMGeometry* pGEM, bool update) {
  LogDebug("Position") << "Constructing AlignableGEMEndcap";
  std::vector<AlignableGEMStation*> tmpGEMStationsInEndcap;
  for (int iec = -1; iec < 2; iec = iec + 2) {
    std::vector<AlignableGEMRing*> tmpGEMRingsInStation;

    for (int ist = 0; ist < 3; ist++) {
      if (ist == 0)
        continue;  //Run3 GEM dosen't have ME0
      if (ist == 2)
        continue;  //Run3 GEM dosen't have GE2/1
      std::vector<AlignableGEMSuperChamber*> tmpGEMSuperChambersInRing;
      int iri = 1;
      int iChamber{0};
      auto vc = pGEM->superChambers();
      for (const auto& det : vc) {
        GEMDetId gemId = det->id();
        int ec = gemId.region();
        int st = gemId.station();
        int ri = gemId.ring();

        if (iec == ec && ist == st && iri == ri) {
          if (update) {
            theGEMEndcaps[iec == -1 ? 0 : 1]->station(ist - 1).ring(iri - 1).superChamber(iChamber).update(det);
          } else {
            AlignableGEMSuperChamber* tmpGEMSuperChamber = new AlignableGEMSuperChamber(det);
            tmpGEMSuperChambersInRing.push_back(tmpGEMSuperChamber);
          }
          ++iChamber;
        }
      }
      if (!update) {
        if (!tmpGEMSuperChambersInRing.empty()) {
          theGEMSuperChambers.insert(
              theGEMSuperChambers.end(), tmpGEMSuperChambersInRing.begin(), tmpGEMSuperChambersInRing.end());
          AlignableGEMRing* tmpGEMRing = new AlignableGEMRing(tmpGEMSuperChambersInRing);
          tmpGEMRingsInStation.push_back(tmpGEMRing);
          tmpGEMSuperChambersInRing.clear();
        }
      }
      if (!update) {
        AlignableGEMStation* tmpGEMStation = new AlignableGEMStation(tmpGEMRingsInStation);
        theGEMRings.insert(theGEMRings.end(), tmpGEMRingsInStation.begin(), tmpGEMRingsInStation.end());
        tmpGEMStationsInEndcap.push_back(tmpGEMStation);
        tmpGEMRingsInStation.clear();
      }
    }
    if (!update) {
      AlignableGEMEndcap* tmpEndcap = new AlignableGEMEndcap(tmpGEMStationsInEndcap);
      theGEMStations.insert(theGEMStations.end(), tmpGEMStationsInEndcap.begin(), tmpGEMStationsInEndcap.end());
      theGEMEndcaps.push_back(tmpEndcap);
      tmpGEMStationsInEndcap.clear();
    }
  }
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::DTLayers() {
  align::Alignables result;

  align::Alignables chambers = DTChambers();
  for (align::Alignables::const_iterator chamberIter = chambers.begin(); chamberIter != chambers.end(); ++chamberIter) {
    align::Alignables superlayers = (*chamberIter)->components();
    for (align::Alignables::const_iterator superlayerIter = superlayers.begin(); superlayerIter != superlayers.end();
         ++superlayerIter) {
      align::Alignables layers = (*superlayerIter)->components();
      for (align::Alignables::const_iterator layerIter = layers.begin(); layerIter != layers.end(); ++layerIter) {
        result.push_back(*layerIter);
      }
    }
  }

  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::DTSuperLayers() {
  align::Alignables result;

  align::Alignables chambers = DTChambers();
  for (align::Alignables::const_iterator chamberIter = chambers.begin(); chamberIter != chambers.end(); ++chamberIter) {
    align::Alignables superlayers = (*chamberIter)->components();
    for (align::Alignables::const_iterator superlayerIter = superlayers.begin(); superlayerIter != superlayers.end();
         ++superlayerIter) {
      result.push_back(*superlayerIter);
    }
  }

  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::DTChambers() {
  align::Alignables result;
  copy(theDTChambers.begin(), theDTChambers.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::DTStations() {
  align::Alignables result;
  copy(theDTStations.begin(), theDTStations.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::DTWheels() {
  align::Alignables result;
  copy(theDTWheels.begin(), theDTWheels.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::DTBarrel() {
  align::Alignables result;
  copy(theDTBarrel.begin(), theDTBarrel.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::CSCLayers() {
  align::Alignables result;

  align::Alignables chambers = CSCChambers();
  for (align::Alignables::const_iterator chamberIter = chambers.begin(); chamberIter != chambers.end(); ++chamberIter) {
    align::Alignables layers = (*chamberIter)->components();
    for (align::Alignables::const_iterator layerIter = layers.begin(); layerIter != layers.end(); ++layerIter) {
      result.push_back(*layerIter);
    }
  }

  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::CSCChambers() {
  align::Alignables result;
  copy(theCSCChambers.begin(), theCSCChambers.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::CSCRings() {
  align::Alignables result;
  copy(theCSCRings.begin(), theCSCRings.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::CSCStations() {
  align::Alignables result;
  copy(theCSCStations.begin(), theCSCStations.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::CSCEndcaps() {
  align::Alignables result;
  copy(theCSCEndcaps.begin(), theCSCEndcaps.end(), back_inserter(result));
  return result;
}

//__________________________________________________________________________________________________
align::Alignables AlignableMuon::GEMEtaPartitions() {
  align::Alignables result;
  align::Alignables superChambers = GEMSuperChambers();
  for (align::Alignables::const_iterator superChamberIter = superChambers.begin();
       superChamberIter != superChambers.end();
       ++superChamberIter) {
    align::Alignables chambers = (*superChamberIter)->components();
    for (align::Alignables::const_iterator chamberIter = chambers.begin(); chamberIter != chambers.end();
         ++chamberIter) {
      align::Alignables etaPartitions = (*chamberIter)->components();
      for (align::Alignables::const_iterator etaPartitionIter = etaPartitions.begin();
           etaPartitionIter != etaPartitions.end();
           ++etaPartitionIter) {
        result.push_back(*etaPartitionIter);
      }
    }
  }
  return result;
}

//__________________________________________________________________________________________________
align::Alignables AlignableMuon::GEMChambers() {
  align::Alignables result;
  align::Alignables superChambers = GEMSuperChambers();
  for (align::Alignables::const_iterator superChamberIter = superChambers.begin();
       superChamberIter != superChambers.end();
       ++superChamberIter) {
    align::Alignables chambers = (*superChamberIter)->components();
    for (align::Alignables::const_iterator chamberIter = chambers.begin(); chamberIter != chambers.end();
         ++chamberIter) {
      result.push_back(*chamberIter);
    }
  }
  return result;
}

//__________________________________________________________________________________________________
align::Alignables AlignableMuon::GEMSuperChambers() {
  align::Alignables result;
  copy(theGEMSuperChambers.begin(), theGEMSuperChambers.end(), back_inserter(result));
  return result;
}

//__________________________________________________________________________________________________
align::Alignables AlignableMuon::GEMRings() {
  align::Alignables result;
  copy(theGEMRings.begin(), theGEMRings.end(), back_inserter(result));
  return result;
}

//__________________________________________________________________________________________________
align::Alignables AlignableMuon::GEMStations() {
  align::Alignables result;
  copy(theGEMStations.begin(), theGEMStations.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMuon::GEMEndcaps() {
  align::Alignables result;
  copy(theGEMEndcaps.begin(), theGEMEndcaps.end(), back_inserter(result));
  return result;
}

//__________________________________________________________________________________________________
void AlignableMuon::recursiveSetMothers(Alignable* alignable) {
  for (const auto& iter : alignable->components()) {
    iter->setMother(alignable);
    recursiveSetMothers(iter);
  }
}

//__________________________________________________________________________________________________
Alignments* AlignableMuon::alignments(void) const {
  align::Alignables comp = this->components();
  Alignments* m_alignments = new Alignments();
  // Add components recursively
  for (align::Alignables::iterator i = comp.begin(); i != comp.end(); i++) {
    Alignments* tmpAlignments = (*i)->alignments();
    std::copy(tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), std::back_inserter(m_alignments->m_align));
    delete tmpAlignments;
  }

  // sort by rawId
  std::sort(m_alignments->m_align.begin(), m_alignments->m_align.end());

  return m_alignments;
}
//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableMuon::alignmentErrors(void) const {
  align::Alignables comp = this->components();
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add components recursively
  for (align::Alignables::iterator i = comp.begin(); i != comp.end(); i++) {
    AlignmentErrorsExtended* tmpAlignmentErrorsExtended = (*i)->alignmentErrors();
    std::copy(tmpAlignmentErrorsExtended->m_alignError.begin(),
              tmpAlignmentErrorsExtended->m_alignError.end(),
              std::back_inserter(m_alignmentErrors->m_alignError));
    delete tmpAlignmentErrorsExtended;
  }

  // sort by rawId
  std::sort(m_alignmentErrors->m_alignError.begin(), m_alignmentErrors->m_alignError.end());

  return m_alignmentErrors;
}
//__________________________________________________________________________________________________
Alignments* AlignableMuon::dtAlignments(void) {
  // Retrieve muon barrel alignments
  Alignments* tmpAlignments = this->DTBarrel().front()->alignments();

  return tmpAlignments;
}
//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableMuon::dtAlignmentErrorsExtended(void) {
  // Retrieve muon barrel alignment errors
  AlignmentErrorsExtended* tmpAlignmentErrorsExtended = this->DTBarrel().front()->alignmentErrors();

  return tmpAlignmentErrorsExtended;
}
//__________________________________________________________________________________________________
Alignments* AlignableMuon::cscAlignments(void) {
  // Retrieve muon endcaps alignments
  Alignments* cscEndCap1 = this->CSCEndcaps().front()->alignments();
  Alignments* cscEndCap2 = this->CSCEndcaps().back()->alignments();
  Alignments* tmpAlignments = new Alignments();

  std::copy(cscEndCap1->m_align.begin(), cscEndCap1->m_align.end(), back_inserter(tmpAlignments->m_align));
  std::copy(cscEndCap2->m_align.begin(), cscEndCap2->m_align.end(), back_inserter(tmpAlignments->m_align));

  return tmpAlignments;
}
//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableMuon::cscAlignmentErrorsExtended(void) {
  // Retrieve muon endcaps alignment errors
  AlignmentErrorsExtended* cscEndCap1Errors = this->CSCEndcaps().front()->alignmentErrors();
  AlignmentErrorsExtended* cscEndCap2Errors = this->CSCEndcaps().back()->alignmentErrors();
  AlignmentErrorsExtended* tmpAlignmentErrorsExtended = new AlignmentErrorsExtended();

  std::copy(cscEndCap1Errors->m_alignError.begin(),
            cscEndCap1Errors->m_alignError.end(),
            back_inserter(tmpAlignmentErrorsExtended->m_alignError));
  std::copy(cscEndCap2Errors->m_alignError.begin(),
            cscEndCap2Errors->m_alignError.end(),
            back_inserter(tmpAlignmentErrorsExtended->m_alignError));

  return tmpAlignmentErrorsExtended;
}

//__________________________________________________________________________________________________
Alignments* AlignableMuon::gemAlignments(void) {
  Alignments* gemEndCap1 = this->GEMEndcaps().front()->alignments();
  Alignments* gemEndCap2 = this->GEMEndcaps().back()->alignments();
  Alignments* tmpAlignments = new Alignments();
  std::copy(gemEndCap1->m_align.begin(), gemEndCap1->m_align.end(), back_inserter(tmpAlignments->m_align));
  std::copy(gemEndCap2->m_align.begin(), gemEndCap2->m_align.end(), back_inserter(tmpAlignments->m_align));
  std::sort(tmpAlignments->m_align.begin(), tmpAlignments->m_align.end());

  return tmpAlignments;
}
//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableMuon::gemAlignmentErrorsExtended(void) {
  // Retrieve muon endcaps alignment errors
  AlignmentErrorsExtended* gemEndCap1Errors = this->GEMEndcaps().front()->alignmentErrors();
  AlignmentErrorsExtended* gemEndCap2Errors = this->GEMEndcaps().back()->alignmentErrors();
  AlignmentErrorsExtended* tmpAlignmentErrorsExtended = new AlignmentErrorsExtended();

  std::copy(gemEndCap1Errors->m_alignError.begin(),
            gemEndCap1Errors->m_alignError.end(),
            back_inserter(tmpAlignmentErrorsExtended->m_alignError));
  std::copy(gemEndCap2Errors->m_alignError.begin(),
            gemEndCap2Errors->m_alignError.end(),
            back_inserter(tmpAlignmentErrorsExtended->m_alignError));
  std::sort(tmpAlignmentErrorsExtended->m_alignError.begin(), tmpAlignmentErrorsExtended->m_alignError.end());

  return tmpAlignmentErrorsExtended;
}
