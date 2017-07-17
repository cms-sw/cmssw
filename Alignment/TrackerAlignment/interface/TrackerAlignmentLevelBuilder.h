#ifndef ALIGNMENT_TRACKERALIGNMENT_INTERFACE_TRACKERALIGNMENTLEVELBUILDER_H_
#define ALIGNMENT_TRACKERALIGNMENT_INTERFACE_TRACKERALIGNMENTLEVELBUILDER_H_

// Original Author:  Max Stark
//         Created:  Wed, 10 Feb 2016 13:48:41 CET

// system includes
#include <set>
#include <map>

// core framework functionality
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// alignment
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignmentLevel.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/TrackerAlignment/interface/TrackerNameSpace.h"

class DetId;
class TrackerTopology;
class TrackerGeometry;

class TrackerAlignmentLevelBuilder {

  //========================== PUBLIC METHODS =================================
  public: //===================================================================

    TrackerAlignmentLevelBuilder(const TrackerTopology*,
                                 const TrackerGeometry*);
    virtual ~TrackerAlignmentLevelBuilder();

    void addDetUnitInfo(const DetId& detId);
    std::vector<align::AlignmentLevels> build();
    const AlignableObjectId& objectIdProvider() const { return alignableObjectId_; }
    const align::TrackerNameSpace& trackerNameSpace() const;

  //========================= PRIVATE METHODS =================================
  private: //==================================================================

    void addPXBDetUnitInfo(const DetId& detId);
    void addPXEDetUnitInfo(const DetId& detId);
    void addTIBDetUnitInfo(const DetId& detId);
    void addTIDDetUnitInfo(const DetId& detId);
    void addTOBDetUnitInfo(const DetId& detId);
    void addTECDetUnitInfo(const DetId& detId);

    align::AlignmentLevels buildPXBAlignmentLevels();
    align::AlignmentLevels buildPXEAlignmentLevels();
    align::AlignmentLevels buildTIBAlignmentLevels();
    align::AlignmentLevels buildTIDAlignmentLevels();
    align::AlignmentLevels buildTOBAlignmentLevels();
    align::AlignmentLevels buildTECAlignmentLevels();

  //========================== PRIVATE DATA ===================================
  //===========================================================================

    const TrackerTopology* trackerTopology_;
    const AlignableObjectId alignableObjectId_;
    align::TrackerNameSpace trackerNameSpace_;
    bool levelsBuilt_{false};

    // PixelBarrel
    std::set<unsigned int> pxbLayerIDs_;
    std::set<unsigned int> pxbLadderIDs_;
    std::set<unsigned int> pxbModuleIDs_;
    std::map<unsigned int, unsigned int> pxbLaddersPerLayer_;

    // PixelEndcap
    std::set<unsigned int> pxeSideIDs_;
    std::set<unsigned int> pxeDiskIDs_;
    std::set<unsigned int> pxeBladeIDs_;
    std::set<unsigned int> pxePanelIDs_;
    std::set<unsigned int> pxeModuleIDs_;

    // TIB
    std::set<unsigned int> tibSideIDs_;
    std::set<unsigned int> tibLayerIDs_;
    std::set<unsigned int> tibStringIDs_;
    std::set<unsigned int> tibModuleIDs_;
    std::map<unsigned int, unsigned int> pxbStringsPerHalfShell_;

    // TID
    std::set<unsigned int> tidSideIDs_;
    std::set<unsigned int> tidWheelIDs_;
    std::set<unsigned int> tidRingIDs_;
    std::set<unsigned int> tidModuleIDs_;
    std::map<unsigned int, unsigned int> tidStringsInnerLayer_;
    std::map<unsigned int, unsigned int> tidStringsOuterLayer_;

    // TOB
    std::set<unsigned int> tobLayerIDs_;
    std::set<unsigned int> tobSideIDs_;
    std::set<unsigned int> tobRodIDs_;
    std::set<unsigned int> tobModuleIDs_;

    // TEC
    std::set<unsigned int> tecSideIDs_;
    std::set<unsigned int> tecWheelIDs_;
    std::set<unsigned int> tecPetalIDs_;
    std::set<unsigned int> tecRingIDs_;
    std::set<unsigned int> tecModuleIDs_;

};

#endif /* ALIGNMENT_TRACKERALIGNMENT_INTERFACE_TRACKERALIGNMENTLEVELBUILDER_H_ */
