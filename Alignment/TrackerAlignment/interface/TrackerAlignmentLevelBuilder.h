#ifndef ALIGNMENT_TRACKERALIGNMENT_INTERFACE_TRACKERALIGNMENTLEVELBUILDER_H_
#define ALIGNMENT_TRACKERALIGNMENT_INTERFACE_TRACKERALIGNMENTLEVELBUILDER_H_

// Original Author:  Max Stark
//         Created:  Wed, 10 Feb 2016 13:48:41 CET

// system includes
#include <set>
#include <map>

// core framework functionality
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// topology
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

// common alignment
#include "Alignment/CommonAlignment/interface/AlignmentLevel.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"



class TrackerAlignmentLevelBuilder {

  //========================== PUBLIC METHODS =================================
  public: //===================================================================

    TrackerAlignmentLevelBuilder(const TrackerTopology*);
    virtual ~TrackerAlignmentLevelBuilder();

    void addDetUnitInfo(const DetId& detId);
    std::vector<align::AlignmentLevels*> build();

  //========================= PRIVATE METHODS =================================
  private: //==================================================================

    void addPXBDetUnitInfo(const DetId& detId);
    void addPXEDetUnitInfo(const DetId& detId);
    void addTIBDetUnitInfo(const DetId& detId);
    void addTIDDetUnitInfo(const DetId& detId);
    void addTOBDetUnitInfo(const DetId& detId);
    void addTECDetUnitInfo(const DetId& detId);

    void buildPXBAlignmentLevels();
    void buildPXEAlignmentLevels();
    void buildTIBAlignmentLevels();
    void buildTIDAlignmentLevels();
    void buildTOBAlignmentLevels();
    void buildTECAlignmentLevels();

  //========================== PRIVATE DATA ===================================
  //===========================================================================

    const TrackerTopology* trackerTopology;

    // sub-detector alignment levels
    align::AlignmentLevels pxb;
    align::AlignmentLevels pxe;
    align::AlignmentLevels tib;
    align::AlignmentLevels tid;
    align::AlignmentLevels tob;
    align::AlignmentLevels tec;

    // all alignment levels of tracker
    std::vector<align::AlignmentLevels*> levels;

    // PixelBarrel
    std::set<unsigned int> pxbLayerIDs;
    std::set<unsigned int> pxbLadderIDs;
    std::set<unsigned int> pxbModuleIDs;
    std::map<unsigned int, unsigned int> pxbLaddersPerLayer;

    // PixelEndcap
    std::set<unsigned int> pxeSideIDs;
    std::set<unsigned int> pxeDiskIDs;
    std::set<unsigned int> pxeBladeIDs;
    std::set<unsigned int> pxePanelIDs;
    std::set<unsigned int> pxeModuleIDs;

    // TIB
    std::set<unsigned int> tibSideIDs;
    std::set<unsigned int> tibLayerIDs;
    std::set<unsigned int> tibStringIDs;
    std::set<unsigned int> tibModuleIDs;
    std::map<unsigned int, unsigned int> pxbStringsPerHalfShell;

    // TID
    std::set<unsigned int> tidSideIDs;
    std::set<unsigned int> tidWheelIDs;
    std::set<unsigned int> tidRingIDs;
    std::set<unsigned int> tidModuleIDs;
    std::map<unsigned int, unsigned int> tidStringsInnerLayer;
    std::map<unsigned int, unsigned int> tidStringsOuterLayer;

    // TOB
    std::set<unsigned int> tobLayerIDs;
    std::set<unsigned int> tobSideIDs;
    std::set<unsigned int> tobRodIDs;
    std::set<unsigned int> tobModuleIDs;

    // TEC
    std::set<unsigned int> tecSideIDs;
    std::set<unsigned int> tecWheelIDs;
    std::set<unsigned int> tecPetalIDs;
    std::set<unsigned int> tecRingIDs;
    std::set<unsigned int> tecModuleIDs;

};

#endif /* ALIGNMENT_TRACKERALIGNMENT_INTERFACE_TRACKERALIGNMENTLEVELBUILDER_H_ */
