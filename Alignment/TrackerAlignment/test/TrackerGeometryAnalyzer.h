#ifndef Alignment_TrackerAlignment_TrackerGeometryAnalyzer_h
#define Alignment_TrackerAlignment_TrackerGeometryAnalyzer_h

// Original Author:  Max Stark
//         Created:  Thu, 14 Jan 2016 11:35:07 CET

// system includes
#include <set>
#include <iostream>
#include <sstream>

// core framework functionality
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// topology and geometry
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// alignment
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

class Alignable;



class TrackerGeometryAnalyzer : public edm::EDAnalyzer {

  //========================== PUBLIC METHODS =================================
  public: //===================================================================

    TrackerGeometryAnalyzer(const edm::ParameterSet&);
    virtual ~TrackerGeometryAnalyzer() {};

    virtual void beginRun(const edm::Run&,   const edm::EventSetup&) override;
    virtual void analyze (const edm::Event&, const edm::EventSetup&) override {};

  //========================= PRIVATE METHODS =================================
  private: //==================================================================

    void setTrackerTopology(const edm::EventSetup&);
    void setTrackerGeometry(const edm::EventSetup&);

    void analyzeTrackerAlignables();
    void analyzeAlignableDetUnits  (Alignable*);
    void analyzeCompositeAlignables(Alignable*);
    int countCompositeAlignables(Alignable*);
    void printAlignableStructure(Alignable*, std::ostringstream&, int indent);

    void analyzeTrackerGeometry();
    void analyzeTrackerGeometryVersion(std::ostringstream&);
    void analyzePXBDetUnit(DetId& detId, std::ostringstream&);
    void analyzePXB();
    void analyzePXEDetUnit(DetId& detId, std::ostringstream&);
    void analyzePXE();
    void analyzeTIBDetUnit(DetId& detId, std::ostringstream&);
    void analyzeTIB();
    void analyzeTIDDetUnit(DetId& detId, std::ostringstream&);
    void analyzeTID();
    void analyzeTOBDetUnit(DetId& detId, std::ostringstream&);
    void analyzeTOB();
    void analyzeTECDetUnit(DetId& detId, std::ostringstream&);
    void analyzeTEC();

  //========================== PRIVATE DATA ===================================
  //===========================================================================

    // config-file parameters
    const bool analyzeAlignables_;
    const bool printTrackerStructure_;
    const int  maxPrintDepth_;
    const bool analyzeGeometry_;
    const bool analyzePXB_;
    const bool analyzePXE_;
    const bool analyzeTIB_;
    const bool analyzeTID_;
    const bool analyzeTOB_;
    const bool analyzeTEC_;

    // topology and geometry
    const TrackerTopology* trackerTopology;
    const TrackerGeometry* trackerGeometry;

    // alignable object ID provider
    AlignableObjectId alignableObjectId_;

    // counter for detUnits
    int numPXBDetUnits = 0;
    int numPXEDetUnits = 0;
    int numTIBDetUnits = 0;
    int numTIDDetUnits = 0;
    int numTOBDetUnits = 0;
    int numTECDetUnits = 0;

    // PixelBarrel counters
    std::set<unsigned int> pxbLayerIDs;
    std::set<unsigned int> pxbLadderIDs;
    std::set<unsigned int> pxbModuleIDs;

    // PixelEndcap counters
    std::set<unsigned int> pxeSideIDs;
    std::set<unsigned int> pxeDiskIDs;
    std::set<unsigned int> pxeBladeIDs;
    std::set<unsigned int> pxePanelIDs;
    std::set<unsigned int> pxeModuleIDs;

    // TIB counters
    std::set<unsigned int> tibSideIDs;
    std::set<unsigned int> tibLayerIDs;
    std::set<unsigned int> tibStringIDs;
    std::set<unsigned int> tibModuleIDs;

    // TID counters
    std::set<unsigned int> tidSideIDs;
    std::set<unsigned int> tidWheelIDs;
    std::set<unsigned int> tidRingIDs;
    std::set<unsigned int> tidModuleIDs;

    // TOB counters
    std::set<unsigned int> tobLayerIDs;
    std::set<unsigned int> tobSideIDs;
    std::set<unsigned int> tobRodIDs;
    std::set<unsigned int> tobModuleIDs;

    // TEC counters
    std::set<unsigned int> tecSideIDs;
    std::set<unsigned int> tecWheelIDs;
    std::set<unsigned int> tecPetalIDs;
    std::set<unsigned int> tecRingIDs;
    std::set<unsigned int> tecModuleIDs;

};

// define this as a plug-in
DEFINE_FWK_MODULE(TrackerGeometryAnalyzer);

#endif /* Alignment_TrackerAlignment_TrackerGeometryAnalyzer_h */
