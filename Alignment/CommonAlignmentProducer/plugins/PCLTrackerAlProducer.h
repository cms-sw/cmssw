#ifndef Alignment_CommonAlignmentProducer_PCLTrackerAlProducer_h
#define Alignment_CommonAlignmentProducer_PCLTrackerAlProducer_h

/// \class AlignmentProducer
///
/// Package     : Alignment/CommonAlignmentProducer
/// Description : calls alignment algorithms
///
///  \author    : Frederic Ronga
///  Revision   : $Revision: 1.27 $
///  last update: $Date: 2012/06/13 16:23:30 $
///  by         : $Author: yana $

// TODO: Update Doxygen description


/*** System includes ***/
#include <vector>
#include <memory>
#include <sstream>

/*** Core framework functionality ***/
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBCommon/interface/Time.h"

/*** Alignment ***/
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"

#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"

/*** Records for ESWatcher ***/
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"

#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorExtendedRcd.h"

/*** Forward declarations ***/
#include "FWCore/Framework/interface/Frameworkfwd.h"

/*** Typedefs ***/
// TODO: Move this at the end of AlignmentMonitorBase.h
typedef std::vector<AlignmentMonitorBase*> AlignmentMonitors;



class PCLTrackerAlProducer : public edm::EDAnalyzer {
  /**************************** PUBLIC METHODS ********************************/
  public: /********************************************************************/

    /// Constructor
    PCLTrackerAlProducer(const edm::ParameterSet&);
    /// Destructor
    virtual ~PCLTrackerAlProducer();

    /*** Code which implements the interface
         Called from outside ***/

    virtual void beginJob() override;
    virtual void endJob()   override;

    virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
    virtual void endRun  (const edm::Run&, const edm::EventSetup&) override;

    virtual void beginLuminosityBlock(const edm::LuminosityBlock&,
                                      const edm::EventSetup&) override;
    virtual void endLuminosityBlock  (const edm::LuminosityBlock&,
                                      const edm::EventSetup&) override;
  
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;



  /************************** PRIVATE METHODS *********************************/
  private: /*******************************************************************/

    /*** Code which is independent of Event & Setup
         Called from constructor ***/

    // TODO: Add missing method description

    void createAlignmentAlgorithm(const edm::ParameterSet&);
    void createMonitors          (const edm::ParameterSet&);
    void createCalibrations      (const edm::ParameterSet&);



    /*** Code which is dependent of Event & Setup
         Called and checked for each Event ***/

    // TODO: Add missing method description

    bool setupChanged(const edm::EventSetup&);
    void initAlignmentAlgorithm(const edm::EventSetup&);
    void initBeamSpot(const edm::Event&);
    void createGeometries(const edm::EventSetup&);
    void applyAlignmentsToDB(const edm::EventSetup&);
    void createAlignables(const TrackerTopology* const);
    void buildParameterStore();
    void applyMisalignment();
    void simpleMisalignment(const Alignables&, const std::string&, float, float, bool);
    void applyAlignmentsToGeometry();

    /// Apply DB constants belonging to (Err)Rcd to geometry,
    /// taking into account 'globalPosition' correction.
    template<class G, class Rcd, class ErrRcd>
    void applyDB(G*, const edm::EventSetup&, const AlignTransform&) const;

    /// Apply DB constants for surface deformations
    template<class G, class DeformationRcd>
    void applyDB(G*, const edm::EventSetup&) const;

    /// read in survey records
    void readInSurveyRcds(const edm::EventSetup&);
    
    /// Add survey info to an alignable
    void addSurveyInfo(Alignable*);



    /*** Code for writing results to database
         Called from endJob() ***/

    // TODO: Add missing method description

    void finish();
    void storeAlignmentsToDB();
    RunRanges makeNonOverlappingRunRanges(const edm::VParameterSet&);

    /// Write alignments and alignment errors for all sub detectors and the
    /// given run number
    void writeForRunRange(cond::Time_t);

    /// Write alignment and/or errors to DB for record names
    /// (removes *globalCoordinates before writing if non-null...).
    /// Takes over ownership of alignments and alignmentErrrors.
    void writeDB(Alignments*, const std::string&, AlignmentErrorsExtended*,
                 const std::string&, const AlignTransform*, cond::Time_t) const;

    /// Write surface deformations (bows & kinks) to DB for given record name
    /// Takes over ownership of alignmentsurfaceDeformations.
    void writeDB(AlignmentSurfaceDeformations*,
                 const std::string&, cond::Time_t) const;



  /*************************** PRIVATE DATA ***********************************/
  /****************************************************************************/

    /*** Alignment data ***/

    AlignmentAlgorithmBase* theAlignmentAlgo;
    Calibrations            theCalibrations;
    AlignmentMonitors       theMonitors;

    AlignmentParameterStore* theAlignmentParameterStore;
    AlignableTracker*        theTrackerAlignables;
    AlignableMuon*           theMuonAlignables;
    AlignableExtras*         theExtraAlignables;

    edm::Handle<reco::BeamSpot> theBeamSpot;
    /// GlobalPositions that might be read from DB, NULL otherwise
    const Alignments* globalPositions_;

    boost::shared_ptr<TrackerGeometry> theTrackerGeometry;
    boost::shared_ptr<DTGeometry>      theMuonDTGeometry;
    boost::shared_ptr<CSCGeometry>     theMuonCSCGeometry;

    int nevent_;



    /*** Parameters from config-file ***/

    edm::ParameterSet theParameterSet;

    const int    stNFixAlignables_;
    const double stRandomShift_, stRandomRotation_;
    const bool   applyDbAlignment_, checkDbAlignmentValidity_;
    const bool   doMisalignmentScenario_;
    const bool   saveToDB_, saveApeToDB_, saveDeformationsToDB_;
    const bool   doTracker_, doMuon_, useExtras_;
    const bool   useSurvey_;

    // map with tracks/trajectories
    const edm::InputTag tjTkAssociationMapTag_;
    // beam spot
    const edm::InputTag beamSpotTag_;
    // LAS beams in edm::Run (ignore if empty)
    const edm::InputTag tkLasBeamTag_;
    // ValueMap containing associtaion cluster - flag
    const edm::InputTag clusterValueMapTag_;



    /*** ESWatcher ***/

    // TODO: Which Record to watch for changes? (see setupChanged()-method)

    edm::ESWatcher<TrackerAlignmentRcd>              watchTrackerAlRcd;
    edm::ESWatcher<TrackerAlignmentErrorExtendedRcd> watchTrackerAlErrorExtRcd;
    edm::ESWatcher<DTAlignmentRcd>                   watchDTAlRcd;
    edm::ESWatcher<DTAlignmentErrorExtendedRcd>      watchDTAlErrExtRcd;
    edm::ESWatcher<CSCAlignmentRcd>                  watchCSCAlRcd;
    edm::ESWatcher<CSCAlignmentErrorExtendedRcd>     watchCSCAlErrExtRcd;

    edm::ESWatcher<TrackerSurfaceDeformationRcd> watchTrackerSurDeRcd;
    edm::ESWatcher<TrackerDigiGeometryRecord>    watchTrackerDigiGeoRcd;
    edm::ESWatcher<GlobalPositionRcd>            watchGlobalPosRcd;
    edm::ESWatcher<IdealGeometryRecord>          watchIdealGeoRcd;

    edm::ESWatcher<TrackerSurveyRcd>              watchTkSurveyRcd_;
    edm::ESWatcher<TrackerSurveyErrorExtendedRcd> watchTkSurveyErrExtRcd_;
    edm::ESWatcher<DTSurveyRcd>                   watchDTSurveyRcd_;
    edm::ESWatcher<DTSurveyErrorExtendedRcd>      watchDTSurveyErrExtRcd_;
    edm::ESWatcher<CSCSurveyRcd>                  watchCSCSurveyRcd_;
    edm::ESWatcher<CSCSurveyErrorExtendedRcd>     watchCSCSurveyErrExtRcd_;



    /*** Survey stuff ***/

    size_t              theSurveyIndex;
    const Alignments*   theSurveyValues;
    const SurveyErrors* theSurveyErrors;
};

#endif
