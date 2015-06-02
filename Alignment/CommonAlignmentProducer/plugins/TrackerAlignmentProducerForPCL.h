#ifndef ALIGNMENT_COMMONALIGNMENTPRODUCER_PLUGINS_ALIGNMENTPRODUCERFORPCL_H_
#define ALIGNMENT_COMMONALIGNMENTPRODUCER_PLUGINS_ALIGNMENTPRODUCERFORPCL_H_

#include <vector>

// Framework
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "DataFormats/Provenance/interface/RunID.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// Alignment
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"

// for watcher
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorRcd.h"


class Alignments;
class IntegratedCalibrationBase;
class SurveyErrors;
namespace edm {
  class Run;
  class LuminosityBlock;
}

class TrackerAlignmentProducerForPCL : public edm::EDAnalyzer {
  public:

    TrackerAlignmentProducerForPCL(const edm::ParameterSet&);
    virtual ~TrackerAlignmentProducerForPCL();

    typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair;
    typedef std::vector<ConstTrajTrackPair>                  ConstTrajTrackPairCollection;
    typedef std::vector<Alignable*>                          Alignables;

    typedef AlignmentAlgorithmBase::RunNumber RunNumber;
    typedef AlignmentAlgorithmBase::RunRange  RunRange;
    typedef std::vector<RunRange>             RunRanges;


    virtual void beginJob(void) override;
    virtual void endJob  (void) override;

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
    virtual void endRun  (const edm::Run&, const edm::EventSetup&) override;

    virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
    virtual void endLuminosityBlock  (const edm::LuminosityBlock&, const edm::EventSetup&) override;

  private:

    void init(const edm::EventSetup&);
    void finish();

    void createGeometries  (const edm::EventSetup&);
    void simpleMisalignment(const Alignables&, const std::string&, float, float, bool);

    /// Apply DB constants belonging to (Err)Rcd to geometry,
    /// taking into account 'globalPosition' correction.
    template<class G, class Rcd, class ErrRcd>
    void applyDB(G*, const edm::EventSetup&, const AlignTransform&) const;
    /// Apply DB constants for surface deformations
    template<class G, class DeformationRcd>
    void applyDB(G*, const edm::EventSetup&) const;

    // write alignments and alignment errors for all sub detectors and
    // the given run number
    void writeForRunRange(cond::Time_t);

    /// Write alignment and/or errors to DB for record names
    /// (removes *globalCoordinates before writing if non-null...).
    /// Takes over ownership of alignments and alignmentErrrors.
    void writeDB(Alignments*, const std::string&, AlignmentErrorsExtended*,
                 const std::string&, const AlignTransform*, cond::Time_t) const;
    /// Write surface deformations (bows & kinks) to DB for given record name
    /// Takes over ownership of alignmentsurfaceDeformations.
    void writeDB(AlignmentSurfaceDeformations*, const std::string&, cond::Time_t) const;

    RunRanges makeNonOverlappingRunRanges(const edm::VParameterSet&);


    /// Add survey info to an alignable
    void addSurveyInfo(Alignable*);
    /// read in survey records
    void readInSurveyRcds(const edm::EventSetup&);

    unsigned int        theSurveyIndex;
    const Alignments*   theSurveyValues;
    const SurveyErrors* theSurveyErrors;


    //            std::unique_ptr<AlignmentAlgorithmBase>     theAlignmentAlgo;
    //            std::unique_ptr<AlignmentParameterStore>    theAlignmentParameterStore;
    //std::vector<std::unique_ptr<IntegratedCalibrationBase>> theCalibrations;

                AlignmentAlgorithmBase*     theAlignmentAlgo;
                AlignmentParameterStore*    theAlignmentParameterStore;
    std::vector<IntegratedCalibrationBase*> theCalibrations;

    edm::ParameterSet theParameterSet;
    AlignableExtras*  theAlignableExtras;
    AlignableTracker* theAlignableTracker;

    //std::shared_ptr<TrackerGeometry> theTracker;
    boost::shared_ptr<TrackerGeometry> theTracker;

    Alignments* globalPositions;

          int  nevent_;
    const bool doTracker_;

    const int  stNFixAlignables_;
    const double stRandomShift_;
    const double stRandomRotation_;

    const bool doMisalignmentScenario_;
    const bool saveToDB;
    const bool saveApeToDB;
    const bool saveDeformationsToDB;
    const bool applyDbAlignment_;
    const bool checkDbAlignmentValidity_;
    const bool useExtras_;
    const bool useSurvey_;

    const edm::InputTag tjTkAssociationMapTag_; // map with tracks/trajectories
    const edm::InputTag beamSpotTag_;           // beam spot
    const edm::InputTag tkLasBeamTag_;          // LAS beams in edm::Run (ignore if empty)
    const edm::InputTag clusterValueMapTag_;    // ValueMap containing associtaion cluster-flag



    edm::ESWatcher<TrackerSurveyRcd> watchTkSurveyRcd_;
    edm::ESWatcher<TrackerSurveyErrorRcd> watchTkSurveyErrRcd_;
};

#endif /* ALIGNMENT_COMMONALIGNMENTPRODUCER_PLUGINS_ALIGNMENTPRODUCERFORPCL_H_ */
