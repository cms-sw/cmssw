#ifndef Alignment_HIPAlignmentAlgorithm_HIPAlignmentAlgorithm_h
#define Alignment_HIPAlignmentAlgorithm_HIPAlignmentAlgorithm_h

#include <vector>
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Riostream.h"

#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
#include "Utilities/General/interface/ClassName.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformationFactory.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Alignment/HIPAlignmentAlgorithm/interface/HIPMonitorConfig.h"
#include "Alignment/HIPAlignmentAlgorithm/interface/HIPAlignableSpecificParameters.h"
#include "TFormula.h"

class TFile;
class TTree;

class HIPAlignmentAlgorithm : public AlignmentAlgorithmBase {
public:
  /// Constructor
  HIPAlignmentAlgorithm(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  /// Destructor
  ~HIPAlignmentAlgorithm() override{};

  /// Call at beginning of job
  void initialize(const edm::EventSetup& setup,
                  AlignableTracker* tracker,
                  AlignableMuon* muon,
                  AlignableExtras* extras,
                  AlignmentParameterStore* store) override;

  /// Call at end of job
  void terminate(const edm::EventSetup& setup) override;

  /// Called at start of new loop
  void startNewLoop(void) override;

  /// Run the algorithm
  void run(const edm::EventSetup& setup, const EventInfo& eventInfo) override;

private:
  bool processHit1D(const AlignableDetOrUnitPtr& alidet,
                    const Alignable* ali,
                    const HIPAlignableSpecificParameters* alispecifics,
                    const TrajectoryStateOnSurface& tsos,
                    const TrackingRecHit* hit,
                    double hitwt);

  bool processHit2D(const AlignableDetOrUnitPtr& alidet,
                    const Alignable* ali,
                    const HIPAlignableSpecificParameters* alispecifics,
                    const TrajectoryStateOnSurface& tsos,
                    const TrackingRecHit* hit,
                    double hitwt);

  int readIterationFile(std::string filename);
  void writeIterationFile(std::string filename, int iter);
  void setAlignmentPositionError(void);
  double calcAPE(double* par, int iter, int function);
  void bookRoot(void);
  void fillAlignablesMonitor(const edm::EventSetup& setup);
  bool calcParameters(Alignable* ali, int setDet, double start, double step);
  void collector(void);
  void collectMonitorTrees(const std::vector<std::string>& filenames);

  HIPAlignableSpecificParameters* findAlignableSpecs(const Alignable* ali);

  // private data members
  const edm::ESGetToken<TrackerTopology, IdealGeometryRecord> topoToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken2_;

  std::unique_ptr<AlignableObjectId> alignableObjectId_;
  AlignmentParameterStore* theAlignmentParameterStore;
  align::Alignables theAlignables;
  std::unique_ptr<AlignableNavigator> theAlignableDetAccessor;

  AlignmentIORoot theIO;
  int ioerr;
  int theIteration;

  // steering parameters

  // verbosity flag
  const bool verbose;
  // Monitor configuration
  HIPMonitorConfig theMonitorConfig;
  const bool doTrackHitMonitoring;
  // names of IO root files
  std::string outfile2, outpath, suvarfilecore, suvarfile, sparameterfile;
  std::string struefile, smisalignedfile, salignedfile, siterationfile, ssurveyfile;

  bool themultiIOV;
  std::vector<unsigned> theIOVrangeSet;

  // alignment position error parameters
  bool theApplyAPE;
  std::vector<edm::ParameterSet> theAPEParameterSet;
  std::vector<std::pair<align::Alignables, std::vector<double> > > theAPEParameters;

  // Default alignment specifications
  // - min number of hits on alignable to calc parameters
  // - max allowed rel error on parameter (else not used)
  // - max allowed pull (residual / uncertainty) on a hit used in alignment
  HIPAlignableSpecificParameters defaultAlignableSpecs;

  bool theApplyCutsPerComponent;
  std::vector<edm::ParameterSet> theCutsPerComponent;
  std::vector<HIPAlignableSpecificParameters> theAlignableSpecifics;

  // collector mode (parallel processing)
  bool isCollector;
  int theCollectorNJobs;
  std::string theCollectorPath;
  int theDataGroup;  // The data type specified in the cfg
  bool trackPs, trackWt, IsCollision, uniEta, rewgtPerAli;
  std::string uniEtaFormula;
  double Scale, cos_cut, col_cut;
  std::vector<double> SetScanDet;

  std::unique_ptr<TFormula> theEtaFormula;

  const std::vector<std::string> surveyResiduals_;
  std::vector<align::StructureType> theLevels;  // for survey residuals

  // root tree variables
  TFile* theTrackHitMonitorIORootFile;
  TTree* theTrackMonitorTree;  // event-wise tree
  TTree* theHitMonitorTree;    // hit-wise tree
  TFile* theAlignablesMonitorIORootFile;
  TTree* theAlignablesMonitorTree;  // alignable-wise tree
  TFile* theSurveyIORootFile;
  TTree* theSurveyTree;  // survey tree

  // common variables for monitor trees
  int m_datatype;

  // variables for alignable-wise tree
  align::ID m2_Id;
  align::StructureType m2_ObjId;
  int m2_Nhit, m2_Type, m2_Layer;
  int m2_datatype;
  float m2_Xpos, m2_Ypos, m2_Zpos;
  SurfaceDeformationFactory::Type m2_dtype;
  unsigned int m2_nsurfdef;
  std::vector<float> m2_surfDef;

  // variables for survey tree
  align::ID m3_Id;
  align::StructureType m3_ObjId;
  float m3_par[6];
};

#endif
