#ifndef L1Trigger_DTTriggerPhase2_MuonPathAnalyzerInChamber_h
#define L1Trigger_DTTriggerPhase2_MuonPathAnalyzerInChamber_h

#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzer.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================
namespace {
  typedef std::array<cmsdt::LATERAL_CASES, cmsdt::NUM_LAYERS_2SL> TLateralities;
}  // namespace
// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathAnalyzerInChamber : public MuonPathAnalyzer {
public:
  // Constructors and destructor
  MuonPathAnalyzerInChamber(const edm::ParameterSet &pset,
                            edm::ConsumesCollector &iC,
                            std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer);
  ~MuonPathAnalyzerInChamber() override;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           std::vector<cmsdt::metaPrimitive> &metaPrimitives) override{};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           std::vector<lat_vector> &lateralities,
           std::vector<cmsdt::metaPrimitive> &metaPrimitives) override{};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> &inMPaths,
           std::vector<cmsdt::metaPrimitive> &outMPaths) override{};
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           MuonPathPtrs &outMPath) override;

  void finish() override;

  // Other public methods
  void setBxTolerance(int t) { bxTolerance_ = t; };
  void setMinHits4Fit(int h) { minHits4Fit_ = h; };
  void setChiSquareThreshold(float ch2Thr) { chiSquareThreshold_ = ch2Thr; };
  void setMinimumQuality(cmsdt::MP_QUALITY q) {
    if (minQuality_ >= cmsdt::LOWQ)
      minQuality_ = q;
  };

  int bxTolerance(void) { return bxTolerance_; };
  int minHits4Fit(void) { return minHits4Fit_; };
  cmsdt::MP_QUALITY minQuality(void) { return minQuality_; };

  bool hasPosRF(int wh, int sec) { return wh > 0 || (wh == 0 && sec % 4 > 1); };

  // Public attributes
  DTGeometry const *dtGeo_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;

  //shift
  std::map<int, float> shiftinfo_;

private:
  // Private methods
  void analyze(MuonPathPtr &inMPath, MuonPathPtrs &outMPaths);

  void setCellLayout(MuonPathPtr &mpath);
  void buildLateralities(MuonPathPtr &mpath);
  void setLateralitiesInMP(MuonPathPtr &mpath, TLateralities lat);
  void setWirePosAndTimeInMP(MuonPathPtr &mpath);
  void calculateFitParameters(MuonPathPtr &mpath,
                              TLateralities lat,
                              int present_layer[cmsdt::NUM_LAYERS_2SL],
                              int &lat_added);

  void evaluateQuality(MuonPathPtr &mPath);
  int totalNumValLateralities_;
  std::vector<TLateralities> lateralities_;
  std::vector<cmsdt::LATQ_TYPE> latQuality_;

  const bool debug_;
  double chi2Th_;
  edm::FileInPath shift_filename_;
  int bxTolerance_;
  cmsdt::MP_QUALITY minQuality_;
  float chiSquareThreshold_;
  short minHits4Fit_;
  int cellLayout_[cmsdt::NUM_LAYERS_2SL];
  bool splitPathPerSL_;

  // global coordinates
  std::shared_ptr<GlobalCoordsObtainer> globalcoordsobtainer_;
};

#endif
