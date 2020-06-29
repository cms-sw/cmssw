#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzerPerSL.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;
// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAnalyzerPerSL::MuonPathAnalyzerPerSL(const ParameterSet &pset, edm::ConsumesCollector &iC)
    : MuonPathAnalyzer(pset, iC),
      bxTolerance_(30),
      minQuality_(LOWQGHOST),
      chiSquareThreshold_(50),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      chi2Th_(pset.getUntrackedParameter<double>("chi2Th")),
      tanPhiTh_(pset.getUntrackedParameter<double>("tanPhiTh")),
      use_LSB_(pset.getUntrackedParameter<bool>("use_LSB")),
      tanPsi_precision_(pset.getUntrackedParameter<double>("tanPsi_precision")),
      x_precision_(pset.getUntrackedParameter<double>("x_precision")) {
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "MuonPathAnalyzer: constructor";

  setChiSquareThreshold(chi2Th_ * 100.);

  //shift
  int rawId;
  shift_filename_ = pset.getParameter<edm::FileInPath>("shift_filename");
  std::ifstream ifin3(shift_filename_.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathAnalyzerPerSL::MuonPathAnalyzerPerSL() -  Cannot find " << shift_filename_.fullPath();
  }
  while (ifin3.good()) {
    ifin3 >> rawId >> shift;
    shiftinfo_[rawId] = shift;
  }

  chosen_sl_ = pset.getUntrackedParameter<int>("trigger_with_sl");

  if (chosen_sl_ != 1 && chosen_sl_ != 3 && chosen_sl_ != 4) {
    LogDebug("MuonPathAnalyzerPerSL") << "chosen sl must be 1,3 or 4(both superlayers)";
    assert(chosen_sl_ != 1 && chosen_sl_ != 3 && chosen_sl_ != 4);  //4 means run using the two superlayers
  }

  dtGeomH = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

MuonPathAnalyzerPerSL::~MuonPathAnalyzerPerSL() {
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "MuonPathAnalyzer: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAnalyzerPerSL::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "MuonPathAnalyzerPerSL::initialiase";

  const MuonGeometryRecord &geom = iEventSetup.get<MuonGeometryRecord>();
  dtGeo_ = &geom.get(dtGeomH);
}

void MuonPathAnalyzerPerSL::run(edm::Event &iEvent,
                                const edm::EventSetup &iEventSetup,
                                MuonPathPtrs &muonpaths,
                                std::vector<metaPrimitive> &metaPrimitives) {
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "MuonPathAnalyzerPerSL: run";

  // fit per SL (need to allow for multiple outputs for a single mpath)
  for (auto &muonpath : muonpaths) {
    analyze(muonpath, metaPrimitives);
  }
}

void MuonPathAnalyzerPerSL::finish() {
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "MuonPathAnalyzer: finish";
};

constexpr int MuonPathAnalyzerPerSL::LAYER_ARRANGEMENTS_[NUM_LAYERS][NUM_CELL_COMB] = {
    {0, 1, 2},
    {1, 2, 3},  // Consecutive groups
    {0, 1, 3},
    {0, 2, 3}  // Non-consecutive groups
};

//------------------------------------------------------------------
//--- Métodos privados
//------------------------------------------------------------------

void MuonPathAnalyzerPerSL::analyze(MuonPathPtr &inMPath, std::vector<metaPrimitive> &metaPrimitives) {
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t starts";

  // LOCATE MPATH
  int selected_Id = 0;
  if (inMPath->primitive(0)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(0)->cameraId();
  else if (inMPath->primitive(1)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(1)->cameraId();
  else if (inMPath->primitive(2)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(2)->cameraId();
  else if (inMPath->primitive(3)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(3)->cameraId();

  DTLayerId thisLId(selected_Id);
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "Building up MuonPathSLId from rawId in the Primitive";
  DTSuperLayerId MuonPathSLId(thisLId.wheel(), thisLId.station(), thisLId.sector(), thisLId.superLayer());
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "The MuonPathSLId is" << MuonPathSLId;

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL")
        << "DTp2:analyze \t\t\t\t In analyze function checking if inMPath->isAnalyzable() " << inMPath->isAnalyzable();

  if (chosen_sl_ < 4 && thisLId.superLayer() != chosen_sl_)
    return;  // avoid running when mpath not in chosen SL (for 1SL fitting)

  auto mPath = std::make_shared<MuonPath>(inMPath);

  if (mPath->isAnalyzable()) {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t yes it is analyzable " << mPath->isAnalyzable();
    setCellLayout(mPath->cellLayout());
    evaluatePathQuality(mPath);
  } else {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t no it is NOT analyzable " << mPath->isAnalyzable();
    return;
  }

  int wi[8], tdc[8], lat[8];
  DTPrimitivePtr Prim0(mPath->primitive(0));
  wi[0] = Prim0->channelId();
  tdc[0] = Prim0->tdcTimeStamp();
  DTPrimitivePtr Prim1(mPath->primitive(1));
  wi[1] = Prim1->channelId();
  tdc[1] = Prim1->tdcTimeStamp();
  DTPrimitivePtr Prim2(mPath->primitive(2));
  wi[2] = Prim2->channelId();
  tdc[2] = Prim2->tdcTimeStamp();
  DTPrimitivePtr Prim3(mPath->primitive(3));
  wi[3] = Prim3->channelId();
  tdc[3] = Prim3->tdcTimeStamp();
  for (int i = 4; i < 8; i++) {
    wi[i] = -1;
    tdc[i] = -1;
    lat[i] = -1;
  }

  DTWireId wireId(MuonPathSLId, 2, 1);

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t checking if it passes the min quality cut "
                                      << mPath->quality() << ">" << minQuality_;
  if (mPath->quality() >= minQuality_) {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t min quality achievedCalidad: " << mPath->quality();
    for (int i = 0; i <= 3; i++) {
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL")
            << "DTp2:analyze \t\t\t\t  Capa: " << mPath->primitive(i)->layerId()
            << " Canal: " << mPath->primitive(i)->channelId() << " TDCTime: " << mPath->primitive(i)->tdcTimeStamp();
    }
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t Starting lateralities loop, totalNumValLateralities: "
                                        << totalNumValLateralities_;

    double best_chi2 = 99999.;
    double chi2_jm_tanPhi = 999;
    double chi2_jm_x = -1;
    double chi2_jm_t0 = -1;
    double chi2_phi = -1;
    double chi2_phiB = -1;
    double chi2_chi2 = -1;
    int chi2_quality = -1;
    int bestLat[8];
    for (int i = 0; i < 8; i++) {
      bestLat[i] = -1;
    }

    for (int i = 0; i < totalNumValLateralities_; i++) {  //here
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t laterality #- " << i;
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " checking quality:";
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL")
            << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " checking mPath Quality=" << mPath->quality();
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL")
            << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " latQuality_[i].val=" << latQuality_[i].valid;
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " before if:";

      if (latQuality_[i].valid and
          (((mPath->quality() == HIGHQ or mPath->quality() == HIGHQGHOST) and latQuality_[i].quality == HIGHQ) or
           ((mPath->quality() == LOWQ or mPath->quality() == LOWQGHOST) and latQuality_[i].quality == LOWQ))) {
        if (debug_)
          LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " inside if";
        mPath->setBxTimeValue(latQuality_[i].bxValue);
        if (debug_)
          LogDebug("MuonPathAnalyzerPerSL")
              << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " settingLateralCombination";
        mPath->setLateralComb(lateralities_[i]);
        if (debug_)
          LogDebug("MuonPathAnalyzerPerSL")
              << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " done settingLateralCombination";

        // Clonamos el objeto analizado.
        auto mpAux = std::make_shared<MuonPath>(mPath);
        lat[0] = mpAux->lateralComb()[0];
        lat[1] = mpAux->lateralComb()[1];
        lat[2] = mpAux->lateralComb()[2];
        lat[3] = mpAux->lateralComb()[3];

        int wiOk[NUM_LAYERS], tdcOk[NUM_LAYERS], latOk[NUM_LAYERS];
        for (int lay = 0; lay < 4; lay++) {
          if (latQuality_[i].invalidateHitIdx == lay) {
            wiOk[lay] = -1;
            tdcOk[lay] = -1;
            latOk[lay] = -1;
          } else {
            wiOk[lay] = wi[lay];
            tdcOk[lay] = tdc[lay];
            latOk[lay] = lat[lay];
          }
        }

        int idxHitNotValid = latQuality_[i].invalidateHitIdx;
        if (idxHitNotValid >= 0) {
          auto dtpAux = std::make_shared<DTPrimitive>();
          mpAux->setPrimitive(dtpAux, idxHitNotValid);
        }

        if (debug_)
          LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t  calculating parameters ";
        calculatePathParameters(mpAux);
        /* 
		 * After calculating the parameters, and if it is a 4-hit fit,
		 * if the resultant chi2 is higher than the programmed threshold, 
		 * the mpath is eliminated and we go to the next element
		 */
        if ((mpAux->quality() == HIGHQ or mpAux->quality() == HIGHQGHOST) &&
            mpAux->chiSquare() > chiSquareThreshold_) {  //check this if!!!
          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL")
                << "DTp2:analyze \t\t\t\t\t  HIGHQ or HIGHQGHOST but min chi2 or Q test not satisfied ";
        } else {
          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t  inside else, returning values: ";
          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t  BX Time = " << mpAux->bxTimeValue();
          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t  BX Id   = " << mpAux->bxNumId();
          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t  XCoor   = " << mpAux->horizPos();
          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t  tan(Phi)= " << mpAux->tanPhi();
          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t  chi2= " << mpAux->chiSquare();
          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t  lateralities = "
                                              << " " << mpAux->lateralComb()[0] << " " << mpAux->lateralComb()[1] << " "
                                              << mpAux->lateralComb()[2] << " " << mpAux->lateralComb()[3];

          DTChamberId ChId(MuonPathSLId.wheel(), MuonPathSLId.station(), MuonPathSLId.sector());

          double jm_tanPhi = -1. * mpAux->tanPhi();  //testing with this line
          if (use_LSB_)
            jm_tanPhi = floor(jm_tanPhi / tanPsi_precision_) * tanPsi_precision_;
          double jm_x =
              (((double)mpAux->horizPos()) / 10.) + x_precision_ * (round(shiftinfo_[wireId.rawId()] / x_precision_));
          if (use_LSB_)
            jm_x = ((double)round(((double)jm_x) / x_precision_)) * x_precision_;
          //changing to chamber frame or reference:
          double jm_t0 = mpAux->bxTimeValue();
          int quality = mpAux->quality();

          //computing phi and phiB
          double z = 0;
          double z1 = Z_POS_SL;
          double z3 = -1. * z1;
          if (ChId.station() == 3 or ChId.station() == 4) {
            z1 = z1 + Z_SHIFT_MB4;
            z3 = z3 + Z_SHIFT_MB4;
          } else if (MuonPathSLId.superLayer() == 1)
            z = z1;
          else if (MuonPathSLId.superLayer() == 3)
            z = z3;

          GlobalPoint jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(LocalPoint(jm_x, 0., z));
          int thisec = MuonPathSLId.sector();
          if (thisec == 13)
            thisec = 4;
          if (thisec == 14)
            thisec = 10;
          double phi = jm_x_cmssw_global.phi() - PHI_CONV * (thisec - 1);
          double psi = atan(jm_tanPhi);
          double phiB = hasPosRF(MuonPathSLId.wheel(), MuonPathSLId.sector()) ? psi - phi : -psi - phi;
          double chi2 = mpAux->chiSquare() * 0.01;  //in cmssw we need cm, 1 cm^2 = 100 mm^2

          if (debug_)
            LogDebug("MuonPathAnalyzerPerSL")
                << "DTp2:analyze \t\t\t\t\t\t\t\t  pushing back metaPrimitive at x=" << jm_x << " tanPhi:" << jm_tanPhi
                << " t0:" << jm_t0;

          if (mpAux->quality() == HIGHQ or
              mpAux->quality() == HIGHQGHOST) {  //keep only the values with the best chi2 among lateralities
            if ((chi2 < best_chi2) && (std::abs(jm_tanPhi) <= tanPhiTh_)) {
              chi2_jm_tanPhi = jm_tanPhi;
              chi2_jm_x = (mpAux->horizPos() / 10.) + shiftinfo_[wireId.rawId()];
              chi2_jm_t0 = mpAux->bxTimeValue();
              chi2_phi = phi;
              chi2_phiB = phiB;
              chi2_chi2 = chi2;
              best_chi2 = chi2;
              chi2_quality = mpAux->quality();
              for (int i = 0; i < 4; i++) {
                bestLat[i] = lat[i];
              }
            }
          } else if (std::abs(jm_tanPhi) <=
                     tanPhiTh_) {  //write the metaprimitive in case no HIGHQ or HIGHQGHOST and tanPhi range
            if (debug_)
              LogDebug("MuonPathAnalyzerPerSL")
                  << "DTp2:analyze \t\t\t\t\t\t\t\t  pushing back metaprimitive no HIGHQ or HIGHQGHOST";
            metaPrimitives.emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                       jm_t0,
                                                       jm_x,
                                                       jm_tanPhi,
                                                       phi,
                                                       phiB,
                                                       chi2,
                                                       quality,
                                                       wiOk[0],
                                                       tdcOk[0],
                                                       latOk[0],
                                                       wiOk[1],
                                                       tdcOk[1],
                                                       latOk[1],
                                                       wiOk[2],
                                                       tdcOk[2],
                                                       latOk[2],
                                                       wiOk[3],
                                                       tdcOk[3],
                                                       latOk[3],
                                                       wi[4],
                                                       tdc[4],
                                                       lat[4],
                                                       wi[5],
                                                       tdc[5],
                                                       lat[5],
                                                       wi[6],
                                                       tdc[6],
                                                       lat[6],
                                                       wi[7],
                                                       tdc[7],
                                                       lat[7],
                                                       -1}));
            if (debug_)
              LogDebug("MuonPathAnalyzerPerSL")
                  << "DTp2:analyze \t\t\t\t\t\t\t\t  done pushing back metaprimitive no HIGHQ or HIGHQGHOST";
          }
        }
      } else {
        if (debug_)
          LogDebug("MuonPathAnalyzerPerSL")
              << "DTp2:analyze \t\t\t\t\t\t\t\t  latQuality_[i].valid and (((mPath->quality()==HIGHQ or "
                 "mPath->quality()==HIGHQGHOST) and latQuality_[i].quality==HIGHQ) or  ((mPath->quality() "
                 "== LOWQ or mPath->quality()==LOWQGHOST) and latQuality_[i].quality==LOWQ)) not passed";
      }
    }
    if (chi2_jm_tanPhi != 999 and std::abs(chi2_jm_tanPhi) < tanPhiTh_) {  //
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t\t\t\t  pushing back best chi2 metaPrimitive";
      metaPrimitives.emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                 chi2_jm_t0,
                                                 chi2_jm_x,
                                                 chi2_jm_tanPhi,
                                                 chi2_phi,
                                                 chi2_phiB,
                                                 chi2_chi2,
                                                 chi2_quality,
                                                 wi[0],
                                                 tdc[0],
                                                 bestLat[0],
                                                 wi[1],
                                                 tdc[1],
                                                 bestLat[1],
                                                 wi[2],
                                                 tdc[2],
                                                 bestLat[2],
                                                 wi[3],
                                                 tdc[3],
                                                 bestLat[3],
                                                 wi[4],
                                                 tdc[4],
                                                 bestLat[4],
                                                 wi[5],
                                                 tdc[5],
                                                 bestLat[5],
                                                 wi[6],
                                                 tdc[6],
                                                 bestLat[6],
                                                 wi[7],
                                                 tdc[7],
                                                 bestLat[7],
                                                 -1}));
    }
  }
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t finishes";
}

void MuonPathAnalyzerPerSL::setCellLayout(const int layout[NUM_LAYERS]) {
  memcpy(cellLayout_, layout, 4 * sizeof(int));

  buildLateralities();
}

/**
 * For a given 4-cell combination (one per layer), all the possible lateralities 
 * combinations that are compatible with a straight line are generated. 
 */
void MuonPathAnalyzerPerSL::buildLateralities(void) {
  LATERAL_CASES(*validCase)[NUM_LAYERS], sideComb[NUM_LAYERS];

  totalNumValLateralities_ = 0;
  /* We generate all the possible lateralities combination for a given group 
     of cells */
  for (int lowLay = LEFT; lowLay <= RIGHT; lowLay++)
    for (int midLowLay = LEFT; midLowLay <= RIGHT; midLowLay++)
      for (int midHigLay = LEFT; midHigLay <= RIGHT; midHigLay++)
        for (int higLay = LEFT; higLay <= RIGHT; higLay++) {
          sideComb[0] = static_cast<LATERAL_CASES>(lowLay);
          sideComb[1] = static_cast<LATERAL_CASES>(midLowLay);
          sideComb[2] = static_cast<LATERAL_CASES>(midHigLay);
          sideComb[3] = static_cast<LATERAL_CASES>(higLay);

          /* If a laterality combination is valid, we store it  */
          if (isStraightPath(sideComb)) {
            validCase = lateralities_ + totalNumValLateralities_;
            memcpy(validCase, sideComb, 4 * sizeof(LATERAL_CASES));

            latQuality_[totalNumValLateralities_].valid = false;
            latQuality_[totalNumValLateralities_].bxValue = 0;
            latQuality_[totalNumValLateralities_].quality = NOPATH;
            latQuality_[totalNumValLateralities_].invalidateHitIdx = -1;

            totalNumValLateralities_++;
          }
        }
}

/**
 * This method checks whether a given combination conform a straight line or not
 */
bool MuonPathAnalyzerPerSL::isStraightPath(LATERAL_CASES sideComb[NUM_LAYERS]) {
  return true;  //trying with all lateralities to be confirmed

  int i, ajustedLayout[NUM_LAYERS], pairDiff[3], desfase[3];

  for (i = 0; i <= 3; i++)
    ajustedLayout[i] = cellLayout_[i] + sideComb[i];
  for (i = 0; i <= 2; i++)
    pairDiff[i] = ajustedLayout[i + 1] - ajustedLayout[i];
  for (i = 0; i <= 1; i++)
    desfase[i] = abs(pairDiff[i + 1] - pairDiff[i]);
  desfase[2] = abs(pairDiff[2] - pairDiff[0]);
  bool resultado = (desfase[0] > 1 or desfase[1] > 1 or desfase[2] > 1);

  return (!resultado);
}
void MuonPathAnalyzerPerSL::evaluatePathQuality(MuonPathPtr &mPath) {
  int totalHighQ = 0, totalLowQ = 0;

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL")
        << "DTp2:evaluatePathQuality \t\t\t\t\t En evaluatePathQuality Evaluando PathQ. Celda base: "
        << mPath->baseChannelId();
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:evaluatePathQuality \t\t\t\t\t Total lateralidades: "
                                      << totalNumValLateralities_;

  mPath->setQuality(NOPATH);

  for (int latIdx = 0; latIdx < totalNumValLateralities_; latIdx++) {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:evaluatePathQuality \t\t\t\t\t Analizando combinacion de lateralidad: "
                                        << lateralities_[latIdx][0] << " " << lateralities_[latIdx][1] << " "
                                        << lateralities_[latIdx][2] << " " << lateralities_[latIdx][3];

    evaluateLateralQuality(latIdx, mPath, &(latQuality_[latIdx]));

    if (latQuality_[latIdx].quality == HIGHQ) {
      totalHighQ++;
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:evaluatePathQuality \t\t\t\t\t\t Lateralidad HIGHQ";
    }
    if (latQuality_[latIdx].quality == LOWQ) {
      totalLowQ++;
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:evaluatePathQuality \t\t\t\t\t\t Lateralidad LOWQ";
    }
  }
  /*
   * Quality stablishment 
   */
  if (totalHighQ == 1) {
    mPath->setQuality(HIGHQ);
  } else if (totalHighQ > 1) {
    mPath->setQuality(HIGHQGHOST);
  } else if (totalLowQ == 1) {
    mPath->setQuality(LOWQ);
  } else if (totalLowQ > 1) {
    mPath->setQuality(LOWQGHOST);
  }
}

void MuonPathAnalyzerPerSL::evaluateLateralQuality(int latIdx, MuonPathPtr &mPath, LATQ_TYPE *latQuality) {
  int layerGroup[3];
  LATERAL_CASES sideComb[3];
  PARTIAL_LATQ_TYPE latQResult[NUM_LAYERS] = {{false, 0}, {false, 0}, {false, 0}, {false, 0}};

  // Default values.
  latQuality->valid = false;
  latQuality->bxValue = 0;
  latQuality->quality = NOPATH;
  latQuality->invalidateHitIdx = -1;

  /* If, for a given laterality combination, the two consecutive 3-layer combinations
     were a valid track, we will have found a right high-quality track, hence 
     it will be unnecessary to check the remaining 2 combinations. 
     In order to mimic the FPGA behaviour, we build a code that analyzes the 4 combinations
     with an additional logic to discriminate the final quality of the track
  */
  for (int i = 0; i <= 3; i++) {
    memcpy(layerGroup, LAYER_ARRANGEMENTS_[i], 3 * sizeof(int));

    // Pick the laterality combination for each cell
    for (int j = 0; j < 3; j++)
      sideComb[j] = lateralities_[latIdx][layerGroup[j]];

    validate(sideComb, layerGroup, mPath, &(latQResult[i]));
  }
  /*
    Impose the condition for a complete laterality combination, that all combinations
    should give the same BX vale to give a consistent track. 
    */
  if (!sameBXValue(latQResult)) {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL")
          << "DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad DESCARTADA. Tolerancia de BX excedida";
    return;
  }

  // two complementary valid tracks => full muon track.
  if ((latQResult[0].latQValid && latQResult[1].latQValid) or (latQResult[0].latQValid && latQResult[2].latQValid) or
      (latQResult[0].latQValid && latQResult[3].latQValid) or (latQResult[1].latQValid && latQResult[2].latQValid) or
      (latQResult[1].latQValid && latQResult[3].latQValid) or (latQResult[2].latQValid && latQResult[3].latQValid)) {
    latQuality->valid = true;

    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t\t Valid BXs";
    long int sumBX = 0, numValid = 0;
    for (int i = 0; i <= 3; i++) {
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:analyze \t\t\t\t\t\t "
                                          << "[" << latQResult[i].bxValue << "," << latQResult[i].latQValid << "]";
      if (latQResult[i].latQValid) {
        sumBX += latQResult[i].bxValue;
        numValid++;
      }
    }

    // mean time of all lateralities.
    if (numValid == 1)
      latQuality->bxValue = sumBX;
    else if (numValid == 2)
      latQuality->bxValue = (sumBX * (MEANTIME_2LAT)) / std::pow(2, 15);
    else if (numValid == 3)
      latQuality->bxValue = (sumBX * (MEANTIME_3LAT)) / std::pow(2, 15);
    else if (numValid == 4)
      latQuality->bxValue = (sumBX * (MEANTIME_4LAT)) / std::pow(2, 15);

    latQuality->quality = HIGHQ;

    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad ACEPTADA. HIGHQ.";
  } else {
    if (latQResult[0].latQValid or latQResult[1].latQValid or latQResult[2].latQValid or latQResult[3].latQValid) {
      latQuality->valid = true;
      latQuality->quality = LOWQ;
      for (int i = 0; i < 4; i++)
        if (latQResult[i].latQValid) {
          latQuality->bxValue = latQResult[i].bxValue;
          latQuality->invalidateHitIdx = omittedHit(i);
          break;
        }

      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad ACEPTADA. LOWQ.";
    } else {
      if (debug_)
        LogDebug("MuonPathAnalyzerPerSL") << "DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad DESCARTADA. NOPATH.";
    }
  }
}

/**
 * Validate, for a layer combination (3), cells and lateralities, if the temporal values
 * fullfill the mean-timer criteria. 
 */
void MuonPathAnalyzerPerSL::validate(LATERAL_CASES sideComb[3],
                                     int layerIndex[3],
                                     MuonPathPtr &mPath,
                                     PARTIAL_LATQ_TYPE *latq) {
  // Valor por defecto.
  latq->bxValue = 0;
  latq->latQValid = false;

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:validate \t\t\t\t\t\t\t In validate, checking muon path for layers: "
                                      << layerIndex[0] << "/" << layerIndex[1] << "/" << layerIndex[2];

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:validate \t\t\t\t\t\t\t Partial lateralities: " << sideComb[0] << "/"
                                      << sideComb[1] << "/" << sideComb[2];

  int validCells = 0;
  for (int j = 0; j < 3; j++)
    if (mPath->primitive(layerIndex[j])->isValidTime())
      validCells++;

  if (validCells != 3) {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:validate \t\t\t\t\t\t\t There is no valid cells.";
    return;
  }

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:validate \t\t\t\t\t\t\t TDC values: "
                                      << mPath->primitive(layerIndex[0])->tdcTimeStamp() << "/"
                                      << mPath->primitive(layerIndex[1])->tdcTimeStamp() << "/"
                                      << mPath->primitive(layerIndex[2])->tdcTimeStamp() << ".";

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:validate \t\t\t\t\t\t\t Valid TIMES: "
                                      << mPath->primitive(layerIndex[0])->isValidTime() << "/"
                                      << mPath->primitive(layerIndex[1])->isValidTime() << "/"
                                      << mPath->primitive(layerIndex[2])->isValidTime() << ".";

  /* Vertical distances  */
  int dVertMI = layerIndex[1] - layerIndex[0];
  int dVertSM = layerIndex[2] - layerIndex[1];

  /* Horizontal distances between lower/middle and middle/upper cells */
  int dHorzMI = cellLayout_[layerIndex[1]] - cellLayout_[layerIndex[0]];
  int dHorzSM = cellLayout_[layerIndex[2]] - cellLayout_[layerIndex[1]];

  /* Pair index of layers that we are using 
     SM => Upper + Middle
     MI => Middle + Lower
     We use pointers to simplify the code */
  int *layPairSM = &layerIndex[1];
  int *layPairMI = &layerIndex[0];

  /* Pair combination of cells to compose the equation. */
  LATERAL_CASES smSides[2], miSides[2];

  /* Considering the index 0 of "sideComb" the laterality of the lower cells is stored,
     we extract the laterality combiantion for SM and MI pairs */

  memcpy(smSides, &sideComb[1], 2 * sizeof(LATERAL_CASES));

  memcpy(miSides, &sideComb[0], 2 * sizeof(LATERAL_CASES));

  long int bxValue = 0;
  int coefsAB[2] = {0, 0}, coefsCD[2] = {0, 0};
  /* It's neccesary to be careful with that pointer's indirection. We need to
     retrieve the lateral coeficientes (+-1) from the lower/middle and
     middle/upper cell's lateral combinations. They are needed to evaluate the
     existance of a possible BX value, following it's calculation equation */
  lateralCoeficients(miSides, coefsAB);
  lateralCoeficients(smSides, coefsCD);

  /* Each of the summs of the 'coefsCD' & 'coefsAB' give always as results 0, +-2
   */

  int denominator = dVertMI * (coefsCD[1] + coefsCD[0]) - dVertSM * (coefsAB[1] + coefsAB[0]);

  if (denominator == 0) {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:validate \t\t\t\t\t\t\t Imposible to calculate BX.";
    return;
  }

  long int sumA = (long int)floor(MAXDRIFT * (dVertMI * dHorzSM - dVertSM * dHorzMI));
  long int numerator =
      (sumA + dVertMI * eqMainBXTerm(smSides, layPairSM, mPath) - dVertSM * eqMainBXTerm(miSides, layPairMI, mPath));

  // These magic numbers are for doing divisions in the FW. 
  // These divisions are done with a precision of 18bits.
  if (denominator == -1*DENOM_TYPE1)
    bxValue = (numerator * (-1*DIVISION_HELPER1)) / std::pow(2, NBITS);
  else if (denominator == -1*DENOM_TYPE2)
    bxValue = (numerator * (-1*DIVISION_HELPER2)) / std::pow(2, NBITS);
  else if (denominator == -1*DENOM_TYPE3)
    bxValue = (numerator * (-1*DIVISION_HELPER3)) / std::pow(2, NBITS);
  else if (denominator == DENOM_TYPE3)
    bxValue = (numerator * (DIVISION_HELPER3)) / std::pow(2, NBITS);
  else if (denominator == DENOM_TYPE2)
    bxValue = (numerator * (DIVISION_HELPER2)) / std::pow(2, NBITS);
  else if (denominator == DENOM_TYPE1)
    bxValue = (numerator * (DIVISION_HELPER1)) / std::pow(2, NBITS);
  else
    LogDebug("MuonPathAnalyzerPerSL") << "Different!";
  if (bxValue < 0) {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL") << "DTp2:validate \t\t\t\t\t\t\t No-valid combination. Negative BX.";
    return;
  }

  for (int i = 0; i < 3; i++)
    if (mPath->primitive(layerIndex[i])->isValidTime()) {
      int diffTime = mPath->primitive(layerIndex[i])->tdcTimeStampNoOffset() - bxValue;

      if (diffTime <= 0 or diffTime > round(MAXDRIFT)) {
        if (debug_)
          LogDebug("MuonPathAnalyzerPerSL")
              << "DTp2:validate \t\t\t\t\t\t\t Invalid BX value. at least one crazt TDC time";
        return;
      }
    }
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:validate \t\t\t\t\t\t\t  BX: " << bxValue;

  /* If you reach here, the BX and partial laterality are considered are valid
     */
  latq->bxValue = bxValue;
  latq->latQValid = true;
}
int MuonPathAnalyzerPerSL::eqMainBXTerm(LATERAL_CASES sideComb[2], int layerIdx[2], MuonPathPtr &mPath) {
  int eqTerm = 0, coefs[2];

  lateralCoeficients(sideComb, coefs);

  eqTerm = coefs[0] * mPath->primitive(layerIdx[0])->tdcTimeStampNoOffset() +
           coefs[1] * mPath->primitive(layerIdx[1])->tdcTimeStampNoOffset();

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:eqMainBXTerm \t\t\t\t\t In eqMainBXTerm EQTerm(BX): " << eqTerm;

  return (eqTerm);
}
int MuonPathAnalyzerPerSL::eqMainTerm(LATERAL_CASES sideComb[2], int layerIdx[2], MuonPathPtr &mPath, int bxValue) {
  int eqTerm = 0, coefs[2];

  lateralCoeficients(sideComb, coefs);

  if (!use_LSB_)
    eqTerm = coefs[0] * (mPath->primitive(layerIdx[0])->tdcTimeStampNoOffset() - bxValue) +
             coefs[1] * (mPath->primitive(layerIdx[1])->tdcTimeStampNoOffset() - bxValue);
  else
    eqTerm = coefs[0] * floor((DRIFT_SPEED / (10 * x_precision_)) *
                              (mPath->primitive(layerIdx[0])->tdcTimeStampNoOffset() - bxValue)) +
             coefs[1] * floor((DRIFT_SPEED / (10 * x_precision_)) *
                              (mPath->primitive(layerIdx[1])->tdcTimeStampNoOffset() - bxValue));

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:\t\t\t\t\t EQTerm(Main): " << eqTerm;

  return (eqTerm);
}

void MuonPathAnalyzerPerSL::lateralCoeficients(LATERAL_CASES sideComb[2], int *coefs) {
  if ((sideComb[0] == LEFT) && (sideComb[1] == LEFT)) {
    *(coefs) = +1;
    *(coefs + 1) = -1;
  } else if ((sideComb[0] == LEFT) && (sideComb[1] == RIGHT)) {
    *(coefs) = +1;
    *(coefs + 1) = +1;
  } else if ((sideComb[0] == RIGHT) && (sideComb[1] == LEFT)) {
    *(coefs) = -1;
    *(coefs + 1) = -1;
  } else if ((sideComb[0] == RIGHT) && (sideComb[1] == RIGHT)) {
    *(coefs) = -1;
    *(coefs + 1) = +1;
  }
}

/**
 * Determines if all valid partial lateral combinations share the same value
 * of 'bxValue'.
 */
bool MuonPathAnalyzerPerSL::sameBXValue(PARTIAL_LATQ_TYPE *latq) {
  bool result = true;

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "Dtp2:sameBXValue bxTolerance_: " << bxTolerance_;

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "Dtp2:sameBXValue \t\t\t\t\t\t d01:" << abs(latq[0].bxValue - latq[1].bxValue);
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "Dtp2:sameBXValue \t\t\t\t\t\t d02:" << abs(latq[0].bxValue - latq[2].bxValue);
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "Dtp2:sameBXValue \t\t\t\t\t\t d03:" << abs(latq[0].bxValue - latq[3].bxValue);
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "Dtp2:sameBXValue \t\t\t\t\t\t d12:" << abs(latq[1].bxValue - latq[2].bxValue);
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "Dtp2:sameBXValue \t\t\t\t\t\t d13:" << abs(latq[1].bxValue - latq[3].bxValue);
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "Dtp2:sameBXValue \t\t\t\t\t\t d23:" << abs(latq[2].bxValue - latq[3].bxValue);

  bool d01, d02, d03, d12, d13, d23;
  d01 = (abs(latq[0].bxValue - latq[1].bxValue) <= bxTolerance_) ? true : false;
  d02 = (abs(latq[0].bxValue - latq[2].bxValue) <= bxTolerance_) ? true : false;
  d03 = (abs(latq[0].bxValue - latq[3].bxValue) <= bxTolerance_) ? true : false;
  d12 = (abs(latq[1].bxValue - latq[2].bxValue) <= bxTolerance_) ? true : false;
  d13 = (abs(latq[1].bxValue - latq[3].bxValue) <= bxTolerance_) ? true : false;
  d23 = (abs(latq[2].bxValue - latq[3].bxValue) <= bxTolerance_) ? true : false;

  /* 4 groups of partial combination of valid lateralities  */
  if ((latq[0].latQValid && latq[1].latQValid && latq[2].latQValid && latq[3].latQValid) && !(d01 && d12 && d23))
    result = false;
  else
      /* 4 posible cases of 3 groups of valid partial lateralities  */
      if (((latq[0].latQValid && latq[1].latQValid && latq[2].latQValid) && !(d01 && d12)) or
          ((latq[0].latQValid && latq[1].latQValid && latq[3].latQValid) && !(d01 && d13)) or
          ((latq[0].latQValid && latq[2].latQValid && latq[3].latQValid) && !(d02 && d23)) or
          ((latq[1].latQValid && latq[2].latQValid && latq[3].latQValid) && !(d12 && d23)))
    result = false;
  else
      /* Lastly, the 4 possible cases of partial valid lateralities */

      if (((latq[0].latQValid && latq[1].latQValid) && !d01) or ((latq[0].latQValid && latq[2].latQValid) && !d02) or
          ((latq[0].latQValid && latq[3].latQValid) && !d03) or ((latq[1].latQValid && latq[2].latQValid) && !d12) or
          ((latq[1].latQValid && latq[3].latQValid) && !d13) or ((latq[2].latQValid && latq[3].latQValid) && !d23))
    result = false;

  return result;
}

/** Calculate the parameters of the detected trayectories */
void MuonPathAnalyzerPerSL::calculatePathParameters(MuonPathPtr &mPath) {
  // The order is important.
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL")
        << "DTp2:calculatePathParameters \t\t\t\t\t\t  calculating calcCellDriftAndXcoor(mPath) ";
  calcCellDriftAndXcoor(mPath);
  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:calculatePathParameters \t\t\t\t\t\t  checking mPath->quality() "
                                      << mPath->quality();
  if (mPath->quality() == HIGHQ or mPath->quality() == HIGHQGHOST) {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL")
          << "DTp2:calculatePathParameters \t\t\t\t\t\t\t  Quality test passed, now calcTanPhiXPosChamber4Hits(mPath) ";
    calcTanPhiXPosChamber4Hits(mPath);
  } else {
    if (debug_)
      LogDebug("MuonPathAnalyzerPerSL")
          << "DTp2:calculatePathParameters \t\t\t\t\t\t\t  Quality test NOT passed calcTanPhiXPosChamber3Hits(mPath) ";
    calcTanPhiXPosChamber3Hits(mPath);
  }

  if (debug_)
    LogDebug("MuonPathAnalyzerPerSL") << "DTp2:calculatePathParameters \t\t\t\t\t\t calcChiSquare(mPath) ";
  calcChiSquare(mPath);
}

void MuonPathAnalyzerPerSL::calcTanPhiXPosChamber(MuonPathPtr &mPath) {
  int layerIdx[2];
  /*
      To calculate path's angle are only necessary two valid primitives.
      This method should be called only when a 'MuonPath' is determined as valid,
      so, at least, three of its primitives must have a valid time.
      With this two comparitions (which can be implemented easily as multiplexors
      in the FPGA) this method ensures to catch two of those valid primitives to
      evaluate the angle.

      The first one is below the middle line of the superlayer, while the other
      one is above this line
    */
  if (mPath->primitive(0)->isValidTime())
    layerIdx[0] = 0;
  else
    layerIdx[0] = 1;

  if (mPath->primitive(3)->isValidTime())
    layerIdx[1] = 3;
  else
    layerIdx[1] = 2;

  /* We identify along which cells' sides the muon travels */
  LATERAL_CASES sideComb[2];
  sideComb[0] = (mPath->lateralComb())[layerIdx[0]];
  sideComb[1] = (mPath->lateralComb())[layerIdx[1]];

  /* Horizontal gap between cells in cell's semi-length units */
  int dHoriz = (mPath->cellLayout())[layerIdx[1]] - (mPath->cellLayout())[layerIdx[0]];

  /* Vertical gap between cells in cell's height units */
  int dVert = layerIdx[1] - layerIdx[0];

  /*-----------------------------------------------------------------*/
  /*--------------------- Phi angle calculation ---------------------*/
  /*-----------------------------------------------------------------*/
  float num = CELL_SEMILENGTH * dHoriz + DRIFT_SPEED * eqMainTerm(sideComb, layerIdx, mPath, mPath->bxTimeValue());

  float denom = CELL_HEIGHT * dVert;
  float tanPhi = num / denom;

  mPath->setTanPhi(tanPhi);

  /*-----------------------------------------------------------------*/
  /*----------------- Horizontal coord. calculation -----------------*/
  /*-----------------------------------------------------------------*/

  /*
      Using known coordinates, relative to superlayer axis reference, (left most
      superlayer side, and middle line between 2nd and 3rd layers), calculating
      horizontal coordinate implies using a basic line equation:
      (y - y0) = (x - x0) * cotg(Phi)
      This horizontal coordinate can be obtained setting y = 0 on last equation,
      and also setting y0 and x0 with the values of a known muon's path cell
      position hit.
      It's enough to use the lower cell (layerIdx[0]) coordinates. So:
      xC = x0 - y0 * tan(Phi)
    */
  float lowerXPHorizPos = mPath->xCoorCell(layerIdx[0]);

  float lowerXPVertPos = 0;  // This is only the absolute value distance.
  if (layerIdx[0] == 0)
    lowerXPVertPos = CELL_HEIGHT + CELL_SEMIHEIGHT;
  else
    lowerXPVertPos = CELL_SEMIHEIGHT;

  mPath->setHorizPos(lowerXPHorizPos + lowerXPVertPos * tanPhi);
}

/**
 * Coordinate and angle calculations for a 4 HITS cases
 */
void MuonPathAnalyzerPerSL::calcTanPhiXPosChamber4Hits(MuonPathPtr &mPath) {
  int x_prec_inv = (int)(1. / (10. * x_precision_));
  int numberOfBits = (int)(round(std::log(x_prec_inv) / std::log(2.)));
  int numerator = 3 * (int)round(mPath->xCoorCell(3) / (10 * x_precision_)) +
                  (int)round(mPath->xCoorCell(2) / (10 * x_precision_)) -
                  (int)round(mPath->xCoorCell(1) / (10 * x_precision_)) -
                  3 * (int)round(mPath->xCoorCell(0) / (10 * x_precision_));
  int CELL_HEIGHT_JM = pow(2, 15) / ((int)(10 * CELL_HEIGHT));
  int tanPhi_x4096 = (numerator * CELL_HEIGHT_JM) >> (3 + numberOfBits);
  mPath->setTanPhi(tanPhi_x4096 * tanPsi_precision_);

  float XPos = (mPath->xCoorCell(0) + mPath->xCoorCell(1) + mPath->xCoorCell(2) + mPath->xCoorCell(3)) / 4;
  mPath->setHorizPos(floor(XPos / (10 * x_precision_)) * 10 * x_precision_);
}

/**
 *  3 HITS cases
 */
void MuonPathAnalyzerPerSL::calcTanPhiXPosChamber3Hits(MuonPathPtr &mPath) {
  int layerIdx[2];
  int x_prec_inv = (int)(1. / (10. * x_precision_));
  int numberOfBits = (int)(round(std::log(x_prec_inv) / std::log(2.)));

  if (mPath->primitive(0)->isValidTime())
    layerIdx[0] = 0;
  else
    layerIdx[0] = 1;

  if (mPath->primitive(3)->isValidTime())
    layerIdx[1] = 3;
  else
    layerIdx[1] = 2;

  /*-----------------------------------------------------------------*/
  /*--------------------- Phi angle calculation ---------------------*/
  /*-----------------------------------------------------------------*/

  int tan_division_denominator_bits = 16;

  int num =
      ((int)((int)(x_prec_inv * mPath->xCoorCell(layerIdx[1])) - (int)(x_prec_inv * mPath->xCoorCell(layerIdx[0])))
       << (12 - numberOfBits));
  int denominator = (layerIdx[1] - layerIdx[0]) * CELL_HEIGHT;
  int denominator_inv = ((int)(0.5 + pow(2, tan_division_denominator_bits) / float(denominator)));

  float tanPhi = ((num * denominator_inv) >> tan_division_denominator_bits) / ((1. / tanPsi_precision_));

  mPath->setTanPhi(tanPhi);

  /*-----------------------------------------------------------------*/
  /*----------------- Horizontal coord. calculation -----------------*/
  /*-----------------------------------------------------------------*/
  float XPos = 0;
  if (mPath->primitive(0)->isValidTime() and mPath->primitive(3)->isValidTime())
    XPos = (mPath->xCoorCell(0) + mPath->xCoorCell(3)) / 2;
  else
    XPos = (mPath->xCoorCell(1) + mPath->xCoorCell(2)) / 2;

  mPath->setHorizPos(floor(XPos / (10 * x_precision_)) * 10 * x_precision_);
}

/**
 * Calculate the drift distances of each wire and the horizontal position 
 */
void MuonPathAnalyzerPerSL::calcCellDriftAndXcoor(MuonPathPtr &mPath) {
  long int drift_speed_new = 889;
  long int drift_dist_um_x4;
  long int wireHorizPos_x4;
  long int pos_mm_x4;
  int x_prec_inv = (int)(1. / (10. * x_precision_));

  for (int i = 0; i <= 3; i++)
    if (mPath->primitive(i)->isValidTime()) {
      drift_dist_um_x4 =
          drift_speed_new * ((long int)mPath->primitive(i)->tdcTimeStampNoOffset() - (long int)mPath->bxTimeValue());
      wireHorizPos_x4 = (long)(mPath->primitive(i)->wireHorizPos() * x_prec_inv);

      if ((mPath->lateralComb())[i] == LEFT)
        pos_mm_x4 = wireHorizPos_x4 - (drift_dist_um_x4 >> 10);
      else
        pos_mm_x4 = wireHorizPos_x4 + (drift_dist_um_x4 >> 10);

      mPath->setXCoorCell(pos_mm_x4 * (10 * x_precision_), i);
      mPath->setDriftDistance(((float)(drift_dist_um_x4 >> 10)) * (10 * x_precision_), i);
    }
}

/**
 * Calculate the quality estimator of each trayectory.
 */
void MuonPathAnalyzerPerSL::calcChiSquare(MuonPathPtr &mPath) {
  int x_prec_inv = (int)(1. / (10. * x_precision_));
  int numberOfBits = (int)(round(std::log(x_prec_inv) / std::log(2.)));
  long int Z_FACTOR[NUM_LAYERS] = {-6, -2, 2, 6};
  for (int i = 0; i < 4; i++) {
    Z_FACTOR[i] = Z_FACTOR[i] * (long int)CELL_HEIGHT;
  }
  long int sum_A = 0, sum_B = 0;
  long int chi2_mm2_x1024 = 0;
  for (int i = 0; i < 4; i++) {
    if (mPath->primitive(i)->isValidTime()) {
      sum_A = (((int)(mPath->xCoorCell(i) / (10 * x_precision_))) - ((int)(mPath->horizPos() / (10 * x_precision_))))
              << (14 - numberOfBits);
      sum_B = Z_FACTOR[i] * ((int)(mPath->tanPhi() / tanPsi_precision_));
      chi2_mm2_x1024 += (sum_A - sum_B) * (sum_A - sum_B);
    }
  }
  chi2_mm2_x1024 = chi2_mm2_x1024 >> 18;

  mPath->setChiSquare(((double)chi2_mm2_x1024 / 1024.));
}

int MuonPathAnalyzerPerSL::omittedHit(int idx) {
  switch (idx) {
    case 0:
      return 3;
    case 1:
      return 0;
    case 2:
      return 2;
    case 3:
      return 1;
  }

  return -1;
}
