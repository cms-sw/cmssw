#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzerInChamber.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;
// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAnalyzerInChamber::MuonPathAnalyzerInChamber(const ParameterSet &pset,
                                                     edm::ConsumesCollector &iC,
                                                     std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer)
    : MuonPathAnalyzer(pset, iC),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      chi2Th_(pset.getUntrackedParameter<double>("chi2Th")),
      shift_filename_(pset.getParameter<edm::FileInPath>("shift_filename")),
      bxTolerance_(30),
      minQuality_(LOWQ),
      chiSquareThreshold_(50),
      minHits4Fit_(pset.getUntrackedParameter<int>("minHits4Fit")),
      splitPathPerSL_(pset.getUntrackedParameter<bool>("splitPathPerSL")) {
  // Obtention of parameters

  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "MuonPathAnalyzer: constructor";

  setChiSquareThreshold(chi2Th_ * 10000.);

  //shift
  int rawId;
  std::ifstream ifin3(shift_filename_.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathAnalyzerInChamber::MuonPathAnalyzerInChamber() -  Cannot find " << shift_filename_.fullPath();
  }
  while (ifin3.good()) {
    ifin3 >> rawId >> shift;
    shiftinfo_[rawId] = shift;
  }

  dtGeomH = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
  globalcoordsobtainer_ = globalcoordsobtainer;
}

MuonPathAnalyzerInChamber::~MuonPathAnalyzerInChamber() {
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "MuonPathAnalyzer: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAnalyzerInChamber::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "MuonPathAnalyzerInChamber::initialiase";

  const MuonGeometryRecord &geom = iEventSetup.get<MuonGeometryRecord>();
  dtGeo_ = &geom.get(dtGeomH);
}

void MuonPathAnalyzerInChamber::run(edm::Event &iEvent,
                                    const edm::EventSetup &iEventSetup,
                                    MuonPathPtrs &muonpaths,
                                    MuonPathPtrs &outmuonpaths) {
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "MuonPathAnalyzerInChamber: run";

  // fit per SL (need to allow for multiple outputs for a single mpath)
  int nMuonPath_counter = 0;
  for (auto muonpath = muonpaths.begin(); muonpath != muonpaths.end(); ++muonpath) {
    if (debug_) {
      LogDebug("MuonPathAnalyzerInChamber")
          << "Full path: " << nMuonPath_counter << " , " << muonpath->get()->nprimitives() << " , "
          << muonpath->get()->nprimitivesUp() << " , " << muonpath->get()->nprimitivesDown();
    }
    ++nMuonPath_counter;

    // Define muonpaths for up/down SL only
    MuonPathPtr muonpathUp_ptr = std::make_shared<MuonPath>();
    muonpathUp_ptr->setNPrimitives(8);
    muonpathUp_ptr->setNPrimitivesUp(muonpath->get()->nprimitivesUp());
    muonpathUp_ptr->setNPrimitivesDown(0);

    MuonPathPtr muonpathDown_ptr = std::make_shared<MuonPath>();
    muonpathDown_ptr->setNPrimitives(8);
    muonpathDown_ptr->setNPrimitivesUp(0);
    muonpathDown_ptr->setNPrimitivesDown(muonpath->get()->nprimitivesDown());

    for (int n = 0; n < muonpath->get()->nprimitives(); ++n) {
      DTPrimitivePtr prim = muonpath->get()->primitive(n);
      // UP
      if (prim->superLayerId() == 3) {
        muonpathUp_ptr->setPrimitive(prim, n);
      }
      // DOWN
      else if (prim->superLayerId() == 1) {
        muonpathDown_ptr->setPrimitive(prim, n);
      }
      // NOT UP NOR DOWN
      else
        continue;
    }

    analyze(*muonpath, outmuonpaths);

    if (splitPathPerSL_) {
      if (muonpathUp_ptr->nprimitivesUp() > 1 && muonpath->get()->nprimitivesDown() > 0)
        analyze(muonpathUp_ptr, outmuonpaths);

      if (muonpathDown_ptr->nprimitivesDown() > 1 && muonpath->get()->nprimitivesUp() > 0)
        analyze(muonpathDown_ptr, outmuonpaths);
    }
  }
}

void MuonPathAnalyzerInChamber::finish() {
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "MuonPathAnalyzer: finish";
};

//------------------------------------------------------------------
//--- Private method
//------------------------------------------------------------------
void MuonPathAnalyzerInChamber::analyze(MuonPathPtr &inMPath, MuonPathPtrs &outMPath) {
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "DTp2:analyze \t\t\t\t starts";

  // Clone the analyzed object
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << inMPath->nprimitives();
  auto mPath = std::make_shared<MuonPath>(inMPath);

  if (debug_) {
    LogDebug("MuonPathAnalyzerInChamber") << "DTp2::analyze, looking at mPath: ";
    for (int i = 0; i < mPath->nprimitives(); i++)
      LogDebug("MuonPathAnalyzerInChamber")
          << mPath->primitive(i)->layerId() << " , " << mPath->primitive(i)->superLayerId() << " , "
          << mPath->primitive(i)->channelId() << " , " << mPath->primitive(i)->laterality();
  }

  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "DTp2:analyze \t\t\t\t\t is Analyzable? ";
  if (!mPath->isAnalyzable())
    return;
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "DTp2:analyze \t\t\t\t\t yes it is analyzable " << mPath->isAnalyzable();

  // first of all, get info from primitives, so we can reduce the number of latereralities:
  buildLateralities(mPath);
  setWirePosAndTimeInMP(mPath);

  std::shared_ptr<MuonPath> mpAux;
  int bestI = -1;
  float best_chi2 = 99999.;
  int added_lat = 0;

  // LOOP for all lateralities:
  for (int i = 0; i < (int)lateralities_.size(); i++) {
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber") << "DTp2:analyze \t\t\t\t\t Start with combination " << i;

    int NTotalHits = NUM_LAYERS_2SL;
    float xwire[NUM_LAYERS_2SL];
    int present_layer[NUM_LAYERS_2SL];
    for (int ii = 0; ii < 8; ii++) {
      xwire[ii] = mPath->xWirePos(ii);
      if (xwire[ii] == 0) {
        present_layer[ii] = 0;
        NTotalHits--;
      } else {
        present_layer[ii] = 1;
      }
    }

    while (NTotalHits >= minHits4Fit_) {
      mPath->setChiSquare(0);
      calculateFitParameters(mPath, lateralities_[i], present_layer, added_lat);
      if (mPath->chiSquare() != 0)
        break;
      NTotalHits--;
    }

    if (mPath->chiSquare() > chiSquareThreshold_)
      continue;

    evaluateQuality(mPath);

    if (mPath->quality() < minQuality_)
      continue;

    double z = 0;
    double jm_x = (mPath->horizPos());
    int selected_Id = 0;
    for (int i = 0; i < mPath->nprimitives(); i++) {
      if (mPath->primitive(i)->isValidTime()) {
        selected_Id = mPath->primitive(i)->cameraId();
        mPath->setRawId(selected_Id);
        break;
      }
    }
    DTLayerId thisLId(selected_Id);
    DTChamberId ChId(thisLId.wheel(), thisLId.station(), thisLId.sector());

    if (thisLId.station() >= 3)
      z += Z_SHIFT_MB4;

    DTSuperLayerId MuonPathSLId(thisLId.wheel(), thisLId.station(), thisLId.sector(), thisLId.superLayer());

    // Count hits in each SL
    int hits_in_SL1 = 0;
    int hits_in_SL3 = 0;
    for (int i = 0; i < mPath->nprimitives(); i++) {
      if (mPath->primitive(i)->isValidTime()) {
        if (i <= 3)
          ++hits_in_SL1;
        else if (i > 3)
          ++hits_in_SL3;
      }
    }

    int SL_for_LUT = 0;
    // Depending on which SL has hits, propagate jm_x to SL1, SL3, or to the center of the chamber
    GlobalPoint jm_x_cmssw_global;
    if (hits_in_SL1 > 2 && hits_in_SL3 <= 2) {
      // Uncorrelated or confirmed with 3 or 4 hits in SL1: propagate to SL1
      jm_x += mPath->tanPhi() * (11.1 + 0.65);
      jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(LocalPoint(jm_x, 0., z + 11.75));
      SL_for_LUT = 1;
    } else if (hits_in_SL1 <= 2 && hits_in_SL3 > 2) {
      // Uncorrelated or confirmed with 3 or 4 hits in SL3: propagate to SL3
      jm_x -= mPath->tanPhi() * (11.1 + 0.65);
      jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(LocalPoint(jm_x, 0., z - 11.75));
      SL_for_LUT = 3;
    } else if (hits_in_SL1 > 2 && hits_in_SL3 > 2) {
      // Correlated: stay at chamber center
      jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(LocalPoint(jm_x, 0., z));
    } else {
      // Not interesting
      continue;
    }

    // Protection against non-converged fits
    if (isnan(jm_x))
      continue;

    // Updating muon-path horizontal position
    mPath->setHorizPos(jm_x);

    // Adjusting sector names for MB4
    int thisec = MuonPathSLId.sector();
    if (thisec == 13)
      thisec = 4;
    if (thisec == 14)
      thisec = 10;

    // Global coordinates from CMSSW
    double phi_cmssw = jm_x_cmssw_global.phi() - PHI_CONV * (thisec - 1);
    double psi = atan(mPath->tanPhi());
    mPath->setPhiCMSSW(phi_cmssw);
    mPath->setPhiBCMSSW(hasPosRF(MuonPathSLId.wheel(), MuonPathSLId.sector()) ? psi - phi_cmssw : -psi - phi_cmssw);

    // Global coordinates from LUTs (firmware-like)
    double phi = -999.;
    double phiB = -999.;
    double x_lut, slope_lut;
    DTSuperLayerId MuonPathSLId1(thisLId.wheel(), thisLId.station(), thisLId.sector(), 1);
    DTSuperLayerId MuonPathSLId3(thisLId.wheel(), thisLId.station(), thisLId.sector(), 3);
    DTWireId wireId1(MuonPathSLId1, 2, 1);
    DTWireId wireId3(MuonPathSLId3, 2, 1);

    // use SL_for_LUT to decide the shift: x-axis origin for LUTs is left chamber side
    double shift_for_lut = 0.;
    if (SL_for_LUT == 1) {
      shift_for_lut = int(10 * shiftinfo_[wireId1.rawId()] * INCREASED_RES_POS_POW);
    } else if (SL_for_LUT == 3) {
      shift_for_lut = int(10 * shiftinfo_[wireId3.rawId()] * INCREASED_RES_POS_POW);
    } else {
      int shift_sl1 = int(round(shiftinfo_[wireId1.rawId()] * INCREASED_RES_POS_POW * 10));
      int shift_sl3 = int(round(shiftinfo_[wireId3.rawId()] * INCREASED_RES_POS_POW * 10));
      if (shift_sl1 < shift_sl3) {
        shift_for_lut = shift_sl1;
      } else
        shift_for_lut = shift_sl3;
    }
    x_lut = double(jm_x) * 10. * INCREASED_RES_POS_POW - shift_for_lut;  // position in cm * precision in JM RF
    slope_lut = -(double(mPath->tanPhi()) * INCREASED_RES_SLOPE_POW);

    auto global_coords = globalcoordsobtainer_->get_global_coordinates(ChId.rawId(), SL_for_LUT, x_lut, slope_lut);
    phi = global_coords[0];
    phiB = global_coords[1];
    mPath->setPhi(phi);
    mPath->setPhiB(phiB);

    if (mPath->chiSquare() < best_chi2 && mPath->chiSquare() > 0) {
      mpAux = std::make_shared<MuonPath>(mPath);
      bestI = i;
      best_chi2 = mPath->chiSquare();
    }
  }
  if (mpAux != nullptr) {
    outMPath.push_back(std::move(mpAux));
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber")
          << "DTp2:analize \t\t\t\t\t Laterality " << bestI << " is the one with smaller chi2";
  } else {
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber")
          << "DTp2:analize \t\t\t\t\t No Laterality found with chi2 smaller than threshold";
  }
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "DTp2:analize \t\t\t\t\t Ended working with this set of lateralities";
}

void MuonPathAnalyzerInChamber::setCellLayout(MuonPathPtr &mpath) {
  for (int i = 0; i <= mpath->nprimitives(); i++) {
    if (mpath->primitive(i)->isValidTime())
      cellLayout_[i] = mpath->primitive(i)->channelId();
    else
      cellLayout_[i] = -99;
  }

  // putting info back into the mpath:
  mpath->setCellHorizontalLayout(cellLayout_);
  for (int i = 0; i <= mpath->nprimitives(); i++) {
    if (cellLayout_[i] >= 0) {
      mpath->setBaseChannelId(cellLayout_[i]);
      break;
    }
  }
}

/**
 * For a combination of up to 8 cells, build all the lateralities to be tested,
 */
void MuonPathAnalyzerInChamber::buildLateralities(MuonPathPtr &mpath) {
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "MPAnalyzer::buildLateralities << setLateralitiesFromPrims ";
  mpath->setLateralCombFromPrimitives();

  totalNumValLateralities_ = 0;
  lateralities_.clear();
  latQuality_.clear();

  /* We generate all the possible laterality combinations compatible with the built 
     group in the previous step*/
  lateralities_.push_back(TLateralities());
  for (int ilat = 0; ilat < NLayers; ilat++) {
    // Get value from input
    LATERAL_CASES lr = (mpath->lateralComb())[ilat];
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber") << "[DEBUG_] Input[" << ilat << "]: " << lr;

    // If left/right fill number
    if (lr != NONE) {
      if (debug_)
        LogDebug("MuonPathAnalyzerInChamber") << "[DEBUG_]   - Adding it to " << lateralities_.size() << " lists...";
      for (unsigned int iall = 0; iall < lateralities_.size(); iall++) {
        lateralities_[iall][ilat] = lr;
      }
    }
    // both possibilites
    else {
      // Get the number of possible options now
      auto ncurrentoptions = lateralities_.size();

      // Duplicate them
      if (debug_)
        LogDebug("MuonPathAnalyzerInChamber") << "[DEBUG_]   - Duplicating " << ncurrentoptions << " lists...";
      copy(lateralities_.begin(), lateralities_.end(), back_inserter(lateralities_));
      if (debug_)
        LogDebug("MuonPathAnalyzerInChamber") << "[DEBUG_]   - Now we have " << lateralities_.size() << " lists...";

      // Asign LEFT to first ncurrentoptions and RIGHT to the last
      for (unsigned int iall = 0; iall < ncurrentoptions; iall++) {
        lateralities_[iall][ilat] = LEFT;
        lateralities_[iall + ncurrentoptions][ilat] = RIGHT;
      }
    }  // else
  }    // Iterate over input array

  totalNumValLateralities_ = (int)lateralities_.size();
  if (totalNumValLateralities_ > 128) {
    // ADD PROTECTION!
    LogDebug("MuonPathAnalyzerInChamber") << "[WARNING]: TOO MANY LATERALITIES TO CHECK !!";
    LogDebug("MuonPathAnalyzerInChamber") << "[WARNING]: skipping this muon";
    lateralities_.clear();
    latQuality_.clear();
    totalNumValLateralities_ = 0;
  }

  // Dump values
  if (debug_) {
    for (unsigned int iall = 0; iall < lateralities_.size(); iall++) {
      LogDebug("MuonPathAnalyzerInChamber") << iall << " -> [";
      for (int ilat = 0; ilat < NLayers; ilat++) {
        if (ilat != 0)
          LogDebug("MuonPathAnalyzerInChamber") << ",";
        LogDebug("MuonPathAnalyzerInChamber") << lateralities_[iall][ilat];
      }
      LogDebug("MuonPathAnalyzerInChamber") << "]";
    }
  }
}

void MuonPathAnalyzerInChamber::setLateralitiesInMP(MuonPathPtr &mpath, TLateralities lat) {
  for (int i = 0; i < 8; i++)
    mpath->primitive(i)->setLaterality(lat[i]);
}

void MuonPathAnalyzerInChamber::setWirePosAndTimeInMP(MuonPathPtr &mpath) {
  int selected_Id = 0;
  for (int i = 0; i < mpath->nprimitives(); i++) {
    if (mpath->primitive(i)->isValidTime()) {
      selected_Id = mpath->primitive(i)->cameraId();
      mpath->setRawId(selected_Id);
      break;
    }
  }
  DTLayerId thisLId(selected_Id);
  DTChamberId chId(thisLId.wheel(), thisLId.station(), thisLId.sector());
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber")
        << "Id " << chId.rawId() << " Wh " << chId.wheel() << " St " << chId.station() << " Se " << chId.sector();
  mpath->setRawId(chId.rawId());

  DTSuperLayerId MuonPathSLId1(thisLId.wheel(), thisLId.station(), thisLId.sector(), 1);
  DTSuperLayerId MuonPathSLId3(thisLId.wheel(), thisLId.station(), thisLId.sector(), 3);
  DTWireId wireId1(MuonPathSLId1, 2, 1);
  DTWireId wireId3(MuonPathSLId3, 2, 1);

  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber")
        << "shift1=" << shiftinfo_[wireId1.rawId()] << " shift3=" << shiftinfo_[wireId3.rawId()];

  float delta = 42000;                                                                      //um
  float zwire[NUM_LAYERS_2SL] = {-13.7, -12.4, -11.1, -9.8002, 9.79999, 11.1, 12.4, 13.7};  // mm
  for (int i = 0; i < mpath->nprimitives(); i++) {
    if (mpath->primitive(i)->isValidTime()) {
      if (i < 4)
        mpath->setXWirePos(10000 * shiftinfo_[wireId1.rawId()] +
                               (mpath->primitive(i)->channelId() + 0.5 * (double)((i + 1) % 2)) * delta,
                           i);
      if (i >= 4)
        mpath->setXWirePos(10000 * shiftinfo_[wireId3.rawId()] +
                               (mpath->primitive(i)->channelId() + 0.5 * (double)((i + 1) % 2)) * delta,
                           i);
      mpath->setZWirePos(zwire[i] * 1000, i);  // in um
      mpath->setTWireTDC(mpath->primitive(i)->tdcTimeStamp() * DRIFT_SPEED, i);
    } else {
      mpath->setXWirePos(0., i);
      mpath->setZWirePos(0., i);
      mpath->setTWireTDC(-1 * DRIFT_SPEED, i);
    }
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber") << mpath->primitive(i)->tdcTimeStamp() << " ";
  }
  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber");
}

void MuonPathAnalyzerInChamber::calculateFitParameters(MuonPathPtr &mpath,
                                                       TLateralities laterality,
                                                       int present_layer[NUM_LAYERS_2SL],
                                                       int &lat_added) {
  // Get number of hits in current muonPath
  int n_hits = 0;
  for (int l = 0; l < 8; ++l) {
    if (present_layer[l] == 1)
      n_hits++;
  }

  // First prepare mpath for fit:
  float xwire[NUM_LAYERS_2SL], zwire[NUM_LAYERS_2SL], tTDCvdrift[NUM_LAYERS_2SL];
  double b[NUM_LAYERS_2SL];
  for (int i = 0; i < 8; i++) {
    xwire[i] = mpath->xWirePos(i);
    zwire[i] = mpath->zWirePos(i);
    tTDCvdrift[i] = mpath->tWireTDC(i);
    b[i] = 1;
  }

  //// NOW Start FITTING:

  // fill hit position
  float xhit[NUM_LAYERS_2SL];
  for (int lay = 0; lay < 8; lay++) {
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber") << "In fitPerLat " << lay << " xwire " << xwire[lay] << " zwire "
                                            << zwire[lay] << " tTDCvdrift " << tTDCvdrift[lay];
    xhit[lay] = xwire[lay] + (-1 + 2 * laterality[lay]) * 1000 * tTDCvdrift[lay];
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber") << "In fitPerLat " << lay << " xhit " << xhit[lay];
  }

  //Proceed with calculation of fit parameters
  double cbscal = 0.0;
  double zbscal = 0.0;
  double czscal = 0.0;
  double bbscal = 0.0;
  double zzscal = 0.0;
  double ccscal = 0.0;

  for (int lay = 0; lay < 8; lay++) {
    if (present_layer[lay] == 0)
      continue;
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber")
          << " For layer " << lay + 1 << " xwire[lay] " << xwire[lay] << " zwire " << zwire[lay] << " b " << b[lay];
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber") << " xhit[lat][lay] " << xhit[lay];
    cbscal = (-1 + 2 * laterality[lay]) * b[lay] + cbscal;
    zbscal = zwire[lay] * b[lay] + zbscal;  //it actually does not depend on laterality
    czscal = (-1 + 2 * laterality[lay]) * zwire[lay] + czscal;

    bbscal = b[lay] * b[lay] + bbscal;          //it actually does not depend on laterality
    zzscal = zwire[lay] * zwire[lay] + zzscal;  //it actually does not depend on laterality
    ccscal = (-1 + 2 * laterality[lay]) * (-1 + 2 * laterality[lay]) + ccscal;
  }

  double cz = 0.0;
  double cb = 0.0;
  double zb = 0.0;
  double zc = 0.0;
  double bc = 0.0;
  double bz = 0.0;

  cz = (cbscal * zbscal - czscal * bbscal) / (zzscal * bbscal - zbscal * zbscal);
  cb = (czscal * zbscal - cbscal * zzscal) / (zzscal * bbscal - zbscal * zbscal);

  zb = (czscal * cbscal - zbscal * ccscal) / (bbscal * ccscal - cbscal * cbscal);
  zc = (zbscal * cbscal - czscal * bbscal) / (bbscal * ccscal - cbscal * cbscal);

  bc = (zbscal * czscal - cbscal * zzscal) / (ccscal * zzscal - czscal * czscal);
  bz = (cbscal * czscal - zbscal * ccscal) / (ccscal * zzscal - czscal * czscal);

  double c_tilde[NUM_LAYERS_2SL];
  double z_tilde[NUM_LAYERS_2SL];
  double b_tilde[NUM_LAYERS_2SL];

  for (int lay = 0; lay < 8; lay++) {
    if (present_layer[lay] == 0)
      continue;
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber")
          << " For layer " << lay + 1 << " xwire[lay] " << xwire[lay] << " zwire " << zwire[lay] << " b " << b[lay];
    c_tilde[lay] = (-1 + 2 * laterality[lay]) + cz * zwire[lay] + cb * b[lay];
    z_tilde[lay] = zwire[lay] + zb * b[lay] + zc * (-1 + 2 * laterality[lay]);
    b_tilde[lay] = b[lay] + bc * (-1 + 2 * laterality[lay]) + bz * zwire[lay];
  }

  //Calculate results per lat
  double xctilde = 0.0;
  double xztilde = 0.0;
  double xbtilde = 0.0;
  double ctildectilde = 0.0;
  double ztildeztilde = 0.0;
  double btildebtilde = 0.0;

  double rect0vdrift = 0.0;
  double recslope = 0.0;
  double recpos = 0.0;

  for (int lay = 0; lay < 8; lay++) {
    if (present_layer[lay] == 0)
      continue;
    xctilde = xhit[lay] * c_tilde[lay] + xctilde;
    ctildectilde = c_tilde[lay] * c_tilde[lay] + ctildectilde;
    xztilde = xhit[lay] * z_tilde[lay] + xztilde;
    ztildeztilde = z_tilde[lay] * z_tilde[lay] + ztildeztilde;
    xbtilde = xhit[lay] * b_tilde[lay] + xbtilde;
    btildebtilde = b_tilde[lay] * b_tilde[lay] + btildebtilde;
  }

  // Results for t0vdrift (BX), slope and position per lat
  rect0vdrift = xctilde / ctildectilde;
  recslope = xztilde / ztildeztilde;
  recpos = xbtilde / btildebtilde;
  if (debug_) {
    LogDebug("MuonPathAnalyzerInChamber") << " In fitPerLat Reconstructed values per lat "
                                          << " rect0vdrift " << rect0vdrift;
    LogDebug("MuonPathAnalyzerInChamber")
        << "rect0 " << rect0vdrift / DRIFT_SPEED << " recBX " << rect0vdrift / DRIFT_SPEED / 25 << " recslope "
        << recslope << " recpos " << recpos;
  }

  //Get t*v and residuals per layer, and chi2 per laterality
  double rectdriftvdrift[NUM_LAYERS_2SL];
  double recres[NUM_LAYERS_2SL];
  double recchi2 = 0.0;
  int sign_tdriftvdrift = {0};
  int incell_tdriftvdrift = {0};
  int physical_slope = {0};

  // Select the worst hit in order to get rid of it
  // Also, check if any hits are close to the wire (rectdriftvdrift[lay] < 0.3 cm)
  // --> in that case, try also changing laterality of such hit
  double maxDif = -1;
  int maxInt = -1;
  int swap_laterality[8] = {0};

  for (int lay = 0; lay < 8; lay++) {
    if (present_layer[lay] == 0)
      continue;
    rectdriftvdrift[lay] = tTDCvdrift[lay] - rect0vdrift / 1000;
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber") << rectdriftvdrift[lay];
    recres[lay] = xhit[lay] - zwire[lay] * recslope - b[lay] * recpos - (-1 + 2 * laterality[lay]) * rect0vdrift;

    // If a hit is too close to the wire, set its corresponding "swap" flag to 1
    if (abs(rectdriftvdrift[lay]) < 3) {
      swap_laterality[lay] = 1;
    }

    if ((present_layer[lay] == 1) && (rectdriftvdrift[lay] < -0.1)) {
      sign_tdriftvdrift = -1;
      if (-0.1 - rectdriftvdrift[lay] > maxDif) {
        maxDif = -0.1 - rectdriftvdrift[lay];
        maxInt = lay;
      }
    }
    if ((present_layer[lay] == 1) && (abs(rectdriftvdrift[lay]) > 21.1)) {
      incell_tdriftvdrift = -1;  //Changed to 2.11 to account for resolution effects
      if (rectdriftvdrift[lay] - 21.1 > maxDif) {
        maxDif = rectdriftvdrift[lay] - 21.1;
        maxInt = lay;
      }
    }
  }

  // Now consider all possible alternative lateralities and push to lateralities_ those
  // we aren't considering yet
  if (lat_added == 0) {
    std::vector<TLateralities> additional_lateralities;
    additional_lateralities.clear();
    additional_lateralities.push_back(laterality);
    // Everytime the swap flag is 1, duplicate all the current elements
    // of additional_lateralities and swap their laterality
    for (int swap = 0; swap < 8; ++swap) {
      if (swap_laterality[swap] == 1) {
        int add_lat_size = int(additional_lateralities.size());
        for (int ll = 0; ll < add_lat_size; ++ll) {
          TLateralities tmp_lat = additional_lateralities[ll];
          if (tmp_lat[swap] == LEFT)
            tmp_lat[swap] = RIGHT;
          else if (tmp_lat[swap] == RIGHT)
            tmp_lat[swap] = LEFT;
          else
            continue;
          additional_lateralities.push_back(tmp_lat);
        }
      }
    }
    // Now compare all the additional lateralities with the lateralities we are considering:
    // if they are not there, add them
    int already_there = 0;
    for (int k = 0; k < int(additional_lateralities.size()); ++k) {
      already_there = 0;
      for (int j = 0; j < int(lateralities_.size()); ++j) {
        if (additional_lateralities[k] == lateralities_[j])
          already_there = 1;
      }
      if (already_there == 0) {
        lateralities_.push_back(additional_lateralities[k]);
      }
    }
    additional_lateralities.clear();
    lat_added = 1;
  }

  if (fabs(recslope / 10) > 1.3)
    physical_slope = -1;

  if (physical_slope == -1 && debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "Combination with UNPHYSICAL slope ";
  if (sign_tdriftvdrift == -1 && debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "Combination with negative tdrift-vdrift ";
  if (incell_tdriftvdrift == -1 && debug_)
    LogDebug("MuonPathAnalyzerInChamber") << "Combination with tdrift-vdrift larger than half cell ";

  for (int lay = 0; lay < 8; lay++) {
    if (present_layer[lay] == 0)
      continue;
    recchi2 = recres[lay] * recres[lay] + recchi2;
  }
  // Chi2/ndof
  if (n_hits > 4)
    recchi2 = recchi2 / (n_hits - 3);

  if (debug_)
    LogDebug("MuonPathAnalyzerInChamber")
        << "In fitPerLat Chi2 " << recchi2 << " with sign " << sign_tdriftvdrift << " within cell "
        << incell_tdriftvdrift << " physical_slope " << physical_slope;

  //LATERALITY IS NOT VALID
  if (true && maxInt != -1) {
    present_layer[maxInt] = 0;
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber") << "We get rid of hit in layer " << maxInt;
  }

  // LATERALITY IS VALID...
  if (!(sign_tdriftvdrift == -1) && !(incell_tdriftvdrift == -1) && !(physical_slope == -1)) {
    mpath->setBxTimeValue((rect0vdrift / DRIFT_SPEED) / 1000);
    mpath->setTanPhi(-1 * recslope / 10);
    mpath->setHorizPos(recpos / 10000);
    mpath->setChiSquare(recchi2 / 100000000);
    setLateralitiesInMP(mpath, laterality);
    if (debug_)
      LogDebug("MuonPathAnalyzerInChamber")
          << "In fitPerLat "
          << "t0 " << mpath->bxTimeValue() << " slope " << mpath->tanPhi() << " pos " << mpath->horizPos() << " chi2 "
          << mpath->chiSquare() << " rawId " << mpath->rawId();
  }
}
void MuonPathAnalyzerInChamber::evaluateQuality(MuonPathPtr &mPath) {
  mPath->setQuality(NOPATH);

  int validHits(0), nPrimsUp(0), nPrimsDown(0);
  for (int i = 0; i < NUM_LAYERS_2SL; i++) {
    if (mPath->primitive(i)->isValidTime()) {
      validHits++;
      if (i < 4)
        nPrimsDown++;
      else if (i >= 4)
        nPrimsUp++;
    }
  }

  mPath->setNPrimitivesUp(nPrimsUp);
  mPath->setNPrimitivesDown(nPrimsDown);

  if (mPath->nprimitivesUp() >= 4 && mPath->nprimitivesDown() >= 4) {
    mPath->setQuality(HIGHHIGHQ);
  } else if ((mPath->nprimitivesUp() == 4 && mPath->nprimitivesDown() == 3) ||
             (mPath->nprimitivesUp() == 3 && mPath->nprimitivesDown() == 4)) {
    mPath->setQuality(HIGHLOWQ);
  } else if ((mPath->nprimitivesUp() == 4 && mPath->nprimitivesDown() <= 2 && mPath->nprimitivesDown() > 0) ||
             (mPath->nprimitivesUp() <= 2 && mPath->nprimitivesUp() > 0 && mPath->nprimitivesDown() == 4)) {
    mPath->setQuality(CHIGHQ);
  } else if ((mPath->nprimitivesUp() == 3 && mPath->nprimitivesDown() == 3)) {
    mPath->setQuality(LOWLOWQ);
  } else if ((mPath->nprimitivesUp() == 3 && mPath->nprimitivesDown() <= 2 && mPath->nprimitivesDown() > 0) ||
             (mPath->nprimitivesUp() <= 2 && mPath->nprimitivesUp() > 0 && mPath->nprimitivesDown() == 3) ||
             (mPath->nprimitivesUp() == 2 && mPath->nprimitivesDown() == 2)) {
    mPath->setQuality(CLOWQ);
  } else if (mPath->nprimitivesUp() >= 4 || mPath->nprimitivesDown() >= 4) {
    mPath->setQuality(HIGHQ);
  } else if (mPath->nprimitivesUp() == 3 || mPath->nprimitivesDown() == 3) {
    mPath->setQuality(LOWQ);
  }
}
