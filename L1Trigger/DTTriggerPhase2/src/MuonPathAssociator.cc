#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAssociator.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyticAnalyzer.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAssociator::MuonPathAssociator(const ParameterSet &pset,
                                       edm::ConsumesCollector &iC,
                                       std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer)
    : debug_(pset.getUntrackedParameter<bool>("debug")),
      clean_chi2_correlation_(pset.getParameter<bool>("clean_chi2_correlation")),
      useBX_correlation_(pset.getParameter<bool>("useBX_correlation")),
      allow_confirmation_(pset.getParameter<bool>("allow_confirmation")),
      dT0_correlate_TP_(pset.getParameter<double>("dT0_correlate_TP")),
      dBX_correlate_TP_(pset.getParameter<int>("dBX_correlate_TP")),
      dTanPsi_correlate_TP_(pset.getParameter<double>("dTanPsi_correlate_TP")),
      minx_match_2digis_(pset.getParameter<double>("minx_match_2digis")),
      chi2corTh_(pset.getParameter<double>("chi2corTh")) {
  // Obtention of parameters

  if (debug_)
    LogDebug("MuonPathAssociator") << "MuonPathAssociator: constructor";

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

  dtGeomH_ = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
  globalcoordsobtainer_ = globalcoordsobtainer;
}

MuonPathAssociator::~MuonPathAssociator() {
  if (debug_)
    LogDebug("MuonPathAssociator") << "MuonPathAssociator: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAssociator::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("MuonPathAssociator") << "MuonPathAssociator::initialiase";

  auto geom = iEventSetup.getHandle(dtGeomH_);
  dtGeo_ = &(*geom);
}

void MuonPathAssociator::run(edm::Event &iEvent,
                             const edm::EventSetup &iEventSetup,
                             edm::Handle<DTDigiCollection> digis,
                             std::vector<metaPrimitive> &inMPaths,
                             std::vector<metaPrimitive> &outMPaths) {
  if (dT0_correlate_TP_)
    correlateMPaths(digis, inMPaths, outMPaths);
  else {
    outMPaths.insert(outMPaths.end(), inMPaths.begin(), inMPaths.end());
  }
}

void MuonPathAssociator::finish() {
  if (debug_)
    LogDebug("MuonPathAssociator") << "MuonPathAssociator: finish";
};

void MuonPathAssociator::correlateMPaths(edm::Handle<DTDigiCollection> dtdigis,
                                         std::vector<metaPrimitive> &inMPaths,
                                         std::vector<metaPrimitive> &outMPaths) {
  if (debug_)
    LogDebug("MuonPathAssociator") << "starting correlation";

  for (int wh = -2; wh <= 2; wh++) {      //wheel num: -2, -1, 0, +1, +2
    for (int st = 1; st <= 4; st++) {     //station num (MB): 1, 2, 3, 4
      for (int se = 1; se <= 14; se++) {  //sector number: 1-12, special sectors 13, 14 to account for bigger MB4s
        if (se >= 13 && st != 4)
          continue;

        DTChamberId ChId(wh, st, se);
        DTSuperLayerId sl1Id(wh, st, se, 1);
        DTSuperLayerId sl3Id(wh, st, se, 3);

        //filterSL1
        std::vector<metaPrimitive> SL1metaPrimitives;
        for (const auto &metaprimitiveIt : inMPaths) {
          if (metaprimitiveIt.rawId == sl1Id.rawId())
            SL1metaPrimitives.push_back(metaprimitiveIt);
        }

        //filterSL3
        std::vector<metaPrimitive> SL3metaPrimitives;
        for (const auto &metaprimitiveIt : inMPaths) {
          if (metaprimitiveIt.rawId == sl3Id.rawId())
            SL3metaPrimitives.push_back(metaprimitiveIt);
        }

        if (SL1metaPrimitives.empty() and SL3metaPrimitives.empty())
          continue;

        if (debug_)
          LogDebug("MuonPathAssociator") << "correlating " << SL1metaPrimitives.size() << " metaPrim in SL1 and "
                                         << SL3metaPrimitives.size() << " in SL3 for " << sl3Id;

        bool at_least_one_correlation = false;
        bool at_least_one_SL1_confirmation = false;
        bool at_least_one_SL3_confirmation = false;

        bool useFitSL1[SL1metaPrimitives.size()];
        for (unsigned int i = 0; i < SL1metaPrimitives.size(); i++)
          useFitSL1[i] = false;
        bool useFitSL3[SL3metaPrimitives.size()];
        for (unsigned int i = 0; i < SL3metaPrimitives.size(); i++)
          useFitSL3[i] = false;

        //SL1-SL3
        vector<metaPrimitive> chamberMetaPrimitives;
        vector<metaPrimitive> confirmedMetaPrimitives;
        vector<metaPrimitive> normalMetaPrimitives;
        int sl1 = 0;
        int sl3 = 0;
        for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end();
             ++SL1metaPrimitive, sl1++, sl3 = 0) {
          if (clean_chi2_correlation_)
            at_least_one_correlation = false;
          for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end();
               ++SL3metaPrimitive, sl3++) {
            if (std::abs(SL1metaPrimitive->tanPhi - SL3metaPrimitive->tanPhi) > dTanPsi_correlate_TP_)
              continue;  //TanPsi match, SliceTest only
            if (useBX_correlation_) {
              if (abs(round(SL1metaPrimitive->t0 / (float)LHC_CLK_FREQ) -
                      round(SL3metaPrimitive->t0 / (float)LHC_CLK_FREQ)) > dBX_correlate_TP_)
                continue;  //BX match
            } else {
              if (std::abs(SL1metaPrimitive->t0 - SL3metaPrimitive->t0) >= dT0_correlate_TP_)
                continue;  //time match
            }
            long int PosSL1 = (int)round(INCREASED_RES_POS_POW * 10 * SL1metaPrimitive->x);
            long int PosSL3 = (int)round(INCREASED_RES_POS_POW * 10 * SL3metaPrimitive->x);
            double NewSlope = -999.;

            long int pos = (PosSL3 + PosSL1) / 2;
            // FW always rounds down (e.g 29.5 -> 29, -29.5 -> -30). For negative numbers, we don't do the same.
            // Let's fix it (this also happens for the slope)
            if (((PosSL3 + PosSL1) % 2 != 0) && (pos < 0)) {
              pos--;
            }

            long int difPos_mm_x4 = PosSL3 - PosSL1;
            long int tanPsi_x4096_x128 = (difPos_mm_x4)*VERT_PHI1_PHI3_INV;
            long int tanpsi = tanPsi_x4096_x128 / ((long int)pow(2, 5 + INCREASED_RES_POS));
            if (tanpsi < 0 && tanPsi_x4096_x128 % ((long int)pow(2, 5 + INCREASED_RES_POS)) != 0)
              tanpsi--;
            NewSlope = -tanpsi / (double)INCREASED_RES_SLOPE_POW;
            double MeanT0 = (SL1metaPrimitive->t0 + SL3metaPrimitive->t0) / 2;
            double MeanPos = (PosSL3 + PosSL1) / (2. * INCREASED_RES_POS_POW * 10);

            DTSuperLayerId SLId1(SL1metaPrimitive->rawId);
            DTSuperLayerId SLId3(SL3metaPrimitive->rawId);
            DTWireId wireId1(SLId1, 2, 1);
            DTWireId wireId3(SLId3, 2, 1);

            int shift_sl1 = int(round(shiftinfo_[wireId1.rawId()] * INCREASED_RES_POS_POW * 10));
            int shift_sl3 = int(round(shiftinfo_[wireId3.rawId()] * INCREASED_RES_POS_POW * 10));
            if (shift_sl1 < shift_sl3) {
              pos -= shift_sl1;
            } else
              pos -= shift_sl3;

            int wi[8], tdc[8], lat[8];
            wi[0] = SL1metaPrimitive->wi1;
            tdc[0] = SL1metaPrimitive->tdc1;
            lat[0] = SL1metaPrimitive->lat1;
            wi[1] = SL1metaPrimitive->wi2;
            tdc[1] = SL1metaPrimitive->tdc2;
            lat[1] = SL1metaPrimitive->lat2;
            wi[2] = SL1metaPrimitive->wi3;
            tdc[2] = SL1metaPrimitive->tdc3;
            lat[2] = SL1metaPrimitive->lat3;
            wi[3] = SL1metaPrimitive->wi4;
            tdc[3] = SL1metaPrimitive->tdc4;
            lat[3] = SL1metaPrimitive->lat4;
            wi[4] = SL3metaPrimitive->wi1;
            tdc[4] = SL3metaPrimitive->tdc1;
            lat[4] = SL3metaPrimitive->lat1;
            wi[5] = SL3metaPrimitive->wi2;
            tdc[5] = SL3metaPrimitive->tdc2;
            lat[5] = SL3metaPrimitive->lat2;
            wi[6] = SL3metaPrimitive->wi3;
            tdc[6] = SL3metaPrimitive->tdc3;
            lat[6] = SL3metaPrimitive->lat3;
            wi[7] = SL3metaPrimitive->wi4;
            tdc[7] = SL3metaPrimitive->tdc4;
            lat[7] = SL3metaPrimitive->lat4;

            long int chi2 = 0;

            long int Z_FACTOR_CORR[8] = {-6, -2, 2, 6, -6, -2, 2, 6};

            for (int i = 0; i < 8; i++) {
              int sign = 2 * (i / 4) - 1;
              Z_FACTOR_CORR[i] = Z_FACTOR_CORR[i] * CELL_HEIGHT + CH_CENTER_TO_MID_SL_X2 * sign;
            }
            long int sum_A, sum_B;
            for (int i = 0; i < NUM_LAYERS_2SL; i++) {
              long int shift, slTime;
              if (i / NUM_LAYERS == 0) {  // layers 0 - 3 -> SL1
                shift = shift_sl1;
                slTime = SL1metaPrimitive->t0;
              } else {  // layers 4 - 7 -> SL3
                shift = shift_sl3;
                slTime = SL3metaPrimitive->t0;
              }
              if (wi[i] != -1) {
                long int drift_dist_um_x4 = DRIFT_SPEED_X4 * (((long int)tdc[i]) - slTime);
                long int wireHorizPos_x4 =
                    (CELL_LENGTH * wi[i] + ((i + 1) % 2) * CELL_SEMILENGTH) * INCREASED_RES_POS_POW;
                long int pos_mm_x4;

                if (lat[i] == 0) {
                  pos_mm_x4 = wireHorizPos_x4 - (drift_dist_um_x4 >> 10);
                } else {
                  pos_mm_x4 = wireHorizPos_x4 + (drift_dist_um_x4 >> 10);
                }
                sum_A = shift + pos_mm_x4 - (long int)round(MeanPos * 10 * INCREASED_RES_POS_POW);
                sum_A = sum_A << (14 - INCREASED_RES_POS);
                sum_B = Z_FACTOR_CORR[i] * (long int)round(-NewSlope * INCREASED_RES_SLOPE_POW);
                chi2 += ((sum_A - sum_B) * (sum_A - sum_B)) >> 2;
              }
            }

            double newChi2 = (double)(chi2 >> INCREASED_RES_POS_POW) / (1024. * 100.);

            if (newChi2 > chi2corTh_)
              continue;

            // Fill the used vectors
            useFitSL1[sl1] = true;
            useFitSL3[sl3] = true;

            int quality = 0;
            if (SL3metaPrimitive->quality == LOWQ and SL1metaPrimitive->quality == LOWQ)
              quality = LOWLOWQ;

            if ((SL3metaPrimitive->quality == HIGHQ && SL1metaPrimitive->quality == LOWQ) or
                (SL1metaPrimitive->quality == HIGHQ && SL3metaPrimitive->quality == LOWQ))
              quality = HIGHLOWQ;

            if (SL3metaPrimitive->quality == HIGHQ && SL1metaPrimitive->quality == HIGHQ)
              quality = HIGHHIGHQ;

            double phi = -999.;
            double phiB = -999.;
            double phi_cmssw = -999.;
            double phiB_cmssw = -999.;
            double z = 0;
            if (ChId.station() >= 3)
              z = Z_SHIFT_MB4;
            GlobalPoint jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(
                LocalPoint(MeanPos, 0., z));  //Jm_x is already extrapolated to the middle of the SL
            int thisec = ChId.sector();
            if (se == 13)
              thisec = 4;
            if (se == 14)
              thisec = 10;
            phi_cmssw = jm_x_cmssw_global.phi() - PHI_CONV * (thisec - 1);
            double psi = atan(NewSlope);
            phiB_cmssw = hasPosRF(ChId.wheel(), ChId.sector()) ? psi - phi_cmssw : -psi - phi_cmssw;

            auto global_coords = globalcoordsobtainer_->get_global_coordinates(ChId.rawId(), 0, pos, tanpsi);
            phi = global_coords[0];
            phiB = global_coords[1];

            if (!clean_chi2_correlation_)
              outMPaths.emplace_back(ChId.rawId(),
                                     MeanT0,
                                     MeanPos,
                                     NewSlope,
                                     phi,
                                     phiB,
                                     phi_cmssw,
                                     phiB_cmssw,
                                     newChi2,
                                     quality,
                                     SL1metaPrimitive->wi1,
                                     SL1metaPrimitive->tdc1,
                                     SL1metaPrimitive->lat1,
                                     SL1metaPrimitive->wi2,
                                     SL1metaPrimitive->tdc2,
                                     SL1metaPrimitive->lat2,
                                     SL1metaPrimitive->wi3,
                                     SL1metaPrimitive->tdc3,
                                     SL1metaPrimitive->lat3,
                                     SL1metaPrimitive->wi4,
                                     SL1metaPrimitive->tdc4,
                                     SL1metaPrimitive->lat4,
                                     SL3metaPrimitive->wi1,
                                     SL3metaPrimitive->tdc1,
                                     SL3metaPrimitive->lat1,
                                     SL3metaPrimitive->wi2,
                                     SL3metaPrimitive->tdc2,
                                     SL3metaPrimitive->lat2,
                                     SL3metaPrimitive->wi3,
                                     SL3metaPrimitive->tdc3,
                                     SL3metaPrimitive->lat3,
                                     SL3metaPrimitive->wi4,
                                     SL3metaPrimitive->tdc4,
                                     SL3metaPrimitive->lat4);
            else
              chamberMetaPrimitives.emplace_back(ChId.rawId(),
                                                 MeanT0,
                                                 MeanPos,
                                                 NewSlope,
                                                 phi,
                                                 phiB,
                                                 phi_cmssw,
                                                 phiB_cmssw,
                                                 newChi2,
                                                 quality,
                                                 SL1metaPrimitive->wi1,
                                                 SL1metaPrimitive->tdc1,
                                                 SL1metaPrimitive->lat1,
                                                 SL1metaPrimitive->wi2,
                                                 SL1metaPrimitive->tdc2,
                                                 SL1metaPrimitive->lat2,
                                                 SL1metaPrimitive->wi3,
                                                 SL1metaPrimitive->tdc3,
                                                 SL1metaPrimitive->lat3,
                                                 SL1metaPrimitive->wi4,
                                                 SL1metaPrimitive->tdc4,
                                                 SL1metaPrimitive->lat4,
                                                 SL3metaPrimitive->wi1,
                                                 SL3metaPrimitive->tdc1,
                                                 SL3metaPrimitive->lat1,
                                                 SL3metaPrimitive->wi2,
                                                 SL3metaPrimitive->tdc2,
                                                 SL3metaPrimitive->lat2,
                                                 SL3metaPrimitive->wi3,
                                                 SL3metaPrimitive->tdc3,
                                                 SL3metaPrimitive->lat3,
                                                 SL3metaPrimitive->wi4,
                                                 SL3metaPrimitive->tdc4,
                                                 SL3metaPrimitive->lat4);

            at_least_one_correlation = true;
          }

          if (at_least_one_correlation == false &&
              allow_confirmation_ == true) {  //no correlation was found, trying with pairs of two digis in the other SL
            int matched_digis = 0;
            double minx = minx_match_2digis_;
            double min2x = minx_match_2digis_;
            int best_tdc = -1;
            int next_tdc = -1;
            int best_wire = -1;
            int next_wire = -1;
            int best_layer = -1;
            int next_layer = -1;
            int best_lat = -1;
            int next_lat = -1;
            int lat = -1;
            for (const auto &dtLayerId_It : *dtdigis) {
              const DTLayerId dtLId = dtLayerId_It.first;
              // creating a new DTSuperLayerId object to compare with the required SL id
              const DTSuperLayerId dtSLId(dtLId.wheel(), dtLId.station(), dtLId.sector(), dtLId.superLayer());
              if (dtSLId.rawId() != sl3Id.rawId())
                continue;
              double l_shift = 0;
              if (dtLId.layer() == 4)
                l_shift = X_POS_L4;
              else if (dtLId.layer() == 3)
                l_shift = X_POS_L3;
              else if (dtLId.layer() == 2)
                l_shift = -1 * X_POS_L3;
              else if (dtLId.layer() == 1)
                l_shift = -1 * X_POS_L4;
              double x_inSL3 = SL1metaPrimitive->x - SL1metaPrimitive->tanPhi * (VERT_PHI1_PHI3 + l_shift);
              for (auto digiIt = (dtLayerId_It.second).first; digiIt != (dtLayerId_It.second).second; ++digiIt) {
                DTWireId wireId(dtLId, (*digiIt).wire());
                if ((*digiIt).time() < SL1metaPrimitive->t0)
                  continue;
                double x_wire =
                    shiftinfo_[wireId.rawId()] + ((*digiIt).time() - SL1metaPrimitive->t0) * DRIFT_SPEED / 10.;
                double x_wire_left =
                    shiftinfo_[wireId.rawId()] - ((*digiIt).time() - SL1metaPrimitive->t0) * DRIFT_SPEED / 10.;
                lat = 1;
                if (std::abs(x_inSL3 - x_wire) > std::abs(x_inSL3 - x_wire_left)) {
                  x_wire = x_wire_left;  //choose the closest laterality
                  lat = 0;
                }
                if (std::abs(x_inSL3 - x_wire) < minx) {
                  // different layer than the stored in best, hit added, matched_digis++;. This approach in somewhat
                  // buggy, as we could have stored as best LayerX -> LayerY -> LayerX, and this should
                  // count only as 2 hits. However, as we confirm with at least 2 hits, having 2 or more
                  // makes no difference
                  if (dtLId.layer() != best_layer) {
                    minx = std::abs(x_inSL3 - x_wire);
                    next_wire = best_wire;
                    next_tdc = best_tdc;
                    next_layer = best_layer;
                    next_lat = best_lat;
                    matched_digis++;
                  }
                  best_wire = (*digiIt).wire();
                  best_tdc = (*digiIt).time();
                  best_layer = dtLId.layer();
                  best_lat = lat;

                } else if ((std::abs(x_inSL3 - x_wire) >= minx) && (std::abs(x_inSL3 - x_wire) < min2x)) {
                  // same layer than the stored in best, no hit added
                  if (dtLId.layer() == best_layer)
                    continue;
                  // different layer than the stored in next, hit added. This approach in somewhat
                  // buggy, as we could have stored as next LayerX -> LayerY -> LayerX, and this should
                  // count only as 2 hits. However, as we confirm with at least 2 hits, having 2 or more
                  // makes no difference
                  matched_digis++;
                  // whether the layer is the same for this hit and the stored in next, we substitute
                  // the one stored and modify the min distance
                  min2x = std::abs(x_inSL3 - x_wire);
                  next_wire = (*digiIt).wire();
                  next_tdc = (*digiIt).time();
                  next_layer = dtLId.layer();
                  next_lat = lat;
                }
              }
            }
            if (matched_digis >= 2 and best_layer != -1 and next_layer != -1) {
              int new_quality = CHIGHQ;
              if (SL1metaPrimitive->quality == LOWQ)
                new_quality = CLOWQ;

              int wi1 = -1;
              int tdc1 = -1;
              int lat1 = -1;
              int wi2 = -1;
              int tdc2 = -1;
              int lat2 = -1;
              int wi3 = -1;
              int tdc3 = -1;
              int lat3 = -1;
              int wi4 = -1;
              int tdc4 = -1;
              int lat4 = -1;

              if (next_layer == 1) {
                wi1 = next_wire;
                tdc1 = next_tdc;
                lat1 = next_lat;
              }
              if (next_layer == 2) {
                wi2 = next_wire;
                tdc2 = next_tdc;
                lat2 = next_lat;
              }
              if (next_layer == 3) {
                wi3 = next_wire;
                tdc3 = next_tdc;
                lat3 = next_lat;
              }
              if (next_layer == 4) {
                wi4 = next_wire;
                tdc4 = next_tdc;
                lat4 = next_lat;
              }

              if (best_layer == 1) {
                wi1 = best_wire;
                tdc1 = best_tdc;
                lat1 = best_lat;
              }
              if (best_layer == 2) {
                wi2 = best_wire;
                tdc2 = best_tdc;
                lat2 = best_lat;
              }
              if (best_layer == 3) {
                wi3 = best_wire;
                tdc3 = best_tdc;
                lat3 = best_lat;
              }
              if (best_layer == 4) {
                wi4 = best_wire;
                tdc4 = best_tdc;
                lat4 = best_lat;
              }

              if (!clean_chi2_correlation_)
                outMPaths.emplace_back(metaPrimitive({ChId.rawId(),
                                                      SL1metaPrimitive->t0,
                                                      SL1metaPrimitive->x,
                                                      SL1metaPrimitive->tanPhi,
                                                      SL1metaPrimitive->phi,
                                                      SL1metaPrimitive->phiB,
                                                      SL1metaPrimitive->phi_cmssw,
                                                      SL1metaPrimitive->phiB_cmssw,
                                                      SL1metaPrimitive->chi2,
                                                      new_quality,
                                                      SL1metaPrimitive->wi1,
                                                      SL1metaPrimitive->tdc1,
                                                      SL1metaPrimitive->lat1,
                                                      SL1metaPrimitive->wi2,
                                                      SL1metaPrimitive->tdc2,
                                                      SL1metaPrimitive->lat2,
                                                      SL1metaPrimitive->wi3,
                                                      SL1metaPrimitive->tdc3,
                                                      SL1metaPrimitive->lat3,
                                                      SL1metaPrimitive->wi4,
                                                      SL1metaPrimitive->tdc4,
                                                      SL1metaPrimitive->lat4,
                                                      wi1,
                                                      tdc1,
                                                      lat1,
                                                      wi2,
                                                      tdc2,
                                                      lat2,
                                                      wi3,
                                                      tdc3,
                                                      lat3,
                                                      wi4,
                                                      tdc4,
                                                      lat4,
                                                      -1}));
              else
                confirmedMetaPrimitives.emplace_back(metaPrimitive({ChId.rawId(),
                                                                    SL1metaPrimitive->t0,
                                                                    SL1metaPrimitive->x,
                                                                    SL1metaPrimitive->tanPhi,
                                                                    SL1metaPrimitive->phi,
                                                                    SL1metaPrimitive->phiB,
                                                                    SL1metaPrimitive->phi_cmssw,
                                                                    SL1metaPrimitive->phiB_cmssw,
                                                                    SL1metaPrimitive->chi2,
                                                                    new_quality,
                                                                    SL1metaPrimitive->wi1,
                                                                    SL1metaPrimitive->tdc1,
                                                                    SL1metaPrimitive->lat1,
                                                                    SL1metaPrimitive->wi2,
                                                                    SL1metaPrimitive->tdc2,
                                                                    SL1metaPrimitive->lat2,
                                                                    SL1metaPrimitive->wi3,
                                                                    SL1metaPrimitive->tdc3,
                                                                    SL1metaPrimitive->lat3,
                                                                    SL1metaPrimitive->wi4,
                                                                    SL1metaPrimitive->tdc4,
                                                                    SL1metaPrimitive->lat4,
                                                                    wi1,
                                                                    tdc1,
                                                                    lat1,
                                                                    wi2,
                                                                    tdc2,
                                                                    lat2,
                                                                    wi3,
                                                                    tdc3,
                                                                    lat3,
                                                                    wi4,
                                                                    tdc4,
                                                                    lat4,
                                                                    -1}));
              useFitSL1[sl1] = true;
              at_least_one_SL1_confirmation = true;
            }
          }
        }

        //finish SL1-SL3

        //SL3-SL1
        sl3 = 0;
        for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end();
             ++SL3metaPrimitive, sl3++) {
          if (useFitSL3[sl3])
            continue;
          if ((at_least_one_correlation == false || clean_chi2_correlation_) &&
              allow_confirmation_) {  //no correlation was found, trying with pairs of two digis in the other SL

            int matched_digis = 0;
            double minx = minx_match_2digis_;
            double min2x = minx_match_2digis_;
            int best_tdc = -1;
            int next_tdc = -1;
            int best_wire = -1;
            int next_wire = -1;
            int best_layer = -1;
            int next_layer = -1;
            int best_lat = -1;
            int next_lat = -1;
            int lat = -1;

            for (const auto &dtLayerId_It : *dtdigis) {
              const DTLayerId dtLId = dtLayerId_It.first;
              // creating a new DTSuperLayerId object to compare with the required SL id
              const DTSuperLayerId dtSLId(dtLId.wheel(), dtLId.station(), dtLId.sector(), dtLId.superLayer());
              if (dtSLId.rawId() != sl1Id.rawId())
                continue;
              double l_shift = 0;
              if (dtLId.layer() == 4)
                l_shift = X_POS_L4;
              if (dtLId.layer() == 3)
                l_shift = X_POS_L3;
              if (dtLId.layer() == 2)
                l_shift = -1 * X_POS_L3;
              if (dtLId.layer() == 1)
                l_shift = -1 * X_POS_L4;
              double x_inSL1 = SL3metaPrimitive->x + SL3metaPrimitive->tanPhi * (VERT_PHI1_PHI3 - l_shift);
              for (auto digiIt = (dtLayerId_It.second).first; digiIt != (dtLayerId_It.second).second; ++digiIt) {
                DTWireId wireId(dtLId, (*digiIt).wire());
                if ((*digiIt).time() < SL3metaPrimitive->t0)
                  continue;
                double x_wire =
                    shiftinfo_[wireId.rawId()] + ((*digiIt).time() - SL3metaPrimitive->t0) * DRIFT_SPEED / 10.;
                double x_wire_left =
                    shiftinfo_[wireId.rawId()] - ((*digiIt).time() - SL3metaPrimitive->t0) * DRIFT_SPEED / 10.;
                lat = 1;
                if (std::abs(x_inSL1 - x_wire) > std::abs(x_inSL1 - x_wire_left)) {
                  x_wire = x_wire_left;  //choose the closest laterality
                  lat = 0;
                }
                if (std::abs(x_inSL1 - x_wire) < minx) {
                  // different layer than the stored in best, hit added, matched_digis++;. This approach in somewhat
                  // buggy, as we could have stored as best LayerX -> LayerY -> LayerX, and this should
                  // count only as 2 hits. However, as we confirm with at least 2 hits, having 2 or more
                  // makes no difference
                  if (dtLId.layer() != best_layer) {
                    minx = std::abs(x_inSL1 - x_wire);
                    next_wire = best_wire;
                    next_tdc = best_tdc;
                    next_layer = best_layer;
                    next_lat = best_lat;
                    matched_digis++;
                  }
                  best_wire = (*digiIt).wire();
                  best_tdc = (*digiIt).time();
                  best_layer = dtLId.layer();
                  best_lat = lat;
                } else if ((std::abs(x_inSL1 - x_wire) >= minx) && (std::abs(x_inSL1 - x_wire) < min2x)) {
                  // same layer than the stored in best, no hit added
                  if (dtLId.layer() == best_layer)
                    continue;
                  // different layer than the stored in next, hit added. This approach in somewhat
                  // buggy, as we could have stored as next LayerX -> LayerY -> LayerX, and this should
                  // count only as 2 hits. However, as we confirm with at least 2 hits, having 2 or more
                  // makes no difference
                  matched_digis++;
                  // whether the layer is the same for this hit and the stored in next, we substitute
                  // the one stored and modify the min distance
                  min2x = std::abs(x_inSL1 - x_wire);
                  next_wire = (*digiIt).wire();
                  next_tdc = (*digiIt).time();
                  next_layer = dtLId.layer();
                  next_lat = lat;
                }
              }
            }
            if (matched_digis >= 2 and best_layer != -1 and next_layer != -1) {
              int new_quality = CHIGHQ;
              if (SL3metaPrimitive->quality == LOWQ)
                new_quality = CLOWQ;

              int wi1 = -1;
              int tdc1 = -1;
              int lat1 = -1;
              int wi2 = -1;
              int tdc2 = -1;
              int lat2 = -1;
              int wi3 = -1;
              int tdc3 = -1;
              int lat3 = -1;
              int wi4 = -1;
              int tdc4 = -1;
              int lat4 = -1;

              if (next_layer == 1) {
                wi1 = next_wire;
                tdc1 = next_tdc;
                lat1 = next_lat;
              }
              if (next_layer == 2) {
                wi2 = next_wire;
                tdc2 = next_tdc;
                lat2 = next_lat;
              }
              if (next_layer == 3) {
                wi3 = next_wire;
                tdc3 = next_tdc;
                lat3 = next_lat;
              }
              if (next_layer == 4) {
                wi4 = next_wire;
                tdc4 = next_tdc;
                lat4 = next_lat;
              }

              if (best_layer == 1) {
                wi1 = best_wire;
                tdc1 = best_tdc;
                lat1 = best_lat;
              }
              if (best_layer == 2) {
                wi2 = best_wire;
                tdc2 = best_tdc;
                lat2 = best_lat;
              }
              if (best_layer == 3) {
                wi3 = best_wire;
                tdc3 = best_tdc;
                lat3 = best_lat;
              }
              if (best_layer == 4) {
                wi4 = best_wire;
                tdc4 = best_tdc;
                lat4 = best_lat;
              }

              if (!clean_chi2_correlation_)
                outMPaths.push_back(metaPrimitive({ChId.rawId(),
                                                   SL3metaPrimitive->t0,
                                                   SL3metaPrimitive->x,
                                                   SL3metaPrimitive->tanPhi,
                                                   SL3metaPrimitive->phi,
                                                   SL3metaPrimitive->phiB,
                                                   SL3metaPrimitive->phi_cmssw,
                                                   SL3metaPrimitive->phiB_cmssw,
                                                   SL3metaPrimitive->chi2,
                                                   new_quality,
                                                   wi1,
                                                   tdc1,
                                                   lat1,
                                                   wi2,
                                                   tdc2,
                                                   lat2,
                                                   wi3,
                                                   tdc3,
                                                   lat3,
                                                   wi4,
                                                   tdc4,
                                                   lat4,
                                                   SL3metaPrimitive->wi1,
                                                   SL3metaPrimitive->tdc1,
                                                   SL3metaPrimitive->lat1,
                                                   SL3metaPrimitive->wi2,
                                                   SL3metaPrimitive->tdc2,
                                                   SL3metaPrimitive->lat2,
                                                   SL3metaPrimitive->wi3,
                                                   SL3metaPrimitive->tdc3,
                                                   SL3metaPrimitive->lat3,
                                                   SL3metaPrimitive->wi4,
                                                   SL3metaPrimitive->tdc4,
                                                   SL3metaPrimitive->lat4,
                                                   -1}));
              else
                confirmedMetaPrimitives.push_back(metaPrimitive({ChId.rawId(),
                                                                 SL3metaPrimitive->t0,
                                                                 SL3metaPrimitive->x,
                                                                 SL3metaPrimitive->tanPhi,
                                                                 SL3metaPrimitive->phi,
                                                                 SL3metaPrimitive->phiB,
                                                                 SL3metaPrimitive->phi_cmssw,
                                                                 SL3metaPrimitive->phiB_cmssw,
                                                                 SL3metaPrimitive->chi2,
                                                                 new_quality,
                                                                 wi1,
                                                                 tdc1,
                                                                 lat1,
                                                                 wi2,
                                                                 tdc2,
                                                                 lat2,
                                                                 wi3,
                                                                 tdc3,
                                                                 lat3,
                                                                 wi4,
                                                                 tdc4,
                                                                 lat4,
                                                                 SL3metaPrimitive->wi1,
                                                                 SL3metaPrimitive->tdc1,
                                                                 SL3metaPrimitive->lat1,
                                                                 SL3metaPrimitive->wi2,
                                                                 SL3metaPrimitive->tdc2,
                                                                 SL3metaPrimitive->lat2,
                                                                 SL3metaPrimitive->wi3,
                                                                 SL3metaPrimitive->tdc3,
                                                                 SL3metaPrimitive->lat3,
                                                                 SL3metaPrimitive->wi4,
                                                                 SL3metaPrimitive->tdc4,
                                                                 SL3metaPrimitive->lat4,
                                                                 -1}));
              useFitSL3[sl3] = true;
              at_least_one_SL3_confirmation = true;
            }
          }
        }
        // Start correlation cleaning
        if (clean_chi2_correlation_) {
          if (debug_)
            LogDebug("MuonPathAssociator") << "Pushing back correlated MPs to the MPs collection";
          removeSharingFits(chamberMetaPrimitives, outMPaths);
        }
        if (clean_chi2_correlation_) {
          if (debug_)
            LogDebug("MuonPathAssociator") << "Pushing back confirmed MPs to the complete vector";
          removeSharingHits(confirmedMetaPrimitives, chamberMetaPrimitives, outMPaths);
        }

        //finish SL3-SL1
        if (at_least_one_correlation == false || clean_chi2_correlation_) {
          if (debug_ && !at_least_one_correlation)
            LogDebug("MuonPathAssociator")
                << "correlation we found zero correlations, adding both collections as they are to the outMPaths";
          if (debug_)
            LogDebug("MuonPathAssociator")
                << "correlation sizes:" << SL1metaPrimitives.size() << " " << SL3metaPrimitives.size();
          if (at_least_one_SL1_confirmation == false || clean_chi2_correlation_) {
            sl1 = 0;
            for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end();
                 ++SL1metaPrimitive, sl1++) {
              if (useFitSL1[sl1])
                continue;

              DTSuperLayerId SLId(SL1metaPrimitive->rawId);
              DTChamberId(SLId.wheel(), SLId.station(), SLId.sector());
              metaPrimitive newSL1metaPrimitive = {ChId.rawId(),
                                                   SL1metaPrimitive->t0,
                                                   SL1metaPrimitive->x,
                                                   SL1metaPrimitive->tanPhi,
                                                   SL1metaPrimitive->phi,
                                                   SL1metaPrimitive->phiB,
                                                   SL1metaPrimitive->phi_cmssw,
                                                   SL1metaPrimitive->phiB_cmssw,
                                                   SL1metaPrimitive->chi2,
                                                   SL1metaPrimitive->quality,
                                                   SL1metaPrimitive->wi1,
                                                   SL1metaPrimitive->tdc1,
                                                   SL1metaPrimitive->lat1,
                                                   SL1metaPrimitive->wi2,
                                                   SL1metaPrimitive->tdc2,
                                                   SL1metaPrimitive->lat2,
                                                   SL1metaPrimitive->wi3,
                                                   SL1metaPrimitive->tdc3,
                                                   SL1metaPrimitive->lat3,
                                                   SL1metaPrimitive->wi4,
                                                   SL1metaPrimitive->tdc4,
                                                   SL1metaPrimitive->lat4,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1};

              bool ok = true;
              for (auto &metaPrimitive : chamberMetaPrimitives) {
                if (!isNotAPrimo(newSL1metaPrimitive, metaPrimitive)) {
                  ok = false;
                  break;
                }
              }
              if (!ok)
                continue;

              if (!clean_chi2_correlation_)
                outMPaths.push_back(newSL1metaPrimitive);
              else
                normalMetaPrimitives.push_back(newSL1metaPrimitive);
            }
          }
          if (at_least_one_SL3_confirmation == false || clean_chi2_correlation_) {
            sl3 = 0;
            for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end();
                 ++SL3metaPrimitive, sl3++) {
              if (useFitSL3[sl3])
                continue;
              DTSuperLayerId SLId(SL3metaPrimitive->rawId);
              DTChamberId(SLId.wheel(), SLId.station(), SLId.sector());
              metaPrimitive newSL3metaPrimitive = {ChId.rawId(),
                                                   SL3metaPrimitive->t0,
                                                   SL3metaPrimitive->x,
                                                   SL3metaPrimitive->tanPhi,
                                                   SL3metaPrimitive->phi,
                                                   SL3metaPrimitive->phiB,
                                                   SL3metaPrimitive->phi_cmssw,
                                                   SL3metaPrimitive->phiB_cmssw,
                                                   SL3metaPrimitive->chi2,
                                                   SL3metaPrimitive->quality,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   SL3metaPrimitive->wi1,
                                                   SL3metaPrimitive->tdc1,
                                                   SL3metaPrimitive->lat1,
                                                   SL3metaPrimitive->wi2,
                                                   SL3metaPrimitive->tdc2,
                                                   SL3metaPrimitive->lat2,
                                                   SL3metaPrimitive->wi3,
                                                   SL3metaPrimitive->tdc3,
                                                   SL3metaPrimitive->lat3,
                                                   SL3metaPrimitive->wi4,
                                                   SL3metaPrimitive->tdc4,
                                                   SL3metaPrimitive->lat4,
                                                   -1};

              if (!clean_chi2_correlation_)
                outMPaths.push_back(newSL3metaPrimitive);
              else
                normalMetaPrimitives.push_back(newSL3metaPrimitive);
            }
          }
        }

        SL1metaPrimitives.clear();
        SL1metaPrimitives.erase(SL1metaPrimitives.begin(), SL1metaPrimitives.end());
        SL3metaPrimitives.clear();
        SL3metaPrimitives.erase(SL3metaPrimitives.begin(), SL3metaPrimitives.end());

        vector<metaPrimitive> auxMetaPrimitives;
        if (clean_chi2_correlation_) {
          if (debug_)
            LogDebug("MuonPathAssociator") << "Pushing back normal MPs to the auxiliar vector";
          removeSharingHits(normalMetaPrimitives, confirmedMetaPrimitives, auxMetaPrimitives);
        }
        if (clean_chi2_correlation_) {
          if (debug_)
            LogDebug("MuonPathAssociator") << "Pushing back normal MPs to the MPs collection";
          removeSharingHits(auxMetaPrimitives, chamberMetaPrimitives, outMPaths);
        }
      }
    }
  }

  //eta TP we do not correlate with other superlayer in the same chamber so we forward them all
  std::vector<metaPrimitive> SL2metaPrimitives;

  for (int wh = -2; wh <= 2; wh++) {
    for (int st = 1; st <= 4; st++) {
      for (int se = 1; se <= 14; se++) {
        if (se >= 13 && st != 4)
          continue;

        DTChamberId ChId(wh, st, se);
        DTSuperLayerId sl2Id(wh, st, se, 2);

        //filterSL2 etaTP
        for (auto metaprimitiveIt = inMPaths.begin(); metaprimitiveIt != inMPaths.end(); ++metaprimitiveIt)
          if (metaprimitiveIt->rawId == sl2Id.rawId()) {
            SL2metaPrimitives.push_back(*metaprimitiveIt);
            if (debug_)
              printmPC(*metaprimitiveIt);
            outMPaths.push_back(*metaprimitiveIt);
          }
      }
    }
  }

  LogDebug("MuonPathAssociator") << "\t etaTP: added " << SL2metaPrimitives.size() << "to outMPaths" << std::endl;

  SL2metaPrimitives.clear();
  SL2metaPrimitives.erase(SL2metaPrimitives.begin(), SL2metaPrimitives.end());
}

void MuonPathAssociator::removeSharingFits(vector<metaPrimitive> &chamberMPaths, vector<metaPrimitive> &allMPaths) {
  bool useFit[chamberMPaths.size()];
  for (unsigned int i = 0; i < chamberMPaths.size(); i++) {
    useFit[i] = true;
  }
  for (unsigned int i = 0; i < chamberMPaths.size(); i++) {
    if (debug_)
      LogDebug("MuonPathAssociator") << "Looking at prim" << i;
    if (!useFit[i])
      continue;
    for (unsigned int j = i + 1; j < chamberMPaths.size(); j++) {
      if (debug_)
        LogDebug("MuonPathAssociator") << "Comparing with prim " << j;
      if (!useFit[j])
        continue;
      metaPrimitive first = chamberMPaths[i];
      metaPrimitive second = chamberMPaths[j];
      if (shareFit(first, second)) {
        if (first.quality > second.quality)
          useFit[j] = false;
        else if (first.quality < second.quality)
          useFit[i] = false;
        else {
          if (first.chi2 < second.chi2)
            useFit[j] = false;
          else {
            useFit[i] = false;
            break;
          }
        }
      }
    }
    if (useFit[i]) {
      if (debug_)
        printmPC(chamberMPaths[i]);
      allMPaths.push_back(chamberMPaths[i]);
    }
  }
  if (debug_)
    LogDebug("MuonPathAssociator") << "---Swapping chamber---";
}

void MuonPathAssociator::removeSharingHits(std::vector<metaPrimitive> &firstMPaths,
                                           std::vector<metaPrimitive> &secondMPaths,
                                           std::vector<metaPrimitive> &allMPaths) {
  for (auto &firstMP : firstMPaths) {
    if (debug_)
      LogDebug("MuonPathAssociator") << "----------------------------------";
    if (debug_)
      LogDebug("MuonPathAssociator") << "Turn for ";
    if (debug_)
      printmPC(firstMP);
    bool ok = true;
    for (auto &secondMP : secondMPaths) {
      if (debug_)
        LogDebug("MuonPathAssociator") << "Comparing with ";
      if (debug_)
        printmPC(secondMP);
      if (!isNotAPrimo(firstMP, secondMP)) {
        ok = false;
        break;
      }
    }
    if (ok) {
      allMPaths.push_back(firstMP);
      if (debug_)
        printmPC(firstMP);
    }
    if (debug_)
      LogDebug("MuonPathAssociator") << "----------------------------------";
  }
}

bool MuonPathAssociator::shareFit(metaPrimitive first, metaPrimitive second) {
  bool lay1 = (first.wi1 == second.wi1) && (first.tdc1 = second.tdc1);
  bool lay2 = (first.wi2 == second.wi2) && (first.tdc2 = second.tdc2);
  bool lay3 = (first.wi3 == second.wi3) && (first.tdc3 = second.tdc3);
  bool lay4 = (first.wi4 == second.wi4) && (first.tdc4 = second.tdc4);
  bool lay5 = (first.wi5 == second.wi5) && (first.tdc5 = second.tdc5);
  bool lay6 = (first.wi6 == second.wi6) && (first.tdc6 = second.tdc6);
  bool lay7 = (first.wi7 == second.wi7) && (first.tdc7 = second.tdc7);
  bool lay8 = (first.wi8 == second.wi8) && (first.tdc8 = second.tdc8);

  if (lay1 && lay2 && lay3 && lay4) {
    if (lay5 || lay6 || lay7 || lay8)
      return true;
    else
      return false;
  } else if (lay5 && lay6 && lay7 && lay8) {
    if (lay1 || lay2 || lay3 || lay4)
      return true;
    else
      return false;
  } else
    return false;
}

bool MuonPathAssociator::isNotAPrimo(metaPrimitive first, metaPrimitive second) {
  int hitsSL1 = (first.wi1 != -1) + (first.wi2 != -1) + (first.wi3 != -1) + (first.wi4 != -1);
  int hitsSL3 = (first.wi5 != -1) + (first.wi6 != -1) + (first.wi7 != -1) + (first.wi8 != -1);

  bool lay1 = (first.wi1 == second.wi1) && (first.tdc1 = second.tdc1) && (first.wi1 != -1);
  bool lay2 = (first.wi2 == second.wi2) && (first.tdc2 = second.tdc2) && (first.wi2 != -1);
  bool lay3 = (first.wi3 == second.wi3) && (first.tdc3 = second.tdc3) && (first.wi3 != -1);
  bool lay4 = (first.wi4 == second.wi4) && (first.tdc4 = second.tdc4) && (first.wi4 != -1);
  bool lay5 = (first.wi5 == second.wi5) && (first.tdc5 = second.tdc5) && (first.wi5 != -1);
  bool lay6 = (first.wi6 == second.wi6) && (first.tdc6 = second.tdc6) && (first.wi6 != -1);
  bool lay7 = (first.wi7 == second.wi7) && (first.tdc7 = second.tdc7) && (first.wi7 != -1);
  bool lay8 = (first.wi8 == second.wi8) && (first.tdc8 = second.tdc8) && (first.wi8 != -1);

  return (((!lay1 && !lay2 && !lay3 && !lay4) || hitsSL1 < 3) && ((!lay5 && !lay6 && !lay7 && !lay8) || hitsSL3 < 3));
}

void MuonPathAssociator::printmPC(metaPrimitive mP) {
  DTChamberId ChId(mP.rawId);
  LogDebug("MuonPathAssociator") << ChId << "\t"
                                 << " " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " "
                                 << setw(2) << left << mP.wi3 << " " << setw(2) << left << mP.wi4 << " " << setw(2)
                                 << left << mP.wi5 << " " << setw(2) << left << mP.wi6 << " " << setw(2) << left
                                 << mP.wi7 << " " << setw(2) << left << mP.wi8 << " " << setw(5) << left << mP.tdc1
                                 << " " << setw(5) << left << mP.tdc2 << " " << setw(5) << left << mP.tdc3 << " "
                                 << setw(5) << left << mP.tdc4 << " " << setw(5) << left << mP.tdc5 << " " << setw(5)
                                 << left << mP.tdc6 << " " << setw(5) << left << mP.tdc7 << " " << setw(5) << left
                                 << mP.tdc8 << " " << setw(2) << left << mP.lat1 << " " << setw(2) << left << mP.lat2
                                 << " " << setw(2) << left << mP.lat3 << " " << setw(2) << left << mP.lat4 << " "
                                 << setw(2) << left << mP.lat5 << " " << setw(2) << left << mP.lat6 << " " << setw(2)
                                 << left << mP.lat7 << " " << setw(2) << left << mP.lat8 << " " << setw(10) << right
                                 << mP.x << " " << setw(9) << left << mP.tanPhi << " " << setw(5) << left << mP.t0
                                 << " " << setw(13) << left << mP.chi2 << " \n";
}
