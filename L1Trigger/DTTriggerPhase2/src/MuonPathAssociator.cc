#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAssociator.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzerPerSL.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAssociator::MuonPathAssociator(const ParameterSet &pset, edm::ConsumesCollector &iC) {
  // Obtention of parameters
  debug = pset.getUntrackedParameter<bool>("debug");
  clean_chi2_correlation = pset.getUntrackedParameter<bool>("clean_chi2_correlation");
  use_LSB = pset.getUntrackedParameter<bool>("use_LSB");
  tanPsi_precision = pset.getUntrackedParameter<double>("tanPsi_precision");
  x_precision = pset.getUntrackedParameter<double>("x_precision");
  useBX_correlation = pset.getUntrackedParameter<bool>("useBX_correlation");
  allow_confirmation = pset.getUntrackedParameter<bool>("allow_confirmation");
  dT0_correlate_TP = pset.getUntrackedParameter<double>("dT0_correlate_TP");
  dBX_correlate_TP = pset.getUntrackedParameter<int>("dBX_correlate_TP");
  dTanPsi_correlate_TP = pset.getUntrackedParameter<double>("dTanPsi_correlate_TP");
  minx_match_2digis = pset.getUntrackedParameter<double>("minx_match_2digis");
  chi2corTh = pset.getUntrackedParameter<double>("chi2corTh");

  if (debug)
    cout << "MuonPathAssociator: constructor" << endl;

  //shift
  int rawId;
  shift_filename = pset.getParameter<edm::FileInPath>("shift_filename");
  std::ifstream ifin3(shift_filename.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathAnalyzerPerSL::MuonPathAnalyzerPerSL() -  Cannot find " << shift_filename.fullPath();
  }
  while (ifin3.good()) {
    ifin3 >> rawId >> shift;
    shiftinfo[rawId] = shift;
  }

  dtGeomH = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

MuonPathAssociator::~MuonPathAssociator() {
  if (debug)
    cout << "MuonPathAssociator: destructor" << endl;
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAssociator::initialise(const edm::EventSetup &iEventSetup) {
  if (debug)
    cout << "MuonPathAssociator::initialiase" << endl;

  const MuonGeometryRecord &geom = iEventSetup.get<MuonGeometryRecord>();
  dtGeo_ = &geom.get(dtGeomH);
}

void MuonPathAssociator::run(edm::Event &iEvent,
                             const edm::EventSetup &iEventSetup,
                             edm::Handle<DTDigiCollection> digis,
                             std::vector<metaPrimitive> &inMPaths,
                             std::vector<metaPrimitive> &outMPaths) {
  if (dT0_correlate_TP)
    correlateMPaths(digis, inMPaths, outMPaths);
  else {
    for (auto metaPrimitiveIt = inMPaths.begin(); metaPrimitiveIt != inMPaths.end(); ++metaPrimitiveIt)
      outMPaths.push_back(*metaPrimitiveIt);
  }
}

void MuonPathAssociator::finish() {
  if (debug)
    cout << "MuonPathAssociator: finish" << endl;
};

void MuonPathAssociator::correlateMPaths(edm::Handle<DTDigiCollection> dtdigis,
                                         std::vector<metaPrimitive> &inMPaths,
                                         std::vector<metaPrimitive> &outMPaths) {
  int x_prec_inv = (int)(1. / (10. * x_precision));
  int numberOfBits = (int)(round(std::log(x_prec_inv) / std::log(2.)));

  //Silvia's code for correlationg filteredMetaPrimitives

  if (debug)
    std::cout << "starting correlation" << std::endl;

  for (int wh = -2; wh <= 2; wh++) {
    for (int st = 1; st <= 4; st++) {
      for (int se = 1; se <= 14; se++) {
        if (se >= 13 && st != 4)
          continue;

        DTChamberId ChId(wh, st, se);
        DTSuperLayerId sl1Id(wh, st, se, 1);
        DTSuperLayerId sl3Id(wh, st, se, 3);

        //filterSL1
        std::vector<metaPrimitive> SL1metaPrimitives;
        for (auto metaprimitiveIt = inMPaths.begin(); metaprimitiveIt != inMPaths.end(); ++metaprimitiveIt)
          if (metaprimitiveIt->rawId == sl1Id.rawId())
            SL1metaPrimitives.push_back(*metaprimitiveIt);

        //filterSL3
        std::vector<metaPrimitive> SL3metaPrimitives;
        for (auto metaprimitiveIt = inMPaths.begin(); metaprimitiveIt != inMPaths.end(); ++metaprimitiveIt)
          if (metaprimitiveIt->rawId == sl3Id.rawId())
            SL3metaPrimitives.push_back(*metaprimitiveIt);

        if (SL1metaPrimitives.size() == 0 and SL3metaPrimitives.size() == 0)
          continue;

        if (debug)
          std::cout << "correlating " << SL1metaPrimitives.size() << " metaPrim in SL1 and " << SL3metaPrimitives.size()
                    << " in SL3 for " << sl3Id << std::endl;

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
        std::vector<metaPrimitive> chamberMetaPrimitives;
        std::vector<metaPrimitive> confirmedMetaPrimitives;
        std::vector<metaPrimitive> normalMetaPrimitives;
        int sl1 = 0;
        int sl3 = 0;
        for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end();
             ++SL1metaPrimitive, sl1++, sl3 = -1) {
          if (clean_chi2_correlation)
            at_least_one_correlation = false;
          for (auto SL3metaPrimitive = SL3metaPrimitives.begin(); SL3metaPrimitive != SL3metaPrimitives.end();
               ++SL3metaPrimitive, sl3++) {
            if (fabs(SL1metaPrimitive->tanPhi - SL3metaPrimitive->tanPhi) > dTanPsi_correlate_TP)
              continue;  //TanPsi match, SliceTest only
            if (useBX_correlation) {
              if (abs(round(SL1metaPrimitive->t0 / 25.) - round(SL3metaPrimitive->t0 / 25.)) > dBX_correlate_TP)
                continue;  //BX match
            } else {
              if (fabs(SL1metaPrimitive->t0 - SL3metaPrimitive->t0) >= dT0_correlate_TP)
                continue;  //time match
            }
            long int PosSL1 = (int)round(10 * SL1metaPrimitive->x / (10 * x_precision));
            long int PosSL3 = (int)round(10 * SL3metaPrimitive->x / (10 * x_precision));
            double NewSlope = -999.;
            if (use_LSB) {
              long int newConstant = (int)(139.5 * 4);
              long int difPos_mm_x4 = PosSL3 - PosSL1;
              long int tanPsi_x4096_x128 = (difPos_mm_x4)*newConstant;
              long int tanPsi_x4096 = tanPsi_x4096_x128 / ((long int)pow(2, 5 + numberOfBits));
              if (tanPsi_x4096 < 0 && tanPsi_x4096_x128 % ((long int)pow(2, 5 + numberOfBits)) != 0)
                tanPsi_x4096--;
              NewSlope = -tanPsi_x4096 * tanPsi_precision;
            }
            double MeanT0 = (SL1metaPrimitive->t0 + SL3metaPrimitive->t0) / 2;
            double MeanPos = (PosSL3 + PosSL1) / (2. / (x_precision));
            if (use_LSB) {
              MeanPos = MeanPos / x_precision;         // get the precision
              MeanPos = round(10. * MeanPos) / 10.;    //avoid rounding effects
              MeanPos = floor(MeanPos) * x_precision;  // return to the normal value
            }

            DTSuperLayerId SLId1(SL1metaPrimitive->rawId);
            DTSuperLayerId SLId3(SL3metaPrimitive->rawId);
            DTWireId wireId1(SLId1, 2, 1);
            DTWireId wireId3(SLId3, 2, 1);

            //double xH[8], xReco[8];
            int wi[8], tdc[8], lat[8];
            //for (int i = 0; i<8; i++){ xH[i]=0; xReco[i]=0;}
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

            long int CH_CENTER_TO_MID_SL_P = (long int)(117.5 * 4);
            long int Z_FACTOR_CORR[8] = {-6, -2, 2, 6, -6, -2, 2, 6};

            for (int i = 0; i < 8; i++) {
              int sign = 2 * (i / 4) - 1;
              Z_FACTOR_CORR[i] = Z_FACTOR_CORR[i] * CELL_HEIGHT + CH_CENTER_TO_MID_SL_P * sign;
            }
            long int sum_A, sum_B;
            for (int i = 0; i < 8; i++) {
              long int shift, slTime;
              if (i / 4 == 0) {
                shift = round(shiftinfo[wireId1.rawId()] / x_precision);
                slTime = SL1metaPrimitive->t0;
              } else {
                shift = round(shiftinfo[wireId3.rawId()] / x_precision);
                slTime = SL3metaPrimitive->t0;
              }
              if (wi[i] != -1) {
                long int drift_speed_new = 889;
                long int drift_dist_um_x4 = drift_speed_new * (((long int)tdc[i]) - slTime);
                long int wireHorizPos_x4 = (42 * wi[i] + ((i + 1) % 2) * 21) / (10 * x_precision);
                long int pos_mm_x4;

                if (lat[i] == 0) {
                  pos_mm_x4 = wireHorizPos_x4 - (drift_dist_um_x4 >> 10);
                } else {
                  pos_mm_x4 = wireHorizPos_x4 + (drift_dist_um_x4 >> 10);
                }
                sum_A = shift + pos_mm_x4 - (long int)round(MeanPos / x_precision);
                sum_A = sum_A << (14 - numberOfBits);
                sum_B = Z_FACTOR_CORR[i] * (long int)round(-NewSlope / tanPsi_precision);
                chi2 += ((sum_A - sum_B) * (sum_A - sum_B)) >> 2;
              }
            }

            double newChi2 = (double)(chi2 >> 16) / (1024. * 100.);

            if (newChi2 > chi2corTh)
              continue;

            // Fill the used vectors
            useFitSL1[sl1] = true;
            useFitSL3[sl3] = true;

            int quality = 0;
            if (SL3metaPrimitive->quality <= 2 and SL1metaPrimitive->quality <= 2)
              quality = 6;

            if ((SL3metaPrimitive->quality >= 3 && SL1metaPrimitive->quality <= 2) or
                (SL1metaPrimitive->quality >= 3 && SL3metaPrimitive->quality <= 2))
              quality = 8;

            if (SL3metaPrimitive->quality >= 3 && SL1metaPrimitive->quality >= 3)
              quality = 9;

            double z = 0;
            if (ChId.station() >= 3)
              z = -1.8;
            GlobalPoint jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(
                LocalPoint(MeanPos, 0., z));  //Jm_x is already extrapolated to the middle of the SL
            int thisec = ChId.sector();
            if (se == 13)
              thisec = 4;
            if (se == 14)
              thisec = 10;
            double phi = jm_x_cmssw_global.phi() - 0.5235988 * (thisec - 1);
            double psi = atan(NewSlope);
            double phiB = hasPosRF(ChId.wheel(), ChId.sector()) ? psi - phi : -psi - phi;

            if (!clean_chi2_correlation)
              outMPaths.push_back(metaPrimitive({ChId.rawId(),
                                                 MeanT0,
                                                 MeanPos,
                                                 NewSlope,
                                                 phi,
                                                 phiB,
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
                                                 SL3metaPrimitive->lat4,
                                                 -1}));
            else
              chamberMetaPrimitives.push_back(metaPrimitive({ChId.rawId(),
                                                             MeanT0,
                                                             MeanPos,
                                                             NewSlope,
                                                             phi,
                                                             phiB,
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
                                                             SL3metaPrimitive->lat4,
                                                             -1}));
            at_least_one_correlation = true;
          }

          if (at_least_one_correlation == false &&
              allow_confirmation == true) {  //no correlation was found, trying with pairs of two digis in the other SL
            int matched_digis = 0;
            double minx = minx_match_2digis;
            double min2x = minx_match_2digis;
            int best_tdc = -1;
            int next_tdc = -1;
            int best_wire = -1;
            int next_wire = -1;
            int best_layer = -1;
            int next_layer = -1;
            int best_lat = -1;
            int next_lat = -1;
            int lat = -1;

            for (auto dtLayerId_It = dtdigis->begin(); dtLayerId_It != dtdigis->end(); ++dtLayerId_It) {
              const DTLayerId dtLId = (*dtLayerId_It).first;
              DTSuperLayerId dtSLId(dtLId);
              if (dtSLId.rawId() != sl3Id.rawId())
                continue;
              double l_shift = 0;
              if (dtLId.layer() == 4)
                l_shift = 1.95;
              if (dtLId.layer() == 3)
                l_shift = 0.65;
              if (dtLId.layer() == 2)
                l_shift = -0.65;
              if (dtLId.layer() == 1)
                l_shift = -1.95;
              double x_inSL3 = SL1metaPrimitive->x - SL1metaPrimitive->tanPhi * (23.5 + l_shift);
              for (auto digiIt = ((*dtLayerId_It).second).first; digiIt != ((*dtLayerId_It).second).second; ++digiIt) {
                DTWireId wireId(dtLId, (*digiIt).wire());
                int x_wire = shiftinfo[wireId.rawId()] + ((*digiIt).time() - SL1metaPrimitive->t0) * 0.00543;
                int x_wire_left = shiftinfo[wireId.rawId()] - ((*digiIt).time() - SL1metaPrimitive->t0) * 0.00543;
                lat = 1;
                if (fabs(x_inSL3 - x_wire) > fabs(x_inSL3 - x_wire_left)) {
                  x_wire = x_wire_left;  //choose the closest laterality
                  lat = 0;
                }
                if (fabs(x_inSL3 - x_wire) < minx) {
                  minx = fabs(x_inSL3 - x_wire);
                  next_wire = best_wire;
                  next_tdc = best_tdc;
                  next_layer = best_layer;
                  next_lat = best_lat;

                  best_wire = (*digiIt).wire();
                  best_tdc = (*digiIt).time();
                  best_layer = dtLId.layer();
                  best_lat = lat;
                  matched_digis++;
                } else if ((fabs(x_inSL3 - x_wire) >= minx) && (fabs(x_inSL3 - x_wire) < min2x)) {
                  min2x = fabs(x_inSL3 - x_wire);
                  next_wire = (*digiIt).wire();
                  next_tdc = (*digiIt).time();
                  next_layer = dtLId.layer();
                  next_lat = lat;
                  matched_digis++;
                }
              }
            }
            if (matched_digis >= 2 and best_layer != -1 and next_layer != -1) {
              int new_quality = 7;
              if (SL1metaPrimitive->quality <= 2)
                new_quality = 5;

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

              if (!clean_chi2_correlation)
                outMPaths.push_back(metaPrimitive({ChId.rawId(),
                                                   SL1metaPrimitive->t0,
                                                   SL1metaPrimitive->x,
                                                   SL1metaPrimitive->tanPhi,
                                                   SL1metaPrimitive->phi,
                                                   SL1metaPrimitive->phiB,
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
                confirmedMetaPrimitives.push_back(metaPrimitive({ChId.rawId(),
                                                                 SL1metaPrimitive->t0,
                                                                 SL1metaPrimitive->x,
                                                                 SL1metaPrimitive->tanPhi,
                                                                 SL1metaPrimitive->phi,
                                                                 SL1metaPrimitive->phiB,
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
          /*for (auto SL1metaPrimitive = SL1metaPrimitives.begin(); SL1metaPrimitive != SL1metaPrimitives.end(); ++SL1metaPrimitive){
			if(fabs(SL1metaPrimitive->t0-SL3metaPrimitive->t0) < dT0_correlate_TP){//time match
			    //this comb was already filled up in the previous loop now we just want to know if there was at least one match
			    at_least_one_correlation=true;
			}
		    } */
          if (useFitSL3[sl3])
            continue;
          if ((at_least_one_correlation == false || clean_chi2_correlation) &&
              allow_confirmation) {  //no correlation was found, trying with pairs of two digis in the other SL

            int matched_digis = 0;
            double minx = minx_match_2digis;
            double min2x = minx_match_2digis;
            int best_tdc = -1;
            int next_tdc = -1;
            int best_wire = -1;
            int next_wire = -1;
            int best_layer = -1;
            int next_layer = -1;
            int best_lat = -1;
            int next_lat = -1;
            int lat = -1;

            for (auto dtLayerId_It = dtdigis->begin(); dtLayerId_It != dtdigis->end(); ++dtLayerId_It) {
              const DTLayerId dtLId = (*dtLayerId_It).first;
              DTSuperLayerId dtSLId(dtLId);
              if (dtSLId.rawId() != sl1Id.rawId())
                continue;
              double l_shift = 0;
              if (dtLId.layer() == 4)
                l_shift = 1.95;
              if (dtLId.layer() == 3)
                l_shift = 0.65;
              if (dtLId.layer() == 2)
                l_shift = -0.65;
              if (dtLId.layer() == 1)
                l_shift = -1.95;
              double x_inSL1 = SL3metaPrimitive->x + SL3metaPrimitive->tanPhi * (23.5 - l_shift);
              for (auto digiIt = ((*dtLayerId_It).second).first; digiIt != ((*dtLayerId_It).second).second; ++digiIt) {
                DTWireId wireId(dtLId, (*digiIt).wire());
                int x_wire = shiftinfo[wireId.rawId()] + ((*digiIt).time() - SL3metaPrimitive->t0) * 0.00543;
                int x_wire_left = shiftinfo[wireId.rawId()] - ((*digiIt).time() - SL3metaPrimitive->t0) * 0.00543;
                lat = 1;
                if (fabs(x_inSL1 - x_wire) > fabs(x_inSL1 - x_wire_left)) {
                  x_wire = x_wire_left;  //choose the closest laterality
                  lat = 0;
                }
                if (fabs(x_inSL1 - x_wire) < minx) {
                  minx = fabs(x_inSL1 - x_wire);
                  next_wire = best_wire;
                  next_tdc = best_tdc;
                  next_layer = best_layer;
                  next_lat = best_lat;

                  best_wire = (*digiIt).wire();
                  best_tdc = (*digiIt).time();
                  best_layer = dtLId.layer();
                  best_lat = lat;
                  matched_digis++;
                } else if ((fabs(x_inSL1 - x_wire) >= minx) && (fabs(x_inSL1 - x_wire) < min2x)) {
                  minx = fabs(x_inSL1 - x_wire);
                  next_wire = (*digiIt).wire();
                  next_tdc = (*digiIt).time();
                  next_layer = dtLId.layer();
                  next_lat = lat;
                  matched_digis++;
                }
              }
            }
            if (matched_digis >= 2 and best_layer != -1 and next_layer != -1) {
              int new_quality = 7;
              if (SL3metaPrimitive->quality <= 2)
                new_quality = 5;

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

              if (!clean_chi2_correlation)
                outMPaths.push_back(metaPrimitive({ChId.rawId(),
                                                   SL3metaPrimitive->t0,
                                                   SL3metaPrimitive->x,
                                                   SL3metaPrimitive->tanPhi,
                                                   SL3metaPrimitive->phi,
                                                   SL3metaPrimitive->phiB,
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
        if (clean_chi2_correlation) {
          if (debug)
            cout << "Pushing back correlated MPs to the MPs collection" << endl;
          removeSharingFits(chamberMetaPrimitives, outMPaths);
        }
        if (clean_chi2_correlation) {
          if (debug)
            cout << "Pushing back confirmed MPs to the complete vector" << endl;
          removeSharingHits(confirmedMetaPrimitives, chamberMetaPrimitives, outMPaths);
        }

        //finish SL3-SL1
        if (at_least_one_correlation == false || clean_chi2_correlation) {
          if (debug && !at_least_one_correlation)
            std::cout << "correlation we found zero correlations, adding both collections as they are to the outMPaths"
                      << std::endl;
          if (debug)
            std::cout << "correlation sizes:" << SL1metaPrimitives.size() << " " << SL3metaPrimitives.size()
                      << std::endl;
          if (at_least_one_SL1_confirmation == false || clean_chi2_correlation) {
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

              if (!clean_chi2_correlation)
                outMPaths.push_back(newSL1metaPrimitive);
              else
                normalMetaPrimitives.push_back(newSL1metaPrimitive);
            }
          }
          if (at_least_one_SL3_confirmation == false || clean_chi2_correlation) {
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

              if (!clean_chi2_correlation)
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

        std::vector<metaPrimitive> auxMetaPrimitives;
        if (clean_chi2_correlation) {
          if (debug)
            cout << "Pushing back normal MPs to the auxiliar vector" << endl;
          removeSharingHits(normalMetaPrimitives, confirmedMetaPrimitives, auxMetaPrimitives);
        }
        if (clean_chi2_correlation) {
          if (debug)
            cout << "Pushing back normal MPs to the MPs collection" << endl;
          removeSharingHits(auxMetaPrimitives, chamberMetaPrimitives, outMPaths);
        }
      }
    }
  }
}

void MuonPathAssociator::removeSharingFits(std::vector<metaPrimitive> &chamberMPaths,
                                           std::vector<metaPrimitive> &allMPaths) {
  bool useFit[chamberMPaths.size()];
  for (unsigned int i = 0; i < chamberMPaths.size(); i++) {
    useFit[i] = true;
  }
  for (unsigned int i = 0; i < chamberMPaths.size(); i++) {
    if (debug)
      cout << "Looking at prim" << i << endl;
    if (!useFit[i])
      continue;
    for (unsigned int j = i + 1; j < chamberMPaths.size(); j++) {
      if (debug)
        cout << "Comparing with prim " << j << endl;
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
      if (debug)
        printmPC(chamberMPaths[i]);
      allMPaths.push_back(chamberMPaths[i]);
    }
  }
  if (debug)
    cout << "---Swapping chamber---" << endl;
}

void MuonPathAssociator::removeSharingHits(std::vector<metaPrimitive> &firstMPaths,
                                           std::vector<metaPrimitive> &secondMPaths,
                                           std::vector<metaPrimitive> &allMPaths) {
  for (auto &firstMP : firstMPaths) {
    if (debug)
      cout << "----------------------------------" << endl;
    if (debug)
      cout << "Turn for " << endl;
    if (debug)
      printmPC(firstMP);
    bool ok = true;
    for (auto &secondMP : secondMPaths) {
      if (debug)
        cout << "Comparing with " << endl;
      if (debug)
        printmPC(secondMP);
      if (!isNotAPrimo(firstMP, secondMP)) {
        ok = false;
        break;
      }
    }
    if (ok) {
      allMPaths.push_back(firstMP);
      if (debug)
        printmPC(firstMP);
    }
    if (debug)
      cout << "----------------------------------" << endl;
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

  if (((!lay1 && !lay2 && !lay3 && !lay4) || hitsSL1 < 3) && ((!lay5 && !lay6 && !lay7 && !lay8) || hitsSL3 < 3))
    return true;
  else
    return false;
}

void MuonPathAssociator::printmPC(metaPrimitive mP) {
  DTChamberId ChId(mP.rawId);
  std::cout << ChId << "\t"
            << " " << setw(2) << left << mP.wi1 << " " << setw(2) << left << mP.wi2 << " " << setw(2) << left << mP.wi3
            << " " << setw(2) << left << mP.wi4 << " " << setw(2) << left << mP.wi5 << " " << setw(2) << left << mP.wi6
            << " " << setw(2) << left << mP.wi7 << " " << setw(2) << left << mP.wi8 << " " << setw(5) << left << mP.tdc1
            << " " << setw(5) << left << mP.tdc2 << " " << setw(5) << left << mP.tdc3 << " " << setw(5) << left
            << mP.tdc4 << " " << setw(5) << left << mP.tdc5 << " " << setw(5) << left << mP.tdc6 << " " << setw(5)
            << left << mP.tdc7 << " " << setw(5) << left << mP.tdc8 << " " << setw(2) << left << mP.lat1 << " "
            << setw(2) << left << mP.lat2 << " " << setw(2) << left << mP.lat3 << " " << setw(2) << left << mP.lat4
            << " " << setw(2) << left << mP.lat5 << " " << setw(2) << left << mP.lat6 << " " << setw(2) << left
            << mP.lat7 << " " << setw(2) << left << mP.lat8 << " " << setw(10) << right << mP.x << " " << setw(9)
            << left << mP.tanPhi << " " << setw(5) << left << mP.t0 << " " << setw(13) << left << mP.chi2 << endl;
}
/*
  void MuonPathAssociator::associate(MuonPath *mpath) {
  
  // First try to match 
  if (mpath->getNPrimitivesUp()>=3 && mpath->getNPrimitivesDown()>=3) {
  if(fabs(mpath->getBxTimeValue(0)-mpath->getBxTimeValue(2)) < dT0_correlate_TP) { //time match
  float PosSL1=mpath->getHorizPos(0);
  float PosSL3=mpath->getHorizPos(2);
  float NewSlope=(PosSL1-PosSL3)/23.5;     
  float MeanT0=(mpath->getBxTimeValue(0)+mpath->getBxTimeValue(2))/2;
  float MeanPos=(PosSL3+PosSL1)/2;
  float newChi2=(mpath->getChiSq(0)+mpath->getChiSq(2))*0.5;//to be recalculated
  MP_QUALITY quality=NOPATH;
      
  if (mpath->getQuality(0) <=LOWQ and mpath->getQuality(2) <=LOWQ)  quality=LOWLOWQ;
  if ((mpath->getQuality(0) >=HIGHQ and mpath->getQuality(2) <=LOWQ) or 
  (mpath->getQuality(0) <=LOWQ and mpath->getQuality(2) >=HIGHQ))
  quality=HIGHLOWQ;
  if (mpath->getQuality(0) >=3 and mpath->getQuality(2) >=3)  quality=HIGHHIGHQ;
      
  DTChamberId ChId(mpath->getRawId());
  GlobalPoint jm_x_cmssw_global = dtGeo_.chamber(ChId)->toGlobal(LocalPoint(MeanPos,0.,0.));//jm_x is already extrapolated to the middle of the SL
  int thisec = ChId.sector();
  float phi= jm_x_cmssw_global.phi()-0.5235988*(thisec-1);
  float psi=atan(NewSlope);
  float phiB=(hasPosRF(ChId.wheel(),ChId.sector())) ? psi-phi :-psi-phi ;
			
  mpath->setBxTimeValue(MeanT0);
  mpath->setTanPhi(NewSlope);
  mpath->setHorizPos(MeanPos);
  mpath->setPhi(phi);
  mpath->setPhiB(phiB);
  mpath->setChiSq(newChi2);
  mpath->setQuality(quality);
  }
  }
  else if (mpath->getNPrimitivesUp()>=3 && mpath->getNPrimitivesDown()<3 && mpath->getNPrimitivesDown()>0 ) {
  // IF this is not the case try to confirm with other SL: 
  mpath->setBxTimeValue(mpath->getBxTimeValue(2));
  mpath->setTanPhi(mpath->getTanPhi(2));
  mpath->setHorizPos(mpath->getHorizPos(2));
  mpath->setPhi(mpath->getPhi(2));
  mpath->setPhiB(mpath->getPhiB(2));
  mpath->setChiSq(mpath->getChiSq(2));

  if (mpath->getQuality(2) == HIGHQ) 
  mpath->setQuality(CHIGHQ);
  else if (mpath->getQuality(2) == LOWQ) 
  mpath->setQuality(CLOWQ);
    
  }
  else if (mpath->getNPrimitivesDown()>=3 && mpath->getNPrimitivesDown()<3 && mpath->getNPrimitivesDown()>0 ) {
  // IF this is not the case try to confirm with other SL: 
  mpath->setBxTimeValue(mpath->getBxTimeValue(2));
  mpath->setTanPhi(mpath->getTanPhi(2));
  mpath->setHorizPos(mpath->getHorizPos(2));
  mpath->setPhi(mpath->getPhi(2));
  mpath->setPhiB(mpath->getPhiB(2));
  mpath->setChiSq(mpath->getChiSq(2));
  mpath->setQuality(CHIGHQ);

  if (mpath->getQuality(0) == HIGHQ) 
  mpath->setQuality(CHIGHQ);
  else if (mpath->getQuality(0) == LOWQ) 
  mpath->setQuality(CLOWQ);
    
  }
  
  }

*/
