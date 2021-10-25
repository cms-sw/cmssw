#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorDisk.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorOverlap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

TrackletCalculator::TrackletCalculator(string name, Settings const& settings, Globals* globals)
    : TrackletCalculatorBase(name, settings, globals) {
  for (unsigned int ilayer = 0; ilayer < N_LAYER; ilayer++) {
    vector<TrackletProjectionsMemory*> tmp(settings.nallstubs(ilayer), nullptr);
    trackletprojlayers_.push_back(tmp);
  }

  for (unsigned int idisk = 0; idisk < N_DISK; idisk++) {
    vector<TrackletProjectionsMemory*> tmp(settings.nallstubs(idisk + N_LAYER), nullptr);
    trackletprojdisks_.push_back(tmp);
  }

  initLayerDisksandISeed(layerdisk1_, layerdisk2_, iSeed_);

  // set TC index
  iTC_ = name_[7] - 'A';

  TCIndex_ = (iSeed_ << 4) + iTC_;
  assert(TCIndex_ >= 0 && TCIndex_ <= (int)settings_.ntrackletmax());

  if (settings_.usephicritapprox()) {
    double phicritFactor =
        0.5 * settings_.rcrit() * globals_->ITC_L1L2()->rinv_final.K() / globals_->ITC_L1L2()->phi0_final.K();
    if (std::abs(phicritFactor - 2.) > 0.25)
      edm::LogPrint("Tracklet")
          << "TrackletCalculator::TrackletCalculator phicrit approximation may be invalid! Please check.";
  }

  // write the drinv and invt inverse tables
  if ((settings_.writeInvTable() || settings_.writeHLSInvTable() || settings_.writeTable()) && iTC_ == 0) {
    void (*writeLUT)(const VarInv&, const string&) = nullptr;
    if (settings.writeInvTable()) {  // Verilog version
      writeLUT = [](const VarInv& x, const string& basename) -> void {
        ofstream fs(basename + ".tab");
        return x.writeLUT(fs, VarBase::verilog);
      };
    } else {  // HLS version
      writeLUT = [](const VarInv& x, const string& basename) -> void {
        ofstream fs(basename + ".tab");
        return x.writeLUT(fs, VarBase::hls);
      };
    }
    writeInvTable(writeLUT);
  }

  // write the firmware design for the calculation of the tracklet parameters
  // and projections
  if ((settings_.writeVerilog() || settings_.writeHLS()) && iTC_ == 0) {
    void (*writeDesign)(const vector<VarBase*>&, const string&) = nullptr;
    if (settings.writeVerilog()) {  // Verilog version
      writeDesign = [](const vector<VarBase*>& v, const string& basename) -> void {
        ofstream fs(basename + ".v");
        return VarBase::verilog_print(v, fs);
      };
    } else {  // HLS version
      writeDesign = [](const vector<VarBase*>& v, const string& basename) -> void {
        ofstream fs(basename + ".cpp");
        return VarBase::hls_print(v, fs);
      };
    }
    writeFirmwareDesign(writeDesign);
  }
}

void TrackletCalculator::addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory) {
  outputProj = dynamic_cast<TrackletProjectionsMemory*>(memory);
  assert(outputProj != nullptr);
}

void TrackletCalculator::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "trackpar") {
    auto* tmp = dynamic_cast<TrackletParametersMemory*>(memory);
    assert(tmp != nullptr);
    trackletpars_ = tmp;
    return;
  }

  if (output.substr(0, 7) == "projout") {
    //output is on the form 'projoutL2PHIC' or 'projoutD3PHIB'
    auto* tmp = dynamic_cast<TrackletProjectionsMemory*>(memory);
    assert(tmp != nullptr);

    unsigned int layerdisk = output[8] - '1';   //layer or disk counting from 0
    unsigned int phiregion = output[12] - 'A';  //phiregion counting from 0

    if (output[7] == 'L') {
      assert(layerdisk < N_LAYER);
      assert(phiregion < trackletprojlayers_[layerdisk].size());
      //check that phiregion not already initialized
      assert(trackletprojlayers_[layerdisk][phiregion] == nullptr);
      trackletprojlayers_[layerdisk][phiregion] = tmp;
      return;
    }

    if (output[7] == 'D') {
      assert(layerdisk < N_DISK);
      assert(phiregion < trackletprojdisks_[layerdisk].size());
      //check that phiregion not already initialized
      assert(trackletprojdisks_[layerdisk][phiregion] == nullptr);
      trackletprojdisks_[layerdisk][phiregion] = tmp;
      return;
    }
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void TrackletCalculator::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "innerallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    innerallstubs_.push_back(tmp);
    return;
  }
  if (input == "outerallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    outerallstubs_.push_back(tmp);
    return;
  }
  if (input.substr(0, 8) == "stubpair") {
    auto* tmp = dynamic_cast<StubPairsMemory*>(memory);
    assert(tmp != nullptr);
    stubpairs_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find intput : " << input;
}

void TrackletCalculator::execute(unsigned int iSector, double phimin, double phimax) {
  unsigned int countall = 0;
  unsigned int countsel = 0;

  phimin_ = phimin;
  phimax_ = phimax;
  iSector_ = iSector;

  //Helpfull to have for debugging the HLS code - will keep here for now.
  //bool print = (iSector == 3) && (getName() == "TC_L1L2G");
  //print = false;

  for (auto& stubpair : stubpairs_) {
    if (trackletpars_->nTracklets() >= settings_.ntrackletmax()) {
      edm::LogVerbatim("Tracklet") << "Will break on too many tracklets in " << getName();
      break;
    }
    for (unsigned int i = 0; i < stubpair->nStubPairs(); i++) {
      countall++;
      const Stub* innerFPGAStub = stubpair->getVMStub1(i).stub();
      const L1TStub* innerStub = innerFPGAStub->l1tstub();

      const Stub* outerFPGAStub = stubpair->getVMStub2(i).stub();
      const L1TStub* outerStub = outerFPGAStub->l1tstub();

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "TrackletCalculator execute " << getName() << "[" << iSector << "]";
      }

      if (innerFPGAStub->layerdisk() < N_LAYER && (getName() != "TC_D1L2A" && getName() != "TC_D1L2B")) {
        if (outerFPGAStub->layerdisk() >= N_LAYER) {
          //overlap seeding
          bool accept = overlapSeeding(outerFPGAStub, outerStub, innerFPGAStub, innerStub);
          if (accept)
            countsel++;
        } else {
          //barrel+barrel seeding
          bool accept = barrelSeeding(innerFPGAStub, innerStub, outerFPGAStub, outerStub);
          if (accept)
            countsel++;
        }
      } else {
        if (outerFPGAStub->layerdisk() >= N_LAYER) {
          //disk+disk seeding
          bool accept = diskSeeding(innerFPGAStub, innerStub, outerFPGAStub, outerStub);
          if (accept)
            countsel++;
        } else if (innerFPGAStub->layerdisk() >= N_LAYER) {
          //layer+disk seeding
          bool accept = overlapSeeding(innerFPGAStub, innerStub, outerFPGAStub, outerStub);
          if (accept)
            countsel++;
        } else {
          throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " invalid seeding";
        }
      }

      if (trackletpars_->nTracklets() >= settings_.ntrackletmax()) {
        edm::LogVerbatim("Tracklet") << "Will break on number of tracklets in " << getName();
        break;
      }

      if (countall >= settings_.maxStep("TC")) {
        if (settings_.debugTracklet())
          edm::LogVerbatim("Tracklet") << "Will break on MAXTC 1";
        break;
      }
      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "TrackletCalculator execute done";
      }
    }
    if (countall >= settings_.maxStep("TC")) {
      if (settings_.debugTracklet())
        edm::LogVerbatim("Tracklet") << "Will break on MAXTC 2";
      break;
    }
  }

  if (settings_.writeMonitorData("TC")) {
    globals_->ofstream("trackletcalculator.txt") << getName() << " " << countall << " " << countsel << endl;
  }
}

void TrackletCalculator::writeInvTable(void (*writeLUT)(const VarInv&, const string&)) {
  switch (iSeed_) {
    case 0:  // L1L2
      writeLUT(globals_->ITC_L1L2()->drinv, settings_.tablePath() + "TC_L1L2_drinv");
      writeLUT(globals_->ITC_L1L2()->invt, settings_.tablePath() + "TC_L1L2_invt");
      break;
    case 1:  // L2L3
      writeLUT(globals_->ITC_L2L3()->drinv, settings_.tablePath() + "TC_L2L3_drinv");
      writeLUT(globals_->ITC_L2L3()->invt, settings_.tablePath() + "TC_L2L3_invt");
      break;
    case 2:  // L3L4
      writeLUT(globals_->ITC_L3L4()->drinv, settings_.tablePath() + "TC_L3L4_drinv");
      writeLUT(globals_->ITC_L3L4()->invt, settings_.tablePath() + "TC_L3L4_invt");
      break;
    case 3:  // L5L6
      writeLUT(globals_->ITC_L5L6()->drinv, settings_.tablePath() + "TC_L5L6_drinv");
      writeLUT(globals_->ITC_L5L6()->invt, settings_.tablePath() + "TC_L5L6_invt");
      break;
    case 4:  // D1D2
      writeLUT(globals_->ITC_F1F2()->drinv, settings_.tablePath() + "TC_F1F2_drinv");
      writeLUT(globals_->ITC_F1F2()->invt, settings_.tablePath() + "TC_F1F2_invt");
      writeLUT(globals_->ITC_B1B2()->drinv, settings_.tablePath() + "TC_B1B2_drinv");
      writeLUT(globals_->ITC_B1B2()->invt, settings_.tablePath() + "TC_B1B2_invt");
      break;
    case 5:  // D3D4
      writeLUT(globals_->ITC_F3F4()->drinv, settings_.tablePath() + "TC_F3F4_drinv");
      writeLUT(globals_->ITC_F3F4()->invt, settings_.tablePath() + "TC_F3F4_invt");
      writeLUT(globals_->ITC_B3B4()->drinv, settings_.tablePath() + "TC_B3B4_drinv");
      writeLUT(globals_->ITC_B3B4()->invt, settings_.tablePath() + "TC_B3B4_invt");
      break;
    case 6:  // L1D1
      writeLUT(globals_->ITC_L1F1()->drinv, settings_.tablePath() + "TC_L1F1_drinv");
      writeLUT(globals_->ITC_L1F1()->invt, settings_.tablePath() + "TC_L1F1_invt");
      writeLUT(globals_->ITC_L1B1()->drinv, settings_.tablePath() + "TC_L1B1_drinv");
      writeLUT(globals_->ITC_L1B1()->invt, settings_.tablePath() + "TC_L1B1_invt");
      break;
    case 7:  // L2D1
      writeLUT(globals_->ITC_L2F1()->drinv, settings_.tablePath() + "TC_L2F1_drinv");
      writeLUT(globals_->ITC_L2F1()->invt, settings_.tablePath() + "TC_L2F1_invt");
      writeLUT(globals_->ITC_L2B1()->drinv, settings_.tablePath() + "TC_L2B1_drinv");
      writeLUT(globals_->ITC_L2B1()->invt, settings_.tablePath() + "TC_L2B1_invt");
      break;
  }
}

void TrackletCalculator::writeFirmwareDesign(void (*writeDesign)(const vector<VarBase*>&, const string&)) {
  switch (iSeed_) {
    case 0:  // L1L2
    {
      const vector<VarBase*> v = {&globals_->ITC_L1L2()->rinv_final,     &globals_->ITC_L1L2()->phi0_final,
                                  &globals_->ITC_L1L2()->t_final,        &globals_->ITC_L1L2()->z0_final,
                                  &globals_->ITC_L1L2()->phiL_0_final,   &globals_->ITC_L1L2()->phiL_1_final,
                                  &globals_->ITC_L1L2()->phiL_2_final,   &globals_->ITC_L1L2()->phiL_3_final,
                                  &globals_->ITC_L1L2()->zL_0_final,     &globals_->ITC_L1L2()->zL_1_final,
                                  &globals_->ITC_L1L2()->zL_2_final,     &globals_->ITC_L1L2()->zL_3_final,
                                  &globals_->ITC_L1L2()->der_phiL_final, &globals_->ITC_L1L2()->der_zL_final,
                                  &globals_->ITC_L1L2()->phiD_0_final,   &globals_->ITC_L1L2()->phiD_1_final,
                                  &globals_->ITC_L1L2()->phiD_2_final,   &globals_->ITC_L1L2()->phiD_3_final,
                                  &globals_->ITC_L1L2()->phiD_4_final,   &globals_->ITC_L1L2()->rD_0_final,
                                  &globals_->ITC_L1L2()->rD_1_final,     &globals_->ITC_L1L2()->rD_2_final,
                                  &globals_->ITC_L1L2()->rD_3_final,     &globals_->ITC_L1L2()->rD_4_final,
                                  &globals_->ITC_L1L2()->der_phiD_final, &globals_->ITC_L1L2()->der_rD_final};
      writeDesign(v, "TC_L1L2");
    } break;
    case 1:  // L2L3
    {
      const vector<VarBase*> v = {&globals_->ITC_L2L3()->rinv_final,     &globals_->ITC_L2L3()->phi0_final,
                                  &globals_->ITC_L2L3()->t_final,        &globals_->ITC_L2L3()->z0_final,
                                  &globals_->ITC_L2L3()->phiL_0_final,   &globals_->ITC_L2L3()->phiL_1_final,
                                  &globals_->ITC_L2L3()->phiL_2_final,   &globals_->ITC_L2L3()->phiL_3_final,
                                  &globals_->ITC_L2L3()->zL_0_final,     &globals_->ITC_L2L3()->zL_1_final,
                                  &globals_->ITC_L2L3()->zL_2_final,     &globals_->ITC_L2L3()->zL_3_final,
                                  &globals_->ITC_L2L3()->der_phiL_final, &globals_->ITC_L2L3()->der_zL_final,
                                  &globals_->ITC_L2L3()->phiD_0_final,   &globals_->ITC_L2L3()->phiD_1_final,
                                  &globals_->ITC_L2L3()->phiD_2_final,   &globals_->ITC_L2L3()->phiD_3_final,
                                  &globals_->ITC_L2L3()->phiD_4_final,   &globals_->ITC_L2L3()->rD_0_final,
                                  &globals_->ITC_L2L3()->rD_1_final,     &globals_->ITC_L2L3()->rD_2_final,
                                  &globals_->ITC_L2L3()->rD_3_final,     &globals_->ITC_L2L3()->rD_4_final,
                                  &globals_->ITC_L2L3()->der_phiD_final, &globals_->ITC_L2L3()->der_rD_final};
      writeDesign(v, "TC_L2L3");
    } break;
    case 2:  // L3L4
    {
      const vector<VarBase*> v = {&globals_->ITC_L3L4()->rinv_final,     &globals_->ITC_L3L4()->phi0_final,
                                  &globals_->ITC_L3L4()->t_final,        &globals_->ITC_L3L4()->z0_final,
                                  &globals_->ITC_L3L4()->phiL_0_final,   &globals_->ITC_L3L4()->phiL_1_final,
                                  &globals_->ITC_L3L4()->phiL_2_final,   &globals_->ITC_L3L4()->phiL_3_final,
                                  &globals_->ITC_L3L4()->zL_0_final,     &globals_->ITC_L3L4()->zL_1_final,
                                  &globals_->ITC_L3L4()->zL_2_final,     &globals_->ITC_L3L4()->zL_3_final,
                                  &globals_->ITC_L3L4()->der_phiL_final, &globals_->ITC_L3L4()->der_zL_final,
                                  &globals_->ITC_L3L4()->phiD_0_final,   &globals_->ITC_L3L4()->phiD_1_final,
                                  &globals_->ITC_L3L4()->phiD_2_final,   &globals_->ITC_L3L4()->phiD_3_final,
                                  &globals_->ITC_L3L4()->phiD_4_final,   &globals_->ITC_L3L4()->rD_0_final,
                                  &globals_->ITC_L3L4()->rD_1_final,     &globals_->ITC_L3L4()->rD_2_final,
                                  &globals_->ITC_L3L4()->rD_3_final,     &globals_->ITC_L3L4()->rD_4_final,
                                  &globals_->ITC_L3L4()->der_phiD_final, &globals_->ITC_L3L4()->der_rD_final};
      writeDesign(v, "TC_L3L4");
    } break;
    case 3:  // L5L6
    {
      const vector<VarBase*> v = {&globals_->ITC_L5L6()->rinv_final,     &globals_->ITC_L5L6()->phi0_final,
                                  &globals_->ITC_L5L6()->t_final,        &globals_->ITC_L5L6()->z0_final,
                                  &globals_->ITC_L5L6()->phiL_0_final,   &globals_->ITC_L5L6()->phiL_1_final,
                                  &globals_->ITC_L5L6()->phiL_2_final,   &globals_->ITC_L5L6()->phiL_3_final,
                                  &globals_->ITC_L5L6()->zL_0_final,     &globals_->ITC_L5L6()->zL_1_final,
                                  &globals_->ITC_L5L6()->zL_2_final,     &globals_->ITC_L5L6()->zL_3_final,
                                  &globals_->ITC_L5L6()->der_phiL_final, &globals_->ITC_L5L6()->der_zL_final,
                                  &globals_->ITC_L5L6()->phiD_0_final,   &globals_->ITC_L5L6()->phiD_1_final,
                                  &globals_->ITC_L5L6()->phiD_2_final,   &globals_->ITC_L5L6()->phiD_3_final,
                                  &globals_->ITC_L5L6()->phiD_4_final,   &globals_->ITC_L5L6()->rD_0_final,
                                  &globals_->ITC_L5L6()->rD_1_final,     &globals_->ITC_L5L6()->rD_2_final,
                                  &globals_->ITC_L5L6()->rD_3_final,     &globals_->ITC_L5L6()->rD_4_final,
                                  &globals_->ITC_L5L6()->der_phiD_final, &globals_->ITC_L5L6()->der_rD_final};
      writeDesign(v, "TC_L5L6");
    } break;
    case 4:  // D1D2
    {
      const vector<VarBase*> v = {&globals_->ITC_F1F2()->rinv_final,     &globals_->ITC_F1F2()->phi0_final,
                                  &globals_->ITC_F1F2()->t_final,        &globals_->ITC_F1F2()->z0_final,
                                  &globals_->ITC_F1F2()->phiL_0_final,   &globals_->ITC_F1F2()->phiL_1_final,
                                  &globals_->ITC_F1F2()->phiL_2_final,   &globals_->ITC_F1F2()->zL_0_final,
                                  &globals_->ITC_F1F2()->zL_1_final,     &globals_->ITC_F1F2()->zL_2_final,
                                  &globals_->ITC_F1F2()->der_phiL_final, &globals_->ITC_F1F2()->der_zL_final,
                                  &globals_->ITC_F1F2()->phiD_0_final,   &globals_->ITC_F1F2()->phiD_1_final,
                                  &globals_->ITC_F1F2()->phiD_2_final,   &globals_->ITC_F1F2()->rD_0_final,
                                  &globals_->ITC_F1F2()->rD_1_final,     &globals_->ITC_F1F2()->rD_2_final,
                                  &globals_->ITC_F1F2()->der_phiD_final, &globals_->ITC_F1F2()->der_rD_final};
      writeDesign(v, "TC_F1F2");
    }
      {
        const vector<VarBase*> v = {&globals_->ITC_B1B2()->rinv_final,     &globals_->ITC_B1B2()->phi0_final,
                                    &globals_->ITC_B1B2()->t_final,        &globals_->ITC_B1B2()->z0_final,
                                    &globals_->ITC_B1B2()->phiL_0_final,   &globals_->ITC_B1B2()->phiL_1_final,
                                    &globals_->ITC_B1B2()->phiL_2_final,   &globals_->ITC_B1B2()->zL_0_final,
                                    &globals_->ITC_B1B2()->zL_1_final,     &globals_->ITC_B1B2()->zL_2_final,
                                    &globals_->ITC_B1B2()->der_phiL_final, &globals_->ITC_B1B2()->der_zL_final,
                                    &globals_->ITC_B1B2()->phiD_0_final,   &globals_->ITC_B1B2()->phiD_1_final,
                                    &globals_->ITC_B1B2()->phiD_2_final,   &globals_->ITC_B1B2()->rD_0_final,
                                    &globals_->ITC_B1B2()->rD_1_final,     &globals_->ITC_B1B2()->rD_2_final,
                                    &globals_->ITC_B1B2()->der_phiD_final, &globals_->ITC_B1B2()->der_rD_final};
        writeDesign(v, "TC_B1B2");
      }
      break;
    case 5:  // D3D4
    {
      const vector<VarBase*> v = {&globals_->ITC_F3F4()->rinv_final,     &globals_->ITC_F3F4()->phi0_final,
                                  &globals_->ITC_F3F4()->t_final,        &globals_->ITC_F3F4()->z0_final,
                                  &globals_->ITC_F3F4()->phiL_0_final,   &globals_->ITC_F3F4()->phiL_1_final,
                                  &globals_->ITC_F3F4()->phiL_2_final,   &globals_->ITC_F3F4()->zL_0_final,
                                  &globals_->ITC_F3F4()->zL_1_final,     &globals_->ITC_F3F4()->zL_2_final,
                                  &globals_->ITC_F3F4()->der_phiL_final, &globals_->ITC_F3F4()->der_zL_final,
                                  &globals_->ITC_F3F4()->phiD_0_final,   &globals_->ITC_F3F4()->phiD_1_final,
                                  &globals_->ITC_F3F4()->phiD_2_final,   &globals_->ITC_F3F4()->rD_0_final,
                                  &globals_->ITC_F3F4()->rD_1_final,     &globals_->ITC_F3F4()->rD_2_final,
                                  &globals_->ITC_F3F4()->der_phiD_final, &globals_->ITC_F3F4()->der_rD_final};
      writeDesign(v, "TC_F3F4");
    }
      {
        const vector<VarBase*> v = {&globals_->ITC_B3B4()->rinv_final,     &globals_->ITC_B3B4()->phi0_final,
                                    &globals_->ITC_B3B4()->t_final,        &globals_->ITC_B3B4()->z0_final,
                                    &globals_->ITC_B3B4()->phiL_0_final,   &globals_->ITC_B3B4()->phiL_1_final,
                                    &globals_->ITC_B3B4()->phiL_2_final,   &globals_->ITC_B3B4()->zL_0_final,
                                    &globals_->ITC_B3B4()->zL_1_final,     &globals_->ITC_B3B4()->zL_2_final,
                                    &globals_->ITC_B3B4()->der_phiL_final, &globals_->ITC_B3B4()->der_zL_final,
                                    &globals_->ITC_B3B4()->phiD_0_final,   &globals_->ITC_B3B4()->phiD_1_final,
                                    &globals_->ITC_B3B4()->phiD_2_final,   &globals_->ITC_B3B4()->rD_0_final,
                                    &globals_->ITC_B3B4()->rD_1_final,     &globals_->ITC_B3B4()->rD_2_final,
                                    &globals_->ITC_B3B4()->der_phiD_final, &globals_->ITC_B3B4()->der_rD_final};
        writeDesign(v, "TC_B3B4");
      }
      break;
    case 6:  // L1D1
    {
      const vector<VarBase*> v = {&globals_->ITC_L1F1()->rinv_final,     &globals_->ITC_L1F1()->phi0_final,
                                  &globals_->ITC_L1F1()->t_final,        &globals_->ITC_L1F1()->z0_final,
                                  &globals_->ITC_L1F1()->phiL_0_final,   &globals_->ITC_L1F1()->phiL_1_final,
                                  &globals_->ITC_L1F1()->phiL_2_final,   &globals_->ITC_L1F1()->zL_0_final,
                                  &globals_->ITC_L1F1()->zL_1_final,     &globals_->ITC_L1F1()->zL_2_final,
                                  &globals_->ITC_L1F1()->der_phiL_final, &globals_->ITC_L1F1()->der_zL_final,
                                  &globals_->ITC_L1F1()->phiD_0_final,   &globals_->ITC_L1F1()->phiD_1_final,
                                  &globals_->ITC_L1F1()->phiD_2_final,   &globals_->ITC_L1F1()->phiD_3_final,
                                  &globals_->ITC_L1F1()->rD_0_final,     &globals_->ITC_L1F1()->rD_1_final,
                                  &globals_->ITC_L1F1()->rD_2_final,     &globals_->ITC_L1F1()->rD_3_final,
                                  &globals_->ITC_L1F1()->der_phiD_final, &globals_->ITC_L1F1()->der_rD_final};
      writeDesign(v, "TC_L1F1");
    }
      {
        const vector<VarBase*> v = {&globals_->ITC_L1B1()->rinv_final,     &globals_->ITC_L1B1()->phi0_final,
                                    &globals_->ITC_L1B1()->t_final,        &globals_->ITC_L1B1()->z0_final,
                                    &globals_->ITC_L1B1()->phiL_0_final,   &globals_->ITC_L1B1()->phiL_1_final,
                                    &globals_->ITC_L1B1()->phiL_2_final,   &globals_->ITC_L1B1()->zL_0_final,
                                    &globals_->ITC_L1B1()->zL_1_final,     &globals_->ITC_L1B1()->zL_2_final,
                                    &globals_->ITC_L1B1()->der_phiL_final, &globals_->ITC_L1B1()->der_zL_final,
                                    &globals_->ITC_L1B1()->phiD_0_final,   &globals_->ITC_L1B1()->phiD_1_final,
                                    &globals_->ITC_L1B1()->phiD_2_final,   &globals_->ITC_L1B1()->phiD_3_final,
                                    &globals_->ITC_L1B1()->rD_0_final,     &globals_->ITC_L1B1()->rD_1_final,
                                    &globals_->ITC_L1B1()->rD_2_final,     &globals_->ITC_L1B1()->rD_3_final,
                                    &globals_->ITC_L1B1()->der_phiD_final, &globals_->ITC_L1B1()->der_rD_final};
        writeDesign(v, "TC_L1B1");
      }
      break;
    case 7:  // L2D1
    {
      const vector<VarBase*> v = {&globals_->ITC_L2F1()->rinv_final,     &globals_->ITC_L2F1()->phi0_final,
                                  &globals_->ITC_L2F1()->t_final,        &globals_->ITC_L2F1()->z0_final,
                                  &globals_->ITC_L2F1()->phiL_0_final,   &globals_->ITC_L2F1()->phiL_1_final,
                                  &globals_->ITC_L2F1()->phiL_2_final,   &globals_->ITC_L2F1()->zL_0_final,
                                  &globals_->ITC_L2F1()->zL_1_final,     &globals_->ITC_L2F1()->zL_2_final,
                                  &globals_->ITC_L2F1()->der_phiL_final, &globals_->ITC_L2F1()->der_zL_final,
                                  &globals_->ITC_L2F1()->phiD_0_final,   &globals_->ITC_L2F1()->phiD_1_final,
                                  &globals_->ITC_L2F1()->phiD_2_final,   &globals_->ITC_L2F1()->phiD_3_final,
                                  &globals_->ITC_L2F1()->rD_0_final,     &globals_->ITC_L2F1()->rD_1_final,
                                  &globals_->ITC_L2F1()->rD_2_final,     &globals_->ITC_L2F1()->rD_3_final,
                                  &globals_->ITC_L2F1()->der_phiD_final, &globals_->ITC_L2F1()->der_rD_final};
      writeDesign(v, "TC_L2F1");
    }
      {
        const vector<VarBase*> v = {&globals_->ITC_L2B1()->rinv_final,     &globals_->ITC_L2B1()->phi0_final,
                                    &globals_->ITC_L2B1()->t_final,        &globals_->ITC_L2B1()->z0_final,
                                    &globals_->ITC_L2B1()->phiL_0_final,   &globals_->ITC_L2B1()->phiL_1_final,
                                    &globals_->ITC_L2B1()->phiL_2_final,   &globals_->ITC_L2B1()->zL_0_final,
                                    &globals_->ITC_L2B1()->zL_1_final,     &globals_->ITC_L2B1()->zL_2_final,
                                    &globals_->ITC_L2B1()->der_phiL_final, &globals_->ITC_L2B1()->der_zL_final,
                                    &globals_->ITC_L2B1()->phiD_0_final,   &globals_->ITC_L2B1()->phiD_1_final,
                                    &globals_->ITC_L2B1()->phiD_2_final,   &globals_->ITC_L2B1()->phiD_3_final,
                                    &globals_->ITC_L2B1()->rD_0_final,     &globals_->ITC_L2B1()->rD_1_final,
                                    &globals_->ITC_L2B1()->rD_2_final,     &globals_->ITC_L2B1()->rD_3_final,
                                    &globals_->ITC_L2B1()->der_phiD_final, &globals_->ITC_L2B1()->der_rD_final};
        writeDesign(v, "TC_L2B1");
      }
      break;
  }
}
