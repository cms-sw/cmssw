#include "L1Trigger/TrackFindingTracklet/interface/imath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace trklet;

bool VarBase::calculate(int debug_level) {
  bool ok1 = true;
  bool ok2 = true;
  bool ok3 = true;

  if (p1_)
    ok1 = p1_->calculate(debug_level);
  if (p2_)
    ok2 = p2_->calculate(debug_level);
  if (p3_)
    ok3 = p3_->calculate(debug_level);

  bool all_ok = debug_level && ok1 && ok2 && ok3;
  long int ival_prev = ival_;

  local_calculate();

  val_ = ival_ * K_;

#ifdef IMATH_ROOT
  if (globals_->use_root) {
    if (h_ == 0) {
      globals_->h_file_->cd();
      std::string hname = "h_" + name_;
      h_ = (TH2F *)globals_->h_file_->Get(hname.c_str());
      if (h_ == 0) {
        h_precision_ = 0.5 * h_nbins_ * K_;
        std::string st = name_ + ";fval;fval-ival*K";
        h_ = new TH2F(hname.c_str(), name_.c_str(), h_nbins_, -range(), range(), h_nbins_, -h_precision_, h_precision_);
        if (debug_level == 3)
          edm::LogVerbatim("Tracklet") << " booking histogram " << hname;
      }
    }
    if (ival_ != ival_prev || op_ == "def" || op_ == "const")
      h_->Fill(fval_, K_ * ival_ - fval_);
  }
#endif

  if (debug_level)
    calcDebug(debug_level, ival_prev, all_ok);

  return all_ok;
}

void VarBase::calcDebug(int debug_level, long int ival_prev, bool &all_ok) {
  if (fval_ > maxval_)
    maxval_ = fval_;
  if (fval_ < minval_)
    minval_ = fval_;

  bool todump = false;
  int nmax = sizeof(long int) * 8;
  int ns = nmax - nbits_;
  long int itest = ival_;
  itest = l1t::bitShift(itest, ns);
  itest = itest >> ns;
  if (itest != ival_) {
    if (debug_level == 3 || (ival_ != ival_prev && all_ok)) {
      edm::LogVerbatim("Tracklet") << "imath: truncated value mismatch!! " << ival_ << " != " << itest;
      todump = true;
    }
    all_ok = false;
  }

  float ftest = val_;
  float tolerance = 0.1 * std::abs(fval_);
  if (tolerance < 2 * K_)
    tolerance = 2 * K_;
  if (std::abs(ftest - fval_) > tolerance) {
    if (debug_level == 3 || (ival_ != ival_prev && (all_ok && (op_ != "inv" || debug_level >= 2)))) {
      edm::LogVerbatim("Tracklet") << "imath: **GROSS** value mismatch!! " << fval_ << " != " << ftest;
      if (op_ == "inv")
        edm::LogVerbatim("Tracklet") << p1_->dump() << "\n-----------------------------------";
      todump = true;
    }
    all_ok = false;
  }
  if (todump)
    edm::LogVerbatim("Tracklet") << dump();
}

void VarFlag::calculate_step() {
  int max_step = 0;
  for (const auto &cut : cuts_) {
    if (!cut->cut_var())
      continue;
    if (cut->cut_var()->latency() + cut->cut_var()->step() > max_step)
      max_step = cut->cut_var()->latency() + cut->cut_var()->step();
  }
  step_ = max_step;
}

//
//  local calculations
//

void VarAdjustK::local_calculate() {
  fval_ = p1_->fval();
  ival_ = p1_->ival();
  if (lr_ > 0)
    ival_ = ival_ >> lr_;
  else if (lr_ < 0)
    ival_ = l1t::bitShift(ival_, (-lr_));
}

void VarAdjustKR::local_calculate() {
  fval_ = p1_->fval();
  ival_ = p1_->ival();
  if (lr_ > 0)
    ival_ = ((ival_ >> (lr_ - 1)) + 1) >> 1;  //rounding
  else if (lr_ < 0)
    ival_ = l1t::bitShift(ival_, (-lr_));
}

void VarAdd::local_calculate() {
  fval_ = p1_->fval() + p2_->fval();
  long int i1 = p1_->ival();
  long int i2 = p2_->ival();
  if (shift1 > 0)
    i1 = l1t::bitShift(i1, shift1);
  if (shift2 > 0)
    i2 = l1t::bitShift(i2, shift2);
  ival_ = i1 + i2;
  if (ps_ > 0)
    ival_ = ival_ >> ps_;
}

void VarSubtract::local_calculate() {
  fval_ = p1_->fval() - p2_->fval();
  long int i1 = p1_->ival();
  long int i2 = p2_->ival();
  if (shift1 > 0)
    i1 = l1t::bitShift(i1, shift1);
  if (shift2 > 0)
    i2 = l1t::bitShift(i2, shift2);
  ival_ = i1 - i2;
  if (ps_ > 0)
    ival_ = ival_ >> ps_;
}

void VarNounits::local_calculate() {
  fval_ = p1_->fval();
  ival_ = (p1_->ival() * cI_) >> ps_;
}

void VarTimesC::local_calculate() {
  fval_ = p1_->fval() * cF_;
  ival_ = (p1_->ival() * cI_) >> ps_;
}

void VarNeg::local_calculate() {
  fval_ = -p1_->fval();
  ival_ = -p1_->ival();
}

void VarShift::local_calculate() {
  fval_ = p1_->fval() * pow(2, -shift_);
  ival_ = p1_->ival();
  if (shift_ > 0)
    ival_ = ival_ >> shift_;
  if (shift_ < 0)
    ival_ = l1t::bitShift(ival_, (-shift_));
}

void VarShiftround::local_calculate() {
  fval_ = p1_->fval() * pow(2, -shift_);
  ival_ = p1_->ival();
  if (shift_ > 0)
    ival_ = ((ival_ >> (shift_ - 1)) + 1) >> 1;
  if (shift_ < 0)
    ival_ = l1t::bitShift(ival_, (-shift_));
}

void VarMult::local_calculate() {
  fval_ = p1_->fval() * p2_->fval();
  ival_ = (p1_->ival() * p2_->ival()) >> ps_;
}

void VarDSPPostadd::local_calculate() {
  fval_ = p1_->fval() * p2_->fval() + p3_->fval();
  ival_ = p3_->ival();
  if (shift3_ > 0)
    ival_ = l1t::bitShift(ival_, shift3_);
  if (shift3_ < 0)
    ival_ = ival_ >> (-shift3_);
  ival_ += p1_->ival() * p2_->ival();
  ival_ = ival_ >> ps_;
}

void VarInv::local_calculate() {
  fval_ = 1. / (offset_ + p1_->fval());
  ival_ = LUT[ival_to_addr(p1_->ival())];
}
