//
// Integer representation of floating point arithmetic suitable for FPGA designs
//
// Author: Yuri Gershtein
// Date:   March 2018
//

#include "L1Trigger/TrackFindingTracklet/interface/imath.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>

using namespace trklet;

std::string VarBase::itos(int i) { return std::to_string(i); }

std::string VarBase::kstring() const {
  char s[1024];
  std::string t = "";
  for (const auto &Kmap : Kmap_) {
    sprintf(s, "^(%i)", Kmap.second);
    std::string t0(s);
    t = t + Kmap.first + t0;
  }
  return t;
}

void VarBase::analyze() {
  if (!readytoanalyze_)
    return;

  double u = maxval_;
  if (u < -minval_)
    u = -minval_;

  int iu = log2(range() / u);
  if (iu > 1) {
    char slog[1024];
    sprintf(slog,
            "analyzing %s: range %g is much larger then %g. suggest cutting by a factor of 2^%i",
            name_.c_str(),
            range(),
            u,
            iu);
    edm::LogVerbatim("Tracklet") << slog;
  }
#ifdef IMATH_ROOT
  char slog[100];
  if (h_) {
    double eff = h_->Integral() / h_->GetEntries();
    if (eff < 0.99) {
      sprintf(slog, "analyzing %s: range is too small, contains %f", name_.c_str(), eff);
      edm::LogVerbatim("Tracklet") << slog;
      h_->Print();
    }
    globals_->h_file_->cd();
    TCanvas *c = new TCanvas();
    c->cd();
    h_->Draw("colz");
    h_->Write();
  } else {
    if (globals_->use_root) {
      sprintf(slog, "analyzing %s: no histogram!\n", name_.c_str());
      edm::LogVerbatim("Tracklet") << slog;
    }
  }
#endif

  if (p1_)
    p1_->analyze();
  if (p2_)
    p2_->analyze();

  readytoanalyze_ = false;
}

std::string VarBase::dump() {
  char s[1024];
  std::string u = kstring();
  sprintf(
      s,
      "Name = %s \t Op = %s \t nbits = %i \n       ival = %li \t fval = %g \t K = %g Range = %f\n       units = %s\n",
      name_.c_str(),
      op_.c_str(),
      nbits_,
      ival_,
      fval_,
      K_,
      range(),
      u.c_str());
  std::string t(s);
  return t;
}

void VarBase::dump_msg() {
  char s[2048];
  std::string u = kstring();
  sprintf(s,
          "Name = %s \t Op = %s \t nbits = %i \n       ival = %li \t fval = %g \t K = %g Range = %f\n       units = "
          "%s\n       step = %i, latency = %i\n",
          name_.c_str(),
          op_.c_str(),
          nbits_,
          ival_,
          fval_,
          K_,
          range(),
          u.c_str(),
          step_,
          latency_);
  std::string t(s);
  edm::LogVerbatim("Tracklet") << t;
  if (p1_)
    p1_->dump_msg();
  if (p2_)
    p2_->dump_msg();
}

void VarAdjustK::adjust(double Knew, double epsilon, bool do_assert, int nbits) {
  //WARNING!!!
  //THIS METHID CAN BE USED ONLY FOR THE FINAL ANSWER
  //THE CHANGE IN CONSTANT CAN NOT BE PROPAGATED UP THE CALCULATION TREE

  K_ = p1_->K();
  Kmap_ = p1_->Kmap();
  double r = Knew / K_;

  lr_ = (r > 1) ? log2(r) + epsilon : log2(r);
  K_ = K_ * pow(2, lr_);
  if (do_assert)
    assert(std::abs(Knew / K_ - 1) < epsilon);

  if (nbits > 0)
    nbits_ = nbits;
  else
    nbits_ = p1_->nbits() - lr_;

  Kmap_["2"] = Kmap_["2"] + lr_;
}

void VarInv::initLUT(double offset) {
  offset_ = offset;
  double offsetI = lround(offset_ / p1_->K());
  for (int i = 0; i < Nelements_; ++i) {
    int i1 = addr_to_ival(i);
    LUT[i] = gen_inv(offsetI + i1);
  }
}

void VarBase::makeready() {
  pipe_counter_ = 0;
  pipe_delays_.clear();
  readytoprint_ = true;
  readytoanalyze_ = true;
  usedasinput_ = false;
  if (p1_)
    p1_->makeready();
  if (p2_)
    p2_->makeready();
  if (p3_)
    p3_->makeready();
}

bool VarBase::has_delay(int i) {
  //dumb sequential search
  for (int pipe_delay : pipe_delays_)
    if (pipe_delay == i)
      return true;
  return false;
}

std::string VarBase::pipe_delay(VarBase *v, int nbits, int delay) {
  //have we been delayed by this much already?
  if (v->has_delay(delay))
    return "";
  v->add_delay(delay);
  std::string name = v->name();
  std::string name_delayed = name + "_delay" + itos(delay);
  std::string out = "wire signed [" + itos(nbits - 1) + ":0] " + name_delayed + ";\n";
  out = out + pipe_delay_wire(v, name_delayed, nbits, delay);
  return out;
}
std::string VarBase::pipe_delays(const int step) {
  std::string out = "";
  if (p1_)
    out += p1_->pipe_delays(step);
  if (p2_)
    out += p2_->pipe_delays(step);
  if (p3_)
    out += p3_->pipe_delays(step);

  int l = step - latency_ - step_;
  return (out + pipe_delay(this, nbits(), l));
}
std::string VarBase::pipe_delay_wire(VarBase *v, std::string name_delayed, int nbits, int delay) {
  std::string name = v->name();
  std::string name_pipe = name + "_pipe" + itos(v->pipe_counter());
  v->pipe_increment();
  std::string out = "pipe_delay #(.STAGES(" + itos(delay) + "), .WIDTH(" + itos(nbits) + ")) " + name_pipe +
                    "(.clk(clk), .val_in(" + name + "), .val_out(" + name_delayed + "));\n";
  return out;
}

void VarBase::inputs(std::vector<VarBase *> *vd) {
  if (op_ == "def" && !usedasinput_) {
    usedasinput_ = true;
    vd->push_back(this);
  } else {
    if (p1_)
      p1_->inputs(vd);
    if (p2_)
      p2_->inputs(vd);
    if (p3_)
      p3_->inputs(vd);
  }
}

#ifdef IMATH_ROOT
TTree *VarBase::addToTree(imathGlobals *globals, VarBase *v, char *s) {
  if (globals->h_file_ == 0) {
    globals->h_file_ = new TFile("imath.root", "RECREATE");
    edm::LogVerbatim("Tracklet") << "recreating file imath.root";
  }
  globals->h_file_->cd();
  TTree *tt = (TTree *)globals->h_file_->Get("tt");
  if (tt == 0) {
    tt = new TTree("tt", "");
    edm::LogVerbatim("Tracklet") << "creating TTree tt";
  }
  std::string si = v->name() + "_i";
  std::string sf = v->name() + "_f";
  std::string sv = v->name();
  if (s != 0) {
    std::string prefix(s);
    si = prefix + si;
    sf = prefix + sf;
    sv = prefix + sv;
  }
  if (!tt->GetBranchStatus(si.c_str())) {
    tt->Branch(si.c_str(), (Long64_t *)&(v->ival_));
    tt->Branch(sf.c_str(), &(v->fval_));
    tt->Branch(sv.c_str(), &(v->val_));
  }

  if (v->p1_)
    addToTree(globals, v->p1_, s);
  if (v->p2_)
    addToTree(globals, v->p2_, s);
  if (v->p3_)
    addToTree(globals, v->p3_, s);

  return tt;
}
TTree *VarBase::addToTree(imathGlobals *globals, double *v, char *s) {
  if (globals->h_file_ == 0) {
    globals->h_file_ = new TFile("imath.root", "RECREATE");
    edm::LogVerbatim("Tracklet") << "recreating file imath.root";
  }
  globals->h_file_->cd();
  TTree *tt = (TTree *)globals->h_file_->Get("tt");
  if (tt == 0) {
    tt = new TTree("tt", "");
    edm::LogVerbatim("Tracklet") << "creating TTree tt";
  }
  tt->Branch(s, v);
  return tt;
}
TTree *VarBase::addToTree(imathGlobals *globals, int *v, char *s) {
  if (globals->h_file_ == 0) {
    globals->h_file_ = new TFile("imath.root", "RECREATE");
    edm::LogVerbatim("Tracklet") << "recreating file imath.root";
  }
  globals->h_file_->cd();
  TTree *tt = (TTree *)globals->h_file_->Get("tt");
  if (tt == 0) {
    tt = new TTree("tt", "");
    edm::LogVerbatim("Tracklet") << "creating TTree tt";
  }
  tt->Branch(s, v);
  return tt;
}
void VarBase::fillTree(imathGlobals *globals) {
  if (globals->h_file_ == 0)
    return;
  globals->h_file_->cd();
  TTree *tt = (TTree *)globals->h_file_->Get("tt");
  if (tt == 0)
    return;
  tt->Fill();
}
void VarBase::writeTree(imathGlobals *globals) {
  if (globals->h_file_ == 0)
    return;
  globals->h_file_->cd();
  TTree *tt = (TTree *)globals->h_file_->Get("tt");
  if (tt == 0)
    return;
  tt->Write();
}

#endif

void VarCut::local_passes(std::map<const VarBase *, std::vector<bool> > &passes,
                          const std::map<const VarBase *, std::vector<bool> > *const previous_passes) const {
  const int lower_cut = lower_cut_ / cut_var_->K();
  const int upper_cut = upper_cut_ / cut_var_->K();
  if (!previous_passes || (previous_passes && !previous_passes->count(cut_var_))) {
    if (!passes.count(cut_var_))
      passes[cut_var_];
    passes.at(cut_var_).push_back(cut_var_->ival() > lower_cut && cut_var_->ival() < upper_cut);
  }
}

bool VarBase::local_passes() const {
  bool passes = false;
  for (const auto &cut : cuts_) {
    const VarCut *const cast_cut = (VarCut *)cut;
    const int lower_cut = cast_cut->lower_cut() / K_;
    const int upper_cut = cast_cut->upper_cut() / K_;
    passes = passes || (ival_ > lower_cut && ival_ < upper_cut);
    if (globals_->printCutInfo_) {
      edm::LogVerbatim("Tracklet") << "  " << name_ << " "
                                   << ((ival_ > lower_cut && ival_ < upper_cut) ? "PASSES" : "FAILS")
                                   << " (required: " << lower_cut * K_ << " < " << ival_ * K_ << " < " << upper_cut * K_
                                   << ")";
    }
  }
  return passes;
}

void VarBase::passes(std::map<const VarBase *, std::vector<bool> > &passes,
                     const std::map<const VarBase *, std::vector<bool> > *const previous_passes) const {
  if (p1_)
    p1_->passes(passes, previous_passes);
  if (p2_)
    p2_->passes(passes, previous_passes);
  if (p3_)
    p3_->passes(passes, previous_passes);

  for (const auto &cut : cuts_) {
    const VarCut *const cast_cut = (VarCut *)cut;
    const int lower_cut = cast_cut->lower_cut() / K_;
    const int upper_cut = cast_cut->upper_cut() / K_;
    if (!previous_passes || (previous_passes && !previous_passes->count(this))) {
      if (!passes.count(this))
        passes[this];
      passes.at(this).push_back(ival_ > lower_cut && ival_ < upper_cut);
      if (globals_->printCutInfo_) {
        edm::LogVerbatim("Tracklet") << "  " << name_ << " "
                                     << ((ival_ > lower_cut && ival_ < upper_cut) ? "PASSES" : "FAILS")
                                     << " (required: " << lower_cut * K_ << " < " << ival_ * K_ << " < "
                                     << upper_cut * K_ << ")";
      }
    }
  }
}

void VarBase::add_cut(VarCut *cut, const bool call_set_cut_var) {
  cuts_.push_back(cut);
  if (call_set_cut_var)
    cut->set_cut_var(this, false);
}

void VarCut::set_cut_var(VarBase *cut_var, const bool call_add_cut) {
  cut_var_ = cut_var;
  if (call_add_cut)
    cut_var->add_cut(this, false);
  if (parent_flag_)
    parent_flag_->calculate_step();
}

void VarFlag::add_cut(VarBase *cut, const bool call_set_parent_flag) {
  cuts_.push_back(cut);
  if (cut->op() == "cut" && call_set_parent_flag) {
    VarCut *const cast_cut = (VarCut *)cut;
    cast_cut->set_parent_flag(this, false);
  }
  calculate_step();
}

void VarCut::set_parent_flag(VarFlag *parent_flag, const bool call_add_cut) {
  parent_flag_ = parent_flag;
  if (call_add_cut)
    parent_flag->add_cut(this, false);
}

VarBase *VarBase::cut_var() {
  if (op_ == "cut")
    return cut_var_;
  else
    return this;
}

bool VarFlag::passes() {
  if (globals_->printCutInfo_) {
    edm::LogVerbatim("Tracklet") << "Checking if " << name_ << " passes...";
  }

  std::map<const VarBase *, std::vector<bool> > passes0, passes1;
  for (const auto &cut : cuts_) {
    if (cut->op() != "cut")
      continue;
    const VarCut *const cast_cut = (VarCut *)cut;
    cast_cut->local_passes(passes0);
  }
  for (const auto &cut : cuts_) {
    if (cut->op() != "cut")
      cut->passes(passes1, &passes0);
    else {
      if (cut->cut_var()->p1())
        cut->cut_var()->p1()->passes(passes1, &passes0);
      if (cut->cut_var()->p2())
        cut->cut_var()->p2()->passes(passes1, &passes0);
      if (cut->cut_var()->p3())
        cut->cut_var()->p3()->passes(passes1, &passes0);
    }
  }

  bool passes = true;
  for (const auto &cut_var : passes0) {
    bool local_passes = false;
    for (const auto pass : cut_var.second)
      local_passes = local_passes || pass;
    passes = passes && local_passes;
  }
  for (const auto &cut_var : passes1) {
    bool local_passes = false;
    for (const auto pass : cut_var.second)
      local_passes = local_passes || pass;
    passes = passes && local_passes;
  }

  if (globals_->printCutInfo_) {
    edm::LogVerbatim("Tracklet") << name_ << " " << (passes ? "PASSES" : "FAILS");
  }

  return passes;
}
