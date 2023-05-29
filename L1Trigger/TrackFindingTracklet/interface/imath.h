// // Integer representation of floating point arithmetic suitable for FPGA designs
//
// Author: Yuri Gershtein
// Date:   March 2018
//
// Functionality:
//
//  *note* all integers are assumed to be signed
//
//  all variables have units, stored in a map <string,int>, with string a unit (i.e. "phi") and int the power
//                   "2" is always present in the map, and it's int pair is referred to as 'shift'
//                   units are properly combined / propagated through calculations
//                   adding/subtracting variables with different units throws an exception
//                   adding/subtracting variables with different shifts is allowed and is handled correctly
//
// calculate() method re-calculates the variable double and int values based on its operands
//                   returns false in case of overflows and/or mismatches between double and int calculations.
//
// the maximum and minimum values that the variable assumes are stored and updated each time calculate() is called
// if IMATH_ROOT is defined, all values are also stored in a histogram
//
// VarDef     (string name, string units, double fmax, double K):
//                   define variable with bit value fval = K*ival, and maximum absolute value fmax.
//                   calculates nbins on its own
//                   one can assign value to it using set_ methods
//
// VarParam   (string name, string units, double fval, int nbits):
//                   define a parameter. K is calculated based on the fval and nbits
//
//         or (string name, std::string units, double fval, double K):
//                   define a parameer with bit value fval = K*ival.
//                   calculates nbins on its own
//
// VarAdd     (string name, VarBase *p1, VarBase *p2, double range = -1, int nmax = 18):
// VarSubtract(string name, VarBase *p1, VarBase *p2, double range = -1, int nmax = 18):
//                   add/subtract variables. Bit length increases by 1, but capped at nmax
//                   if range>0 specified, bit length is decreased to drop unnecessary high bits
//
// VarMult    (string name, VarBase *p1, VarBase *p2, double range = -1, int nmax = 18):
//                   multiplication. Bit length is a sum of the lengths of the operads, but capped at nmax
//                   if range>0 specified, bit length is decreased to drop unnecessary high bits or post-shift is reduced
//
// VarTimesC  (string name, VarBase *p1, double cF, int ps = 17):
//                   multiplication by a constant. Bit length stays the same
//                   ps defines number of bits used to represent the constant
//
// VarDSPPostadd (string name, VarBase *p1, VarBase *p2, VarBase *p3, double range = -1, int nmax = 18):
//                   explicit instantiation of the 3-clock DSP postaddition: p1*p2+p3
//                   range and nmax have the same meaning as for the VarMult.
//
// VarShift  (string name, VarBase *p1, int shift):
//                   shifts the variable right by shift (equivalent to multiplication by pow(2, -shift));
//                   Units stay the same, nbits are adjusted.
//
// VarShiftround  (string name, VarBase *p1, int shift):
//                   shifts the variable right by shift, but doing rounding, i.e.
//                   (p>>(shift-1)+1)>>1;
//                   Units stay the same, nbits are adjusted.
//
// VarNeg    (string name, VarBase *p1):
//                   multiplies the variable by -1
//
// VarInv     (string name, VarBase *p1, double offset, int nbits, int n, unsigned int shift, mode m, int nbaddr=-1):
//                   LUT-based inversion, f = 1./(offset + f1) and  i = 2^n / (offsetI + i1)
//                   nbits is the width of the LUT (signed)
//                   m is from enum mode {pos, neg, both} and refers to possible sign values of f
//                            for pos and neg, the most significant bit of p1 (i.e. the sign bit) is ignored
//                   shift is a shift applied in i1<->address conversions (used to reduce size of LUT)
//                   nbaddr: if not specified, it is taken to be equal to p1->nbits()
//
//
// VarNounits (string name, VarBase *p1, int ps = 17):
//                   convert a number with units to a number - needed for trig function expansion (i.e. 1 - 0.5*phi^2)
//                   ps is a number of bits to represent the unit conversion constant
//
// VarAdjustK (string name, VarBase *p1, double Knew, double epsilon = 1e-5, bool do_assert = false, int nbits = -1)
//                   adjust variable shift so the K is as close to Knew as possible (needed for bit length adjustments)
//                   if do_assert is true, throw an exeption if Knew/Kold is not a power of two
//                   epsilon is a comparison precision, nbits forces the bit length (possibly discarding MSBs)
//
// VarAdjustKR (string name, VarBase *p1, double Knew, double epsilon = 1e-5, bool do_assert = false, int nbits = -1)
//                   - same as adjustK(), but with rounding, and therefore latency=1
//
// bool calculate(int debug_level) runs through the entire formula tree recalculating both ineteger and floating point values
//                     returns true if everything is OK, false if obvious problems with the calculation exist, i.e
//                                  -  integer value does not fit into the alotted number of bins
//                                  -  integer value is more then 10% or more then 2 away from fval_/K_
//                     debug_level:  0 - no warnings
//                                   1 - limited warning
//                                   2 - as 1, but also include explicit warnings when LUT was used out of its range
//                                   3 - maximum complaints level
//
// VarFlag (string name, VarBase *cut_var, VarBase *...)
//
//                    flag to apply cuts defined for any variable. When output as Verilog, the flag
//                    is true if and only if the following conditions are all true:
//                       1) the cut defined by each VarCut pointer in the argument list must be passed
//                       by the associated variable
//                       2) each VarBase pointer in the argument list that is not also a VarCut
//                       pointer must pass all of its associated cuts
//                       3) all children of the variables in the argument list must pass all of their
//                       associated cuts
//                    The VarFlag::passes() method replicates the behavior of the output Verilog,
//                    returning true if and only if the above conditions are all true. The
//                    VarBase::local_passes() method can be used to query if a given variable passes
//                    its associated cuts, regardless of whether its children do.
//
#ifndef L1Trigger_TrackFindingTracklet_interface_imath_h
#define L1Trigger_TrackFindingTracklet_interface_imath_h

//use root if uncommented
//#ifndef CMSSW_GIT_HASH
//#define IMATH_ROOT
//#endif

#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <sstream>
#include <string>
#include <cassert>
#include <set>

#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/L1TCommon/interface/BitShift.h"

#ifdef IMATH_ROOT
#include <TH2F.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TTree.h>
#endif

//operation latencies for proper HDL pipelining
#define MULT_LATENCY 1
#define LUT_LATENCY 2
#define DSP_LATENCY 3

// Print out information on the pass/fail status of all variables. Warning:
// this outputs a lot of information for busy events!

namespace trklet {

  struct imathGlobals {
    bool printCutInfo_{false};
#ifdef IMATH_ROOT
    TFile *h_file_ = new TFile("imath.root", "RECREATE");
    bool use_root;
#endif
  };

  class VarCut;
  class VarFlag;

  class VarBase {
  public:
    VarBase(imathGlobals *globals, std::string name, VarBase *p1, VarBase *p2, VarBase *p3, int l) {
      globals_ = globals;
      p1_ = p1;
      p2_ = p2;
      p3_ = p3;
      name_ = name;
      latency_ = l;
      int step1 = (p1) ? p1->step() + p1->latency() : 0;
      int step2 = (p2) ? p2->step() + p2->latency() : 0;
      step_ = std::max(step1, step2);

      cuts_.clear();
      cut_var_ = nullptr;

      pipe_counter_ = 0;
      pipe_delays_.clear();

      minval_ = std::numeric_limits<double>::max();
      maxval_ = -std::numeric_limits<double>::max();
      readytoprint_ = true;
      readytoanalyze_ = true;
      usedasinput_ = false;
      Kmap_.clear();
      Kmap_["2"] = 0;  // initially, zero shift
#ifdef IMATH_ROOT
      h_ = 0;
      h_nbins_ = 1024;
      h_precision_ = 0.02;
#endif
    }
    virtual ~VarBase() {
#ifdef IMATH_ROOT
      if (globals_->h_file_) {
        globals_->h_file_->ls();
        globals_->h_file_->Close();
        globals_->h_file_ = 0;
      }
#endif
    }

    static struct Verilog {
    } verilog;
    static struct HLS {
    } hls;

    std::string kstring() const;
    std::string name() const { return name_; }
    std::string op() const { return op_; }
    VarBase *p1() const { return p1_; }
    VarBase *p2() const { return p2_; }
    VarBase *p3() const { return p3_; }
    double fval() const { return fval_; }
    long int ival() const { return ival_; }

    bool local_passes() const;
    void passes(std::map<const VarBase *, std::vector<bool> > &passes,
                const std::map<const VarBase *, std::vector<bool> > *const previous_passes = nullptr) const;
    void print_cuts(std::map<const VarBase *, std::set<std::string> > &cut_strings,
                    const int step,
                    Verilog,
                    const std::map<const VarBase *, std::set<std::string> > *const previous_cut_strings = nullptr) const;
    void print_cuts(std::map<const VarBase *, std::set<std::string> > &cut_strings,
                    const int step,
                    HLS,
                    const std::map<const VarBase *, std::set<std::string> > *const previous_cut_strings = nullptr) const;
    void add_cut(VarCut *cut, const bool call_set_cut_var = true);
    VarBase *cut_var();
    // observed range of fval_ (only filled if debug_level > 0)
    double minval() const { return minval_; }
    double maxval() const { return maxval_; }
    void analyze();
#ifdef IMATH_ROOT
    TH2F *h() { return h_; }
#endif
    void reset() {
      minval_ = std::numeric_limits<double>::max();
      maxval_ = -std::numeric_limits<double>::max();
#ifdef IMATH_ROOT
      h_->Clear();
#endif
    }

    int nbits() const { return nbits_; }
    std::map<std::string, int> Kmap() const { return Kmap_; }
    double range() const { return (1 << (nbits_ - 1)) * K_; }  // everything is signed
    double K() const { return K_; };
    int shift() const { return Kmap_.at("2"); }

    void makeready();
    int step() const { return step_; }
    int latency() const { return latency_; }
    void add_latency(unsigned int l) { latency_ += l; }  //only call before using the variable in calculation!
    bool calculate(int debug_level = 0);
    virtual void local_calculate() {}
    void calcDebug(int debug_level, long int ival_prev, bool &all_ok);
    virtual void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) {
      fs << "// VarBase here. Soemthing is wrong!! " << l1 << ", " << l2 << ", " << l3 << "\n";
    }
    virtual void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) {
      fs << "// VarBase here. Soemthing is wrong!! " << l1 << ", " << l2 << ", " << l3 << "\n";
    }
    void print_step(int step, std::ofstream &fs, Verilog);
    void print_step(int step, std::ofstream &fs, HLS);
    void print_all(std::ofstream &fs, Verilog);
    void print_all(std::ofstream &fs, HLS);
    void print_truncation(std::string &t, const std::string &o1, const int ps, Verilog) const;
    void print_truncation(std::string &t, const std::string &o1, const int ps, HLS) const;
    void inputs(std::vector<VarBase *> *vd);  //collect all inputs

    int pipe_counter() { return pipe_counter_; }
    void pipe_increment() { pipe_counter_++; }
    void add_delay(int i) { pipe_delays_.push_back(i); }
    bool has_delay(int i);  //returns true if already have this variable delayed.
    static void verilog_print(const std::vector<VarBase *> &v, std::ofstream &fs) { design_print(v, fs, verilog); }
    static void hls_print(const std::vector<VarBase *> &v, std::ofstream &fs) { design_print(v, fs, hls); }
    static void design_print(const std::vector<VarBase *> &v, std::ofstream &fs, Verilog);
    static void design_print(const std::vector<VarBase *> &v, std::ofstream &fs, HLS);
    static std::string pipe_delay(VarBase *v, int nbits, int delay);
    std::string pipe_delays(const int step);
    static std::string pipe_delay_wire(VarBase *v, std::string name_delayed, int nbits, int delay);

#ifdef IMATH_ROOT
    static TTree *addToTree(imathGlobals *globals, VarBase *v, char *s = 0);
    static TTree *addToTree(imathGlobals *globals, int *v, char *s);
    static TTree *addToTree(imathGlobals *globals, double *v, char *s);
    static void fillTree(imathGlobals *globals);
    static void writeTree(imathGlobals *globals);
#endif

    void dump_msg();
    std::string dump();
    static std::string itos(int i);

  protected:
    imathGlobals *globals_;
    std::string name_;
    VarBase *p1_;
    VarBase *p2_;
    VarBase *p3_;
    std::string op_;  // operation
    int latency_;     // number of clock cycles for the operation (for HDL output)
    int step_;        // step number in the calculation (for HDL output)

    double fval_;    // exact calculation
    long int ival_;  // integer calculation
    double val_;     // integer calculation converted to double, ival_*K

    std::vector<VarBase *> cuts_;
    VarBase *cut_var_;

    int nbits_;
    double K_;
    std::map<std::string, int> Kmap_;

    int pipe_counter_;
    std::vector<int> pipe_delays_;

    bool readytoanalyze_;
    bool readytoprint_;
    bool usedasinput_;

    double minval_;
    double maxval_;
#ifdef IMATH_ROOT
    void set_hist_pars(int n = 256, double p = 0.05) {
      h_nbins_ = n;
      h_precision_ = p;
    }
    int h_nbins_;
    double h_precision_;
    TH2F *h_;
#endif
  };

  class VarAdjustK : public VarBase {
  public:
    VarAdjustK(imathGlobals *globals,
               std::string name,
               VarBase *p1,
               double Knew,
               double epsilon = 1e-5,
               bool do_assert = false,
               int nbits = -1)
        : VarBase(globals, name, p1, nullptr, nullptr, 0) {
      op_ = "adjustK";
      K_ = p1->K();
      Kmap_ = p1->Kmap();

      double r = Knew / K_;

      lr_ = (r > 1) ? log2(r) + epsilon : log2(r);
      K_ = K_ * pow(2, lr_);
      if (do_assert)
        assert(std::abs(Knew / K_ - 1) < epsilon);

      if (nbits > 0)
        nbits_ = nbits;
      else
        nbits_ = p1->nbits() - lr_;

      Kmap_["2"] = Kmap_["2"] + lr_;
    }

    ~VarAdjustK() override = default;

    void adjust(double Knew, double epsilon = 1e-5, bool do_assert = false, int nbits = -1);

    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int lr_;
  };

  class VarAdjustKR : public VarBase {
  public:
    VarAdjustKR(imathGlobals *globals,
                std::string name,
                VarBase *p1,
                double Knew,
                double epsilon = 1e-5,
                bool do_assert = false,
                int nbits = -1)
        : VarBase(globals, name, p1, nullptr, nullptr, 1) {
      op_ = "adjustKR";
      K_ = p1->K();
      Kmap_ = p1->Kmap();

      double r = Knew / K_;

      lr_ = (r > 1) ? log2(r) + epsilon : log2(r);
      K_ = K_ * pow(2, lr_);
      if (do_assert)
        assert(std::abs(Knew / K_ - 1) < epsilon);

      if (nbits > 0)
        nbits_ = nbits;
      else
        nbits_ = p1->nbits() - lr_;

      Kmap_["2"] = Kmap_["2"] + lr_;
    }

    ~VarAdjustKR() override = default;

    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int lr_;
  };

  class VarParam : public VarBase {
  public:
    VarParam(imathGlobals *globals, std::string name, double fval, int nbits)
        : VarBase(globals, name, nullptr, nullptr, nullptr, 0) {
      op_ = "const";
      nbits_ = nbits;
      int l = log2(std::abs(fval)) + 1.9999999 - nbits;
      Kmap_["2"] = l;
      K_ = pow(2, l);
      fval_ = fval;
      ival_ = fval / K_;
    }
    VarParam(imathGlobals *globals, std::string name, std::string units, double fval, double K)
        : VarBase(globals, name, nullptr, nullptr, nullptr, 0) {
      op_ = "const";
      K_ = K;
      nbits_ = log2(fval / K) + 1.999999;  //plus one to round up
      if (!units.empty())
        Kmap_[units] = 1;
      else {
        //defining a constant, K should be a power of two
        int l = log2(K);
        if (std::abs(pow(2, l) / K - 1) > 1e-5) {
          char slog[100];
          snprintf(slog, 100, "defining unitless constant, yet K is not a power of 2! %g, %g", K, pow(2, l));
          edm::LogVerbatim("Tracklet") << slog;
        }
        Kmap_["2"] = l;
      }
    }

    ~VarParam() override = default;

    void set_fval(double fval) {
      fval_ = fval;
      if (fval > 0)
        ival_ = fval / K_ + 0.5;
      else
        ival_ = fval / K_ - 0.5;
      val_ = ival_ * K_;
    }
    void set_ival(int ival) {
      ival_ = ival;
      fval_ = ival * K_;
      val_ = fval_;
    }
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;
  };

  class VarDef : public VarBase {
  public:
    //construct from scratch
    VarDef(imathGlobals *globals, std::string name, std::string units, double fmax, double K)
        : VarBase(globals, name, nullptr, nullptr, nullptr, 1) {
      op_ = "def";
      K_ = K;
      nbits_ = log2(fmax / K) + 1.999999;  //plus one to round up
      if (!units.empty())
        Kmap_[units] = 1;
      else {
        //defining a constant, K should be a power of two
        int l = log2(K);
        if (std::abs(pow(2, l) / K - 1) > 1e-5) {
          char slog[100];
          snprintf(slog, 100, "defining unitless constant, yet K is not a power of 2! %g, %g", K, pow(2, l));
          edm::LogVerbatim("Tracklet") << slog;
        }
        Kmap_["2"] = l;
      }
    }
    //construct from abother variable (all provenance info is lost!)
    VarDef(imathGlobals *globals, std::string name, VarBase *p) : VarBase(globals, name, nullptr, nullptr, nullptr, 1) {
      op_ = "def";
      K_ = p->K();
      nbits_ = p->nbits();
      Kmap_ = p->Kmap();
    }
    void set_fval(double fval) {
      fval_ = fval;
      if (fval > 0)
        ival_ = fval / K_;
      else
        ival_ = fval / K_ - 1;
      val_ = ival_ * K_;
    }
    void set_ival(int ival) {
      ival_ = ival;
      fval_ = ival * K_;
      val_ = ival_ * K_;
    }
    ~VarDef() override = default;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;
  };

  class VarAdd : public VarBase {
  public:
    VarAdd(imathGlobals *globals, std::string name, VarBase *p1, VarBase *p2, double range = -1, int nmax = 18)
        : VarBase(globals, name, p1, p2, nullptr, 1) {
      op_ = "add";

      std::map<std::string, int> map1 = p1->Kmap();
      std::map<std::string, int> map2 = p2->Kmap();
      int s1 = map1["2"];
      int s2 = map2["2"];

      //first check if the constants are all lined up
      //go over the two maps subtracting the units
      for (const auto &it : map2) {
        if (map1.find(it.first) == map1.end())
          map1[it.first] = -it.second;
        else
          map1[it.first] = map1[it.first] - it.second;
      }

      char slog[100];

      //assert if different
      for (const auto &it : map1) {
        if (it.second != 0) {
          if (it.first != "2") {
            snprintf(
                slog, 100, "VarAdd: bad units! %s^%i for variable %s", (it.first).c_str(), it.second, name_.c_str());
            edm::LogVerbatim("Tracklet") << slog;
            p1->dump_msg();
            p2->dump_msg();
            throw cms::Exception("BadConfig") << "imath units are different!";
          }
        }
      }

      double ki1 = p1->K() / pow(2, s1);
      double ki2 = p2->K() / pow(2, s2);
      //those should be the same
      if (std::abs(ki1 / ki2 - 1.) > 1e-6) {
        snprintf(slog, 100, "VarAdd: bad constants! %f %f for variable %s", ki1, ki2, name_.c_str());
        edm::LogVerbatim("Tracklet") << slog;
        p1->dump_msg();
        p2->dump_msg();
        throw cms::Exception("BadConfig") << "imath constants are different!";
      }
      //everything checks out!

      Kmap_ = p1->Kmap();

      int s0 = s1 < s2 ? s1 : s2;
      shift1 = s1 - s0;
      shift2 = s2 - s0;

      int n1 = p1->nbits() + shift1;
      int n2 = p2->nbits() + shift2;
      int n0 = 1 + (n1 > n2 ? n1 : n2);

      //before shifting, check the range
      if (range > 0) {
        n0 = log2(range / ki1 / pow(2, s0)) + 1e-9;
        n0 = n0 + 2;
      }

      if (n0 <= nmax) {  //if it fits, we're done
        ps_ = 0;
        Kmap_["2"] = s0;
        nbits_ = n0;
      } else {
        ps_ = n0 - nmax;
        Kmap_["2"] = s0 + ps_;
        nbits_ = nmax;
      }

      K_ = ki1 * pow(2, Kmap_["2"]);
    }
    ~VarAdd() override = default;
    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int ps_;
    int shift1;
    int shift2;
  };

  class VarSubtract : public VarBase {
  public:
    VarSubtract(imathGlobals *globals, std::string name, VarBase *p1, VarBase *p2, double range = -1, int nmax = 18)
        : VarBase(globals, name, p1, p2, nullptr, 1) {
      op_ = "subtract";

      std::map<std::string, int> map1 = p1->Kmap();
      std::map<std::string, int> map2 = p2->Kmap();
      int s1 = map1["2"];
      int s2 = map2["2"];

      //first check if the constants are all lined up go over the two maps subtracting the units
      for (const auto &it : map2) {
        if (map1.find(it.first) == map1.end())
          map1[it.first] = -it.second;
        else
          map1[it.first] = map1[it.first] - it.second;
      }

      char slog[100];

      //assert if different
      for (const auto &it : map1) {
        if (it.second != 0) {
          if (it.first != "2") {
            snprintf(
                slog, 100, "VarAdd: bad units! %s^%i for variable %s", (it.first).c_str(), it.second, name_.c_str());
            edm::LogVerbatim("Tracklet") << slog;
            p1->dump_msg();
            p2->dump_msg();
            throw cms::Exception("BadConfig") << "imath units are different!";
          }
        }
      }

      double ki1 = p1->K() / pow(2, s1);
      double ki2 = p2->K() / pow(2, s2);
      //those should be the same
      if (std::abs(ki1 / ki2 - 1.) > 1e-6) {
        snprintf(slog, 100, "VarAdd: bad constants! %f %f for variable %s", ki1, ki2, name_.c_str());
        edm::LogVerbatim("Tracklet") << slog;
        p1->dump_msg();
        p2->dump_msg();
        throw cms::Exception("BadConfig") << "imath constants are different!";
      }
      //everything checks out!

      Kmap_ = p1->Kmap();

      int s0 = s1 < s2 ? s1 : s2;
      shift1 = s1 - s0;
      shift2 = s2 - s0;

      int n1 = p1->nbits() + shift1;
      int n2 = p2->nbits() + shift2;
      int n0 = 1 + (n1 > n2 ? n1 : n2);

      //before shifting, check the range
      if (range > 0) {
        n0 = log2(range / ki1 / pow(2, s0)) + 1e-9;
        n0 = n0 + 2;
      }

      if (n0 <= nmax) {  //if it fits, we're done
        ps_ = 0;
        Kmap_["2"] = s0;
        nbits_ = n0;
      } else {
        ps_ = n0 - nmax;
        Kmap_["2"] = s0 + ps_;
        nbits_ = nmax;
      }

      K_ = ki1 * pow(2, Kmap_["2"]);
    }

    ~VarSubtract() override = default;

    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int ps_;
    int shift1;
    int shift2;
  };

  class VarNounits : public VarBase {
  public:
    VarNounits(imathGlobals *globals, std::string name, VarBase *p1, int ps = 17)
        : VarBase(globals, name, p1, nullptr, nullptr, MULT_LATENCY) {
      op_ = "nounits";
      ps_ = ps;
      nbits_ = p1->nbits();

      int s1 = p1->shift();
      double ki = p1->K() / pow(2, s1);
      int m = log2(ki);

      K_ = pow(2, s1 + m);
      Kmap_["2"] = s1 + m;
      double c = ki * pow(2, -m);
      cI_ = c * pow(2, ps_);
    }
    ~VarNounits() override = default;

    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int ps_;
    int cI_;
  };

  class VarShiftround : public VarBase {
  public:
    VarShiftround(imathGlobals *globals, std::string name, VarBase *p1, int shift)
        : VarBase(globals, name, p1, nullptr, nullptr, 1) {  // latency is one because there is an addition
      op_ = "shiftround";
      shift_ = shift;

      nbits_ = p1->nbits() - shift;
      Kmap_ = p1->Kmap();
      K_ = p1->K();
    }
    ~VarShiftround() override = default;

    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int shift_;
  };

  class VarShift : public VarBase {
  public:
    VarShift(imathGlobals *globals, std::string name, VarBase *p1, int shift)
        : VarBase(globals, name, p1, nullptr, nullptr, 0) {
      op_ = "shift";
      shift_ = shift;

      nbits_ = p1->nbits() - shift;
      Kmap_ = p1->Kmap();
      K_ = p1->K();
    }
    ~VarShift() override = default;
    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int shift_;
  };

  class VarNeg : public VarBase {
  public:
    VarNeg(imathGlobals *globals, std::string name, VarBase *p1) : VarBase(globals, name, p1, nullptr, nullptr, 1) {
      op_ = "neg";
      nbits_ = p1->nbits();
      Kmap_ = p1->Kmap();
      K_ = p1->K();
    }
    ~VarNeg() override = default;
    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;
  };

  class VarTimesC : public VarBase {
  public:
    VarTimesC(imathGlobals *globals, std::string name, VarBase *p1, double cF, int ps = 17)
        : VarBase(globals, name, p1, nullptr, nullptr, MULT_LATENCY) {
      op_ = "timesC";
      cF_ = cF;
      ps_ = ps;

      nbits_ = p1->nbits();
      Kmap_ = p1->Kmap();
      K_ = p1->K();

      int s1 = Kmap_["2"];
      double l = log2(std::abs(cF));
      if (l > 0)
        l += 0.999999;
      int m = l;

      cI_ = cF * pow(2, ps - m);
      K_ = K_ * pow(2, m);
      Kmap_["2"] = s1 + m;
    }
    ~VarTimesC() override = default;
    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int ps_;
    int cI_;
    double cF_;
  };

  class VarMult : public VarBase {
  public:
    VarMult(imathGlobals *globals, std::string name, VarBase *p1, VarBase *p2, double range = -1, int nmax = 18)
        : VarBase(globals, name, p1, p2, nullptr, MULT_LATENCY) {
      op_ = "mult";

      const std::map<std::string, int> map1 = p1->Kmap();
      const std::map<std::string, int> map2 = p2->Kmap();
      for (const auto &it : map1) {
        if (Kmap_.find(it.first) == Kmap_.end())
          Kmap_[it.first] = it.second;
        else
          Kmap_[it.first] = Kmap_[it.first] + it.second;
      }
      for (const auto &it : map2) {
        if (Kmap_.find(it.first) == Kmap_.end())
          Kmap_[it.first] = it.second;
        else
          Kmap_[it.first] = Kmap_[it.first] + it.second;
      }
      K_ = p1->K() * p2->K();
      int s0 = Kmap_["2"];

      int n0 = p1->nbits() + p2->nbits();
      if (range > 0) {
        n0 = log2(range / K_) + 1e-9;
        n0 = n0 + 2;
      }
      if (n0 < nmax) {
        ps_ = 0;
        nbits_ = n0;
      } else {
        ps_ = n0 - nmax;
        nbits_ = nmax;
        Kmap_["2"] = s0 + ps_;
        K_ = K_ * pow(2, ps_);
      }
    }
    ~VarMult() override = default;
    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int ps_;
  };

  class VarDSPPostadd : public VarBase {
  public:
    VarDSPPostadd(
        imathGlobals *globals, std::string name, VarBase *p1, VarBase *p2, VarBase *p3, double range = -1, int nmax = 18)
        : VarBase(globals, name, p1, p2, p3, DSP_LATENCY) {
      op_ = "DSP_postadd";

      //first, get constants for the p1*p2
      std::map<std::string, int> map1 = p1->Kmap();
      std::map<std::string, int> map2 = p2->Kmap();
      for (const auto &it : map2) {
        if (map1.find(it.first) == map1.end())
          map1[it.first] = it.second;
        else
          map1[it.first] = map1[it.first] + it.second;
      }
      double k0 = p1->K() * p2->K();
      int s0 = map1["2"];

      //now addition
      std::map<std::string, int> map3 = p3->Kmap();
      int s3 = map3["2"];

      //first check if the constants are all lined up
      //go over the two maps subtracting the units
      for (const auto &it : map3) {
        if (map1.find(it.first) == map1.end())
          map1[it.first] = -it.second;
        else
          map1[it.first] = map1[it.first] - it.second;
      }

      char slog[100];

      //assert if different
      for (const auto &it : map1) {
        if (it.second != 0) {
          if (it.first != "2") {
            snprintf(slog,
                     100,
                     "VarDSPPostadd: bad units! %s^%i for variable %s",
                     (it.first).c_str(),
                     it.second,
                     name_.c_str());
            edm::LogVerbatim("Tracklet") << slog;
            p1->dump_msg();
            p2->dump_msg();
            p3->dump_msg();
            throw cms::Exception("BadConfig") << "imath units are different!";
          }
        }
      }

      double ki1 = k0 / pow(2, s0);
      double ki2 = p3->K() / pow(2, s3);
      //those should be the same
      if (std::abs(ki1 / ki2 - 1.) > 1e-6) {
        snprintf(slog, 100, "VarDSPPostadd: bad constants! %f %f for variable %s", ki1, ki2, name_.c_str());
        edm::LogVerbatim("Tracklet") << slog;
        p1->dump_msg();
        p2->dump_msg();
        p3->dump_msg();
        throw cms::Exception("BadConfig") << "imath constants are different!";
      }
      //everything checks out!

      shift3_ = s3 - s0;
      if (shift3_ < 0) {
        throw cms::Exception("BadConfig") << "imath VarDSPPostadd: loosing precision on C in A*B+C: " << shift3_;
      }

      Kmap_ = p3->Kmap();
      Kmap_["2"] = Kmap_["2"] - shift3_;

      int n12 = p1->nbits() + p2->nbits();
      int n3 = p3->nbits() + shift3_;
      int n0 = 1 + (n12 > n3 ? n12 : n3);

      //before shifting, check the range
      if (range > 0) {
        n0 = log2(range / ki2 / pow(2, s3)) + 1e-9;
        n0 = n0 + 2;
      }

      if (n0 <= nmax) {  //if it fits, we're done
        ps_ = 0;
        nbits_ = n0;
      } else {
        ps_ = n0 - nmax;
        Kmap_["2"] = Kmap_["2"] + ps_;
        nbits_ = nmax;
      }

      K_ = ki2 * pow(2, Kmap_["2"]);
    }
    ~VarDSPPostadd() override = default;

    void local_calculate() override;
    using VarBase::print;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;

  protected:
    int ps_;
    int shift3_;
  };

  class VarInv : public VarBase {
  public:
    enum mode { pos, neg, both };

    VarInv(imathGlobals *globals,
           std::string name,
           VarBase *p1,
           double offset,
           int nbits,
           int n,
           unsigned int shift,
           mode m,
           int nbaddr = -1)
        : VarBase(globals, name, p1, nullptr, nullptr, LUT_LATENCY) {
      op_ = "inv";
      offset_ = offset;
      nbits_ = nbits;
      n_ = n;
      shift_ = shift;
      m_ = m;
      if (nbaddr < 0)
        nbaddr = p1->nbits();
      nbaddr_ = nbaddr - shift;
      if (m_ != mode::both)
        nbaddr_--;
      Nelements_ = 1 << nbaddr_;
      mask_ = Nelements_ - 1;
      ashift_ = sizeof(int) * 8 - nbaddr_;

      const std::map<std::string, int> map1 = p1->Kmap();
      for (const auto &it : map1)
        Kmap_[it.first] = -it.second;
      Kmap_["2"] = Kmap_["2"] - n;
      K_ = pow(2, -n) / p1->K();

      LUT = new int[Nelements_];
      double offsetI = lround(offset_ / p1_->K());
      for (int i = 0; i < Nelements_; ++i) {
        int i1 = addr_to_ival(i);
        LUT[i] = gen_inv(offsetI + i1);
      }
    }
    ~VarInv() override { delete[] LUT; }

    void set_mode(mode m) { m_ = m; }
    void initLUT(double offset);
    double offset() { return offset_; }
    double Ioffset() { return offset_ / p1_->K(); }

    void local_calculate() override;
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void writeLUT(std::ofstream &fs) const { writeLUT(fs, verilog); }
    void writeLUT(std::ofstream &fs, Verilog) const;
    void writeLUT(std::ofstream &fs, HLS) const;

    int ival_to_addr(int ival) { return ((ival >> shift_) & mask_); }
    int addr_to_ival(int addr) {
      switch (m_) {
        case mode::pos:
          return l1t::bitShift(addr, shift_);
        case mode::neg:
          return l1t::bitShift((addr - Nelements_), shift_);
        case mode::both:
          return l1t::bitShift(addr, ashift_) >> (ashift_ - shift_);
      }
      assert(0);
    }
    int gen_inv(int i) {
      unsigned int ms = sizeof(int) * 8 - nbits_;
      int lut = 0;
      if (i > 0) {
        int i1 = i + (1 << shift_) - 1;
        int lut1 = (lround((1 << n_) / i) << ms) >> ms;
        int lut2 = (lround((1 << n_) / (i1)) << ms) >> ms;
        lut = 0.5 * (lut1 + lut2);
      } else if (i < -1) {
        int i1 = i + (1 << shift_) - 1;
        int i2 = i;
        int lut1 = (lround((1 << n_) / i1) << ms) >> ms;
        int lut2 = (lround((1 << n_) / i2) << ms) >> ms;
        lut = 0.5 * (lut1 + lut2);
      }
      return lut;
    }

  protected:
    double offset_;
    int n_;
    mode m_;
    unsigned int shift_;
    unsigned int mask_;
    unsigned int ashift_;
    int Nelements_;
    int nbaddr_;

    int *LUT;
  };

  class VarCut : public VarBase {
  public:
    VarCut(imathGlobals *globals, double lower_cut, double upper_cut)
        : VarBase(globals, "", nullptr, nullptr, nullptr, 0),
          lower_cut_(lower_cut),
          upper_cut_(upper_cut),
          parent_flag_(nullptr) {
      op_ = "cut";
    }

    VarCut(imathGlobals *globals, VarBase *cut_var, double lower_cut, double upper_cut)
        : VarCut(globals, lower_cut, upper_cut) {
      set_cut_var(cut_var);
    }
    ~VarCut() override = default;

    double lower_cut() const { return lower_cut_; }
    double upper_cut() const { return upper_cut_; }

    void local_passes(std::map<const VarBase *, std::vector<bool> > &passes,
                      const std::map<const VarBase *, std::vector<bool> > *const previous_passes = nullptr) const;
    using VarBase::print;
    void print(std::map<const VarBase *, std::set<std::string> > &cut_strings,
               const int step,
               Verilog,
               const std::map<const VarBase *, std::set<std::string> > *const previous_cut_strings = nullptr) const;
    void print(std::map<const VarBase *, std::set<std::string> > &cut_strings,
               const int step,
               HLS,
               const std::map<const VarBase *, std::set<std::string> > *const previous_cut_strings = nullptr) const;

    void set_parent_flag(VarFlag *parent_flag, const bool call_add_cut);
    VarFlag *parent_flag() { return parent_flag_; }
    void set_cut_var(VarBase *cut_var, const bool call_add_cut = true);

  protected:
    double lower_cut_;
    double upper_cut_;
    VarFlag *parent_flag_;
  };

  class VarFlag : public VarBase {
  public:
    template <class... Args>
    VarFlag(imathGlobals *globals, std::string name, VarBase *cut, Args... args)
        : VarBase(globals, name, nullptr, nullptr, nullptr, 0) {
      op_ = "flag";
      nbits_ = 1;
      add_cuts(cut, args...);
    }

    template <class... Args>
    void add_cuts(VarBase *cut, Args... args) {
      add_cut(cut);
      add_cuts(args...);
    }

    void add_cuts(VarBase *cut) { add_cut(cut); }

    void add_cut(VarBase *cut, const bool call_set_parent_flag = true);

    void calculate_step();
    bool passes();
    void print(std::ofstream &fs, Verilog, int l1 = 0, int l2 = 0, int l3 = 0) override;
    void print(std::ofstream &fs, HLS, int l1 = 0, int l2 = 0, int l3 = 0) override;
  };
};  // namespace trklet
#endif
