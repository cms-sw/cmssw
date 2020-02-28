///
/// \class L1TwinMuxParams
///
/// Description: Placeholder for BMTF parameters
///
///
/// \author: Giannis Flouris
/// mod : g karathanasis

#ifndef L1TTwinMuxParams_h
#define L1TTwinMuxParams_h

#include <memory>
#include <iostream>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"

class L1TTwinMuxParams {
public:
  L1TTwinMuxParams();
  enum { Version = 1 };

  class Node {
  public:
    std::string type_;
    unsigned version_;
    std::vector<double> dparams_;
    std::vector<unsigned> uparams_;
    std::vector<int> iparams_;
    std::vector<std::string> sparams_;
    Node() {
      type_ = "unspecified";
      version_ = 0;
    }
    COND_SERIALIZABLE;
  };
  enum { CONFIG = 0, NUM_TM_PARAM_NODES = 7 };
  enum {
    USERPCBXFORDTBELOWQUALITY,
    DphiWindow,
    UseOnlyRPC,
    UseOnlyDT,
    UseLowQDT,
    CorrectDTBxwRPC,
    Verbose,
    NUM_CONFIG_PARAMS
  };

  ~L1TTwinMuxParams() {}

  // FW version
  unsigned fwVersion() const { return fwVersion_; }
  void setFwVersion(unsigned fwVersion) { fwVersion_ = fwVersion; }

  void set_USERPCBXFORDTBELOWQUALITY(int par1) { pnodes_[CONFIG].iparams_[USERPCBXFORDTBELOWQUALITY] = par1; }
  int get_USERPCBXFORDTBELOWQUALITY() const { return pnodes_[CONFIG].iparams_[USERPCBXFORDTBELOWQUALITY]; }

  void set_UseOnlyRPC(int par1) { pnodes_[CONFIG].iparams_[UseOnlyRPC] = par1; }
  int get_UseOnlyRPC() const { return pnodes_[CONFIG].iparams_[UseOnlyRPC]; }

  void set_UseOnlyDT(int par1) { pnodes_[CONFIG].iparams_[UseOnlyDT] = par1; }
  int get_UseOnlyDT() const { return pnodes_[CONFIG].iparams_[UseOnlyDT]; }

  void set_UseLowQDT(int par1) { pnodes_[CONFIG].iparams_[UseLowQDT] = par1; }
  int get_UseLowQDT() const { return pnodes_[CONFIG].iparams_[UseLowQDT]; }

  void set_CorrectDTBxwRPC(int par1) { pnodes_[CONFIG].iparams_[CorrectDTBxwRPC] = par1; }
  int get_CorrectDTBxwRPC() const { return pnodes_[CONFIG].iparams_[CorrectDTBxwRPC]; }

  void set_Verbose(int par1) { pnodes_[CONFIG].iparams_[Verbose] = par1; }
  int get_Verbose() const { return pnodes_[CONFIG].iparams_[Verbose]; }

  void set_DphiWindowBxShift(int par1) { pnodes_[CONFIG].iparams_[DphiWindow] = par1; }
  int get_DphiWindowBxShift() const { return pnodes_[CONFIG].iparams_[DphiWindow]; }

  // print parameters to stream:
  void print(std::ostream&) const;
  friend std::ostream& operator<<(std::ostream& o, const L1TTwinMuxParams& p) {
    p.print(o);
    return o;
  }

private:
  unsigned version_;
  unsigned fwVersion_;

  std::vector<Node> pnodes_;
  // std::vector here is just so we can use "blob" in DB and evade max size limitations...

  COND_SERIALIZABLE;
};
#endif
