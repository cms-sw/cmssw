#ifndef L1Trigger_TrackFindingTracklet_interface_VMStubsTEMemory_h
#define L1Trigger_TrackFindingTracklet_interface_VMStubsTEMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubTE.h"

#include <string>
#include <vector>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;
  class TrackletLUT;

  class VMStubsTEMemory : public MemoryBase {
  public:
    VMStubsTEMemory(std::string name, Settings const& settings);

    ~VMStubsTEMemory() override = default;

    void resize(int nbins) { stubsbinnedvm_.resize(nbins); }

    bool addVMStub(VMStubTE vmstub, int bin);

    bool addVMStub(VMStubTE vmstub);

    unsigned int nVMStubs() const { return stubsvm_.size(); }

    unsigned int nVMStubsBinned(unsigned int bin) const { return stubsbinnedvm_[bin].size(); }

    unsigned int nBin() const { return stubsbinnedvm_.size(); }

    const VMStubTE& getVMStubTE(unsigned int i) const { return stubsvm_[i]; }

    const VMStubTE& getVMStubTEBinned(unsigned int bin, unsigned int i) const { return stubsbinnedvm_[bin][i]; }

    void clean() override;

    void writeStubs(bool first, unsigned int iSector);

    int phibin() const { return phibin_; }

    void getPhiRange(double& phimin, double& phimax, unsigned int iSeed, unsigned int inner);

    void setother(VMStubsTEMemory* other) { other_ = other; }

    VMStubsTEMemory* other() { return other_; }

    void setbendtable(const TrackletLUT& bendtable);

  private:
    int layer_;
    int disk_;
    int layerdisk_;
    int phibin_;
    VMStubsTEMemory* other_;
    bool overlap_;
    bool extra_;
    bool extended_;  // for the L2L3->D1 and D1D2->L2
    bool isinner_;   // is inner layer/disk for TE purpose

    TrackletLUT bendtable_;

    std::vector<VMStubTE> stubsvm_;
    std::vector<std::vector<VMStubTE> > stubsbinnedvm_;
  };

};  // namespace trklet
#endif
