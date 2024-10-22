#ifndef L1Trigger_Phase2L1ParticleFlow_corrector_h
#define L1Trigger_Phase2L1ParticleFlow_corrector_h
#include <TGraph.h>
#include <TH1.h>
#include <string>
#include <vector>

class TDirectory;

namespace l1t {
  class PFCluster;
}

namespace l1tpf {
  class corrector {
  public:
    corrector() : is2d_(false), neta_(0), nemf_(0), emfMax_(-1), emulate_(false) {}
    corrector(const std::string &iFile, float emfMax = -1, bool debug = false, bool emulate = false);
    corrector(const std::string &iFile,
              const std::string &directory,
              float emfMax = -1,
              bool debug = false,
              bool emulate = false);
    corrector(TDirectory *src, float emfMax = -1, bool debug = false, bool emulate = false);
    // create an empty corrector (you'll need to fill the graphs later)
    corrector(const TH1 *index, float emfMax = -1);
    ~corrector();

    // no copy, but can move
    corrector(const corrector &corr) = delete;
    corrector &operator=(const corrector &corr) = delete;
    corrector(corrector &&corr);
    corrector &operator=(corrector &&corr);

    float correctedPt(float et, float emEt, float eta) const;
    float correctedPt(float et, float eta) const { return correctedPt(et, 0, eta); }
    void correctPt(l1t::PFCluster &cluster, float preserveEmEt = true) const;

    bool valid() const { return (index_.get() != nullptr); }

    // set the graph (note: it is cloned, and the corrector owns the clone)
    void setGraph(const TGraph &graph, int ieta, int iemf = 0);

    bool is2d() const { return is2d_; }
    unsigned int neta() const { return neta_; }
    unsigned int nemf() const { return nemf_; }
    // access the index histogram
    const TH1 &getIndex() const { return *index_; }
    // access the graphs (owned by the corrector, may be null)
    TGraph *getGraph(int ieta, int iemf = 0) { return corrections_[ieta * nemf_ + iemf]; }
    const TGraph *getGraph(int ieta, int iemf = 0) const { return corrections_[ieta * nemf_ + iemf]; }

    // store the corrector
    void writeToFile(const std::string &filename, const std::string &directory) const;
    // store the corrector
    void writeToFile(TDirectory *dest) const;

  private:
    std::unique_ptr<TH1> index_;
    std::vector<TGraph *> corrections_;
    std::vector<TH1 *> correctionsEmulated_;
    bool is2d_;
    unsigned int neta_, nemf_;
    float emfMax_;
    bool emulate_;

    void init_(const std::string &iFile, const std::string &directory, bool debug, bool emulate);
    void init_(TDirectory *src, bool debug);
    void initEmulation_(TDirectory *src, bool debug);
  };
}  // namespace l1tpf
#endif
