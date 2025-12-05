#ifndef L1Trigger_Phase2L1ParticleFlow_corrector_h
#define L1Trigger_Phase2L1ParticleFlow_corrector_h
#include <string>
#include <vector>
#include <memory>

#ifdef CMSSW_GIT_HASH
#define L1PF_USE_ROOT
#endif

class PFCluster;

// Define this macro to enable ROOT-dependent interfaces
#ifdef L1PF_USE_ROOT
#include <TGraph.h>
#include <TH1.h>
class TDirectory;
#endif

namespace l1tpf {
  class corrector {
  public:
    enum class EmulationMode { None, Correction, CorrectedPt };

    corrector()
        : is2d_(false),
          neta_(0),
          nemf_(0),
          emfMax_(-1),
          emulate_(false),
          debug_(false),
          emulationMode_(l1tpf::corrector::EmulationMode::CorrectedPt) {}
    corrector(const std::string &iFile,
              float emfMax = -1,
              bool debug = false,
              bool emulate = false,
              l1tpf::corrector::EmulationMode emulationMode = l1tpf::corrector::EmulationMode::CorrectedPt);

#ifdef L1PF_USE_ROOT
    // ROOT-based constructors (guarded)
    corrector(const std::string &iFile,
              const std::string &directory,
              float emfMax = -1,
              bool debug = false,
              bool emulate = false,
              l1tpf::corrector::EmulationMode emulationMode = l1tpf::corrector::EmulationMode::CorrectedPt);
    corrector(TDirectory *src,
              float emfMax = -1,
              bool debug = false,
              bool emulate = false,
              l1tpf::corrector::EmulationMode emulationMode = l1tpf::corrector::EmulationMode::CorrectedPt);
    // create an empty corrector (you'll need to fill the graphs later)
    corrector(const TH1 *index, float emfMax = -1);
#endif
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
    bool is2d() const { return is2d_; }
    unsigned int neta() const { return neta_; }
    unsigned int nemf() const { return nemf_; }

    // access the index histogram (ROOT only)
#ifdef L1PF_USE_ROOT
    // set the graph (note: it is cloned, and the corrector owns the clone)
    void setGraph(const TGraph &graph, int ieta, int iemf = 0);

    const TH1 &getIndex() const { return *index_; }
    // access the graphs (owned by the corrector, may be null)
    TGraph *getGraph(int ieta, int iemf = 0) { return corrections_[ieta * nemf_ + iemf]; }
    const TGraph *getGraph(int ieta, int iemf = 0) const { return corrections_[ieta * nemf_ + iemf]; }
    // store the corrector
    void writeToFile(TDirectory *dest) const;
#endif

    // store the corrector
    void writeToFile(const std::string &filename, const std::string &directory) const;

  private:
    // Storage: ROOT or JSON-backed
#ifdef L1PF_USE_ROOT
    std::unique_ptr<TH1> index_;
    std::vector<TGraph *> corrections_;
    std::vector<TH1 *> correctionsEmulated_;
    void init_(const std::string &iFile, const std::string &directory, bool debug, bool emulate);
    void init_(TDirectory *src, bool debug);
    void initGraphs_(TDirectory *src, bool debug);
    void initEmulation_(TDirectory *src, bool debug);

#else
    struct Binning1D {
      std::vector<float> edges;
    };
    struct Binning2D {
      Binning1D eta, emf;
    };
    struct EmulHistogram {
      std::vector<float> binEdges;
      std::vector<float> values;
    };
    std::unique_ptr<Binning2D> index_;
    std::vector<EmulHistogram> correctionsJson_;
    // Init from JSON (ROOT-free)
    void initJson_(const std::string &jsonFile, bool debug, bool emulate);
#endif
    bool is2d_;
    unsigned int neta_, nemf_;
    float emfMax_;
    bool emulate_;
    bool debug_;
    l1tpf::corrector::EmulationMode emulationMode_;
  };
}  // namespace l1tpf
#endif
