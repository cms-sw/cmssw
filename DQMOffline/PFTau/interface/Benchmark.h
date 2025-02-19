#ifndef RecoParticleFlow_Benchmark_Benchmark_h
#define RecoParticleFlow_Benchmark_Benchmark_h

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


#include <string>

class TH1;
class TH1F;
class TH2;

class TH2F;
class TDirectory;

class DQMStore; 


/// abstract base class 
class Benchmark{

 public:

  class PhaseSpace {
  public:
    int n; 
    float m;
    float M;
    PhaseSpace() : n(1), m(0), M(1) {}
    PhaseSpace( int n, float m, float M):n(n), m(m), M(M) {}
  };

  enum Mode {
    DEFAULT,
    DQMOFFLINE,
    VALIDATION
  };

  static DQMStore *DQM_; 

  Benchmark(Mode mode = DEFAULT) : 
    dir_(0), mode_(mode), 
    ptMin_(0), ptMax_(10e10), 
    etaMin_(-10), etaMax_(10), 
    phiMin_(-10), phiMax_(10) {}

  virtual ~Benchmark();

  void setParameters( Mode mode) { mode_ = mode;}
  
  void setRange( float ptMin, float ptMax, 
		 float etaMin, float etaMax, 
		 float phiMin, float phiMax ) {
    ptMin_ = ptMin; ptMax_ = ptMax; etaMin_ = etaMin; etaMax_ = etaMax;
    phiMin_ = phiMin; phiMax_ = phiMax;
  }

  bool isInRange(float pt, float eta, float phi) const {
    return pt>ptMin_ && pt<ptMax_ && eta>etaMin_ && eta<etaMax_ && phi>phiMin_ && phi<phiMax_ ? true : false; 
  }

  virtual void setDirectory(TDirectory* dir);
  
  /// write to the TFile, in plain ROOT mode. No need to call this function in DQM mode
  void write();

 protected:

  /// book a 1D histogram, either with DQM or plain root. 
  TH1F* book1D(const char* histname, const char* title, 
	       int nbins, float xmin, float xmax);

  /// book a 2D histogram, either with DQM or plain root.
  TH2F* book2D(const char* histname, const char* title, 
	       int nbinsx, float xmin, float xmax,
	       int nbinsy, float ymin, float ymax ); 

  /// book a 2D histogram, either with DQM or plain root.
  TH2F* book2D(const char* histname, const char* title, 
	       int nbinsx, float* xbins,
	       int nbinsy, float ymin, float ymax ); 

 
  TDirectory* dir_;

  Mode        mode_;

  float       ptMin_;
  float       ptMax_;
  float       etaMin_;
  float       etaMax_;
  float       phiMin_;
  float       phiMax_;


};

#endif 
