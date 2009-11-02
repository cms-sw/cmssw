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
    PhaseSpace( int n, float m, float M):n(n), m(m), M(M) {}
  };

  enum Mode {
    DEFAULT,
    COARSE,
    FINE
  };

  static DQMStore *DQM_; 

  Benchmark(Mode mode = DEFAULT) : dir_(0), mode_(mode) {}
  virtual ~Benchmark();


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

 
  TDirectory* dir_;

  Mode        mode_;
};

#endif 
