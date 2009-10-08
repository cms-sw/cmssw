#ifndef RecoParticleFlow_Benchmark_Benchmark_h
#define RecoParticleFlow_Benchmark_Benchmark_h

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


#include <string>

class TH1;
class TH1F;
class TH2;
class TH2F;
class TFile; 

class DQMStore; 


/// abstract base class 
class Benchmark{

 public:

  Benchmark() : file_(0), DQM_(0) {}
  virtual ~Benchmark();


  void write(std::string Filename);
  
  /// call this function after construction to use the plain root mode
  void setfile(TFile *file) {
    file_ = file;
    DQM_ = 0; 
  }

 protected:

  ///book a 1D histogram, either with DQM or plain root. 
  TH1F* book1D(const char* histname, const char* title, 
	       int nbins, float xmin, float xmax);

  ///book a 2D histogram, either with DQM or plain root.
  TH2F* book2D(const char* histname, const char* title, 
	       int nbinsx, float xmin, float xmax,
	       int nbinsy, float ymin, float ymax ); 

  ///COLIN replace by shared_ptr
  TFile* file_;

  /// must be initialized in child EDAnalyzers. Otherwise: plain root mode
  DQMStore *DQM_; 
  
  

};

#endif 
