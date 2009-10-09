#ifndef RecoParticleFlow_Benchmark_Benchmark_h
#define RecoParticleFlow_Benchmark_Benchmark_h

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


#include <string>

class TH1;
class TH1F;
class TH2;
class TH2F;
class TFile; 
class TDirectory;

class DQMStore; 


/// abstract base class 
class Benchmark{

 public:

  Benchmark() : file_(0), DQM_(0) {}
  virtual ~Benchmark();


  
  /// call this function after construction to use the plain root mode
  ///COLIN this interface is in principle quite dangerous, as it allows to modify the TFile from inside the class. Any other possibility?
  void setFile(TFile *file);

  void setDirectory(TDirectory* dir);
  
  /// write to the TFile, in plain ROOT mode. No need to call this function in DQM mode
  void write();

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

  TDirectory* dir_;

  /// must be initialized in child EDAnalyzers. Otherwise: plain root mode
  DQMStore *DQM_; 
  
  

};

#endif 
