#ifndef LA_FILLER_FITTER_H
#define LA_FILLER_FITTER_H

#include <string>
#include <vector>
#include <map>
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include <TTree.h>
#include "SymmetryFit.h"
class Book;

class LA_Filler_Fitter {

 public:

  enum Method { WIDTH  =1<<0, FIRST_METHOD=1<<0, 
		RATIO  =1<<1,  
		SQRTVAR=1<<2, 
		SYMM   =1<<3,
		MULTI  =1<<4, LAST_METHOD=1<<4};
  static std::string method(Method m,bool fit=true) { 
    switch(m) {
    case WIDTH:  return "_width_prof";
    case RATIO:  return std::string("_tanLA")+(fit?  "_ratio":"");
    case SQRTVAR:return "_sqrtVar_prof";
    case SYMM:   return fit? SymmetryFit::name("_symm") : "_symm";
    case MULTI:   return fit? SymmetryFit::name("_multi") : "_multi";
    default: return "_UNKNOWN";
    }
  }

  struct Result { 
    float reco,recoErr,measure,measureErr,calibratedMeasurement,calibratedError,field,chi2; 
    unsigned ndof,entries; 
    Result() : reco(1000), recoErr(0), 
	       measure(1000), measureErr(0), 
	       calibratedMeasurement(0), calibratedError(0), 
	       field(0), chi2(0), ndof(0), entries(0) {}
  };
  
  struct EnsembleSummary {
    unsigned samples;
    float truth, 
      meanMeasured,   SDmeanMeasured,
      sigmaMeasured,  SDsigmaMeasured,
      meanUncertainty,SDmeanUncertainty,
      pull,           SDpull;
    EnsembleSummary() : samples(0),truth(0),
		        meanMeasured(0),SDmeanMeasured(0),
		        sigmaMeasured(0),SDsigmaMeasured(0),
		        meanUncertainty(0),SDmeanUncertainty(0),
		        pull(0),SDpull(0) {}
  };
  
  LA_Filler_Fitter(int methods, int M, int N, double low, double up, unsigned max=0) :
    ensembleSize_(M),
    ensembleBins_(N),ensembleLow_(low),ensembleUp_(up),
    byLayer_(false),byModule_(false),
    maxEvents_(max),
    methods_(methods)
    {};
  
  LA_Filler_Fitter(int methods, bool layer, bool module, unsigned max=0) : 
    ensembleSize_(0),
    ensembleBins_(0),ensembleLow_(0),ensembleUp_(0),
    byLayer_(layer),byModule_(module),
    maxEvents_(max),
    methods_(methods)
    {};
  
  void fill(TTree*, Book&);
  void summarize_ensembles(Book&);
  
  static void fit(Book& book) { 
    make_and_fit_ratio(book); 
    make_and_fit_symmchi2(book);
    make_and_fit_multisymmchi2(book);
    fit_profile(book,method(WIDTH)); 
    fit_profile(book,method(SQRTVAR)); 
  }
  static void make_and_fit_ratio(Book&);
  static void make_and_fit_symmchi2(Book&);
  static void make_and_fit_multisymmchi2(Book&);
  static void fit_profile(Book&, const std::string);
  
  static Result result(Method, const std::string name, const Book&);
  static std::map< std::string,                      Result  >    layer_results(const Book&, const Method);
  static std::map<    uint32_t,                      Result  >   module_results(const Book&, const Method);
  static std::map< std::string,          std::vector<Result> > ensemble_results(const Book&, const Method );
  static std::map< std::string, std::vector<EnsembleSummary> > ensemble_summary(const Book& );

  static std::pair<std::pair<float,float>, std::pair<float,float> > offset_slope(const std::vector<EnsembleSummary>&);
  static float pull(const std::vector<EnsembleSummary>&);

  static std::string subdetLabel(const SiStripDetId);
  static std::string moduleLabel(const SiStripDetId);
  static std::string layerLabel(const SiStripDetId);
  
 private:
  
  static unsigned guess_bin(const TH1* const);
  static unsigned find_rebin(const TH1* const);
  static TH1* rms_from_x_xx(const std::string, const TH1* const, const TH1* const);
  static TH1* subset_probability(const std::string name, const TH1* const , const TH1* const );

  int ensembleSize_, ensembleBins_;
  double ensembleLow_, ensembleUp_;
  bool byLayer_, byModule_;
  Long64_t maxEvents_;
  int methods_;
};

std::ostream& operator<<(std::ostream&, const LA_Filler_Fitter::Result&);
std::ostream& operator<<(std::ostream&, const LA_Filler_Fitter::EnsembleSummary&);

#endif
