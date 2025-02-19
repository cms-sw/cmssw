#ifndef LA_FILLER_FITTER_H
#define LA_FILLER_FITTER_H

#include <string>
#include <vector>
#include <map>
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "CalibTracker/SiStripCommon/interface/Book.h"
#include <TTree.h>
#include "SymmetryFit.h"
class TProfile;

class LA_Filler_Fitter {

 public:

 //ensemble constructor
  LA_Filler_Fitter(int methods, int M, int N, double low, double up, unsigned max) :
    ensembleSize_(M),
    ensembleBins_(N),ensembleLow_(low),ensembleUp_(up),
    byLayer_(true),byModule_(false),
    localYbin_(0),
    stripsPerBin_(0),
    maxEvents_(max),
    methods_(methods) {};
    
  //measurement constructor
  LA_Filler_Fitter(int methods, bool layer, bool module, float localybin, unsigned stripbin, unsigned max) : 
    ensembleSize_(0),
    ensembleBins_(0),ensembleLow_(0),ensembleUp_(0),
    byLayer_(layer),byModule_(module),
    localYbin_(localybin),
    stripsPerBin_(stripbin),
    maxEvents_(max),
    methods_(methods) {};
  
  enum Method { WIDTH =1<<0, FIRST_METHOD=1<<0, 
		PROB1 =1<<1,
		AVGV2 =1<<2,  
		AVGV3 =1<<3, 
		RMSV2 =1<<4,
		RMSV3 =1<<5, LAST_METHOD=1<<5};

  static std::string method(Method m,bool fit=true) { 
    switch(m) {
    case WIDTH: return      "_width_prof";
    case PROB1: return fit? SymmetryFit::name(method(m,0)): "_prob_w1"   ;  
    case AVGV2: return fit? SymmetryFit::name(method(m,0)): "_avg_var_w2";  
    case AVGV3: return fit? SymmetryFit::name(method(m,0)): "_avg_var_w3";  
    case RMSV2: return fit? SymmetryFit::name(method(m,0)): "_rms_var_w2";  
    case RMSV3: return fit? SymmetryFit::name(method(m,0)): "_rms_var_w3";  
    default: return "_UNKNOWN";
    }
  }

  struct Result { 
    std::pair<float,float> reco,measured,calMeasured;
    float field,chi2;
    unsigned ndof,entries; 
    Result() : reco(1000,0), measured(1000,0), calMeasured(1000,0),
	       field(0), chi2(0), ndof(0), entries(0) {}
  };
  
  struct EnsembleSummary {
    std::pair<float,float> meanMeasured, sigmaMeasured, meanUncertainty, pull;
    float truth;
    unsigned samples;
    EnsembleSummary() : meanMeasured(0,0), sigmaMeasured(0,0), meanUncertainty(0,0), pull(0,0),
			truth(0), samples(0) {}
  };
  
  //Located in src/LA_Filler.cc
  void fill(TTree*, Book&) const;
  void fill_one_cluster(Book&,
			const poly<std::string>&,
			const unsigned,	const float, const float, const float, const float) const;
  poly<std::string> allAndOne(const unsigned width) const;
  poly<std::string> varWidth(const unsigned width) const;
  poly<std::string> granularity(const SiStripDetId, const float, const Long64_t, const float, const unsigned) const;
  static std::string subdetLabel(const SiStripDetId);
  static std::string moduleLabel(const SiStripDetId);
  static std::string layerLabel(const SiStripDetId);
  static unsigned layer_index(bool TIB, bool stereo, unsigned layer) { return  layer + (TIB?0:6) +(stereo?1:0) + ( (layer>2)?1:(layer==1)?-1:0 );}

  //Located in src/LA_Fitter.cc
  static void fit(Book& book) { make_and_fit_symmchi2(book); fit_width_profile(book); }
  static void make_and_fit_symmchi2(Book&);
  static void fit_width_profile(Book&);
  static TH1* rms_profile(const std::string, const TProfile* const);
  static TH1* subset_probability(const std::string name, const TH1* const , const TH1* const );
  static unsigned find_rebin(const TH1* const);

  //Located in src/LA_Results.cc
  void summarize_ensembles(Book&) const;
  static Result result(Method, const std::string name, const Book&);
  static std::map< std::string,                      Result  >    layer_results(const Book&, const Method);
  static std::map<    uint32_t,                      Result  >   module_results(const Book&, const Method);
  static std::map< std::string,          std::vector<Result> > ensemble_results(const Book&, const Method );
  static std::map< std::string, std::vector<EnsembleSummary> > ensemble_summary(const Book& );
  static std::pair<std::pair<float,float>, std::pair<float,float> > offset_slope(const std::vector<EnsembleSummary>&);
  static float pull(const std::vector<EnsembleSummary>&);
    
 private:
  
  const int ensembleSize_, ensembleBins_;
  const double ensembleLow_, ensembleUp_;
  const bool byLayer_, byModule_;
  const float localYbin_;
  const unsigned stripsPerBin_;
  const Long64_t maxEvents_;
  const int methods_;
};

std::ostream& operator<<(std::ostream&, const LA_Filler_Fitter::Result&);
std::ostream& operator<<(std::ostream&, const LA_Filler_Fitter::EnsembleSummary&);

#endif
