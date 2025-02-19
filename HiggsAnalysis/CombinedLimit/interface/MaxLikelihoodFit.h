#ifndef HiggsAnalysis_CombinedLimit_MaxLikelihoodFit_h
#define HiggsAnalysis_CombinedLimit_MaxLikelihoodFit_h
/** \class MaxLikelihoodFit
 *
 * Do a ML fit of the data with background and signal+background hypothesis and print out diagnostics plots 
 *
 * \author Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "../interface/FitterAlgoBase.h"
#include "TTree.h"
class MaxLikelihoodFit : public FitterAlgoBase {
public:
  MaxLikelihoodFit() ;
  virtual const std::string & name() const {
    static const std::string name("MaxLikelihoodFit");
    return name;
  }
  ~MaxLikelihoodFit();
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;
  virtual void setToyNumber(const int) ;
  virtual void setNToys(const int);

protected:
  virtual bool runSpecific(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);

  static std::string name_;

  static std::string minos_;

  static bool justFit_, noErrors_;
  static std::string out_; 
  static bool        makePlots_;
  static float       rebinFactor_;
  static std::string signalPdfNames_, backgroundPdfNames_;
  static bool        saveNormalizations_;
  static bool        reuseParams_;
  int currentToy_, nToys;
  int fitStatus_, numbadnll_;
  double mu_, nll_nll0_, nll_bonly_,nll_sb_;
  std::auto_ptr<TFile> fitOut;
  double* globalObservables_;
  double* nuisanceParameters_;

  TTree *t_fit_b_, *t_fit_sb_;
   
  void getNormalizations(RooAbsPdf *pdf, const RooArgSet &obs, RooArgSet &out);
  void createFitResultTrees(const RooStats::ModelConfig &);
  void setFitResultTrees(const RooArgSet *, double *);
};


#endif
