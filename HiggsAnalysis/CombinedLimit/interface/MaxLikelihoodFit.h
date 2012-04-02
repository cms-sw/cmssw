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

class MaxLikelihoodFit : public FitterAlgoBase {
public:
  MaxLikelihoodFit() ;
  virtual const std::string & name() const {
    static const std::string name("MaxLikelihoodFit");
    return name;
  }
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

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

  void getNormalizations(RooAbsPdf *pdf, const RooArgSet &obs, RooArgSet &out);
};


#endif
