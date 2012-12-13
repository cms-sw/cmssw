#ifndef HiggsAnalysis_CombinedLimit_LimitAlgo_h
#define HiggsAnalysis_CombinedLimit_LimitAlgo_h
/** \class LimitAlgo
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), Giovanni Petrucciani (UCSD)
 *
 *
 */
#include <boost/program_options.hpp>
#include <string>
class RooWorkspace;
class RooAbsData;
namespace RooStats { class ModelConfig; }

class LimitAlgo {
public:
  LimitAlgo() { }
  LimitAlgo(const char * desc) : options_(desc) { }
  virtual ~LimitAlgo() { }
  virtual void applyOptions(const boost::program_options::variables_map &vm) { }
  virtual void applyDefaultOptions() { }
  virtual void setToyNumber(const int) { }
  virtual void setNToys(const int) { }
  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) = 0;
  virtual const std::string & name() const = 0;
  const boost::program_options::options_description & options() const {
    return options_;
  }
protected:
  boost::program_options::options_description options_;
};

#endif
