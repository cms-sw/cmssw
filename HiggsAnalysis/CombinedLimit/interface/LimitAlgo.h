#ifndef HiggsAnalysis_CombinedLimit_LimitAlgo_h
#define HiggsAnalysis_CombinedLimit_LimitAlgo_h
/** \class LimitAlgo
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN)
 *
 *
 */
#include <boost/program_options.hpp>
#include <string>
class RooWorkspace;
class RooAbsData;

class LimitAlgo {
public:
  LimitAlgo() { }
  virtual void options(const boost::program_options::variables_map &vm) { }
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit) = 0;
  virtual const std::string & name() const = 0;
  virtual const boost::program_options::options_description & options() const = 0;
};

#endif
