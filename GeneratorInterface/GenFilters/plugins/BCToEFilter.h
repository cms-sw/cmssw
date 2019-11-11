#ifndef BCToEFilter_h
#define BCToEFilter_h

/** \class BCToEFilter
 *
 *  BCToEFilter 
 *
 * \author J Lamb, UCSB
 * this is just the wrapper around the filtering algorithm
 * found in BCToEFilterAlgo
 * 
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/GenFilters/plugins/BCToEFilterAlgo.h"

class BCToEFilter : public edm::EDFilter {
public:
  explicit BCToEFilter(const edm::ParameterSet&);
  ~BCToEFilter() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  std::unique_ptr<BCToEFilterAlgo> BCToEAlgo_;
};
#endif
