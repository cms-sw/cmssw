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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/GenFilters/plugins/BCToEFilterAlgo.h"

class BCToEFilter : public edm::global::EDFilter<> {
public:
  explicit BCToEFilter(const edm::ParameterSet&);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const BCToEFilterAlgo BCToEAlgo_;
};
#endif
