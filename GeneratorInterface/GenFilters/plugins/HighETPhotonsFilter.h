#ifndef HighETPhotonsFilter_h
#define HighETPhotonsFilter_h

/** \class HighETPhotonsFilter
 *
 *  HighETPhotonsFilter 
 *
 * \author J Lamb, UCSB
 * this is just the wrapper around the filtering algorithm
 * found in HighETPhotonsFilterAlgo
 * 
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/GenFilters/plugins/HighETPhotonsFilterAlgo.h"

class HighETPhotonsFilter : public edm::EDFilter {
public:
  explicit HighETPhotonsFilter(const edm::ParameterSet &);
  ~HighETPhotonsFilter() override;

  bool filter(edm::Event &, const edm::EventSetup &) override;

private:
  HighETPhotonsFilterAlgo *HighETPhotonsAlgo_;
};
#endif
