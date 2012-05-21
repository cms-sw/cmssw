/*
 * Selects taus by rho corrected PT
 *
 * Author: Evan K. Friis, UW Madison
 *
 * */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDFilter.h"

class PFTauPtCutRhoCorrectedSelector : public edm::EDFilter {
  public:
    PFTauPtCutRhoCorrectedSelector(const edm::ParameterSet& pset);
    virtual ~PFTauPtCutRhoCorrectedSelector(){}
    bool filter(edm::Event& evt, const edm::EventSetup& es);
  private:
    edm::InputTag src_;
    edm::InputTag srcRho_;
    double effectiveArea_;
    double minPt_;
    bool filter_;
};

