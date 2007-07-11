#ifndef HiggsAnalysis_HiggsAnalysisSkimming
#define HiggsAnalysis_HiggsAnalysisSkimming

/* \class HiggsAnalysisSkimming
 *
 *
 * EDFilter to select Higgs events based on the
 * different filters (4 leptons, 2 gammas, ...).
 *
 * At this stage, the L3 trigger isn't setup, so mimic L3 trigger
 * selection using full reconstruction
 *
 * \author Dominique Fortin - UC Riverside
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDFilter.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/Event.h>

class HiggsAnalysisSkimType;

class HiggsAnalysisSkimming : public edm::EDFilter {
  
 public:
  // Constructor
  explicit HiggsAnalysisSkimming(const edm::ParameterSet&);

  // Destructor
  virtual ~HiggsAnalysisSkimming();

  /// Get event properties to send to builder to fill seed collection
  virtual bool filter(edm::Event&, const edm::EventSetup&);


 private:
  bool debug;

  // Class for performing skim
  HiggsAnalysisSkimType* skimFilter;

  int nEvents,
      nSelectedEvents;

};

#endif
