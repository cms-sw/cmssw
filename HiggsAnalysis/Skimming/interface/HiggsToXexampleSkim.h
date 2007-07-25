#ifndef HiggsAnalysis_HiggsToXexampleSkim
#define HiggsAnalysis_HiggsToXexampleSkim

/** \class HiggsToXexampleSkim
 *
 * This is a template for developers to create their own skim 
 *
 * \author Dominique Fortin - UC Riverside
 *
 */

// user include files
#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimType.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/InputTag.h"


class HiggsToXexampleSkim : public HiggsAnalysisSkimType {
  
 public:
  // Constructor
  explicit HiggsToXexampleSkim(const edm::ParameterSet&);

  // Destructor
  virtual ~HiggsToXexampleSkim(){};

  /// Get event properties to send to builder to fill seed collection
  virtual bool skim(edm::Event&, const edm::EventSetup& );


 private:
  bool debug;

  // Cuts 
  float dummyCut;

  // tag for sample label
  edm::InputTag recTrackLabel;
  edm::InputTag theGLBMuonLabel;
  edm::InputTag thePixelGsfELabel;
};

#endif
