#ifndef VBFHiggsTo2TauLJetSkim_h
#define VBFHiggsTo2TauLJetSkim_h

/** \class VBFHiggsTo2TauLJetSkim
 *
 * From HiggsAnalysis/Skimming/interface/HiggsToXexample.h
 *
 * \author S.Greder
 *
 */

#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimType.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/Event.h>
// #include "FWCore/ParameterSet/interface/InputTag.h"


class VBFHiggsTo2TauLJetSkim : public HiggsAnalysisSkimType 
{
  
 public:
  // Constructor
  explicit VBFHiggsTo2TauLJetSkim(const edm::ParameterSet&);

  // Destructor
  virtual ~VBFHiggsTo2TauLJetSkim(){};

  /// Get event properties to send to builder to fill seed collection
  virtual bool skim(edm::Event&, const edm::EventSetup&);

 private:
  bool debug;

  // Cuts 
};

#endif
