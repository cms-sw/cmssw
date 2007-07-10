#ifndef HiggsAnalysisSkimming_HiggsAnalysisSkimType
#define HiggsAnalysisSkimming_HiggsAnalysisSkimType

/** \class HiggsAnalysisSkimType
 *
 * An abstract base class for algorithmic classes used to
 * create skims for different Higgs analyses, e.g. 
 * H -> ZZ -> 4 leptons.
 *
 * \author Dominique Fortin - UC Riverside
 *
 */

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>

class HiggsAnalysisSkimType {
 public:
  // Constructor
  explicit HiggsAnalysisSkimType( const edm::ParameterSet& ) {};
  // Destructor
  virtual ~HiggsAnalysisSkimType() {};

  /** Run the desired skim */
  virtual bool skim( edm::Event& event, const edm::EventSetup&, int& theTrigger ) = 0;

 private:
 
};

#endif

