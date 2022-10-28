#ifndef CommonTools_ParticleFlow_PFMETAlgo_
#define CommonTools_ParticleFlow_PFMETAlgo_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//not a fwd declaration, to save the pain to the user to include the necessary DF header as well
#include "DataFormats/METReco/interface/MET.h"

/**\class PFMETAlgo 
\brief Computes the MET from a collection of PFCandidates. 

\author Colin Bernet
\date   february 2008
*/

namespace pf2pat {

  class PFMETAlgo {
  public:
    explicit PFMETAlgo(const edm::ParameterSet&);

    reco::MET produce(const reco::PFCandidateCollection& pfCandidates) const;

  private:
    /// HF calibration factor (in 31X applied by PFProducer)
    double hfCalibFactor_;

    /// verbose ?
    bool verbose_;
  };
}  // namespace pf2pat

#endif
