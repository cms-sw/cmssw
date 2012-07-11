#ifndef METAlgo_h
#define METAlgo_h

/** \class METAlgo
 *
 * Calculates MET for given input CaloTower collection.
 * Does corrections based on supplied parameters.
 *
 * \author M. Schmitt, R. Cavanaugh, The University of Florida
 *
 * \version   1st Version May 14, 2005
 ************************************************************/

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

class METAlgo 
{
 public:
  //typedef std::vector<const reco::Candidate*> InputCollection;
  typedef std::vector<const reco::Candidate> InputCollection;
  METAlgo();
  virtual ~METAlgo();
  //virtual void run(const reco::CandidateCollection*, CommonMETData*,  double );
  virtual void run(edm::Handle<edm::View<reco::Candidate> >, CommonMETData*,  double );
 private:
};

#endif // METAlgo_h

/*  LocalWords:  METAlgo
 */
