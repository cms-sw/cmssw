#ifndef TCMETAlgo_h
#define TCMETAlgo_h

/** \class TCMETAlgo
 *
 * Calculates TCMET based on ... (add details here)
 *
 * \author    F. Golf and A. Yagil
 *
 * \version   1st Version November 12, 2008 
 ************************************************************/

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/MET.h"
class TCMETAlgo 
{
 public:
  typedef std::vector<const reco::Candidate> InputCollection;
  TCMETAlgo();
  virtual ~TCMETAlgo();
  reco::MET addInfo(edm::Handle<edm::View<reco::Candidate> >, CommonMETData tcmet, bool NoHF, double globaleThreshold);
 private:
};

#endif // TCMETAlgo_h

