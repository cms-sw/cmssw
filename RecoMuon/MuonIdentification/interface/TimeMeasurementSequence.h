#ifndef MuonIdentification_TimeMeasurementSequence_h
#define MuonIdentification_TimeMeasurementSequence_h

/** \class reco::TimeMeasurementSequence TimeMeasurementSequence.h RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h
 *  
 * A class holding a set of individual time measurements along the muon trajectory
 *
 * \author Piotr Traczyk, CERN
 *
 *
 */

#include <vector>

class TimeMeasurementSequence {
public:
  std::vector<double> dstnc;
  std::vector<double> local_t0;
  std::vector<double> weightTimeVtx;
  std::vector<double> weightInvbeta;

  double totalWeightInvbeta;
  double totalWeightTimeVtx;

  TimeMeasurementSequence() : totalWeightInvbeta(0), totalWeightTimeVtx(0) {}
};

#endif
