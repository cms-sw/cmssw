#ifndef MVAMETDATA_UMKFIMU
#define MVAMETDATA_UMKFIMU

/*
 * =====================================================================================
 *
 *       Filename:  MVAMETData.h
 *
 *    Description:  Intermediate data formats used in MVA MET calculation.
 *
 *         Author:  Phil Harris, CERN
 *
 * =====================================================================================
 */


#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {

struct JetInfo
{
  JetInfo()
    : p4_(0.,0.,0.,0.),
      mva_(0.),
      neutralEnFrac_(0.)
  {}
  ~JetInfo() {}
  reco::Candidate::LorentzVector p4_;
  double mva_;
  double neutralEnFrac_;

  friend bool operator<(const reco::JetInfo&, const reco::JetInfo&);
};

bool operator<(const JetInfo&, const JetInfo&);

}

#endif /* end of include guard: MVAMETDATA_UMKFIMU */
