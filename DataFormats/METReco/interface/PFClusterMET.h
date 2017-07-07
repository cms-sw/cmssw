#ifndef METReco_PFClusterMET_h
#define METReco_PFClusterMET_h

/*
class: PFClusterMET
description:  MET made from Particle Flow clusters
authors: Salvatore Rappoccio
date: 28-Dec-2010
*/

#include "DataFormats/METReco/interface/MET.h"
namespace reco
{
  class PFClusterMET:  public MET {
  public:
    PFClusterMET() ;
    PFClusterMET( double sumet_,
		  const LorentzVector& fP4, const Point& fVertex )
      : MET( sumet_, fP4, fVertex )  {}

    ~PFClusterMET() override {}
    

  };
}
#endif
