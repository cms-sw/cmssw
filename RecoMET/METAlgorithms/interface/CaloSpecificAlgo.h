#ifndef METProducers_CaloMETInfo_h
#define METProducers_CaloMETInfo_h

/// Adds Calorimeter specific information to MET base class
/// Author: R. Cavanaugh (taken from F.Ratnikov, UMd)
/// 6 June, 2006

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

class CaloSpecificAlgo 
{
 public:
  /// Make CaloMET. Assumes MET is made from CaloTowerCandidates
  reco::CaloMET addInfo(CommonMETData met);
};

#endif
