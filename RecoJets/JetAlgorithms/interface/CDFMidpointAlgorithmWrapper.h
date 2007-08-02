#ifndef JetAlgorithms_CDFMidpointAlgorithmWrapper_h
#define JetAlgorithms_CDFMidpointAlgorithmWrapper_h

/**
 * Interface to external CDF MidpointCone algorithm
 * F.Ratnikov, UMd, June 19, 2007
 * Redesigned on Aug. 1, 2007 by F.R.
 * $Id: CMSIterativeConeAlgorithm.cc,v 1.8 2007/07/20 18:46:38 fedor Exp $
 **/

#include "RecoJets/JetAlgorithms/interface/FastJetBaseWrapper.h"

namespace fastjet {
  class CDFMidPointPlugin;
}

class CDFMidpointAlgorithmWrapper : public FastJetBaseWrapper {
 public:
  CDFMidpointAlgorithmWrapper(const edm::ParameterSet& fConfig);
  virtual ~CDFMidpointAlgorithmWrapper();
 protected:
  virtual void makeJetDefinition (const edm::ParameterSet& fConfig);
 private:
  fastjet::CDFMidPointPlugin* mPlugin;
};

#endif
