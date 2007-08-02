#ifndef JetAlgorithms_SISConeAlgorithmWrapper_h
#define JetAlgorithms_SISConeAlgorithmWrapper_h

/**
 * Interface to Seedless Infrared Safe Cone algorithm (http://projects.hepforge.org/siscone)
 * F.Ratnikov, UMd, June 22, 2007
 * Redesigned on Aug. 1, 2007 by F.R.
 * $Id: SISConeAlgorithmWrapper.h,v 1.2 2007/08/02 17:42:57 fedor Exp $
 **/

#include "RecoJets/JetAlgorithms/interface/FastJetBaseWrapper.h"

namespace fastjet {
  class SISConePlugin;
}


class SISConeAlgorithmWrapper : public FastJetBaseWrapper {
 public:
  SISConeAlgorithmWrapper(const edm::ParameterSet& fConfig);
  virtual ~SISConeAlgorithmWrapper();
 private:
  fastjet::SISConePlugin* mPlugin;
};

#endif
