#ifndef JetAlgorithms_SISConeAlgorithmWrapper_h
#define JetAlgorithms_SISConeAlgorithmWrapper_h

/**
 * Interface to Seedless Infrared Safe Cone algorithm (http://projects.hepforge.org/siscone)
 * F.Ratnikov, UMd, June 22, 2007
 * Redesigned on Aug. 1, 2007 by F.R.
 * $Id: CMSIterativeConeAlgorithm.cc,v 1.8 2007/07/20 18:46:38 fedor Exp $
 **/

#include "RecoJets/JetAlgorithms/interface/FastJetBaseWrapper.h"

namespace fastjet {
  class SISConePlugin;
}


class SISConeAlgorithmWrapper : public FastJetBaseWrapper {
 public:
  SISConeAlgorithmWrapper(const edm::ParameterSet& fConfig);
  virtual ~SISConeAlgorithmWrapper();
 protected:
  virtual void makeJetDefinition (const edm::ParameterSet& fConfig);
 private:
  fastjet::SISConePlugin* mPlugin;
};

#endif
