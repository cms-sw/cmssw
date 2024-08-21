#ifndef FWCore_FWLite_BareRootProductGetter_h
#define FWCore_FWLite_BareRootProductGetter_h

#include "FWCore/FWLite/interface/BareRootProductGetterBase.h"

class BareRootProductGetter : public BareRootProductGetterBase {
public:
  BareRootProductGetter() = default;

private:
  TFile* currentFile() const override;
};

#endif
