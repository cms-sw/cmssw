#ifndef PhysicsTools_RooStatsCms_SterneBinomialInterval_h
#define PhysicsTools_RooStatsCms_SterneBinomialInterval_h

#if (defined (STANDALONE) or defined (__CINT__) )
#include "BinomialNoncentralInterval.h"
#else
#include "PhysicsTools/RooStatsCms/interface/BinomialNoncentralInterval.h"
#endif

struct SterneSorter {
  bool operator()(const BinomialProbHelper& l, const BinomialProbHelper& r) const {
    return l.prob() > r.prob();
  }
};

class SterneBinomialInterval : public BinomialNoncentralInterval<SterneSorter> {
  const char* name() const { return "Feldman-Cousins"; }

#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(SterneBinomialInterval,1)
#endif
};

#endif
