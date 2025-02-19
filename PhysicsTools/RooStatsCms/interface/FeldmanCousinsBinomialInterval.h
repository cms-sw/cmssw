#ifndef PhysicsTools_RooStatsCms_FeldmanCousinsBinomialInterval_h
#define PhysicsTools_RooStatsCms_FeldmanCousinsBinomialInterval_h

#if (defined (STANDALONE) or defined (__CINT__) )
#include "BinomialNoncentralInterval.h"
#else
#include "PhysicsTools/RooStatsCms/interface/BinomialNoncentralInterval.h"
#endif

struct FeldmanCousinsSorter {
  bool operator()(const BinomialProbHelper& l, const BinomialProbHelper& r) const {
    return l.lratio() > r.lratio();
  }
};

class FeldmanCousinsBinomialInterval : public BinomialNoncentralInterval<FeldmanCousinsSorter> {
  const char* name() const { return "Feldman-Cousins"; }

#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(FeldmanCousinsBinomialInterval,1)
#endif
};

#endif
