#ifndef ROO_SIM_OPT_PDF
#define ROO_SIM_OPT_PDF

/** Trick RooSimultaneous */

#include "RooSimultaneous.h"

class RooSimultaneousOpt : public RooSimultaneous {
public:
	RooSimultaneousOpt(){};
  RooSimultaneousOpt(const char* name, const char* title, RooAbsCategoryLValue& indexCat) :
        RooSimultaneous(name, title, indexCat) {}
  RooSimultaneousOpt(const RooSimultaneous &any, const char* name=0) :
        RooSimultaneous(any, name) {}
  RooSimultaneousOpt(const RooSimultaneousOpt &other, const char* name=0) :
        RooSimultaneous(other, name) {}

  virtual RooSimultaneousOpt *clone(const char* name=0) const { return new RooSimultaneousOpt(*this, name); }

  virtual RooAbsReal* createNLL(RooAbsData& data, const RooLinkedList& cmdList) ;

private:
  ClassDef(RooSimultaneousOpt,1) // Variant of RooSimultaneous that can put together binned and unbinned stuff 

};

#endif
