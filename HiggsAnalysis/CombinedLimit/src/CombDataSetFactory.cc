#include "../interface/CombDataSetFactory.h"

#include <cmath>

CombDataSetFactory::CombDataSetFactory(const RooArgSet &vars, RooCategory &cat) :
    vars_(vars), cat_(&cat) 
{
    if (vars_.contains(cat)) vars_.remove(cat);
}

CombDataSetFactory::~CombDataSetFactory() {}

void CombDataSetFactory::addSet(const char *label, RooDataHist *set) {
    map_[label] = set;
}

void CombDataSetFactory::addSet(const char *label, RooDataSet *set) {
    mapUB_[label] = set;
}


RooDataHist *CombDataSetFactory::done(const char *name, const char *title) {
    return new RooDataHist(name,title,vars_,*cat_,map_);
    map_.clear();
}

RooDataSet *CombDataSetFactory::doneUnbinned(const char *name, const char *title) {
  using namespace RooFit;
  return new RooDataSet(name,title,vars_,Index(*cat_),Import(mapUB_));
    mapUB_.clear();
}


ClassImp(CombDataSetFactory)
