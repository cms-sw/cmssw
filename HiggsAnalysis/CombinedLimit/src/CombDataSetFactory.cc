#include "HiggsAnalysis/CombinedLimit/interface/CombDataSetFactory.h"

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

RooDataHist *CombDataSetFactory::done(const char *name, const char *title) {
    return new RooDataHist(name,title,vars_,*cat_,map_);
    map_.clear();
}

ClassImp(CombDataSetFactory)
