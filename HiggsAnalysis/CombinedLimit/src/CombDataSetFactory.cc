#include "../interface/CombDataSetFactory.h"
#include "../interface/utils.h"

#include <cmath>

CombDataSetFactory::CombDataSetFactory(const RooArgSet &vars, RooCategory &cat) :
    vars_(vars), cat_(&cat), weight_(0)
{
    if (vars_.contains(cat)) vars_.remove(cat);
}

CombDataSetFactory::~CombDataSetFactory() {}

void CombDataSetFactory::addSetBin(const char *label, RooDataHist *set) {
    map_[label] = set;
}

void CombDataSetFactory::addSetAny(const char *label, RooDataHist *set) {
    if (weight_ == 0) weight_ = new RooRealVar("_weight_","",1);
    RooDataSet *data = new RooDataSet(TString(set->GetName())+"_unbin", "", RooArgSet(*set->get(), *weight_), "_weight_");
    for (int i = 0, n = set->numEntries(); i < n; ++i) {
        const RooArgSet *entry = set->get(i);
        data->add(*entry, set->weight());
    }
    mapUB_[label] = data;
}


void CombDataSetFactory::addSetAny(const char *label, RooDataSet *set) {
    mapUB_[label] = set;
}


RooDataHist *CombDataSetFactory::done(const char *name, const char *title) {
    return new RooDataHist(name,title,vars_,*cat_,map_);
    map_.clear();
}

RooDataSet *CombDataSetFactory::doneUnbinned(const char *name, const char *title) {
    using namespace RooFit; 
    RooDataSet *ret = 0;
    if (weight_) {
        RooArgSet varsPlusWeight(vars_); varsPlusWeight.add(*weight_);
        ret = new RooDataSet(name,title,varsPlusWeight,Index(*cat_),Import(mapUB_),WeightVar(*weight_));
    } else {
        ret = new RooDataSet(name,title,vars_,Index(*cat_),Import(mapUB_));
    }
    mapUB_.clear();
    return ret;
}


ClassImp(CombDataSetFactory)
