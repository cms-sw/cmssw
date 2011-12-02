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
#if 0   // This works most of the time, but sometimes weights end up wrong.
        // I don't knopw why the @$^#&^# this happens, but it nevertheless 
        // means that I can't use the constructor with the map directly
        //
        for (std::map<std::string, RooDataSet *>::iterator it = mapUB_.begin(), ed = mapUB_.end(); it != ed; ++it) {
            RooDataSet *data = it->second;
            if (!data->isWeighted()) {
                weight_->setVal(1.0);
                data->addColumn(*weight_);
                it->second = new RooDataSet(data->GetName(), data->GetTitle(), data, *data->get(), /*cut=*/(char*)0, weight_->GetName());
            }
            //std::cout << "\n\n\n========== DATASET " << it->first << " =============" << std::endl;
            //utils::printRDH(it->second);
        }
        RooArgSet varsPlusWeight(vars_); varsPlusWeight.add(*weight_);
        ret = new RooDataSet(name,title,varsPlusWeight,Index(*cat_),Import(mapUB_),WeightVar(*weight_));
        //std::cout << "\n\n\n========== COMBINED DATASET =============" << std::endl;
        //utils::printRDH(ret);
#else
        RooArgSet varsPlusCat(vars_); varsPlusCat.add(*cat_);
        RooArgSet varsPlusWeight(varsPlusCat); varsPlusWeight.add(*weight_);
        ret = new RooDataSet(name,title,varsPlusWeight,WeightVar(*weight_));
        for (std::map<std::string, RooDataSet *>::iterator it = mapUB_.begin(), ed = mapUB_.end(); it != ed; ++it) {
            //std::cout << "\n\n\n========== DATASET " << it->first << " =============" << std::endl;
            //utils::printRDH(it->second);
            cat_->setLabel(it->first.c_str());
            RooDataSet *data = it->second;
            for (unsigned int i = 0, n = data->numEntries(); i < n; ++i) {
                varsPlusCat = *data->get(i);
                ret->add(varsPlusCat, data->weight());
            }
        }
        //std::cout << "\n\n\n========== COMBINED DATASET =============" << std::endl;
        //utils::printRDH(ret);
#endif
    } else {
        ret = new RooDataSet(name,title,vars_,Index(*cat_),Import(mapUB_));
    }
    mapUB_.clear();
    return ret;
}


ClassImp(CombDataSetFactory)
