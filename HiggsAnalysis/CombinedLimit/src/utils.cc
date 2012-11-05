#include "../interface/utils.h"
#include "../interface/RooSimultaneousOpt.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <typeinfo>
#include <stdexcept>

#include <TIterator.h>
#include <TString.h>

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooRealVar.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>
#include <RooPlot.h>
#include <RooStats/ModelConfig.h>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

using namespace std;

void utils::printRDH(RooAbsData *data) {
  std::vector<std::string> varnames, catnames;
  const RooArgSet *b0 = data->get();
  TIterator *iter = b0->createIterator();
  for (RooAbsArg *a = 0; (a = (RooAbsArg *)iter->Next()) != 0; ) {
    if (a->InheritsFrom("RooRealVar")) {
      varnames.push_back(a->GetName());
    } else if (a->InheritsFrom("RooCategory")) {
      catnames.push_back(a->GetName());
    }
  }
  delete iter;
  size_t nv = varnames.size(), nc = catnames.size();
  printf(" bin  ");
  for (size_t j = 0; j < nv; ++j) { printf("%16.16s  ", varnames[j].c_str()); }
  for (size_t j = 0; j < nc; ++j) { printf("%16.16s  ", catnames[j].c_str()); }
  printf("  weight\n");
  for (int i = 0, nb = data->numEntries(); i < nb; ++i) {
    const RooArgSet *bin = data->get(i);
    printf("%4d  ",i);
    for (size_t j = 0; j < nv; ++j) { printf("%16g  ",    bin->getRealValue(varnames[j].c_str())); }
    for (size_t j = 0; j < nc; ++j) { printf("%16.16s  ", bin->getCatLabel(catnames[j].c_str())); }
    printf("%8.3f\n", data->weight());
  }
}

void utils::printRAD(const RooAbsData *d) {
  if (d->InheritsFrom("RooDataHist") || d->numEntries() != 1) printRDH(const_cast<RooAbsData*>(d));
  else d->get(0)->Print("V");
}

void utils::printPdf(RooAbsPdf *pdf) {
  std::cout << "Pdf " << pdf->GetName() << " parameters." << std::endl;
  std::auto_ptr<RooArgSet> params(pdf->getVariables());
  params->Print("V");
}


void utils::printPdf(RooStats::ModelConfig &mc) {
  std::cout << "ModelConfig " << mc.GetName() << " (" << mc.GetTitle() << "): pdf parameters." << std::endl;
  std::auto_ptr<RooArgSet> params(mc.GetPdf()->getVariables());
  params->Print("V");
}

void utils::printPdf(RooWorkspace *w, const char *pdfName) {
  std::cout << "PDF " << pdfName << " parameters." << std::endl;
  std::auto_ptr<RooArgSet> params(w->pdf(pdfName)->getVariables());
  params->Print("V");
}

RooAbsPdf *utils::factorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &constraints) {
    assert(&pdf);
    const std::type_info & id = typeid(pdf);
    if (id == typeid(RooProdPdf)) {
        //std::cout << " pdf is product pdf " << pdf.GetName() << std::endl;
        RooProdPdf *prod = dynamic_cast<RooProdPdf *>(&pdf);
        RooArgList newFactors; RooArgSet newOwned;
        RooArgList list(prod->pdfList());
        bool needNew = false;
        for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            RooAbsPdf *newpdf = factorizePdf(observables, *pdfi, constraints);
            //std::cout << "    for " << pdfi->GetName() << "   newpdf  " << (newpdf == 0 ? "null" : (newpdf == pdfi ? "old" : "new"))  << std::endl;
            if (newpdf == 0) { needNew = true; continue; }
            if (newpdf != pdfi) { needNew = true; newOwned.add(*newpdf); }
            newFactors.add(*newpdf);
        }
        if (!needNew) { copyAttributes(pdf, *prod); return prod; }
        else if (newFactors.getSize() == 0) return 0;
        else if (newFactors.getSize() == 1) {
            RooAbsPdf *ret = (RooAbsPdf *) newFactors.first()->Clone(TString::Format("%s_obsOnly", pdf.GetName()));
            copyAttributes(pdf, *ret);
            return ret;
        }
        RooProdPdf *ret = new RooProdPdf(TString::Format("%s_obsOnly", pdf.GetName()), "", newFactors);
        ret->addOwnedComponents(newOwned);
        copyAttributes(pdf, *ret);
        return ret;
    } else if (id == typeid(RooSimultaneous) || id == typeid(RooSimultaneousOpt)) {
        RooSimultaneous *sim  = dynamic_cast<RooSimultaneous *>(&pdf);
        RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone();
        int nbins = cat->numBins((const char *)0);
        TObjArray factorizedPdfs(nbins); RooArgSet newOwned;
        bool needNew = false;
        for (int ic = 0, nc = nbins; ic < nc; ++ic) {
            cat->setBin(ic);
            RooAbsPdf *pdfi = sim->getPdf(cat->getLabel());
            RooAbsPdf *newpdf = factorizePdf(observables, *pdfi, constraints);
            factorizedPdfs[ic] = newpdf;
            if (newpdf == 0) { throw std::runtime_error(std::string("ERROR: channel ") + cat->getLabel() + " factorized to zero."); }
            if (newpdf != pdfi) { needNew = true; newOwned.add(*newpdf); }
        }
        RooSimultaneous *ret = sim;
        if (needNew) {
            ret = new RooSimultaneous(TString::Format("%s_obsOnly", pdf.GetName()), "", (RooAbsCategoryLValue&) sim->indexCat());
            for (int ic = 0, nc = nbins; ic < nc; ++ic) {
                cat->setBin(ic);
                RooAbsPdf *newpdf = (RooAbsPdf *) factorizedPdfs[ic];
                if (newpdf) ret->addPdf(*newpdf, cat->getLabel());
            }
            ret->addOwnedComponents(newOwned);
        }
        delete cat;
        if (id == typeid(RooSimultaneousOpt)) {
            RooSimultaneousOpt *newret = new RooSimultaneousOpt(*ret);
            newret->addOwnedComponents(RooArgSet(*ret));
            ret = newret;
        }
        copyAttributes(pdf, *ret);
        return ret;
    } else if (pdf.dependsOn(observables)) {
        return &pdf;
    } else {
        if (!constraints.contains(pdf)) constraints.add(pdf);
        return 0;
    }

}

void utils::factorizePdf(RooStats::ModelConfig &model, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints, bool debug) {
    return factorizePdf(*model.GetObservables(), pdf, obsTerms, constraints, debug);
}
void utils::factorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints, bool debug) {
    assert(&pdf);
    const std::type_info & id = typeid(pdf);
    if (id == typeid(RooProdPdf)) {
        RooProdPdf *prod = dynamic_cast<RooProdPdf *>(&pdf);
        RooArgList list(prod->pdfList());
        for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            factorizePdf(observables, *pdfi, obsTerms, constraints);
        }
    } else if (id == typeid(RooSimultaneous) || id == typeid(RooSimultaneousOpt)) {
        RooSimultaneous *sim  = dynamic_cast<RooSimultaneous *>(&pdf);
        RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone();
        for (int ic = 0, nc = cat->numBins((const char *)0); ic < nc; ++ic) {
            cat->setBin(ic);
            RooAbsPdf *pdfi = sim->getPdf(cat->getLabel());
            if (pdfi != 0) factorizePdf(observables, *pdfi, obsTerms, constraints);
        }
        delete cat;
    } else if (pdf.dependsOn(observables)) {
        if (!obsTerms.contains(pdf)) obsTerms.add(pdf);
    } else {
        if (!constraints.contains(pdf)) constraints.add(pdf);
    }
}
void utils::factorizeFunc(const RooArgSet &observables, RooAbsReal &func, RooArgList &obsTerms, RooArgList &constraints, bool debug) {
    RooAbsPdf *pdf = dynamic_cast<RooAbsPdf *>(&func);
    if (pdf != 0) { 
        factorizePdf(observables, *pdf, obsTerms, constraints, debug); 
        return; 
    }
    const std::type_info & id = typeid(func);
    if (id == typeid(RooProduct)) {
        RooProduct *prod = dynamic_cast<RooProduct *>(&func);
        RooArgSet components(prod->components());
        //std::cout << "Function " << func.GetName() << " is a RooProduct with " << components.getSize() << " components." << std::endl;
        std::auto_ptr<TIterator> iter(components.createIterator());
        for (RooAbsReal *funci = (RooAbsReal *) iter->Next(); funci != 0; funci = (RooAbsReal *) iter->Next()) {
            //std::cout << "  component " << funci->GetName() << " of type " << funci->ClassName() << std::endl;
            factorizeFunc(observables, *funci, obsTerms, constraints);
        }
    } else if (func.dependsOn(observables)) {
        if (!obsTerms.contains(func)) obsTerms.add(func);
    } else {
        if (!constraints.contains(func)) constraints.add(func);
    }
}

RooAbsPdf *utils::makeNuisancePdf(RooStats::ModelConfig &model, const char *name) { 
    return utils::makeNuisancePdf(*model.GetPdf(), *model.GetObservables(), name);
}

RooAbsPdf *utils::makeNuisancePdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name) { 
    assert(&pdf);
    RooArgList obsTerms, constraints;
    factorizePdf(observables, pdf, obsTerms, constraints);
    if (constraints.getSize() == 0) return 0;
    return new RooProdPdf(name,"", constraints);
}

RooAbsPdf *utils::makeObsOnlyPdf(RooStats::ModelConfig &model, const char *name) { 
    RooArgList obsTerms, constraints;
    factorizePdf(model, *model.GetPdf(), obsTerms, constraints);
    return new RooProdPdf(name,"", obsTerms);
}

RooAbsPdf *utils::fullClonePdf(const RooAbsPdf *pdf, RooArgSet &holder, bool cloneLeafNodes) {
  // Clone all FUNC compents by copying all branch nodes
  RooArgSet tmp("RealBranchNodeList") ;
  pdf->branchNodeServerList(&tmp);
  tmp.snapshot(holder, cloneLeafNodes); 
  // Find the top level FUNC in the snapshot list
  return (RooAbsPdf*) holder.find(pdf->GetName());
}
RooAbsReal *utils::fullCloneFunc(const RooAbsReal *pdf, RooArgSet &holder, bool cloneLeafNodes) {
  // Clone all FUNC compents by copying all branch nodes
  RooArgSet tmp("RealBranchNodeList") ;
  pdf->branchNodeServerList(&tmp);
  tmp.snapshot(holder, cloneLeafNodes); 
  // Find the top level FUNC in the snapshot list
  return (RooAbsReal*) holder.find(pdf->GetName());
}


void utils::getClients(const RooAbsCollection &values, const RooAbsCollection &allObjects, RooAbsCollection &clients) {
    std::auto_ptr<TIterator> iterAll(allObjects.createIterator());
    std::auto_ptr<TIterator> iterVal(values.createIterator());
    for (RooAbsArg *v = (RooAbsArg *) iterVal->Next(); v != 0; v = (RooAbsArg *) iterVal->Next()) {
        if (typeid(*v) != typeid(RooRealVar)) continue;
        std::auto_ptr<TIterator> clientIter(v->clientIterator());
        for (RooAbsArg *a = (RooAbsArg *) clientIter->Next(); a != 0; a = (RooAbsArg *) clientIter->Next()) {
            if (allObjects.containsInstance(*a) && !clients.containsInstance(*a)) clients.add(*a);
        }
    }
}

bool utils::setAllConstant(const RooAbsCollection &coll, bool constant) {
    bool changed = false;
    std::auto_ptr<TIterator> iter(coll.createIterator());
    for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
        RooRealVar *v = dynamic_cast<RooRealVar *>(a);
        if (v && (v->isConstant() != constant)) {
            changed = true;
            v->setConstant(constant);
        }
    }
    return changed;
}

bool utils::checkModel(const RooStats::ModelConfig &model, bool throwOnFail) {
    bool ok = true; std::ostringstream errors; 
    std::auto_ptr<TIterator> iter;
    RooAbsPdf *pdf = model.GetPdf(); if (pdf == 0) throw std::invalid_argument("Model without Pdf");
    RooArgSet allowedToFloat; 
    if (model.GetObservables() == 0) { 
        ok = false; errors << "ERROR: model does not define observables.\n"; 
        std::cout << errors.str() << std::endl;
        if (throwOnFail) throw std::invalid_argument(errors.str()); else return false; 
    } else {
        allowedToFloat.add(*model.GetObservables());
    }
    if (model.GetParametersOfInterest() == 0) { 
        ok = false; errors << "ERROR: model does not define parameters of interest.\n";  
    } else {
        iter.reset(model.GetParametersOfInterest()->createIterator());
        for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
            RooRealVar *v = dynamic_cast<RooRealVar *>(a);
            if (!v) { ok = false; errors << "ERROR: parameter of interest " << a->GetName() << " is a " << a->ClassName() << " and not a RooRealVar\n"; continue; }
            if (v->isConstant()) { ok = false; errors << "ERROR: parameter of interest " << a->GetName() << " is constant\n"; continue; }
            if (!pdf->dependsOn(*v)) { ok = false; errors << "ERROR: pdf does not depend on parameter of interest " << a->GetName() << "\n"; continue; }
            allowedToFloat.add(*v);
        }
    }
    if (model.GetNuisanceParameters() != 0) { 
        iter.reset(model.GetNuisanceParameters()->createIterator());
        for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
            RooRealVar *v = dynamic_cast<RooRealVar *>(a);
            if (!v) { ok = false; errors << "ERROR: nuisance parameter " << a->GetName() << " is a " << a->ClassName() << " and not a RooRealVar\n"; continue; }
            if (v->isConstant()) { ok = false; errors << "ERROR: nuisance parameter " << a->GetName() << " is constant\n"; continue; }
            if (!pdf->dependsOn(*v)) { errors << "WARNING: pdf does not depend on nuisance parameter " << a->GetName() << "\n"; continue; }
            allowedToFloat.add(*v);
        }
    }
    if (model.GetGlobalObservables() != 0) { 
        iter.reset(model.GetGlobalObservables()->createIterator());
        for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
            RooRealVar *v = dynamic_cast<RooRealVar *>(a);
            if (!v) { ok = false; errors << "ERROR: global observable " << a->GetName() << " is a " << a->ClassName() << " and not a RooRealVar\n"; continue; }
            if (!v->isConstant()) { ok = false; errors << "ERROR: global observable " << a->GetName() << " is not constant\n"; continue; }
            if (!pdf->dependsOn(*v)) { errors << "WARNING: pdf does not depend on global observable " << a->GetName() << "\n"; continue; }
        }
    }
    std::auto_ptr<RooArgSet> params(pdf->getParameters(*model.GetObservables()));
    iter.reset(params->createIterator());
    for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
        if (a->getAttribute("flatParam") && a->isConstant()) {
            ok = false; errors << "ERROR: parameter " << a->GetName() << " is declared as flatParam but is constant.\n";
        }
        if (a->isConstant() || allowedToFloat.contains(*a)) continue;
        if (a->getAttribute("flatParam")) {
            RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
            if (rrv->getVal() > rrv->getMax() || rrv->getVal() < rrv->getMin()) {
                ok = false; errors << "ERROR: flatParam " << rrv->GetName() << " has a value " << rrv->getVal() << 
                                      " outside of the defined range [" << rrv->getMin() << ", " << rrv->getMax() << "]\n";
            }
        } else {
            errors << "WARNING: pdf parameter " << a->GetName() << " (type " << a->ClassName() << ") is not allowed to float (it's not nuisance, poi, observable or global observable\n"; 
        }
        RooRealVar *v = dynamic_cast<RooRealVar *>(a);
        if (v != 0 && !v->isConstant() && (v->getVal() < v->getMin() || v->getVal() > v->getMax())) {
            ok = false; errors << "ERROR: parameter " << a->GetName() << " has value " << v->getVal() << " outside range [ " << v->getMin() << " , " << v->getMax() << " ]\n"; 
        }
    }
    iter.reset();
    std::cout << errors.str() << std::endl;
    if (!ok && throwOnFail) throw std::invalid_argument(errors.str()); 
    return ok;
}

RooSimultaneous * utils::rebuildSimPdf(const RooArgSet &observables, RooSimultaneous *sim) {
    RooArgList constraints;
    RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone();
    int nbins = cat->numBins((const char *)0);
    TObjArray factorizedPdfs(nbins); 
    RooArgSet newOwned;
    for (int ic = 0, nc = nbins; ic < nc; ++ic) {
        cat->setBin(ic);
        RooAbsPdf *pdfi = sim->getPdf(cat->getLabel());
        if (pdfi == 0) { factorizedPdfs[ic] = 0; continue; }
        RooAbsPdf *newpdf = factorizePdf(observables, *pdfi, constraints);
        factorizedPdfs[ic] = newpdf;
        if (newpdf == 0) { continue; }
        if (newpdf != pdfi) { newOwned.add(*newpdf);  }
    }
    RooSimultaneous *ret = new RooSimultaneous(TString::Format("%s_reloaded", sim->GetName()), "", (RooAbsCategoryLValue&) sim->indexCat());
    for (int ic = 0, nc = nbins; ic < nc; ++ic) {
        cat->setBin(ic);
        RooAbsPdf *newpdf = (RooAbsPdf *) factorizedPdfs[ic];
        if (newpdf) {
            if (constraints.getSize() > 0) {
                RooArgList allfactors(constraints); allfactors.add(*newpdf);
                RooProdPdf *newerpdf = new RooProdPdf(TString::Format("%s_plus_constr", newpdf->GetName()), "", allfactors);
                ret->addPdf(*newerpdf, cat->getLabel());
                copyAttributes(*newpdf, *newerpdf);
                newOwned.add(*newerpdf);
            } else {
                ret->addPdf(*newpdf, cat->getLabel());
            }
        }
    }
    ret->addOwnedComponents(newOwned);
    copyAttributes(*sim, *ret);
    delete cat;
    return ret;
}

void utils::copyAttributes(const RooAbsArg &from, RooAbsArg &to) {
    if (&from == &to) return;
    const std::set<std::string> attribs = from.attributes();
    if (!attribs.empty()) {
        for (std::set<std::string>::const_iterator it = attribs.begin(), ed = attribs.end(); it != ed; ++it) to.setAttribute(it->c_str());
    }
    const std::map<std::string, std::string> strattribs = from.stringAttributes();
    if (!strattribs.empty()) {
        for (std::map<std::string,std::string>::const_iterator it = strattribs.begin(), ed = strattribs.end(); it != ed; ++it) to.setStringAttribute(it->first.c_str(), it->second.c_str());
    }
}

void utils::guessChannelMode(RooSimultaneous &simPdf, RooAbsData &simData, bool verbose) 
{
    RooAbsCategoryLValue &cat = const_cast<RooAbsCategoryLValue &>(simPdf.indexCat());
    TList *split = simData.split(cat, kTRUE);
    for (int i = 0, n = cat.numBins((const char *)0); i < n; ++i) {
        cat.setBin(i);
        RooAbsPdf *pdf = simPdf.getPdf(cat.getLabel());
        if (pdf->getAttribute("forceGenBinned") || pdf->getAttribute("forceGenUnbinned")) {
            if (verbose) std::cout << " - " << cat.getLabel() << " has generation mode already set" << std::endl;
            continue;
        }
        RooAbsData *spl = (RooAbsData *) split->FindObject(cat.getLabel());
        if (spl == 0) { 
            if (verbose) std::cout << " - " << cat.getLabel() << " has no dataset, cannot guess" << std::endl; 
            continue;
        }
        if (spl->numEntries() != spl->sumEntries()) {
            if (verbose) std::cout << " - " << cat.getLabel() << " has " << spl->numEntries() << " num entries of sum " << spl->sumEntries() << ", mark as binned" << std::endl;
            pdf->setAttribute("forceGenBinned");
        } else {
            if (verbose) std::cout << " - " << cat.getLabel() << " has " << spl->numEntries() << " num entries of sum " << spl->sumEntries() << ", mark as unbinned" << std::endl;
            pdf->setAttribute("forceGenUnbinned");
        }
    }
}

std::vector<RooPlot *>
utils::makePlots(const RooAbsPdf &pdf, const RooAbsData &data, const char *signalSel, const char *backgroundSel, float rebinFactor) {
    std::vector<RooPlot *> ret;
    RooArgList constraints;
    RooAbsPdf *facpdf = factorizePdf(*data.get(0), const_cast<RooAbsPdf &>(pdf), constraints);

    const std::type_info & id = typeid(*facpdf);
    if (id == typeid(RooSimultaneous) || id == typeid(RooSimultaneousOpt)) {
        const RooSimultaneous *sim  = dynamic_cast<const RooSimultaneous *>(&pdf);
        const RooAbsCategoryLValue &cat = (RooAbsCategoryLValue &) sim->indexCat();
        TList *datasets = data.split(cat, true);
        TIter next(datasets);
        for (RooAbsData *ds = (RooAbsData *) next(); ds != 0; ds = (RooAbsData *) next()) {
            RooAbsPdf *pdfi  = sim->getPdf(ds->GetName());
            std::auto_ptr<RooArgSet> obs(pdfi->getObservables(ds));
            if (obs->getSize() == 0) break;
            RooRealVar *x = dynamic_cast<RooRealVar *>(obs->first());
            if (x == 0) continue;
            int nbins = x->numBins(); if (nbins == 0) nbins = 100;
            if (nbins/rebinFactor > 6) nbins = ceil(nbins/rebinFactor);
            ret.push_back(x->frame(RooFit::Title(ds->GetName()), RooFit::Bins(nbins)));
            ret.back()->SetName(ds->GetName());
            ds->plotOn(ret.back());
            if (signalSel && strlen(signalSel))         pdfi->plotOn(ret.back(), RooFit::LineColor(209), RooFit::Components(signalSel));
            if (backgroundSel && strlen(backgroundSel)) pdfi->plotOn(ret.back(), RooFit::LineColor(206), RooFit::Components(backgroundSel));
            pdfi->plotOn(ret.back());
            delete ds;
        }
        delete datasets;
    } else if (pdf.canBeExtended()) {
        std::auto_ptr<RooArgSet> obs(pdf.getObservables(&data));
        RooRealVar *x = dynamic_cast<RooRealVar *>(obs->first());
        if (x != 0) {
            ret.push_back(x->frame());
            ret.back()->SetName("data");
            data.plotOn(ret.back());
            if (signalSel && strlen(signalSel))         pdf.plotOn(ret.back(), RooFit::LineColor(209), RooFit::Components(signalSel));
            if (backgroundSel && strlen(backgroundSel)) pdf.plotOn(ret.back(), RooFit::LineColor(206), RooFit::Components(backgroundSel));
            pdf.plotOn(ret.back());
        }
    }
    if (facpdf != &pdf) { delete facpdf; }
    return ret;

}

void utils::CheapValueSnapshot::readFrom(const RooAbsCollection &src) {
    if (&src != src_) {
        src_ = &src;
        values_.resize(src.getSize());
    }
    RooLinkedListIter iter = src.iterator(); int i = 0;
    for (RooAbsArg *a = (RooAbsArg *) iter.Next(); a != 0; a = (RooAbsArg *) iter.Next(), ++i) {
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv == 0) throw std::invalid_argument("Collection to read from contains a non-RooRealVar");
        values_[i] = rrv->getVal();
    }
}

void utils::CheapValueSnapshot::writeTo(const RooAbsCollection &src) const {
    if (&src == src_) {
        RooLinkedListIter iter = src.iterator();  int i = 0;
        for (RooAbsArg *a = (RooAbsArg *) iter.Next(); a != 0; a = (RooAbsArg *) iter.Next(), ++i) {
            RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
            rrv->setVal(values_[i]);
        }
    } else {
        RooLinkedListIter iter = src_->iterator();  int i = 0;
        for (RooAbsArg *a = (RooAbsArg *) iter.Next(); a != 0; a = (RooAbsArg *) iter.Next(), ++i) {
            RooAbsArg *a2 = src.find(a->GetName()); if (a2 == 0) continue;
            RooRealVar *rrv = dynamic_cast<RooRealVar *>(a2);
            rrv->setVal(values_[i]);
        }
    }
}

void utils::CheapValueSnapshot::Print(const char *fmt) const {
    if (src_ == 0) { printf("<NIL>\n"); return; }
    if (fmt[0] == 'V') {
        RooLinkedListIter iter = src_->iterator(); int i = 0;
        for (RooAbsArg *a = (RooAbsArg *) iter.Next(); a != 0; a = (RooAbsArg *) iter.Next(), ++i) {
            printf(" %3d) %-30s = %9.6g\n", i, a->GetName(), values_[i]);
        }
        printf("\n");
    } else {
        src_->Print(fmt);
    }
}

void utils::setPhysicsModelParameters( std::string setPhysicsModelParameterExpression, RooStats::ModelConfig *mc) {

  const RooArgSet * POI = mc->GetParametersOfInterest();
  if (!POI) {
    cout << "setPhysicsModelParameter Warning: ModelConfig " << mc->GetName() << " does not have any parameters of interest. Doing nothing.\n";
    return;
  }
 
  vector<string> SetParameterExpressionList;  
  boost::split(SetParameterExpressionList, setPhysicsModelParameterExpression, boost::is_any_of(","));
  for (UInt_t p = 0; p < SetParameterExpressionList.size(); ++p) {
    vector<string> SetParameterExpression;
    boost::split(SetParameterExpression, SetParameterExpressionList[p], boost::is_any_of("="));
      
    if (SetParameterExpression.size() != 2) {
      std::cout << "Error parsing physics model parameter expression : " << SetParameterExpressionList[p] << endl;
    } else {
      double PhysicsParameterValue = atof(SetParameterExpression[1].c_str());
      RooRealVar *tmpParameter = (RooRealVar*)POI->find(SetParameterExpression[0].c_str());      
      if (tmpParameter) {
        cout << "Set Default Value of Parameter " << SetParameterExpression[0] 
             << " To : " << PhysicsParameterValue << "\n";
        tmpParameter->setVal(PhysicsParameterValue);
      } else {
        std::cout << "Warning: Did not find a parameter with name " << SetParameterExpression[0] << endl;
      }
    }
  }

}

void utils::setPhysicsModelParameterRanges( std::string setPhysicsModelParameterRangeExpression, RooStats::ModelConfig *mc) {

  const RooArgSet * POI = mc->GetParametersOfInterest();
  if (!POI) {
    cout << "setPhysicsModelParameter Warning: ModelConfig " << mc->GetName() << " does not have any parameters of interest. Doing nothing.\n";
    return;
  }
 
  vector<string> SetParameterRangeExpressionList;  
  boost::split(SetParameterRangeExpressionList, setPhysicsModelParameterRangeExpression, boost::is_any_of(":"));
  for (UInt_t p = 0; p < SetParameterRangeExpressionList.size(); ++p) {
    vector<string> SetParameterRangeExpression;
    boost::split(SetParameterRangeExpression, SetParameterRangeExpressionList[p], boost::is_any_of("=,"));
      
    if (SetParameterRangeExpression.size() != 3) {
      std::cout << "Error parsing physics model parameter expression : " << SetParameterRangeExpressionList[p] << endl;
    } else {
      double PhysicsParameterRangeLow = atof(SetParameterRangeExpression[1].c_str());
      double PhysicsParameterRangeHigh = atof(SetParameterRangeExpression[2].c_str());
      RooRealVar *tmpParameter = (RooRealVar*)POI->find(SetParameterRangeExpression[0].c_str());            
      if (tmpParameter) {
        cout << "Set Range of Parameter " << SetParameterRangeExpression[0] 
             << " To : (" << PhysicsParameterRangeLow << "," << PhysicsParameterRangeHigh << ")\n";
        tmpParameter->setRange(PhysicsParameterRangeLow,PhysicsParameterRangeHigh);
      } else {
        std::cout << "Warning: Did not find a parameter with name " << SetParameterRangeExpression[0] << endl;
      }
    }
  }

}
