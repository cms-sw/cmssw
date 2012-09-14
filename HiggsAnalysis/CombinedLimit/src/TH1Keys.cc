#include "../interface/TH1Keys.h"
#include <RooBinning.h>
#include <RooMsgService.h>

#include <stdexcept>
#include <vector>

TH1Keys::TH1Keys() :
    x_(0),
    dataset_(0),
    underflow_(0.0), overflow_(0.0),
    globalScale_(1.0),
    cache_(0),
    isCacheGood_(false)
{
}

TH1Keys::TH1Keys(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup, TString options, Double_t rho) :
    TH1(name,title,nbinsx,xlow,xup),
    min_(xlow), max_(xup),
    x_(new RooRealVar("x", "x", min_, max_)),
    w_(new RooRealVar("w", "w", 1.0)),
    point_(*x_),
    dataset_(new RooDataSet(name, title, RooArgSet(*x_, *w_), "w")),
    underflow_(0.0), overflow_(0.0),
    globalScale_(1.0),
    options_(options),
    rho_(rho),
    cache_(new TH1F("",title,nbinsx,xlow,xup)),
    isCacheGood_(true)
{
    cache_->SetDirectory(0);
    fDimension = 1;
    x_->setBins(nbinsx);
}

TH1Keys::TH1Keys(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins, TString options, Double_t rho) :
    TH1(name,title,nbinsx,xbins),
    min_(xbins[0]), max_(xbins[nbinsx]),
    x_(new RooRealVar("x", "x", min_, max_)),
    w_(new RooRealVar("w", "w", 1.0)),
    point_(*x_),
    dataset_(new RooDataSet(name, title, RooArgSet(*x_, *w_), "w")),
    underflow_(0.0), overflow_(0.0),
    globalScale_(1.0),
    options_(options),
    rho_(rho),
    cache_(new TH1F("",title,nbinsx,xbins)),
    isCacheGood_(true)
{
    cache_->SetDirectory(0);
    fDimension = 1;
    std::vector<Double_t> boundaries(nbinsx+1);
    for (Int_t i = 0; i <= nbinsx; ++i) boundaries[i] = xbins[i];
    x_->setBinning(RooBinning(nbinsx, &boundaries[0]));
}

TH1Keys::TH1Keys(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins, TString options, Double_t rho) :
    TH1(name,title,nbinsx,xbins),
    min_(xbins[0]), max_(xbins[nbinsx]),
    x_(new RooRealVar("x", "x", min_, max_)),
    w_(new RooRealVar("w", "w", 1.0)),
    point_(*x_),
    dataset_(new RooDataSet(name, title, RooArgSet(*x_, *w_), "w")),
    underflow_(0.0), overflow_(0.0),
    globalScale_(1.0),
    options_(options),
    rho_(rho),
    cache_(new TH1F("",title,nbinsx,xbins)),
    isCacheGood_(true)
{
    cache_->SetDirectory(0);
    fDimension = 1;
    x_->setBinning(RooBinning(nbinsx, xbins));
}


TH1Keys::TH1Keys(const TH1Keys &other)  :
    TH1(other),
    min_(other.min_), max_(other.max_),
    x_(new RooRealVar("x", "x", min_, max_)),
    w_(new RooRealVar("w", "w", 1.0)),
    point_(*x_),
    dataset_(new RooDataSet(other.GetName(), other.GetTitle(), RooArgSet(*x_, *w_), "w")),
    underflow_(other.underflow_), overflow_(other.overflow_),
    globalScale_(other.globalScale_),
    options_(other.options_),
    rho_(other.rho_),
    cache_((TH1*)other.cache_->Clone()),
    isCacheGood_(other.isCacheGood_)
{
    fDimension = 1;
    x_->setBinning(other.x_->getBinning());
}


TH1Keys::~TH1Keys() 
{
    delete cache_;
    delete dataset_;
    delete x_;
}

Int_t TH1Keys::Fill(Double_t x, Double_t w)
{
    isCacheGood_ = false;
    if (x >= max_) overflow_ += w;
    else if (x < min_) underflow_ += w;
    else {
        x_->setVal(x);
        dataset_->add(point_, w);
        return 1;
    } 
    return -1;
}

void TH1Keys::FillN(Int_t ntimes, const Double_t *x, const Double_t *w, Int_t stride)
{
    isCacheGood_ = false;
    for (Int_t i = 0; i < ntimes; i += stride) {
        Fill(x[i], w[i]);
    }
}

#if ROOT_VERSION_CODE <  ROOT_VERSION(5,34,00)
void TH1Keys::Add(const TH1 *h1, Double_t c1) 
#else
Bool_t TH1Keys::Add(const TH1 *h1, Double_t c1) 
#endif
{
    if (c1 != 1.0) dont("Add with constant != 1");
    const TH1Keys *other = dynamic_cast<const TH1Keys *>(h1);
    if (other == 0) dont("Add with a non TH1Keys");
    dataset_->append(const_cast<RooDataSet&>(*other->dataset_));
    isCacheGood_ = false;
#if ROOT_VERSION_CODE >=  ROOT_VERSION(5,34,00)
    return true; 
#endif
}

void TH1Keys::Scale(Double_t c1, Option_t *option)
{
    globalScale_ *= c1;
    if (cache_) cache_->Scale(c1);
}

void TH1Keys::Reset(Option_t *option) {
    dataset_->reset();
    overflow_ = underflow_ = 0.0;
    globalScale_ = 1.0;
    cache_->Reset();
    isCacheGood_ = true;
}

// ------------------------------------------------------------

void TH1Keys::FillH1() const
{
    if (dataset_->numEntries() == 0) {
        cache_->Reset(); // make sure it's empty
    } else {
        RooFit::MsgLevel gKill = RooMsgService::instance().globalKillBelow();
        RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
        delete cache_;
        RooNDKeysPdf pdf("","",*x_,*dataset_,options_,rho_);
        cache_ = pdf.createHistogram(GetName(), *x_);
        if (cache_->Integral()) cache_->Scale(1.0/cache_->Integral());
        cache_->SetBinContent(0,                     underflow_);
        cache_->SetBinContent(cache_->GetNbinsX()+1, overflow_);
        cache_->Scale(dataset_->sumEntries() * globalScale_);
        RooMsgService::instance().setGlobalKillBelow(gKill);
    }
    isCacheGood_ = true;
}

void TH1Keys::dont(const char *msg) const {
    TObject::Error("TH1Keys",msg);
    throw std::runtime_error(std::string("Error in TH1Keys: ")+msg);
}
