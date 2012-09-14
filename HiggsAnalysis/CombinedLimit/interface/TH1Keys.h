#ifndef TH1Keys_h
#define TH1Keys_h

#include <TH1.h>
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooDataSet.h>
#include <RooNDKeysPdf.h>

class TH1Keys : public TH1 {
    public:
       TH1Keys();
       TH1Keys(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup, TString options = "a", Double_t rho = 1.5);
       TH1Keys(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins, TString options = "a", Double_t rho = 1.5) ;
       TH1Keys(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins, TString options = "a", Double_t rho = 1.5) ;
       TH1Keys(const TH1Keys &other);
       virtual ~TH1Keys();

       TH1 * GetHisto() { if (!isCacheGood_) FillH1(); return cache_; }
       const TH1 * GetHisto() const { if (!isCacheGood_) FillH1(); return cache_; }

       virtual Int_t    Fill(Double_t x) { return Fill(x,1.0); }
       virtual Int_t    Fill(Double_t x, Double_t w);
       virtual void     FillN(Int_t ntimes, const Double_t *x, const Double_t *w, Int_t stride=1);
#if ROOT_VERSION_CODE <  ROOT_VERSION(5,34,00)
       virtual void     Add(const TH1 *h1, Double_t c1=1);
       virtual void     Add(const TH1 *h, const TH1 *h2, Double_t c1=1, Double_t c2=1) { dont("Add with two arguments"); } 
#else
       virtual Bool_t   Add(const TH1 *h1, Double_t c1=1);
       virtual Bool_t   Add(const TH1 *h, const TH1 *h2, Double_t c1=1, Double_t c2=1) { dont("Add with two arguments"); return false;} 
#endif
       virtual void     AddBinContent(Int_t bin) { AddBinContent(bin, 1.0); }
       virtual void     AddBinContent(Int_t bin, Double_t w) { dont("AddBinContent"); }
       virtual void     Copy(TObject &hnew) const { dont("Copy"); }

       virtual TH1     *DrawCopy(Option_t *option="") const { dont("DrawCopy"); return 0; }

       virtual Double_t GetBinContent(Int_t bin) const { return GetHisto()->GetBinContent(bin); }
       virtual Double_t GetBinContent(Int_t bin, Int_t) const { return GetHisto()->GetBinContent(bin); }
       virtual Double_t GetBinContent(Int_t bin, Int_t, Int_t) const {return GetHisto()->GetBinContent(bin); }

       virtual Double_t GetEntries() const { return dataset_->numEntries(); }

       virtual void     Reset(Option_t *option="") ; 
       virtual void     SetBinContent(Int_t bin, Double_t content) { dont("SetBinContent"); }
       virtual void     SetBinContent(Int_t bin, Int_t, Double_t content)        { SetBinContent(bin,content); }
       virtual void     SetBinContent(Int_t bin, Int_t, Int_t, Double_t content) { SetBinContent(bin,content); }
       virtual void     SetBinsLength(Int_t n=-1) { dont("SetBinLength"); }
       virtual void     Scale(Double_t c1=1, Option_t *option="");

       ClassDef(TH1Keys,1)  //

    private:
        Double_t    min_, max_;
        RooRealVar *x_, *w_;
        RooArgSet   point_;
        RooDataSet *dataset_;
        Double_t    underflow_, overflow_;
        Double_t    globalScale_;

	TString          options_;
        Double_t           rho_;

        mutable TH1 *cache_;
        mutable bool isCacheGood_;

        void FillH1() const;

        void dont(const char *) const ;
}; // class

#endif
