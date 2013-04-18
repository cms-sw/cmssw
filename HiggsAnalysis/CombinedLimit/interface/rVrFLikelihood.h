#ifndef HiggsAnalysis_CombinedLimit_rVrFLikelihood_h
#define HiggsAnalysis_CombinedLimit_rVrFLikelihood_h

#include <RooAbsReal.h>
#include <RooRealProxy.h>
#include <TH2.h>
#include <vector>

class rVrFLikelihood : public RooAbsReal {

    public:
        rVrFLikelihood() {}
        rVrFLikelihood(const char *name, const char *title) ;
        void addChannel(const TH2* chi2, RooAbsReal &muV, RooAbsReal &muF);
        ~rVrFLikelihood() ;

        TObject * clone(const char *newname) const ;

        // Must be public to get dictionaries to compile properly
        struct Channel { 
            Channel() {}
            Channel(rVrFLikelihood *parent, const TH2* chi2_, RooAbsReal &muV_, RooAbsReal &muF_) :
                chi2(chi2_), 
                muV("muV","signal strength modifier for qqH,VH",  parent, muV_),
                muF("muF","signal strength modifier for ggH,ttH", parent, muF_) {}
            const TH2* chi2;
            RooRealProxy muV, muF;
        };

    protected:
        Double_t evaluate() const;

    private:
        std::vector<Channel> channels_;

        ClassDef(rVrFLikelihood,1) // Asymmetric power	
};

#endif
