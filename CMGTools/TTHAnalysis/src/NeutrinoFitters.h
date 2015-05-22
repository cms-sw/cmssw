#ifndef NeutrinoFitters_h
#define NeutrinoFitters_h

#include <TLorentzVector.h>
#include <Minuit2/Minuit2Minimizer.h>
#include <Math/Functor.h>

namespace cmg {

class BaseNeutrinoFitter {
    public:
        BaseNeutrinoFitter() ;
        virtual ~BaseNeutrinoFitter() {}
        void initLep(const TLorentzVector &lp, const TLorentzVector &lm) ;
        void initMET(double met, double metphi, double htJet25) ;
        const TLorentzVector & lp()    const { return lp_;    }
        const TLorentzVector & lm()    const { return lm_;    }
        const TLorentzVector & nu()    const { return nu_;    }
        const TLorentzVector & nubar() const { return nubar_; }
        const TLorentzVector & wp()    const { return wp_;    }
        const TLorentzVector & wm()    const { return wm_;    }
       
        void fit() ; 
        virtual double nll(double costheta, double phi, double costhetabar, double phibar) ; 

        // Functor interface for Minuit
        double eval(const double *x) { return nll(x[0],x[1],x[2],x[3]); }
    protected:
        TLorentzVector lp_, lm_;
        TLorentzVector wp_, wm_;
        TLorentzVector nu_, nubar_;
        double metx_, mety_, metsig2_;

        ROOT::Minuit2::Minuit2Minimizer minimizer_;
        ROOT::Math::Functor functor_;

        double metNll() const ;
        void setWlnu(TLorentzVector &w, TLorentzVector &nu, const TLorentzVector &l, double costheta, double phi);
};

class TwoMTopNeutrinoFitter : public BaseNeutrinoFitter {
    public:
        TwoMTopNeutrinoFitter(double mTop=172.5) : mTopOffset_(mTop-172.5) {}
        ~TwoMTopNeutrinoFitter() {}
        void initBJets(const TLorentzVector &b, const TLorentzVector &bbar) { b_ = b; bbar_ = bbar; }
        const TLorentzVector & b()    const { return b_;    }
        const TLorentzVector & bbar() const { return bbar_; }
        const TLorentzVector & t()    const { return t_;    }
        const TLorentzVector & tbar() const { return tbar_; }
        virtual double nll(double costheta, double phi, double costhetabar, double phibar) ;
    protected:
        TLorentzVector b_, bbar_, t_, tbar_;
        double mTopOffset_;
        double topMassNll(double mass) const ; 
};

} // namespace


#endif
