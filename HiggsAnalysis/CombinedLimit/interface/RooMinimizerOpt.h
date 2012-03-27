#ifndef HiggsAnalysis_CombinedLimit_RooMinimizerOpt
#define HiggsAnalysis_CombinedLimit_RooMinimizerOpt

#if defined(ROO_MINIMIZER) || defined(ROO_MINIMIZER_FCN)
   #error "You cannot include RooMinimizer.h or RooMinimizerFcn.h before RooMinimizerOpt.h"
#else
   #define private protected
   #include <RooMinimizer.h>
   #undef protected
#endif

class RooMinimizerOpt : public RooMinimizer {
    public:
        RooMinimizerOpt(RooAbsReal& function) ;
        Double_t edm();
};

class RooMinimizerFcnOpt : public RooMinimizerFcn {
    public: 
        RooMinimizerFcnOpt(RooAbsReal *funct, RooMinimizer *context,  bool verbose = false);
        virtual ROOT::Math::IBaseFunctionMultiDim* Clone() const;
    protected:
        virtual double DoEval(const double * x) const;
        mutable std::vector<RooRealVar *> _vars;
};

#endif
