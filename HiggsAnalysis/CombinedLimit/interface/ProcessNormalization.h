#ifndef HiggsAnalysis_CombinedLimit_ProcessNormalization_h
#define HiggsAnalysis_CombinedLimit_ProcessNormalization_h

#include <TIterator.h>
#include <RooAbsReal.h>
#include "RooListProxy.h"

//_________________________________________________
/*
BEGIN_HTML
<p>
ProcessNormalization is helper class for implementing process normalizations
</p>
END_HTML
*/
//
class ProcessNormalization : public RooAbsReal {
   public:
      ProcessNormalization() : nominalValue_(1) {}
      ProcessNormalization(const char *name, const char *title, double nominal=1) ;
      ProcessNormalization(const char *name, const char *title, RooAbsReal &nominal) ;
      ProcessNormalization(const ProcessNormalization &other, const char *newname = 0) ;
      ~ProcessNormalization() ;

      TObject * clone(const char *newname) const { return new ProcessNormalization(*this, newname); }

      void setNominalValue(double nominal) { nominalValue_ = nominal; }
      void addLogNormal(double kappa, RooAbsReal &theta) ;
      void addAsymmLogNormal(double kappaLo, double kappaHi, RooAbsReal &theta) ;
      void addOtherFactor(RooAbsReal &factor) ;
      void dump() const ;
    protected:
        Double_t evaluate() const;

    private:
        // ---- PERSISTENT ----
        double nominalValue_;                         
        std::vector<double> logKappa_; // Logarithm of symmetric kappas
        RooListProxy thetaList_;        // List of nuisances for symmetric kappas
        std::vector<std::pair<double,double> > logAsymmKappa_; // Logarithm of asymmetric kappas (low, high)
        RooListProxy asymmThetaList_;                           // List of nuisances for asymmetric kappas
        RooListProxy otherFactorList_;     // Other multiplicative terms 

        // get the kappa for the appropriate x
        Double_t logKappaForX(double x, const std::pair<double,double> &logKappas ) const ;

  ClassDef(ProcessNormalization,1) // Process normalization interpolator 
};

#endif
