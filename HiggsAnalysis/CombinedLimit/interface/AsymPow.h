#ifndef HiggsAnalysis_CombinedLimit_AsymPow_h
#define HiggsAnalysis_CombinedLimit_AsymPow_h

#include <RooAbsReal.h>
#include <RooRealProxy.h>


//_________________________________________________
/*
BEGIN_HTML
<p>
AsymPow is helper class for implementing asymmetric log-normal errors. 
It has two parameters <i>kappa<sub>Low</sub></i>, <i>kappa<sub>High</sub></i> and one variable (<i>theta</i>).
<ul>
<li>for <i>theta &gt; 0</i>, it evaluates to <b>pow</b>(<i>kappa<sub>High</sub></i>, <i>theta</i>). </li>
<li>for <i>theta &lt; 0</i>, it evaluates to <b>pow</b>(<i>kappa<sub>Low</sub></i>, &minus;<i>theta</i>). </li>
</ul>
</p>
END_HTML
*/
//
class AsymPow : public RooAbsReal {

   public:
      AsymPow() {}
      AsymPow(const char *name, const char *title, RooAbsReal &kappaLow, RooAbsReal &kappaHigh, RooAbsReal &theta) ;
      ~AsymPow() ;

      TObject * clone(const char *newname) const ;

    protected:
        Double_t evaluate() const;

    private:
        RooRealProxy kappaLow_, kappaHigh_;
        RooRealProxy theta_;

  ClassDef(AsymPow,1) // Asymmetric power	
};

#endif
