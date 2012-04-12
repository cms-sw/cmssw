#ifndef HiggsAnalysis_CombinedLimit_RooSpline1D_h
#define HiggsAnalysis_CombinedLimit_RooSpline1D_h

#include <RooAbsReal.h>
#include <RooRealProxy.h>
#include <Math/Interpolator.h>

//_________________________________________________
/*
BEGIN_HTML
<p>
RooSpline1D is helper class for smoothing a list of values using an interpolator provided by the ROOT::Math::Interpolator class. 
For the interpolation types, see http://project-mathlibs.web.cern.ch/project-mathlibs/sw/html/group__Interpolation.html
</p>
END_HTML
*/
//
class RooSpline1D : public RooAbsReal {

   public:
      RooSpline1D() {}
      RooSpline1D(const char *name, const char *title, RooAbsReal &xvar, unsigned int npoints, const double *xvals, const double *yvals, const char *algo="CSPLINE") ;
      RooSpline1D(const char *name, const char *title, RooAbsReal &xar, unsigned int npoints, const float *xvals, const float *yvals, const char *algo="CSPLINE") ;
      ~RooSpline1D() ;

      TObject * clone(const char *newname) const ;

    protected:
        Double_t evaluate() const;

    private:
        RooRealProxy xvar_;
        std::vector<double> x_, y_;
        std::string type_;

        mutable ROOT::Math::Interpolator *interp_; //! not to be serialized        
        void init() const ;

  ClassDef(RooSpline1D,1) // Smooth interpolation	
};

#endif
