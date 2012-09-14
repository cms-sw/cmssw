#ifndef HiggsAnalysis_CombinedLimit_RooScaleLOSM_h
#define HiggsAnalysis_CombinedLimit_RooScaleLOSM_h

#include <cmath>
#include <complex>
#include "TMath.h"

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRealProxy.h"

typedef std::complex<double> complexD;

class RooScaleLOSM: public RooAbsReal
{
  public:
    RooScaleLOSM(){};
    RooScaleLOSM(const char *name, const char *title, RooAbsReal &mH);
    ~RooScaleLOSM(){};

    virtual TObject* clone(const char *newname) const = 0;

  protected:
    complexD f(double tau) const;
    inline complexD AmpSpinOneHalf(double tau) const;
    inline complexD AmpSpinOne(double tau) const;
    virtual Double_t evaluate() const = 0;

    RooRealProxy mH_;
    static const double mt_, mW_;
    complexD At_, Ab_, AW_;

    double C_SM_;

  private:

    ClassDef(RooScaleLOSM,1)
};

class RooScaleHGamGamLOSM: public RooScaleLOSM
{
  public:
    RooScaleHGamGamLOSM(){};
    RooScaleHGamGamLOSM(const char *name, const char *title,
    		RooAbsReal &mH, RooAbsReal &ct, RooAbsReal &cW, RooAbsReal &mb, RooAbsReal &cb);
    ~RooScaleHGamGamLOSM(){};

    TObject* clone(const char *newname) const;

  protected:
      Double_t evaluate() const;
      RooRealProxy ct_, cW_, mb_, cb_;

  private:

    ClassDef(RooScaleHGamGamLOSM,1)
};

class RooScaleHGluGluLOSM: public RooScaleLOSM
{
  public:
    RooScaleHGluGluLOSM(){};
    RooScaleHGluGluLOSM(const char *name, const char *title,
    		RooAbsReal &mH, RooAbsReal &ct, RooAbsReal &mb, RooAbsReal &cb);
    ~RooScaleHGluGluLOSM(){};

    TObject* clone(const char *newname) const;

  protected:
      Double_t evaluate() const;
      RooRealProxy ct_, mb_, cb_;

  private:

    ClassDef(RooScaleHGluGluLOSM,1)
};

class RooScaleHGamGamLOSMPlusX: public RooScaleHGamGamLOSM
{
  public:
    RooScaleHGamGamLOSMPlusX(){};
    RooScaleHGamGamLOSMPlusX(const char *name, const char *title,
    		RooAbsReal &mH, RooAbsReal &ct, RooAbsReal &cW, RooAbsReal &mb, RooAbsReal &cb, RooAbsReal &X);
    ~RooScaleHGamGamLOSMPlusX(){};

    TObject* clone(const char *newname) const;

  protected:
      Double_t evaluate() const;
      RooRealProxy X_;

  private:

    ClassDef(RooScaleHGamGamLOSMPlusX,1)
};

class RooScaleHGluGluLOSMPlusX: public RooScaleHGluGluLOSM
{
  public:
    RooScaleHGluGluLOSMPlusX(){};
    RooScaleHGluGluLOSMPlusX(const char *name, const char *title,
    		RooAbsReal &mH, RooAbsReal &ct, RooAbsReal &mb, RooAbsReal &cb, RooAbsReal &X);
    ~RooScaleHGluGluLOSMPlusX(){};

    TObject* clone(const char *newname) const;

  protected:
      Double_t evaluate() const;
      RooRealProxy X_;

  private:

    ClassDef(RooScaleHGluGluLOSMPlusX,1)
};

#endif
