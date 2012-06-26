#include "../interface/RooScaleLOSM.h"

const double RooScaleLOSM::mt_ = 172.50;
const double RooScaleLOSM::mW_ = 80.398;

RooScaleLOSM::RooScaleLOSM(const char *name, const char *title, RooAbsReal &mH):
  RooAbsReal(name, title),
  mH_("mH","",this,mH)
{

}

complexD RooScaleLOSM::f(double tau) const
{
  if(tau <= 1)
  {
    double temp = std::asin(std::sqrt(tau));
    return complexD(temp*temp, 0);
  }
  else
  {
    double temp = std::sqrt(1-1/tau);

    complexD retVal(std::log((1+temp)/(1-temp)), -TMath::Pi());
    retVal *= retVal;
    retVal *= -1.0/4.0;

    return retVal;
  }
}

inline complexD RooScaleLOSM::AmpSpinOneHalf(double tau) const
{
  return (tau + (tau-1.0)*f(tau))/(tau*tau)*2.0;
}

inline complexD RooScaleLOSM::AmpSpinOne(double tau) const
{
  return -1.0*(2.0*tau*tau + 3.0*tau + 3.0*(2.0*tau-1.0)*f(tau))/(tau*tau);
}


ClassImp(RooScaleLOSM)

////////////////////////////////////////////////////////////////////

RooScaleHGamGamLOSM::RooScaleHGamGamLOSM(const char *name, const char *title, RooAbsReal &mH, RooAbsReal &ct, RooAbsReal &cW, RooAbsReal &mb, RooAbsReal &cb):
  RooScaleLOSM(name, title, mH),
  ct_("ct", "Top Quark coupling constant", this, ct),
  cb_("cb", "Bottom Quark coupling constant", this, cb),
  cW_("cW", "W Boson coupling constant", this, cW),
  mb_("mb","",this,mb)
{
	At_ = 3.* (4./9.) * AmpSpinOneHalf( (mH_*mH_)/(4*mt_*mt_) );
	Ab_ = 3.* (1./9.) * AmpSpinOneHalf( (mH_*mH_)/(4*mb_*mb_) );
	AW_ = AmpSpinOne( (mH_*mH_)/(4*mW_*mW_) );
	C_SM_ = norm(At_ + Ab_ + AW_);
}

TObject* RooScaleHGamGamLOSM::clone(const char *newname) const
{
  return new RooScaleHGamGamLOSM(newname, this->GetTitle(),
    const_cast<RooAbsReal &>(mH_.arg()),
    const_cast<RooAbsReal &>(ct_.arg()),
    const_cast<RooAbsReal &>(cW_.arg()),
    const_cast<RooAbsReal &>(mb_.arg()),
    const_cast<RooAbsReal &>(cb_.arg())
  );
}


Double_t RooScaleHGamGamLOSM::evaluate() const
{
	const double ct = ct_;
	const double cb = cb_;
	const double cW = cW_;

	const double C_deviated = norm(ct*At_ + cb*Ab_ + cW*AW_);

	return C_deviated/C_SM_;
}


ClassImp(RooScaleHGamGamLOSM)


////////////////////////////////////////////////////////////////////

RooScaleHGluGluLOSM::RooScaleHGluGluLOSM(const char *name, const char *title, RooAbsReal &mH, RooAbsReal &ct, RooAbsReal &mb, RooAbsReal &cb):
  RooScaleLOSM(name, title, mH),
  ct_("ct", "Top Quark coupling constant", this, ct),
  cb_("cb", "Bottom Quark coupling constant", this, cb),
  mb_("mb","",this,mb)
{
	At_ = AmpSpinOneHalf( (mH_*mH_)/(4*mt_*mt_) );
	Ab_ = AmpSpinOneHalf( (mH_*mH_)/(4*mb_*mb_) );
	C_SM_ = norm(At_ + Ab_);
}

TObject* RooScaleHGluGluLOSM::clone(const char *newname) const
{
  return new RooScaleHGluGluLOSM(newname, this->GetTitle(),
    const_cast<RooAbsReal &>(mH_.arg()),
    const_cast<RooAbsReal &>(ct_.arg()),
    const_cast<RooAbsReal &>(mb_.arg()),
    const_cast<RooAbsReal &>(cb_.arg())
  );
}


Double_t RooScaleHGluGluLOSM::evaluate() const
{
	const double ct = ct_;
	const double cb = cb_;

	const double C_deviated = norm(ct*At_ + cb*Ab_);

	return C_deviated/C_SM_;
}


ClassImp(RooScaleHGluGluLOSM)
