#include "../interface/RooScaleLOSM.h"

const double RooScaleLOSM::mt_ = 172.50;
const double RooScaleLOSM::mW_ = 80.398;

RooScaleLOSM::RooScaleLOSM(const char *name, const char *title, RooAbsReal &mH):
  RooAbsReal(name, title),
  mH_("mH","Higgs boson mass [GeV]",this,mH)
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
  return 2. * (tau + (tau-1.)*f(tau)) / (tau*tau);
}

inline complexD RooScaleLOSM::AmpSpinOne(double tau) const
{
  return -1. * (2.*tau*tau + 3.*tau + 3.*(2.*tau-1.)*f(tau)) / (tau*tau);
}


ClassImp(RooScaleLOSM)

////////////////////////////////////////////////////////////////////
RooScaleHGamGamLOSM::RooScaleHGamGamLOSM(const char *name, const char *title,
		RooAbsReal &mH,
		RooAbsReal &ct, RooAbsReal &cW, RooAbsReal &mb, RooAbsReal &cb):
  RooScaleLOSM(name, title, mH),
  ct_("ct", "Top Quark coupling constant", this, ct),
  cW_("cW", "W Boson coupling constant", this, cW),
  mb_("mb", "(Running) Bottom Quark mass [GeV]",this,mb),
  cb_("cb", "Bottom Quark coupling constant", this, cb)
{
	At_ = (4./3.) * AmpSpinOneHalf( (mH_*mH_)/(4*mt_*mt_) ); // Nc = 3, Qt^2 = 4/9  => 4/3
	Ab_ = (1./3.) * AmpSpinOneHalf( (mH_*mH_)/(4*mb_*mb_) ); // Nc = 3, Qb^2 = 1/9  => 1/3
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
	const double ct = ct_, cb = cb_, cW = cW_;

	const double C_deviated = norm(ct*At_ + cb*Ab_ + cW*AW_);

	return C_deviated/C_SM_;
}

ClassImp(RooScaleHGamGamLOSM)


////////////////////////////////////////////////////////////////////
RooScaleHGluGluLOSM::RooScaleHGluGluLOSM(const char *name, const char *title,
		RooAbsReal &mH,
		RooAbsReal &ct, RooAbsReal &mb, RooAbsReal &cb):
  RooScaleLOSM(name, title, mH),
  ct_("ct", "Top Quark coupling constant", this, ct),
  mb_("mb", "(Running) Bottom Quark mass [GeV]",this,mb),
  cb_("cb", "Bottom Quark coupling constant", this, cb)
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
	const double ct = ct_, cb = cb_;

	const double C_deviated = norm(ct*At_ + cb*Ab_);

	return C_deviated/C_SM_;
}

ClassImp(RooScaleHGluGluLOSM)


////////////////////////////////////////////////////////////////////
RooScaleHGamGamLOSMPlusX::RooScaleHGamGamLOSMPlusX(const char *name, const char *title,
		RooAbsReal &mH,
		RooAbsReal &ct, RooAbsReal &cW, RooAbsReal &mb, RooAbsReal &cb,
		RooAbsReal &X):
	RooScaleHGamGamLOSM(name, title, mH, ct, cW, mb, cb),
	X_("X","Extra amplitude in the photon loop",this,X)
{
}

TObject* RooScaleHGamGamLOSMPlusX::clone(const char *newname) const
{
  return new RooScaleHGamGamLOSMPlusX(newname, this->GetTitle(),
    const_cast<RooAbsReal &>(mH_.arg()),
    const_cast<RooAbsReal &>(ct_.arg()),
    const_cast<RooAbsReal &>(cW_.arg()),
    const_cast<RooAbsReal &>(mb_.arg()),
    const_cast<RooAbsReal &>(cb_.arg()),
    const_cast<RooAbsReal &>(X_.arg())
  );
}


Double_t RooScaleHGamGamLOSMPlusX::evaluate() const
{
	const double ct = ct_, cb = cb_, cW = cW_, X =  X_;

	const double C_deviated = norm(ct*At_ + cb*Ab_ + cW*AW_ + X);

	return C_deviated/C_SM_;
}

ClassImp(RooScaleHGamGamLOSMPlusX)


////////////////////////////////////////////////////////////////////
RooScaleHGluGluLOSMPlusX::RooScaleHGluGluLOSMPlusX(const char *name, const char *title,
		RooAbsReal &mH,
		RooAbsReal &ct, RooAbsReal &mb, RooAbsReal &cb,
		RooAbsReal &X):
  RooScaleHGluGluLOSM(name, title, mH, ct, mb, cb),
  X_("X","Extra amplitude in the gluon loop",this,X)
{
}

TObject* RooScaleHGluGluLOSMPlusX::clone(const char *newname) const
{
  return new RooScaleHGluGluLOSMPlusX(newname, this->GetTitle(),
    const_cast<RooAbsReal &>(mH_.arg()),
    const_cast<RooAbsReal &>(ct_.arg()),
    const_cast<RooAbsReal &>(mb_.arg()),
    const_cast<RooAbsReal &>(cb_.arg()),
    const_cast<RooAbsReal &>(X_.arg())
  );
}


Double_t RooScaleHGluGluLOSMPlusX::evaluate() const
{
	const double ct = ct_, cb = cb_, X =  X_;

	const double C_deviated = norm(ct*At_ + cb*Ab_ + X);

	return C_deviated/C_SM_;
}

ClassImp(RooScaleHGluGluLOSMPlusX)



