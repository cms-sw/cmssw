#ifndef GlobalFitParameters_h
#define GlobalFitParameters_h

// interface class for the Parametrization classes
/// -----------------------------------------------------------------
class TMeasurement
{
public:
  TMeasurement():
    pt  ( 0.),
    EMF ( 0.),
    HadF( 0.),
    OutF( 0.),
    E   ( 0.),
    eta ( 0.),
    phi ( 0.){};
  TMeasurement(TMeasurement* m):
    pt  (m->pt  ),
    EMF (m->EMF ),
    HadF(m->HadF),
    OutF(m->OutF),
    E   (m->E   ),
    eta (m->eta ),
    phi (m->phi ){};

  //all common variables
  double pt;
  double EMF;
  double HadF;
  double OutF;
  double E;
  double eta;//necessary???
  double phi;//necessary???
};

// keeps different parametrizations for the 
// underlying hypothesis of the GlobalFit
/// -----------------------------------------------------------------
class Parametrization 
{
public:
  Parametrization(unsigned int ntowerpars, unsigned int njetpars) : 
    ntowerpars_(ntowerpars), njetpars_(njetpars) {}
  virtual ~Parametrization() {}
  // ----------------------------------------------------------------
  //  correctedTowerEt(TMeasurement *const x, double *const par)
  //  returns the corrected Et of a tower 
  //  input: x->pt   : et  of whole tower
  //         x->EMF  : et  of ECAL  part
  //         x->HadF : et  of HCAL  part
  //         x->OutF : et  of Outer part
  //         x->E    : en  of Outer part
  //         x->eta  : eta of tower
  //         x->phi  : phi of tower
  //  par  : the correction parameters of this tower
  // ----------------------------------------------------------------
  virtual double correctedTowerEt(TMeasurement *const x,double *const par) const = 0;

  // ----------------------------------------------------------------
  //  correctedJetEt(TMeasurement *const x,double *const par)
  //  returns the corrected Et of a jet 
  //  input: x->pt   : et  of uncorrected jet
  //         x->EMF  : et  of ECAL  part
  //         x->HadF : et  of HCAL  part
  //         x->OutF : et  of Outer part
  //         x->E    : en  of Outer part
  //         x->eta  : eta of tower
  //         x->phi  : phi of tower
  //  par:  the correction parameters of this jet
  // ----------------------------------------------------------------
  virtual double correctedJetEt(TMeasurement *const x,double *const par) const = 0;
  virtual const char * name() const = 0;

  unsigned int nTowerPars() const { return ntowerpars_;}
  unsigned int nJetPars() const { return njetpars_;}

private: 
  Parametrization();
  unsigned int  ntowerpars_, njetpars_;
};

// parametrization of the hadronic response 
// by a step function in et
/// -----------------------------------------------------------------
class StepParametrization : public Parametrization { 
public:
  StepParametrization() : Parametrization(12,0) {}
  const char* name() const { return "StepParametrization";}
  
  double correctedTowerEt(TMeasurement *const x,double *par) const {
    double result = 0;
    
    if(x->HadF>=0.0  && x->HadF<=1.0)  result = x->EMF+x->OutF + par[0]*x->HadF; 
    else if (x->HadF>   1.0 && x->HadF<=   2.0) result = x->EMF+x->OutF+par[ 1]*x->HadF;
    else if (x->HadF>   2.0 && x->HadF<=   5.0) result = x->EMF+x->OutF+par[ 2]*x->HadF;
    else if (x->HadF>   5.0 && x->HadF<=  10.0) result = x->EMF+x->OutF+par[ 3]*x->HadF;
    else if (x->HadF>  10.0 && x->HadF<=  20.0) result = x->EMF+x->OutF+par[ 4]*x->HadF;
    else if (x->HadF>  20.0 && x->HadF<=  40.0) result = x->EMF+x->OutF+par[ 5]*x->HadF;
    else if (x->HadF>  40.0 && x->HadF<=  80.0) result = x->EMF+x->OutF+par[ 6]*x->HadF;
    else if (x->HadF>  80.0 && x->HadF<= 160.0) result = x->EMF+x->OutF+par[ 7]*x->HadF;
    else if (x->HadF> 160.0 && x->HadF<= 300.0) result = x->EMF+x->OutF+par[ 8]*x->HadF;
    else if (x->HadF> 300.0 && x->HadF<= 600.0) result = x->EMF+x->OutF+par[ 9]*x->HadF;
    else if (x->HadF> 600.0 && x->HadF<=1000.0) result = x->EMF+x->OutF+par[10]*x->HadF;
    else if (x->HadF>1000.0 )                   result = x->EMF+x->OutF+par[11]*x->HadF;
    return result;
  }
    
  double correctedJetEt(TMeasurement *const x,double *const par) const {
    return  x->pt;   
    //OutOfCone, Dominant, parametrized in Et since cone R lorenz invariant
    //return x->pt * ( 1. + 0.295 * par[0] * exp(- 0.02566 * par[1] * x->pt)); 
    /*
      double result = 0;
      if(x->pt>=0.0  && x->pt<=5.0)          result =  par[0]*x->pt + par[1];
      else if (x->pt>5.0   && x->pt<=20.0)   result =  par[2]*x->pt + par[3];
      else if (x->pt>20.0  && x->pt<=80.0)   result =  par[4]*x->pt + par[5];
      else if (x->pt>80.0 )                 result =  par[6]*x->pt + par[7];
      return result;
    */
  }
};

// parametrization of hadronic response 
// by a step function in en
/// -----------------------------------------------------------------
class StepParametrizationEnergy : public Parametrization { 
public:
  StepParametrizationEnergy() : Parametrization(12,2) {}
  const char* name() const { return "StepParametrizationEnergy";}
  
  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    double result = 0;
    // reweight from et to en
    double e =  x->HadF * x->E / x->pt;
    
    if(e>=0.0  && e<=1.0)  result = x->EMF+x->OutF + par[0]*x->HadF;
    else if (e>   1.0 && e<=   2.0) result = x->EMF+x->OutF+par[ 1]*x->HadF;
    else if (e>   2.0 && e<=   5.0) result = x->EMF+x->OutF+par[ 2]*x->HadF;
    else if (e>   5.0 && e<=  10.0) result = x->EMF+x->OutF+par[ 3]*x->HadF;
    else if (e>  10.0 && e<=  20.0) result = x->EMF+x->OutF+par[ 4]*x->HadF;
    else if (e>  20.0 && e<=  40.0) result = x->EMF+x->OutF+par[ 5]*x->HadF;
    else if (e>  40.0 && e<=  80.0) result = x->EMF+x->OutF+par[ 6]*x->HadF;
    else if (e>  80.0 && e<= 160.0) result = x->EMF+x->OutF+par[ 7]*x->HadF;
    else if (e> 160.0 && e<= 300.0) result = x->EMF+x->OutF+par[ 8]*x->HadF;
    else if (e> 300.0 && e<= 600.0) result = x->EMF+x->OutF+par[ 9]*x->HadF;
    else if (e> 600.0 && e<=1000.0) result = x->EMF+x->OutF+par[10]*x->HadF;
    else if (e>1000.0 )             result = x->EMF+x->OutF+par[11]*x->HadF;
    
    return result;
  }
    
  double correctedJetEt(TMeasurement *const x,double *const par) const {
    //OutOfCone, Dominant, parametrized in Et since cone R lorenz invariant
    return x->pt * ( 1. + 0.295 * par[0] * exp(- 0.02566 * par[1] * x->pt));   
  }
};

// parametrization of hadronic response by a step 
// function with 3 sets of parametrizations for 
// different em fractions
/// -----------------------------------------------------------------
class StepEfracParametrization : public Parametrization {
public:
  StepEfracParametrization() : Parametrization(36,0) {}  //(36,2) {}
  const char* name() const { return "StepEfracParametrization";}

  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    double result=0;
    
    double Efrac = x->EMF/(x->HadF+x->OutF+x->EMF);
    if( Efrac < 0.2 ) {
      if(x->HadF>=0.0 && x->HadF<=1.0)            result = x->EMF+x->OutF+par[ 0]*x->HadF;
      else if (x->HadF>   1.0 && x->HadF<=   2.0) result = x->EMF+x->OutF+par[ 1]*x->HadF;
      else if (x->HadF>   2.0 && x->HadF<=   5.0) result = x->EMF+x->OutF+par[ 2]*x->HadF;
      else if (x->HadF>   5.0 && x->HadF<=  10.0) result = x->EMF+x->OutF+par[ 3]*x->HadF;
      else if (x->HadF>  10.0 && x->HadF<=  20.0) result = x->EMF+x->OutF+par[ 4]*x->HadF;
      else if (x->HadF>  20.0 && x->HadF<=  40.0) result = x->EMF+x->OutF+par[ 5]*x->HadF;
      else if (x->HadF>  40.0 && x->HadF<=  80.0) result = x->EMF+x->OutF+par[ 6]*x->HadF;
      else if (x->HadF>  80.0 && x->HadF<= 160.0) result = x->EMF+x->OutF+par[ 7]*x->HadF;
      else if (x->HadF> 160.0 && x->HadF<= 300.0) result = x->EMF+x->OutF+par[ 8]*x->HadF;
      else if (x->HadF> 300.0 && x->HadF<= 600.0) result = x->EMF+x->OutF+par[ 9]*x->HadF;
      else if (x->HadF> 600.0 && x->HadF<=1000.0) result = x->EMF+x->OutF+par[10]*x->HadF;
      else if (x->HadF>1000.0 )                   result = x->EMF+x->OutF+par[11]*x->HadF;
    } else if (Efrac < 0.5) {
      if(x->HadF>=0.0 && x->HadF<=1.0)            result = x->EMF+x->OutF+par[12]*x->HadF;
      else if (x->HadF>   1.0 && x->HadF<=   2.0) result = x->EMF+x->OutF+par[13]*x->HadF;
      else if (x->HadF>   2.0 && x->HadF<=   5.0) result = x->EMF+x->OutF+par[14]*x->HadF;
      else if (x->HadF>   5.0 && x->HadF<=  10.0) result = x->EMF+x->OutF+par[15]*x->HadF;
      else if (x->HadF>  10.0 && x->HadF<=  20.0) result = x->EMF+x->OutF+par[16]*x->HadF;
      else if (x->HadF>  20.0 && x->HadF<=  40.0) result = x->EMF+x->OutF+par[17]*x->HadF;
      else if (x->HadF>  40.0 && x->HadF<=  80.0) result = x->EMF+x->OutF+par[18]*x->HadF;
      else if (x->HadF>  80.0 && x->HadF<= 160.0) result = x->EMF+x->OutF+par[19]*x->HadF;
      else if (x->HadF> 160.0 && x->HadF<= 300.0) result = x->EMF+x->OutF+par[20]*x->HadF;
      else if (x->HadF> 300.0 && x->HadF<= 600.0) result = x->EMF+x->OutF+par[21]*x->HadF;
      else if (x->HadF> 600.0 && x->HadF<=1000.0) result = x->EMF+x->OutF+par[22]*x->HadF;
      else if (x->HadF>1000.0 )                   result = x->EMF+x->OutF+par[23]*x->HadF;
    } else {
      if(x->HadF>=0.0 && x->HadF<=1.0)            result = x->EMF+x->OutF+par[24]*x->HadF;
      else if (x->HadF>   1.0 && x->HadF<=   2.0) result = x->EMF+x->OutF+par[25]*x->HadF;
      else if (x->HadF>   2.0 && x->HadF<=   5.0) result = x->EMF+x->OutF+par[26]*x->HadF;
      else if (x->HadF>   5.0 && x->HadF<=  10.0) result = x->EMF+x->OutF+par[27]*x->HadF;
      else if (x->HadF>  10.0 && x->HadF<=  20.0) result = x->EMF+x->OutF+par[28]*x->HadF;
      else if (x->HadF>  20.0 && x->HadF<=  40.0) result = x->EMF+x->OutF+par[29]*x->HadF;
      else if (x->HadF>  40.0 && x->HadF<=  80.0) result = x->EMF+x->OutF+par[30]*x->HadF;
      else if (x->HadF>  80.0 && x->HadF<= 160.0) result = x->EMF+x->OutF+par[31]*x->HadF;
      else if (x->HadF> 160.0 && x->HadF<= 300.0) result = x->EMF+x->OutF+par[32]*x->HadF;
      else if (x->HadF> 300.0 && x->HadF<= 600.0) result = x->EMF+x->OutF+par[33]*x->HadF;
      else if (x->HadF> 600.0 && x->HadF<=1000.0) result = x->EMF+x->OutF+par[34]*x->HadF;
      else if (x->HadF>1000.0 )                   result = x->EMF+x->OutF+par[35]*x->HadF;
    }
    return result;
  }
  
  double correctedJetEt(TMeasurement *const x,double *const par) const {
    return  x->pt;   
    //OutOfCone, Dominant, parametrized in Et since cone R lorenz invariant
    //return x->pt * ( 1. + 0.295 * par[0] * exp(- 0.02566 * par[1] * x->pt));   
  }
};

// parametrization of hadronic response by a step 
// function with 3 sets of parametrizations for 
// different em fractions; correction only on jets
/// -----------------------------------------------------------------
class StepJetParametrization : public Parametrization { 
public:
  StepJetParametrization() : Parametrization(0,65) {}
  const char* name() const { return "StepJetParametrization";}

  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    return x->pt;
  }
    
  double correctedJetEt(TMeasurement *const x,double *const par) const {
    double pt     = x->pt;
    double Efrac  = x->EMF/( x->EMF+x->HadF+x->OutF);
    double result = 0.;
    if(Efrac < 0.2) {
      if(pt>=0.0 && pt<=10.0)         result = x->EMF+x->OutF+par[ 0]*x->HadF;
      else if (pt> 10.0 && pt<= 20.0) result = x->EMF+x->OutF+par[ 1]*x->HadF;
      else if (pt> 20.0 && pt<= 30.0) result = x->EMF+x->OutF+par[ 2]*x->HadF;
      else if (pt> 30.0 && pt<= 40.0) result = x->EMF+x->OutF+par[ 3]*x->HadF;
      else if (pt> 40.0 && pt<= 60.0) result = x->EMF+x->OutF+par[ 4]*x->HadF;
      else if (pt> 60.0 && pt<= 80.0) result = x->EMF+x->OutF+par[ 5]*x->HadF;
      else if (pt> 80.0 && pt<=100.0) result = x->EMF+x->OutF+par[ 6]*x->HadF;
      else if (pt>100.0 && pt<=120.0) result = x->EMF+x->OutF+par[ 7]*x->HadF;
      else if (pt>120.0 && pt<=140.0) result = x->EMF+x->OutF+par[ 8]*x->HadF;
      else if (pt>140.0 && pt<=160.0) result = x->EMF+x->OutF+par[ 9]*x->HadF;
      else if (pt>160.0 && pt<=180.0) result = x->EMF+x->OutF+par[10]*x->HadF;
      else if (pt>180.0 && pt<=200.0) result = x->EMF+x->OutF+par[11]*x->HadF;
      else if (pt>200.0 && pt<=225.0) result = x->EMF+x->OutF+par[12]*x->HadF;
      else if (pt>225.0 && pt<=250.0) result = x->EMF+x->OutF+par[13]*x->HadF;
      else if (pt>250.0 && pt<=275.0) result = x->EMF+x->OutF+par[14]*x->HadF;
      else if (pt>275.0 && pt<=300.0) result = x->EMF+x->OutF+par[15]*x->HadF;
      else if (pt>300.0 && pt<=350.0) result = x->EMF+x->OutF+par[16]*x->HadF;
      else if (pt>350.0 && pt<=400.0) result = x->EMF+x->OutF+par[17]*x->HadF;
      else if (pt>400.0 && pt<=500.0) result = x->EMF+x->OutF+par[18]*x->HadF;
      else if (pt>500.0 && pt<=700.0) result = x->EMF+x->OutF+par[19]*x->HadF;
      else if (pt>700.0 )             result = x->EMF+x->OutF+par[20]*x->HadF;
    } else if (Efrac < 0.5) {
      if(pt>=0.0 && pt<=10.0)         result = x->EMF+x->OutF+par[21]*x->HadF;
      else if (pt> 10.0 && pt<= 20.0) result = x->EMF+x->OutF+par[22]*x->HadF;
      else if (pt> 20.0 && pt<= 30.0) result = x->EMF+x->OutF+par[23]*x->HadF;
      else if (pt> 30.0 && pt<= 40.0) result = x->EMF+x->OutF+par[24]*x->HadF;
      else if (pt> 40.0 && pt<= 60.0) result = x->EMF+x->OutF+par[25]*x->HadF;
      else if (pt> 60.0 && pt<= 80.0) result = x->EMF+x->OutF+par[26]*x->HadF;
      else if (pt> 80.0 && pt<=100.0) result = x->EMF+x->OutF+par[27]*x->HadF;
      else if (pt>100.0 && pt<=120.0) result = x->EMF+x->OutF+par[28]*x->HadF;
      else if (pt>120.0 && pt<=140.0) result = x->EMF+x->OutF+par[29]*x->HadF;
      else if (pt>140.0 && pt<=160.0) result = x->EMF+x->OutF+par[30]*x->HadF;
      else if (pt>160.0 && pt<=180.0) result = x->EMF+x->OutF+par[31]*x->HadF;
      else if (pt>180.0 && pt<=200.0) result = x->EMF+x->OutF+par[32]*x->HadF;
      else if (pt>200.0 && pt<=225.0) result = x->EMF+x->OutF+par[33]*x->HadF;
      else if (pt>225.0 && pt<=250.0) result = x->EMF+x->OutF+par[34]*x->HadF;
      else if (pt>250.0 && pt<=275.0) result = x->EMF+x->OutF+par[35]*x->HadF;
      else if (pt>275.0 && pt<=300.0) result = x->EMF+x->OutF+par[36]*x->HadF;
      else if (pt>300.0 && pt<=350.0) result = x->EMF+x->OutF+par[37]*x->HadF;
      else if (pt>350.0 && pt<=400.0) result = x->EMF+x->OutF+par[38]*x->HadF;
      else if (pt>400.0 && pt<=500.0) result = x->EMF+x->OutF+par[39]*x->HadF;
      else if (pt>500.0 && pt<=700.0) result = x->EMF+x->OutF+par[40]*x->HadF;
      else if (pt>700.0 )             result = x->EMF+x->OutF+par[41]*x->HadF;
    } else {
      if(pt>=0.0 && pt<=10.0)   result = x->EMF+x->OutF+par[42]*x->HadF;
      else if (pt> 10.0 && pt<= 20.0) result = x->EMF+x->OutF+par[43]*x->HadF;
      else if (pt> 20.0 && pt<= 30.0) result = x->EMF+x->OutF+par[44]*x->HadF;
      else if (pt> 30.0 && pt<= 40.0) result = x->EMF+x->OutF+par[45]*x->HadF;
      else if (pt> 40.0 && pt<= 60.0) result = x->EMF+x->OutF+par[46]*x->HadF;
      else if (pt> 60.0 && pt<= 80.0) result = x->EMF+x->OutF+par[47]*x->HadF;
      else if (pt> 80.0 && pt<=100.0) result = x->EMF+x->OutF+par[48]*x->HadF;
      else if (pt>100.0 && pt<=120.0) result = x->EMF+x->OutF+par[49]*x->HadF;
      else if (pt>120.0 && pt<=140.0) result = x->EMF+x->OutF+par[50]*x->HadF;
      else if (pt>140.0 && pt<=160.0) result = x->EMF+x->OutF+par[51]*x->HadF;
      else if (pt>160.0 && pt<=180.0) result = x->EMF+x->OutF+par[52]*x->HadF;
      else if (pt>180.0 && pt<=200.0) result = x->EMF+x->OutF+par[53]*x->HadF;
      else if (pt>200.0 && pt<=225.0) result = x->EMF+x->OutF+par[54]*x->HadF;
      else if (pt>225.0 && pt<=250.0) result = x->EMF+x->OutF+par[55]*x->HadF;
      else if (pt>250.0 && pt<=275.0) result = x->EMF+x->OutF+par[56]*x->HadF;
      else if (pt>275.0 && pt<=300.0) result = x->EMF+x->OutF+par[57]*x->HadF;
      else if (pt>300.0 && pt<=350.0) result = x->EMF+x->OutF+par[58]*x->HadF;
      else if (pt>350.0 && pt<=400.0) result = x->EMF+x->OutF+par[59]*x->HadF;
      else if (pt>400.0 && pt<=500.0) result = x->EMF+x->OutF+par[60]*x->HadF;
      else if (pt>500.0 && pt<=700.0) result = x->EMF+x->OutF+par[61]*x->HadF;
      else if (pt>700.0 )             result = x->EMF+x->OutF+par[62]*x->HadF;
    }

    //OutOfCone, Dominant, parametrized in Et since cone R lorenz invariant
    result *= ( 1. + 0.295 * par[63] * exp(- 0.02566 * par[64] * result));   

    return  result;
  }
};


// parametrization of response by some "clever" function
/// -----------------------------------------------------------------
class MyParametrization: public Parametrization {
 public:
  MyParametrization() : Parametrization(3,2) {}
  const char* name() const { return "MyParametrization";}
  
  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    return x->EMF + par[0]*x->HadF + par[1]*log(x->pt) + par[2];
  }
  
  double correctedJetEt(TMeasurement *const x,double *const par) const {
    return par[0]*x->pt + par[1];
  }
};

// Parametrization of response with some ideas from the JetMET group
/// -----------------------------------------------------------------
class JetMETParametrization: public Parametrization {
public:
  JetMETParametrization() : Parametrization(3,5) {}
  const char* name() const { return "JetMETParametrization";}
  
  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    if(par[0] < -10) par[0] =     -10;
    if(par[1] <   0) par[1] = -par[1];
    if(par[2] <   0) par[2] = -par[2];
    return par[1] * x->HadF + par[2] * x->EMF + x->OutF + par[0];
  }
  
  double correctedJetEt(TMeasurement *const x,double *const par) const {
    double logx = log(x->pt);
    if(logx < 0) logx = 0;
    if(par[1] < 0) par[1] *= -1;
    if(par[2] < 0) par[2] *= -1;
    if(par[3] < 0) par[3] *= -1;
    if(par[4] < 0) par[4] *= -1;
    return (par[0] - par[1]/(pow(logx,par[2]) + par[3]) + par[4]/x->pt) * x->pt;  
  }
};

// simple parametrization
/// -----------------------------------------------------------------
class SimpleParametrization: public Parametrization {
public:
  SimpleParametrization() : Parametrization(3,3) {}
  const char* name() const { return "SimpleParametrization";}

  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    if(par[0] < -10) par[0] =     -10;
    if(par[1] <   0) par[1] = -par[1];
    if(par[2] <   0) par[2] = -par[2];
    return par[1] * x->EMF + par[2] * x->HadF + x->OutF + par[0];
  }

  double correctedJetEt(TMeasurement *const x,double *const par) const {
    if(par[0] < 0) par[0] *= -1;
    if(par[1] < 0) par[1] *= -1;
    if(par[2] < 0) par[2] *= -1;
    return x->pt * ( par[2] + par[0] * exp( -par[1] * x->pt ) );  
  }
};

// parametrization for toy MC
/// -----------------------------------------------------------------
class ToyParametrization: public Parametrization {
public:
  ToyParametrization() : Parametrization(1,0) {}
  const char* name() const { return "ToyParametrization";}

  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    if( fabs(x->HadF+x->EMF+x->OutF-x->pt)>0.1){
      return par[0] * x->HadF + x->EMF + x->OutF;
    }
  }

  double correctedJetEt(TMeasurement *const x,double *const par) const {
    return x->pt;  
  }
};

// parametrization of hadronic response by a step function
// optimized for ToyMC pt spectrum 0 - 300 GeV
/// -----------------------------------------------------------------
class ToyStepParametrization : public Parametrization { 
public:
  ToyStepParametrization() : Parametrization(15,0) {}
  const char* name() const { return "ToyStepParametrization";}
  
  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    double result = 0.;
    double pt = x->HadF;
    
    if(pt < 2.)        result = x->EMF+x->OutF+par[ 0]*x->HadF;
    else if(pt <   5.) result = x->EMF+x->OutF+par[ 1]*x->HadF;
    else if(pt <  10.) result = x->EMF+x->OutF+par[ 2]*x->HadF;
    else if(pt <  20.) result = x->EMF+x->OutF+par[ 3]*x->HadF;
    else if(pt <  30.) result = x->EMF+x->OutF+par[ 4]*x->HadF;
    else if(pt <  40.) result = x->EMF+x->OutF+par[ 5]*x->HadF;
    else if(pt <  50.) result = x->EMF+x->OutF+par[ 6]*x->HadF;
    else if(pt <  60.) result = x->EMF+x->OutF+par[ 7]*x->HadF;
    else if(pt <  70.) result = x->EMF+x->OutF+par[ 8]*x->HadF;
    else if(pt <  80.) result = x->EMF+x->OutF+par[ 9]*x->HadF;
    else if(pt <  90.) result = x->EMF+x->OutF+par[10]*x->HadF;
    else if(pt < 100.) result = x->EMF+x->OutF+par[11]*x->HadF;
    else if(pt < 110.) result = x->EMF+x->OutF+par[12]*x->HadF;
    else if(pt < 120.) result = x->EMF+x->OutF+par[13]*x->HadF;
    else               result = x->EMF+x->OutF+par[14]*x->HadF;
    return result;
  }
    
  double correctedJetEt(TMeasurement *const x,double *const par) const {
    return x->pt;
  }
};


// parametrization of Jet hadronic response by a step function
// optimized for ToyMC pt spectrum 0 - 300 GeV
/// -----------------------------------------------------------------
class ToyStepJetParametrization : public Parametrization { 
 public:
  ToyStepJetParametrization() : Parametrization(0,15) {}
  const char* name() const { return "ToyStepJetParametrization";}
  
  double correctedTowerEt(TMeasurement *const x,double *const par) const {
    return x->pt;
  }
  
  double correctedJetEt(TMeasurement *const x,double *const par) const {
    double pt = x->HadF;
    double result = 0.;
    
    if(pt < 25. )      result = x->EMF+x->OutF+par[ 0]*x->HadF;
    else if(pt <  50.) result = x->EMF+x->OutF+par[ 1]*x->HadF;
    else if(pt <  75.) result = x->EMF+x->OutF+par[ 2]*x->HadF;
    else if(pt < 100.) result = x->EMF+x->OutF+par[ 3]*x->HadF;
    else if(pt < 125.) result = x->EMF+x->OutF+par[ 4]*x->HadF;
    else if(pt < 150.) result = x->EMF+x->OutF+par[ 5]*x->HadF;
    else if(pt < 175.) result = x->EMF+x->OutF+par[ 6]*x->HadF;
    else if(pt < 200.) result = x->EMF+x->OutF+par[ 7]*x->HadF;
    else if(pt < 225.) result = x->EMF+x->OutF+par[ 8]*x->HadF;
    else if(pt < 250.) result = x->EMF+x->OutF+par[ 9]*x->HadF;
    else if(pt < 275.) result = x->EMF+x->OutF+par[10]*x->HadF;
    else if(pt < 300.) result = x->EMF+x->OutF+par[11]*x->HadF;
    else if(pt < 325.) result = x->EMF+x->OutF+par[12]*x->HadF;
    else if(pt < 350.) result = x->EMF+x->OutF+par[13]*x->HadF;
    else               result = x->EMF+x->OutF+par[14]*x->HadF;
    return  result;
  }
};

#endif

