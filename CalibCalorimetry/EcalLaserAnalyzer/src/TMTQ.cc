/* 
 *  \class TMTQ
 *
 *  $Date: 2008/04/25 10:24:33 $
 *  \author: Julie Malcles - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMTQ.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMarkov.h>
#include <TMath.h>

using namespace std;

//ClassImp(TMTQ)


// Default Constructor...
TMTQ::TMTQ()
{
  init();
}


// Destructor
TMTQ::~TMTQ()
{
}

void TMTQ::init()
{

  for(int j=0;j<nOutVar;j++){
    cuts[0][j]=0.0;
    cuts[1][j]=10.0e9;
    mom[j]=new TMom();
  }
}

void TMTQ::addEntry(double peak, double sigma, double fit, double ampl, double trise, double fwhm, double fw20, double fw80, double ped, double pedsig, double sliding)
{
  double val[nOutVar];

  val[iPeak]=peak;
  val[iSigma]=sigma;
  val[iFit]=fit;
  val[iAmpl]=ampl;
  val[iTrise]=trise;
  val[iFwhm]=fwhm;
  val[iFw20]=fw20;
  val[iFw80]=fw80;
  val[iPed]=ped;
  val[iPedsig]=pedsig;
  val[iSlide]=sliding;

  for(int ivar=0;ivar<nOutVar;ivar++){
    mom[ivar]->addEntry(val[ivar]);
  }

}
  
void TMTQ::setCut(int ivar, double mean, double sig){

  if(ivar<nOutVar){
    cuts[0][ivar]=mean-2.0*sig;
    cuts[1][ivar]=mean+2.0*sig;
    
    mom[ivar]->setCut(cuts[0][ivar],cuts[1][ivar]);
  }
}

vector<double> TMTQ::get(int ivar){ 

  vector<double> res;
  
  if(ivar<nOutVar){

    res.push_back(mom[ivar]->getMean());
    res.push_back(mom[ivar]->getRMS());
    res.push_back(mom[ivar]->getM3());
    res.push_back(mom[ivar]->getNevt());
    res.push_back(mom[ivar]->getMin());
    res.push_back(mom[ivar]->getMax());

  }
  return res;

}

vector<double>   TMTQ::getPeak(){vector<double> x=get(TMTQ::iPeak); return x;}
vector<double>   TMTQ::getAmpl(){vector<double> x=get(TMTQ::iAmpl); return x;}
vector<double>   TMTQ::getSigma(){vector<double> x=get(TMTQ::iSigma); return x;}
vector<double>   TMTQ::getTrise(){vector<double> x=get(TMTQ::iTrise); return x;}
vector<double>   TMTQ::getFit(){vector<double> x=get(TMTQ::iFit); return x;}
vector<double>   TMTQ::getFwhm(){vector<double> x=get(TMTQ::iFwhm); return x;}
vector<double>   TMTQ::getFw20(){vector<double> x=get(TMTQ::iFw20); return x;}
vector<double>   TMTQ::getFw80(){vector<double> x=get(TMTQ::iFw80); return x;}
vector<double>   TMTQ::getPed(){vector<double> x=get(TMTQ::iPed); return x;}
vector<double>   TMTQ::getPedsig(){vector<double> x=get(TMTQ::iPedsig); return x;}
vector<double>   TMTQ::getSliding(){vector<double> x=get(TMTQ::iSlide); return x;}



