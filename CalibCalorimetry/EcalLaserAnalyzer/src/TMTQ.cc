/* 
 *  \class TMTQ
 *
 *  $Date: 2012/02/09 10:08:10 $
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

std::vector<double> TMTQ::get(int ivar){ 

  std::vector<double> res;
  
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

std::vector<double>   TMTQ::getPeak(){std::vector<double> x=get(TMTQ::iPeak); return x;}
std::vector<double>   TMTQ::getAmpl(){std::vector<double> x=get(TMTQ::iAmpl); return x;}
std::vector<double>   TMTQ::getSigma(){std::vector<double> x=get(TMTQ::iSigma); return x;}
std::vector<double>   TMTQ::getTrise(){std::vector<double> x=get(TMTQ::iTrise); return x;}
std::vector<double>   TMTQ::getFit(){std::vector<double> x=get(TMTQ::iFit); return x;}
std::vector<double>   TMTQ::getFwhm(){std::vector<double> x=get(TMTQ::iFwhm); return x;}
std::vector<double>   TMTQ::getFw20(){std::vector<double> x=get(TMTQ::iFw20); return x;}
std::vector<double>   TMTQ::getFw80(){std::vector<double> x=get(TMTQ::iFw80); return x;}
std::vector<double>   TMTQ::getPed(){std::vector<double> x=get(TMTQ::iPed); return x;}
std::vector<double>   TMTQ::getPedsig(){std::vector<double> x=get(TMTQ::iPedsig); return x;}
std::vector<double>   TMTQ::getSliding(){std::vector<double> x=get(TMTQ::iSlide); return x;}



