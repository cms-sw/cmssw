// $Id: HLLHCEvtVtxGenerator_Fix.cc, v 1.0 2015/03/15 10:32:11 Exp $

#include "IOMC/EventVertexGenerators/interface/HLLHCEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "HepMC/SimpleVector.h"

using namespace std;

namespace {
  
  constexpr double pmass = 0.9382720813e9; // eV
  constexpr double gamma34 = 1.22541670246517764513;    // Gamma(3/4)
  constexpr double gamma14 = 3.62560990822190831193;    // Gamma(1/4)
  constexpr double gamma54 = 0.90640247705547798267;    // Gamma(5/4)
  constexpr double sqrt2 = 1.41421356237;
  constexpr double sqrt2to5 = 5.65685424949;
  constexpr double two_pi=2.0*M_PI;
}

void HLLHCEvtVtxGenerator::fillDescriptions(edm::ConfigurationDescriptions &descriptions)
{
    edm::ParameterSetDescription desc;
    desc.add<double>("MeanXIncm",0.0);
    desc.add<double>("MeanYIncm",0.0);
    desc.add<double>("MeanZIncm",0.0);
    desc.add<double>("TimeOffsetInns",0.0);
    desc.add<double>("EprotonInGeV", 7000.0);
    desc.add<double>("CrossingAngleInurad", 510.0);
    desc.add<double>("CrabbingAngleCrossingInurad", 380.0);
    desc.add<double>("CrabbingAngleSeparationInurad", 0.0);
    desc.add<double>("CrabFrequencyInMHz", 400.0);
    desc.add<bool>("RF800", false);
    desc.add<double>("BetaCrossingPlaneInm", 0.20);
    desc.add<double>("BetaSeparationPlaneInm", 0.20);
    desc.add<double>("HorizontalEmittance", 2.5e-06);
    desc.add<double>("VerticalEmittance", 2.05e-06);
    desc.add<double>("BunchLengthInm", 0.09);
    desc.add<edm::InputTag>("src");
    desc.add<bool>("readDB");
    descriptions.add("HLLHCEvtVtxGenerator",desc);
}

HLLHCEvtVtxGenerator::HLLHCEvtVtxGenerator(const edm::ParameterSet & p )
  : BaseEvtVtxGenerator(p),
    fMeanX(p.getParameter<double>("MeanXIncm")*cm),
    fMeanY(p.getParameter<double>("MeanYIncm")*cm),
    fMeanZ(p.getParameter<double>("MeanZIncm")*cm),
    fTimeOffset(p.getParameter<double>("TimeOffsetInns")*ns*c_light),
    momeV(p.getParameter<double>("EprotonInGeV")*1e9),
    gamma(momeV/pmass + 1.0),
    beta(std::sqrt((1.0 - 1.0/gamma)*((1.0 + 1.0/gamma)))),
    betagamma(beta*gamma),
    phi(p.getParameter<double>("CrossingAngleInurad")*1e-6),    
    wcc(p.getParameter<double>("CrabFrequencyInMHz")*1e6),
    RF800(p.getParameter<bool>("RF800")),
    betx(p.getParameter<double>("BetaCrossingPlaneInm")),
    bets(p.getParameter<double>("BetaSeparationPlaneInm")),
    epsxn(p.getParameter<double>("HorizontalEmittance")),
    epssn(p.getParameter<double>("VerticalEmittance")),
    sigs(p.getParameter<double>("BunchLengthInm")),    
    alphax(p.getParameter<double>("CrabbingAngleCrossingInurad")*1e-6),
    alphay(p.getParameter<double>("CrabbingAngleSeparationInurad")*1e-6),
    oncc(alphax/phi),
    epsx(epsxn/(betagamma)),
    epss(epsx),
    sigx(std::sqrt(epsx*betx)),
    phiCR(oncc*phi)
    
{   
 
}

HLLHCEvtVtxGenerator::~HLLHCEvtVtxGenerator() 
{
}


HepMC::FourVector* HLLHCEvtVtxGenerator::newVertex(CLHEP::HepRandomEngine* engine) {

  double imax=intensity(0.,0.,0.,0.);

  double x(0.),y(0.),z(0.),t(0.),i(0.);
  
  int count=0;
  
  auto shoot = [&](){ return CLHEP::RandFlat::shoot(engine); };
  
  do {
    z=(shoot()-0.5)*6.0*sigs;
    t=(shoot()-0.5)*6.0*sigs;
    x=(shoot()-0.5)*12.0*sigma(0.0,epsxn,betx,betagamma);
    y=(shoot()-0.5)*12.0*sigma(0.0,epssn,bets,betagamma);
    
    i=intensity(x,y,z,t);
    
    if (i>imax)  edm::LogError("Too large intensity") 
                   << "i>imax : "<<i<<" > "<<imax<<endl;
    ++count;
  } while ((i<imax*shoot())&&count<10000);
  
  if (count>9999) edm::LogError("Too many tries ") 
                    << " count : "<<count<<endl;
  
  if(fVertex == 0)
    fVertex = new HepMC::FourVector();
  
  //---convert to mm
  x*=1000.0;
  y*=1000.0;
  z*=1000.0;
  t*=1000.0;
  
  x+=fMeanX;
  y+=fMeanY;
  z+=fMeanZ;
  t+=fTimeOffset;
  
  fVertex->set( x, y, z, t );
  
  return fVertex;
}



double HLLHCEvtVtxGenerator::sigma(double z, double epsilon, double beta, double betagamma) const
{
  double sigma=std::sqrt(epsilon*(beta+z*z/beta)/betagamma);
  
  return sigma;
} 

double HLLHCEvtVtxGenerator::intensity(double x, 
                                       double y, 
                                       double z, 
                                       double t) const {
  //---c in m/s --- remember t is already in meters
  constexpr double c= 2.99792458e+8; // m/s
  
  const double sigmay=sigma(z,epssn,bets,betagamma);
    
  const double alphay_mod=alphay*std::cos(wcc*(z-t)/c);
  
  const double cay=std::cos(alphay_mod);
  const double say=std::sin(alphay_mod);

  const double dy=-(z-t)*say-y*cay;

  const double xzt_density= integrandCC(x,z,t);

  const double norm = two_pi*sigmay;
  
  return ( std::exp(-dy*dy/(sigmay*sigmay))*
           xzt_density/norm );
}

double HLLHCEvtVtxGenerator::integrandCC(double x,
                                         double z, 
                                         double ct) const {  
  constexpr double local_c_light = 2.99792458e8;

  const double k = wcc/local_c_light*two_pi;
  const double k2 = k*k;
  const double cos = std::cos(phi/2.0);
  const double sin = std::sin(phi/2.0);
  const double cos2 = cos*cos;
  const double sin2 = sin*sin;
    
  const double sigx2 = sigx*sigx;
  const double sigmax2=sigx2*(1+z*z/(betx*betx));
    
  const double sigs2 = sigs*sigs;
  
  constexpr double factorRMSgauss4  = 1./sqrt2/gamma34 * gamma14; // # Factor to take rms sigma as input of the supergaussian
  constexpr double NormFactorGauss4 = sqrt2to5 * gamma54 * gamma54;
  
  const double sinCR  = std::sin(phiCR/2.0);
  const double sinCR2 = sinCR*sinCR;
      
  double result = -1.0;
    
  if( !RF800 ) {
    const double norm =2.0/(two_pi*sigs2);
    const double cosks = std::cos(k*z);
    const double sinkct = std::sin(k*ct);
    result = norm*std::exp(-ct*ct/sigs2
                           -z*z*cos2/sigs2
                           -1.0/(4*k2*sigmax2)*(
                                                //-4*cosks*cosks * sinkct*sinkct * sinCR2 // comes from integral over x
                                                -8*z*k*std::sin(k*z)*std::cos(k*ct) * sin * sinCR
                                                +2 * sinCR2
                                                -std::cos(2*k*(z-ct)) * sinCR2
                                                -std::cos(2*k*(z+ct)) * sinCR2
                                                +4*k2*z*z *sin2
                                                )
                           - x*x*(cos2/sigmax2 + sin2/sigs2) // contribution from x integrand
                           + x*ct*sin/sigs2 // contribution from x integrand
                           + 2*x*cos*cosks*sinkct*sinCR/k/sigmax2 // contribution from x integrand
                           //+(2*ct/k)*np.cos(k*s)*np.sin(k*ct) *(sin*sinCR)/(sigs2*cos)  # small term
                           //+ct**2*(sin2/sigs4)/(cos2/sigmax2)				              # small term
                           )/(1.0+(z*z)/(betx*betx))/std::sqrt(1.0+(z*z)/(bets*bets));
      
  } else {
    
    const double norm = 2.0/(NormFactorGauss4*sigs2*factorRMSgauss4);
    const double sigs4=sigs2*sigs2*factorRMSgauss4*factorRMSgauss4;
    const double cosks = std::cos(k*z);
    const double sinct = std::sin(k*ct);
    result = norm*std::exp(
                           -ct*ct*ct*ct/sigs4
                           -z*z*z*z*cos2*cos2/sigs4
                           -6*ct*ct*z*z*cos2/sigs4
                           -sin2/(4*k2*sigmax2)*(
                                                 2
                                                 +4*k2*z*z
                                                 -std::cos(2*k*(z-ct))
                                                 -std::cos(2*k*(z+ct))
                                                 -8*k*s*std::cos(k*ct)*std::sin(k*z) 
                                                 -4 * cosks*cosks * sinct*sinct)
                           )/std::sqrt(1+z*z/(betx*betx))/std::sqrt(1+z*z/(bets*bets));
  }
  
  return result;
}

