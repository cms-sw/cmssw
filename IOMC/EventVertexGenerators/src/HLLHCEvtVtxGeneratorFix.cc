// $Id: HLLHCEvtVtxGenerator_Fix.cc, v 1.0 2015/03/15 10:32:11 Exp $

#include "IOMC/EventVertexGenerators/interface/HLLHCEvtVtxGeneratorFix.h"
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

HLLHCEvtVtxGeneratorFix::HLLHCEvtVtxGeneratorFix(const edm::ParameterSet & p )
    : BaseEvtVtxGenerator(p)
{   
    fRandom = new CLHEP::RandFlat(getEngine());
  
    fMeanX =  p.getParameter<double>("MeanXIncm")*cm;
    fMeanY =  p.getParameter<double>("MeanYIncm")*cm;
    fMeanZ =  p.getParameter<double>("MeanZIncm")*cm;
    fTimeOffset = p.getParameter<double>("TimeOffsetInns")*ns*c_light;

    fEproton = p.getParameter<double>("EprotonInGeV");
    fTheta = p.getParameter<double>("HalfCrossingAngleInurad");

    fAlphax = p.getParameter<double>("CrabAngleCrossingPlaneInurad");
    fOmegax = p.getParameter<double>("CrabFrequencyCrossingPlaneInMHz");
    fEpsilonx = p.getParameter<double>("NormalizedEmittanceCrossingPlaneInum");
    fBetax = p.getParameter<double>("BetaStarCrossingPlaneInm");

    fAlphay = p.getParameter<double>("CrabAngleParallelPlaneInurad");
    fOmegay = p.getParameter<double>("CrabFrequencyParallelPlaneInMHz");
    fEpsilony = p.getParameter<double>("NormalizedEmittanceParallelPlaneInum");
    fBetay = p.getParameter<double>("BetaStarParallelPlaneInm");

    fZsize = p.getParameter<double>("ZsizeInm");
    fProfile = p.getParameter<string>("BeamProfile");

}

HLLHCEvtVtxGeneratorFix::~HLLHCEvtVtxGeneratorFix() 
{
    delete fRandom; 
}


HepMC::FourVector* HLLHCEvtVtxGeneratorFix::newVertex() {

    lhcbeamparams params;

    params.betagamma=fEproton/0.94;  //FIXME 
    params.theta=fTheta*1e-6;
    params.alphax=fAlphax*1e-6;
    params.omegax=fOmegax*1e6;
    params.epsilonx=fEpsilonx*1e-6;
    params.betax=fBetax;
    params.alphay=fAlphay*1e-6;
    params.omegay=fOmegay*1e6;
    params.epsilony=fEpsilony*1e-6;
    params.betay=fBetay;
    params.zsize=fZsize;  
    params.beamprofile=fProfile;


    double imax=p1(0.0,0.0,0.0,0.0,params)*p2(0.0,0.0,0.0,0.0,params);

    double x,y,z,t,i;

    int count=0;

    do
    {
        z=(fRandom->fire()-0.5)*6.0*fZsize;
        t=(fRandom->fire()-0.5)*6.0*fZsize;
        x=(fRandom->fire()-0.5)*12.0*sigma(0.0,params.epsilonx,params.betax,
                                           params.betagamma);
        y=(fRandom->fire()-0.5)*8.0*sigma(0.0,params.epsilony,params.betay,
                                          params.betagamma);

        i=p1(x,y,z,t,params)*p2(x,y,z,t,params);

        if (i>imax)  edm::LogError("Too large intensity") 
                         << "i>imax : "<<i<<" > "<<imax<<endl;
        count++;
    }
    while ((i<imax*fRandom->fire())&&count<10000);
  
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

void HLLHCEvtVtxGeneratorFix::fillDescriptions(edm::ConfigurationDescriptions &descriptions)
{
    edm::ParameterSetDescription desc;
    desc.add<double>("MeanXIncm");
    desc.add<double>("MeanYIncm");
    desc.add<double>("MeanZIncm");
    desc.add<double>("TimeOffsetInns");
    desc.add<double>("EprotonInGeV");
    desc.add<double>("HalfCrossingAngleInurad");
    desc.add<double>("CrabAngleCrossingPlaneInurad");
    desc.add<double>("CrabFrequencyCrossingPlaneInMHz");
    desc.add<double>("NormalizedEmittanceCrossingPlaneInum");
    desc.add<double>("BetaStarCrossingPlaneInm");
    desc.add<double>("CrabAngleParallelPlaneInurad");
    desc.add<double>("CrabFrequencyParallelPlaneInMHz");
    desc.add<double>("NormalizedEmittanceParallelPlaneInum");
    desc.add<double>("BetaStarParallelPlaneInm");
    desc.add<double>("ZsizeInm");
    desc.add<string>("BeamProfile");  
    desc.add<edm::InputTag>("src");
    desc.add<bool>("readDB");
    descriptions.add("HLLHCEvtVtxGeneratorFix",desc);
}

double HLLHCEvtVtxGeneratorFix::sigma(double z, double epsilon, double beta, double betagamma)
{
    double sigma=sqrt(epsilon*(beta+z*z/beta)/betagamma);

    return sigma;
} 

double HLLHCEvtVtxGeneratorFix::rhoz(double z, const lhcbeamparams& params)
{
    static double two_pi=8.0*atan(1.0);

    if (params.beamprofile=="Flat")
    {
        if (fabs(z)<params.zsize)
            return 1.0;
        return 0.0;
    }
    else if (params.beamprofile=="Gauss")
        return exp(-0.5*z*z/(params.zsize*params.zsize))/(sqrt(two_pi)*params.zsize);
    else
    {
        edm::LogError("Wrong BeamProfile") << "BeamProfile: " << params.beamprofile
                                           << " expect either 'Flat' or 'Gauss' " 
                                           << endl;
    }  
    return 0.0; 
}


double HLLHCEvtVtxGeneratorFix::p1(double x, 
				double y, 
				double z, 
				double t, 
				const lhcbeamparams& params)
{
    double sigmax=sigma(z,params.epsilonx,params.betax,params.betagamma);
    double sigmay=sigma(z,params.epsilony,params.betay,params.betagamma);

    //---c in m/s --- remember t is already in meters
    double c=c_light/m*s;
    static double two_pi=8.0*atan(1.0);
  
    double omegax=two_pi*params.omegax;
    double omegay=two_pi*params.omegay;
    double alphax=params.alphax*cos(omegax*(z-t)/c);
    double alphay=params.alphay*cos(omegay*(z-t)/c);

    double cax=cos(alphax);
    double sax=sin(alphax);

    double cay=cos(alphay);
    double say=sin(alphay);

    double ct=cos(params.theta);
    double st=sin(params.theta);

    double dx=(z-t*ct)*(cax*st-sax*ct)-(x-t*st)*(sax*st+cax*ct);

    double dy=-(z-t)*say-y*cay;

    double zrho=rhoz((z-t*ct)*(cax*ct+sax*st)-(x-t*st)*(sax*ct-cax*st),
                     params);

    return exp(-0.5*dx*dx/(sigmax*sigmax))*exp(-0.5*dy*dy/(sigmay*sigmay))*zrho/(two_pi*sigmax*sigmay);
}

double HLLHCEvtVtxGeneratorFix::p2(double x, 
				double y, 
				double z, 
				double t, 
				const lhcbeamparams& params)
{
    double sigmax=sigma(z,params.epsilonx,params.betax,params.betagamma);
    double sigmay=sigma(z,params.epsilony,params.betay,params.betagamma);

    static double two_pi=8.0*atan(1.0);

    double cax=cos(-params.alphax);
    double sax=sin(-params.alphax);

    double cay=cos(params.alphay);
    double say=sin(params.alphay);

    double ct=cos(-params.theta);
    double st=sin(-params.theta);

    double dx=(z+t*ct)*(cax*st-sax*ct)-(x+t*st)*(sax*st+cax*ct);

    double dy=-(z+t)*say-y*cay;

    double zrho=rhoz((z+t*ct)*(cax*ct+sax*st)-(x+t*st)*(sax*ct-cax*st),
                     params);

    return exp(-0.5*dx*dx/(sigmax*sigmax))*exp(-0.5*dy*dy/(sigmay*sigmay))*zrho/(two_pi*sigmax*sigmay); 
}

  
