#include "JetMETCorrections/JetParton/interface/JetPartonCorrector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;

namespace JetPartonNamespace{
class UserPartonMixture{
 public:
  UserPartonMixture(){}
  virtual ~UserPartonMixture(){} 
  virtual double mixt (double et,double eta)
  {
    // The mixture of quark and gluons corresponds to QCD jets.
    double f = 0.2+et/1000.*(1.+.13*exp(eta));
    return f;
  }
};

class ParametrizationJetParton{
  
 public:
  
  ParametrizationJetParton(int thePartonMixture,const vector<double>& x, const vector<double>& y, const vector<double>& z)
  {
    type = thePartonMixture;
    pq = x;
    pg = y;
    pqcd = z;
  }
  
  double value(double arg1, double arg2) const;
  
 private:
  
  int type;
  std::vector<double> pq;
  std::vector<double> pg;
  std::vector<double> pqcd;
  
};

double ParametrizationJetParton::value(double e, double eta)const{
  double enew(e);
  double x = e;
  double y = eta;
  
  if( abs(x-pq[2]) < pq[1]/pq[0] || abs(x-pg[2]) < pg[1]/pg[0] || abs(x-pqcd[2]) < pqcd[1]/pqcd[0] )
    {
      return enew;
    }
  
  double kjetq = pq[0]-pq[1]/(x-pq[2]);
  double kjetg = pg[0]-pg[1]/(x-pg[2]);
  double kjetqcd = pqcd[0]-pqcd[1]/(x-pqcd[2]);
  
  switch(type){
  case 1:
    {
      if( abs(kjetq) > 0.0001 ) enew=e/kjetq;
      break;
    }
  case 2:
    {
      if( abs(kjetg) > 0.0001 ) enew=e/kjetg;
      break;
    }
  case 3:
    {
      if( abs(kjetqcd) > 0.0001 ) enew=e/kjetqcd;
      break;
    }
  case 4:
    {
      cout<<"[Jets] JetPartonCorrector: Warning! Calibration to b-quark - does not implemented yet. Light quark calibration is used instead "<<endl;
      if( abs(kjetq) > 0.0001 ) enew=e/kjetq;
      break;
    }
  case 100:
    {
      UserPartonMixture upm;
      double f = upm.mixt(x,y);
      double kjet=(f*kjetq+kjetg)/(f+1);
      if( abs(kjet) > 0.0001 ) enew=e/kjet;
      break;
    }
    
  default:
    cerr<<"[Jets] JetPartonCorrector: Error! unknown parametrization type = "<<type<<" No correction applied ..."<<endl;
    break;
  }
  return enew;
}

class   JetPartonCalibrationParameterSet{
 public:
  JetPartonCalibrationParameterSet(string tag);
  int neta(){return etavector.size();}
  double eta(int ieta){return etavector[ieta];}
  int type(int ieta){return typevector[ieta];}
  const vector<double>& parameters(int ieta){return pars[ieta];}
  bool valid(){return etavector.size();}
  
 private:
  
  vector<double> etavector;
  vector<int> typevector;
  vector< vector<double> > pars;
};

JetPartonCalibrationParameterSet::JetPartonCalibrationParameterSet(string tag){

  std::string file="JetMETCorrections/JetParton/data/"+tag+".txt";

  edm::FileInPath f1(file);

  std::ifstream in( (f1.fullPath()).c_str() );



  //  if ( f1.isLocal() ){
    string line;
    while( std::getline( in, line) ){
      if(!line.size() || line[0]=='#') continue;
      istringstream linestream(line);
      double par;
      int type;
      linestream>>par>>type;
      etavector.push_back(par);
      typevector.push_back(type);
      pars.push_back(vector<double>());
      while(linestream>>par)pars.back().push_back(par);
    }
    //  }
    //  else
    //    if (tag!="no") { cout<<"The file \""<<file<<"\" was not found in path \""<<f1.fullPath()<<"\"."<<endl; }
}
} // namespace

JetPartonCorrector::JetPartonCorrector(const edm::ParameterSet& fConfig)
{
  thePartonMixture = fConfig.getParameter<int>("MixtureType");
  theJetFinderRadius = fConfig.getParameter<double>("Radius");
  setParameters (fConfig.getParameter <std::string> ("tagName"),theJetFinderRadius,thePartonMixture );
}

JetPartonCorrector::~JetPartonCorrector()
{
  for(ParametersMap::iterator ip=parametrization.begin();ip!=parametrization.end();ip++) delete ip->second;  
}

void JetPartonCorrector::setParameters(std::string aCalibrationType, double aJetFinderRadius, int aPartonMixture )
{
     
     theJetFinderRadius = aJetFinderRadius;
     thePartonMixture = aPartonMixture;
    
     JetPartonNamespace::JetPartonCalibrationParameterSet pset(aCalibrationType);
      
     if((!pset.valid()) && (aCalibrationType != "no"))
       {
	 edm::LogError ("JetPartonCorrector: Jet Corrections not found ") << aCalibrationType << 
	   " not found! Cannot apply any correction ... For JetPlusTrack calibration only radii 0.5 and 0.7 are included for JetParton" << endl;
          return;
       }
     if (aCalibrationType=="no") return;
       
       
     map<int,vector<double> > pq;
     map<int,vector<double> > pg;
     map<int,vector<double> > pqcd;
     int iq = 0;
     int ig = 0;
     int iqcd = 0;
    for(int ieta=0; ieta<pset.neta();ieta++)
    {
     if( pset.type(ieta) == 1 ) {pq[iq] = pset.parameters(ieta); iq++;};
     if( pset.type(ieta) == 2 ) {pg[ig] = pset.parameters(ieta);ig++;};
     if( pset.type(ieta) == 3 ) {pqcd[iqcd] = pset.parameters(ieta);iqcd++;};
    }
    
    for(int ieta=0; ieta<iq;ieta++){
      parametrization[pset.eta(ieta)]=new JetPartonNamespace::ParametrizationJetParton(thePartonMixture,(*pq.find(ieta)).second,(*pg.find(ieta)).second,(*pqcd.find(ieta)).second);    
    }
}
double JetPartonCorrector::correction( const LorentzVector& fJet) const
{
  if(parametrization.empty()) { return 1.; }
  
  double et=fJet.Et();
  double eta=fabs(fJet.Eta());
  
  //if(eta<10) { eta=abs(fJet.getY()); }

  double etnew;
  std::map<double,JetPartonNamespace::ParametrizationJetParton*>::const_iterator ip=parametrization.upper_bound(eta);
  if(ip==parametrization.begin()) 
    { 
      etnew=ip->second->value(et,eta); 
    }
  else if(ip==parametrization.end()) 
    {
      etnew=(--ip)->second->value(et,eta);
    }
  else
    {
      double eta2=ip->first;
      double et2=ip->second->value(et,eta);
      ip--;
      double eta1=ip->first;
      double et1=ip->second->value(et,eta);
      
      etnew=(eta2*et1 - eta1*et2 + eta*et2 - eta*et1)/(eta2-eta1); 
    }
   cout<<" JetParton::The new energy found "<<etnew<<" "<<et<<endl;    
  float mScale = 1000.;
   
  if( et > 0.001) mScale = etnew/et;

  return mScale;
  
  
}
