#include "JetMETCorrections/MCJet/interface/JetCalibratorMCJet.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include <vector>
#include <fstream>
#include <sstream>
using namespace std;

class ParametrizationMCJet{

 public:
  
  ParametrizationMCJet(int ptype, vector<double> parameters):type(ptype),p(parameters){};
  
  double value(double arg) const;

 private:

  int type;
  std::vector<double> p;
};

double ParametrizationMCJet::value(double e)const{

  double enew(e);

  switch(type){
  case 1:
    {
      double a2 = p[6]/(sqrt(fabs(p[7]*p[2] + p[8]))) + p[9];
      double a1 = p[3]*sqrt(p[1] +p[4]) + p[5];
      double w1 = (a1*p[2] - a2*p[1])/(p[2]-p[1]);
      double w2 = (a2-a1)/(p[2]-p[1]);
      double koef = 1.;
      double JetCalibrEt = e;
      double x=e;

      if ( e<10 ) JetCalibrEt=10.;

           for (int ie=0; ie<10; ie++) {
	   
	     if (JetCalibrEt < p[1]) {
	      koef = p[3]*sqrt(JetCalibrEt +p[4]) + p[5];
	     }

	     else if (JetCalibrEt < p[2]) {
	      koef = w1 + JetCalibrEt*w2;
	     }

	     else if (JetCalibrEt > p[2]) {
	      koef = p[6]/(sqrt(fabs(p[7]*JetCalibrEt + p[8]))) + p[9];
	     }
	     
	     JetCalibrEt = x / koef;

	     }

      enew=e/koef;

      break;
    }

  default:
    cerr<<"JetCalibratorMCJet: Error: unknown parametrization type '"<<type<<"' in JetCalibratorMCJet. No correction applied"<<endl;
    break;
  }
  return enew;
};

class   JetCalibrationParameterSetMCJet{
public:
  JetCalibrationParameterSetMCJet(string tag);
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
JetCalibrationParameterSetMCJet::JetCalibrationParameterSetMCJet(string tag){

  //string path(getenv("CMSSW_SEARCH_PATH"));
  //string file="JetCalibrationData/MCJet/"+tag+".txt";
  std::string file="JetMETCorrections/MCJet/data/"+tag+".txt";
  
  edm::FileInPath f1(file);
  
  std::ifstream in( (f1.fullPath()).c_str() );
  
  if ( f1.isLocal() ){
    cout << " Start to read file "<<file<<endl;
    string line;
    while( std::getline( in, line)){
      if(!line.size() || line[0]=='#') continue;
      istringstream linestream(line);
      double par;
      int type;
      linestream>>par>>type;
      
      cout<<" Parameter eta = "<<par<<" Type= "<<type<<endl;
      
      etavector.push_back(par);
      typevector.push_back(type);
      pars.push_back(vector<double>());
      while(linestream>>par)pars.back().push_back(par);
    }
  }
  else
    cout<<"The file \""<<file<<"\" was not found in path \""<<f1.fullPath()<<"\"."<<endl;
}

JetCalibratorMCJet::~JetCalibratorMCJet()
{
  for(std::map<double,ParametrizationMCJet*>::iterator ip=parametrization.begin();ip!=parametrization.end();ip++) delete ip->second;  
}

void JetCalibratorMCJet::setParameters(std::string aCalibrationType)
{ 

  if (theCalibrationType.size()==0)
    {
      theCalibrationType = aCalibrationType; 
      JetCalibrationParameterSetMCJet pset(theCalibrationType);
      if(!pset.valid()){
	cerr<<"[Jets] JetCalibratorMCJet: calibration = "<<theCalibrationType<< " not found! Cannot apply any correction ..." << endl;
      }
      for(int ieta=0; ieta<pset.neta();ieta++)
	parametrization[pset.eta(ieta)]=new ParametrizationMCJet(pset.type(ieta),pset.parameters(ieta));
    }
  else
    {
      cout <<"[Jets] JetCalibratorMCJet: calibration = "<<theCalibrationType<< " was set before - contnuing using this calibration ..." << endl;
    }
}

void JetCalibratorMCJet::run( const CaloJetCollection* theJetIn, std::vector<HepLorentzVector> & theJetOut) 
{
  if(parametrization.empty()) { return; }
  cout<<" The size of the input jet collection "<<theJetIn->size()<<endl;
  if( theJetIn->size() == 0 ) { return; }
  
  for(CaloJetCollection::const_iterator ijet = theJetIn->begin(); ijet != theJetIn->end(); ijet++)
  {
    double et=(*ijet).getEt();
    double eta=abs((*ijet).getEta());
    

    if(eta<10) { eta=abs((*ijet).getY()); }
    
    cout<<" Et and eta of jet "<<et<<" "<<eta<<endl;

    double etnew;
    std::map<double,ParametrizationMCJet*>::const_iterator ip=parametrization.upper_bound(eta);
    if(ip==parametrization.begin()) 
    { 
        etnew=ip->second->value(et); 
    }
      else if(ip==parametrization.end()) 
      {
          etnew=(--ip)->second->value(et);
      }
       else
          {
            double et2=ip->second->value(et);
            etnew=et2;
          }
	 //theJet*=etnew/et;
	 
	 cout<<" The new energy found "<<etnew<<" "<<et<<endl;

	 HepLorentzVector thejet((*ijet).getPx()*etnew/et, (*ijet).getPy()*etnew/et, (*ijet).getPz()*etnew/et,
	                                (*ijet).getE()*etnew/et);
			
	 cout<<" The new jet is created "<<endl;
	 		
         theJetOut.push_back(thejet);
  }
  cout<<" The first jet is added "<<theJetOut.size()<<endl;
}

