#include "JetMETCorrections/TauJet/interface/TauJetCorrector.h"

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

double TauJetCorrector::ParametrizationTauJet::value(double et, double eta)const{

  double x=et;
  double etnew(et);
  double etabound = (*theEtabound.find(type)).second;
  std::vector<double> taus = (*theParam.find(type)).second;
  
       if ( fabs(eta) > etabound) {
          //cout << " ===> ETA outside of range - CORRECTION NOT DONE ***" << endl;
          //cout << " eta = " << eta <<" pt = " << et << " etabound "<<etabound<<endl;
          return et;
        }
  //cout << "correction parameters: " << taus[0] << " " << taus[1] << " " << taus[2] << endl;
  switch(type){
  case 1:
    {
      etnew = 2*(x-taus[0])/(taus[1]+sqrt(taus[1]*taus[1]-4*taus[0]*taus[2]+4*x*taus[2]));
      break;
    }
  case 2:
    {
      etnew = 2*(x-taus[0])/(taus[1]+sqrt(taus[1]*taus[1]-4*taus[0]*taus[2]+4*x*taus[2]));
      break;
    }
  case 3:
    {
      etnew = 2*(x-taus[0])/(taus[1]+sqrt(taus[1]*taus[1]-4*taus[0]*taus[2]+4*x*taus[2]));
      break;
    }

  default:
	edm::LogError("TauJetCorrector: Error: unknown parametrization type ") << type << " in TauJetCorrector. No correction applied" << endl;
	//cerr<<"TauJetCorrector: Error: unknown parametrization type '"<<type<<"' in TauJetCorrector. No correction applied"<<endl;
    break;
  }
  return etnew;
}

class   JetCalibrationParameterSetTauJet{
public:
  JetCalibrationParameterSetTauJet(string tag);
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
JetCalibrationParameterSetTauJet::JetCalibrationParameterSetTauJet(string tag){

  std::string file="JetMETCorrections/TauJet/data/"+tag+".txt";
  
  edm::FileInPath f1(file);
  
  std::ifstream in( (f1.fullPath()).c_str() );
  
  //  if ( f1.isLocal() ){
    //cout << " Start to read file "<<file<<endl;
    string line;
    while( std::getline( in, line)){
      if(!line.size() || line[0]=='#') continue;
      istringstream linestream(line);
      double par;
      int type;
      linestream>>par>>type;
      
      //cout<<" Parameter eta = "<<par<<" Type= "<<type<<endl;
      
      etavector.push_back(par);
      typevector.push_back(type);
      pars.push_back(vector<double>());
      while(linestream>>par)pars.back().push_back(par);
    }
    //  }
    //  else
    //    cout<<"The file \""<<file<<"\" was not found in path \""<<f1.fullPath()<<"\"."<<endl;
}


TauJetCorrector::TauJetCorrector(const edm::ParameterSet& fConfig)
{
  type = fConfig.getParameter<int>("TauTriggerType");
  setParameters (fConfig.getParameter <std::string> ("tagName"),type);
}


TauJetCorrector::~TauJetCorrector()
{
  for(ParametersMap::iterator ip=parametrization.begin();ip!=parametrization.end();ip++) delete ip->second;
}

void TauJetCorrector::setParameters(std::string aCalibrationType, int itype)
{
     //cout<< " Start to set parameters "<<endl;
     type = itype;
     JetCalibrationParameterSetTauJet pset(aCalibrationType);
     
     if((!pset.valid()) && (aCalibrationType!="no"))
       {
	 edm::LogError( "TauJetCorrector:Jet Corrections not found ")<<aCalibrationType<< 
	   " not found! Cannot apply any correction ... For JetPlusTrack calibration only radii 0.5 and 0.7 are included for JetParton" << endl;
         return; 
       }
     if (aCalibrationType=="no") return;
      
     map<int,vector<double> > pq;
     map<int,vector<double> > pg;
     map<int,vector<double> > pqcd;
     map<int,double > etaboundx;
     int iq = 0;
     int ig = 0;
     int iqcd = 0;
     int mtype = 0;
    for(int ieta=0; ieta<pset.neta();ieta++)
    {
     if( pset.type(ieta) == 1 ) {pq[iq] = pset.parameters(ieta);    iq++;  mtype=(int)(pset.type(ieta));}
     if( pset.type(ieta) == 2 ) {pg[ig] = pset.parameters(ieta);    ig++;  mtype=(int)(pset.type(ieta));}
     if( pset.type(ieta) == 3 ) {pqcd[iqcd] = pset.parameters(ieta);iqcd++;mtype=(int)(pset.type(ieta));}
     if( pset.type(ieta) == -1 ) {etaboundx[mtype-1] = pset.eta(ieta);}
    }
    
    //cout<<" Number of parameters "<<iq<<" "<<ig<<" "<<iqcd<<endl;
    int mynum = 0;
      for(int ieta=0; ieta<pset.neta();ieta++)
        {
	//cout<<" New parmetrization "<<ieta<<" "<<pset.type(ieta)<<endl;
	
	 if ( pset.type(ieta) == -1 ) continue;
	 if( ieta < iq+1)
	 {
      	  parametrization[pset.eta(ieta)]=new ParametrizationTauJet(pset.type(ieta),(*pq.find(ieta)).second,
	                                                                          (*etaboundx.find(0)).second);
	   //cout<<" ALL "<<ieta<<" "<<((*pq.find(ieta)).second)[0]<<" "<<((*pq.find(ieta)).second)[1]<<" "<<
	   //((*pq.find(ieta)).second)[2]<<endl;

	 }
	 if( ieta > iq && ieta < iq + ig + 2 ) 
	 {
	  mynum = ieta - iq - 1;
      	  parametrization[pset.eta(ieta)]=new ParametrizationTauJet(pset.type(ieta),(*pg.find(mynum)).second,
	                                                                          (*etaboundx.find(1)).second);
	   //cout<<" One prong "<<((*pg.find(mynum)).second)[0]<<" "<<((*pg.find(mynum)).second)[1]<<" "<<
	   //((*pg.find(mynum)).second)[2]<<endl;
	   
	 }
	 if( ieta > iq + ig + 1) 
	 {
	  mynum = ieta - iq - ig - 2;
	  //cout<<" Mynum "<<mynum<<" "<<ieta<<" "<<pset.type(ieta)<<endl;
      	  parametrization[pset.eta(ieta)]=new ParametrizationTauJet(pset.type(ieta),(*pqcd.find(mynum)).second,
	                                                                          (*etaboundx.find(2)).second);
	   //cout<<" Two prongs "<<((*pqcd.find(mynum)).second)[0]<<" "<<((*pqcd.find(mynum)).second)[1]<<" "<<
	   //((*pqcd.find(mynum)).second)[2]<<endl;
	 }
	}
   //cout<<" Parameters inserted into mAlgorithm "<<endl;								   
}

double TauJetCorrector::correction( const LorentzVector& fJet) const 
{
  //cout<<" Start Apply Corrections "<<endl;
  if(parametrization.empty()) { return 1.; }
  
    double et=fJet.Et();
    double eta=fabs(fJet.Eta());
    
    //cout<<" Et and eta of jet "<<et<<" "<<eta<<endl;

    double etnew;
    std::map<double,ParametrizationTauJet*>::const_iterator ip=parametrization.upper_bound(eta);
    etnew=(--ip)->second->value(et,eta);

	 //cout<<" The new energy found "<<etnew<<" "<<et<<endl;

         float mScale = etnew/et;
	 		
     return mScale;
}

double TauJetCorrector::correction(const reco::Jet& fJet) const {
	return correction(fJet.p4());
}
