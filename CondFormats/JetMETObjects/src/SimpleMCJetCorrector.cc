//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: MCJetCorrector.cc,v 1.4 2007/02/26 20:31:26 fedor Exp $
//
// MC Jet Corrector
//
#include "CondFormats/JetMETObjects/interface/SimpleMCJetCorrector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

using namespace std;

namespace {
  bool debug = false;
  /// Parametrization itself
  class ParametrizationMCJet {
  public:
    ParametrizationMCJet(int ptype, vector<double> parameters):type(ptype),p(parameters){};
    double value(double e) const {
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
	  double etConstantResponse=5.;//Beneath this corrected Et response is held constant
	
	  if ( e<etConstantResponse ) JetCalibrEt=etConstantResponse;  
	  
	  for (int ie=0; ie<10; ie++) { //10 iterations garantees convergence
	    
	    if ( JetCalibrEt<etConstantResponse ) JetCalibrEt=etConstantResponse; 
	    
	    if (JetCalibrEt < p[1]) {
	      koef = p[3]*sqrt(fabs(JetCalibrEt +p[4])) + p[5];
	    }
	    
	    else if (JetCalibrEt < p[2]) {
	      koef = w1 + JetCalibrEt*w2;
	    }
	    
	    else if (JetCalibrEt > p[2]) {
	      koef = p[6]/(sqrt(fabs(p[7]*JetCalibrEt + p[8]))) + p[9];
	    }
	    
	    if(koef<0.1)koef=0.1;
	    JetCalibrEt = x / koef;
	    
	  }
	  
	  enew=e/koef;
	  
	  break;
	}
      default:
	std::cerr << "JetCalibratorMCJet: Error: unknown parametrization type '"
		  << type << "'" << std::endl;
	break;
      }
      return enew;
    }
    
  private:
    int type;
    std::vector<double> p;
  };
  
  /// Calibration parameters
  class   JetCalibrationParameterSetMCJet{
  public:
    JetCalibrationParameterSetMCJet(string tag);
    int neta(){return etavector.size();}
    double eta(int ieta){return etavector[ieta];}
    int type(int ieta){return typevector[ieta];}
    const vector<double>& parameters(int ieta){return pars[ieta];}
    bool valid(){return etavector.size();}
    
  private:
    
    std::vector<double> etavector;
    std::vector<int> typevector;
    std::vector< std::vector<double> > pars;
  };
  JetCalibrationParameterSetMCJet::JetCalibrationParameterSetMCJet(const std::string& fDataFile){

    std::ifstream in( fDataFile.c_str() );
    
    //  if ( f1.isLocal() ){
    if (debug) cout << " Start to read file "<<file<<endl;
    string line;
    while( std::getline( in, line)){
      if(!line.size() || line[0]=='#') continue;
      istringstream linestream(line);
      double par;
      int type;
      linestream>>par>>type;
      
      if (debug) cout<<" Parameter eta = "<<par<<" Type= "<<type<<endl;
      
      etavector.push_back(par);
      typevector.push_back(type);
      pars.push_back(vector<double>());
      while(linestream>>par)pars.back().push_back(par);
    }
  }
  
} // namespace

MCJetCorrector::MCJetCorrector (const std::string& fDataFile) {
  init (fDataFile);
}

void MCJetCorrector::init (const std::string& fDataFile) {
  // clean up map if not empty
  for (ParametersMap::iterator it = mParametrization.begin (); it != mParametrization.end (); it++) {
    delete it->second;
  }
  JetCalibrationParameterSetMCJet pset (fDataFile);
  if(pset.valid()){
    for (int ieta=0; ieta < pset.neta(); ieta++) {
      mParametrization [pset.eta(ieta)]= new ParametrizationMCJet (pset.type(ieta), pset.parameters(ieta));
    }
  }
  else {
    std::cerr << "MCJetCorrector: calibration = " << fDataFile
	      << " not found! Cannot apply any correction ..." << std::endl;
  }
}

MCJetCorrector::~MCJetCorrector () {
  // clean up map
  for (ParametersMap::iterator it = mParametrization.begin (); it != mParametrization.end (); it++) {
    delete it->second;
  }
} 

virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE, ) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEtaPhiE (p4.Pt(), p4.Eta(), p4.Phi(), p4.E());
}


double MCJetCorrector::correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE, ) const {
  if (mParametrization.empty()) {return 1;}
  
  double et=fE / cosh (fEta);
  double eta=fabs (fEta);
  
  
  if (debug) cout<<" Et and eta of jet "<<et<<" "<<eta<<endl;

  double etnew;
  ParametersMap::const_iterator ip=mParametrization.upper_bound(eta);
  if (ip==mParametrization.begin()) { 
    etnew=ip->second->value(et); 
  }
  else if (ip==mParametrization.end()) {
    etnew=(--ip)->second->value(et);
  }
  else {
    double et2=ip->second->value(et);
    etnew=et2;
  }
	 
  if (debug) cout<<" The new energy found "<<etnew<<" "<<et<<endl;
  
  return etnew/et;
}
