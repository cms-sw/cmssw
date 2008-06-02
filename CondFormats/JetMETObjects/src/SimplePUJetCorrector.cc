//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimplePUJetCorrector.cc,v 1.6 2007/12/08 01:55:41 fedor Exp $
//
// PU Jet Corrector
//
#include "CondFormats/JetMETObjects/interface/SimplePUJetCorrector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Math/PtEtaPhiE4D.h"
#include "Math/PxPyPzE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

using namespace std;


namespace {
  bool debug = false;

  /// Parametrization itself
  class ParametrizationPUJet {
  public:
    ParametrizationPUJet(vector<double> parameters):p(parameters){};
    double value(double e, double eta) const {
      double enew(e);
	  double koef = 1; 
	  enew=e/koef;
	  
      return enew;
    }
    
  private:
    std::vector<double> p;
  };
  
  /// Calibration parameters
  class   JetCalibrationParameterSetPUJet{
  public:
    JetCalibrationParameterSetPUJet(const std::string& tag);
    int nlumi(){return lumivector.size();}
//    double eta(int ieta){return etavector[ieta];}
    int lumi(int ieta){return lumivector[ieta];}
    const vector<double>& parameters(int ieta){return pars[ieta];}
    bool valid(){return lumivector.size();}
    
  private:
    
//    std::vector<double> etavector;
    std::vector<int> lumivector;
    std::vector< std::vector<double> > pars;
  };
  JetCalibrationParameterSetPUJet::JetCalibrationParameterSetPUJet(const std::string& fDataFile){

    std::ifstream in( fDataFile.c_str() );
    
    //  if ( f1.isLocal() ){
    if (debug) cout << " Start to read file "<<fDataFile<<endl;
    string line;
    while( std::getline( in, line)){
      if(!line.size() || line[0]=='#') continue;
      istringstream linestream(line);
      double par;
      int type;
      linestream>>type;
      
      if (debug) cout<<" Type= "<<type<<endl;
      
//      etavector.push_back(par);
      lumivector.push_back(type);
      pars.push_back(vector<double>());
      while(linestream>>par)pars.back().push_back(par);
    }
  }
  
} // namespace

SimplePUJetCorrector::SimplePUJetCorrector ()
  : mParametrization (0)
{}

SimplePUJetCorrector::SimplePUJetCorrector (const std::string& fDataFile) 
  : mParametrization (0)
{
  init (fDataFile);
}

void SimplePUJetCorrector::init (const std::string& fDataFile) {
  // clean up map if not empty
  if (mParametrization) {
    for (ParametersMap::iterator it = mParametrization->begin (); it != mParametrization->end (); it++) {
      delete it->second;
    }
    delete mParametrization;
  }
  mParametrization = new ParametersMap ();
  JetCalibrationParameterSetPUJet pset (fDataFile);
  if(pset.valid()){
    for (int ieta=0; ieta < pset.nlumi(); ieta++) {
      (*mParametrization) [pset.lumi(ieta)]= new ParametrizationPUJet (pset.parameters(ieta));
    }
  }
  else {
    std::cerr << "SimplePUJetCorrector: calibration = " << fDataFile
	      << " not found! Cannot apply any correction ..." << std::endl;
  }
}

SimplePUJetCorrector::~SimplePUJetCorrector () {
  // clean up map
  for (ParametersMap::iterator it = mParametrization->begin (); it != mParametrization->end (); it++) {
    delete it->second;
  }
  delete mParametrization;
} 

double SimplePUJetCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEtaPhiE (p4.Pt(), p4.Eta(), p4.Phi(), p4.E());
}

double SimplePUJetCorrector::correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const {
  double costhetainv = cosh (fEta);
  return correctionEtEta (fE/costhetainv, fEta);
}

double SimplePUJetCorrector::correctionEtEta (double fEt, double fEta) const {
  if (!mParametrization || mParametrization->empty()) return 1;
  
  double et=fEt;
  double eta=fabs (fEta);
  
  
  if (debug) cout<<" Et and eta of jet "<<et<<" "<<eta<<endl;
  int lumi_value = 1;
  double etnew;
  ParametersMap::const_iterator ip=mParametrization->upper_bound(lumi_value);
  if (ip==mParametrization->begin()) { 
    etnew=ip->second->value(et,eta); 
  }
  else if (ip==mParametrization->end()) {
    etnew=(--ip)->second->value(et,eta);
  }
  else {
    double et2=ip->second->value(et,eta);
    etnew=et2;
  }
	 
  if (debug) cout<<" The new energy found "<<etnew<<" "<<et<<endl;
  
  return etnew/et;
}
