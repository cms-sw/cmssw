//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleZSPJetCorrector.cc,v 1.2 2009/11/17 17:38:30 kodolova Exp $
//
// ZSP Jet Corrector
//
#include "CondFormats/JetMETObjects/interface/SimpleZSPJetCorrector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

using namespace std;


namespace zsp {

  bool debug = false;

  /// Parametrization itself
  class ParametrizationZSPJet {
  public:
    ParametrizationZSPJet(int ptype, vector<double> parameters):type(ptype),p(parameters){};
    double value(double e) const {
      double enew(e);
      switch(type){
      case 1:
        {
         if (debug) std::cout<<" Case 1 p: "<<p[0]<<" "<<p[1]<<" "<<p[2]<<" "<<p[3]<<std::endl;
         if(p.size()>4) { 
                          if (debug) std::cout<<" Wrong parametrization type: check the input file in CondFormats/JetMETObjects/data " <<std::endl;
                          break;
                        }
   
          double koef = 1. - p[1] + p[2]/((e+p[3])*(e+p[3]));
          enew=e/koef;
          break;
        }
      case 2:
       { 
         if (debug) std::cout<<" Case 2 p: "<<p[0]<<" "<<p[1]<<" "<<p[2]<<" "<<p[3]<<" "<<p[4]<<std::endl;
         if(p.size()<5) { 
                         if (debug) std::cout<<" Wrong parametrization type: check the input file in CondFormats/JetMETObjects/data " <<std::endl;
                         break;
                        }   
         double koef = 1. - p[1]*exp(p[2]*e)-p[3]*exp(p[4]*e);  
         enew=e/koef; 
        break;
      }
       default:
      {
        if (debug) std::cout<<" Wrong parametrization type: check the input file in CondFormats/JetMETObjects/data: No parametrization " <<std::endl; 
        break;
      }
     } // end switch {type}
      return enew;
    }
    int getNPU(){ int npu = (int)p[0]; return npu; }    

  private:
    int type;
    std::vector<double> p;
  };
  
  /// Calibration parameters
  class   JetCalibrationParameterSetZSPJet{
  public:
    JetCalibrationParameterSetZSPJet(const std::string& tag);
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
  JetCalibrationParameterSetZSPJet::JetCalibrationParameterSetZSPJet(const std::string& fDataFile){

    std::ifstream in( fDataFile.c_str() );
    
    //  if ( f1.isLocal() ){
    if (zsp::debug) cout << " Start to read file "<<fDataFile<<endl;
    string line;
    while( std::getline( in, line)){
      if(!line.size() || line[0]=='#') continue;
      istringstream linestream(line);
      double par;
      int type;
      linestream>>par>>type;
      
      if (zsp::debug) cout<<" Parameter eta = "<<par<<" Type= "<<type<<endl;
      
      etavector.push_back(par);
      typevector.push_back(type);
      pars.push_back(vector<double>());
      while(linestream>>par)pars.back().push_back(par);
    }
  }
  
} // namespace

SimpleZSPJetCorrector::SimpleZSPJetCorrector ()
  : mParametrization (0)
{}

SimpleZSPJetCorrector::SimpleZSPJetCorrector (const std::string& fDataFile) 
  : mParametrization (0)
{
  init (fDataFile);
}

void SimpleZSPJetCorrector::init (const std::string& fDataFile) {
  // clean up map if not empty
  if (mParametrization) {
    for (zsp::ZSPParametersMap::iterator it = mParametrization->begin (); it != mParametrization->end (); it++) {
      delete it->second;
    }
    delete mParametrization;
  }
  mParametrization = new zsp::ZSPParametersMap ();
  zsp::JetCalibrationParameterSetZSPJet pset (fDataFile);
  if(pset.valid()){
    for (int ieta=0; ieta < pset.neta(); ieta++) {
      (*mParametrization) [pset.eta(ieta)]= new zsp::ParametrizationZSPJet (pset.type(ieta), pset.parameters(ieta));
    }
  }
  else {
    std::cerr << "SimpleZSPJetCorrector: calibration = " << fDataFile
	      << " not found! Cannot apply any correction ..." << std::endl;
  }
}

SimpleZSPJetCorrector::~SimpleZSPJetCorrector () {
  // clean up map
  for (zsp::ZSPParametersMap::iterator it = mParametrization->begin (); it != mParametrization->end (); it++) {
    delete it->second;
  }
  delete mParametrization;
} 

double SimpleZSPJetCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEtaPhiE (p4.Pt(), p4.Eta(), p4.Phi(), p4.E());
}

double SimpleZSPJetCorrector::correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const {
  double costhetainv = cosh (fEta);
  return correctionEtEtaPhiP (fE/costhetainv, fEta, fPhi, fPt*costhetainv);
}

double SimpleZSPJetCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  if (!mParametrization || mParametrization->empty()) return 1;
  
  double et=fEt;
  double eta=fabs (fEta);
  
  
  if (zsp::debug) cout<<" Et and eta of jet "<<et<<" "<<eta<<endl;

  double etnew;
  zsp::ZSPParametersMap::const_iterator ip=mParametrization->upper_bound(eta);
  if (ip==mParametrization->begin()) { 
    etnew=ip->second->value(et); 
  }
  else if (ip==mParametrization->end()) {
    etnew=(--ip)->second->value(et);
  }
  else {
    double et2=ip->second->value(et);
    etnew=et2;
  }
	 
  if (zsp::debug) cout<<" The new energy found "<<etnew<<" "<<et<<endl;
  
  return etnew/et;
}
