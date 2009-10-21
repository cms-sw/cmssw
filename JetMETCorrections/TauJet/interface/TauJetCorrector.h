#ifndef TauJetCorrector_h
#define TauJetCorrector_h
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include <map>
#include <string>
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

///
/// jet energy corrections from Taujet calibration
///

class TauJetCorrector: public JetCorrector
{
public:  

  TauJetCorrector(const edm::ParameterSet& fParameters);
  virtual ~TauJetCorrector();
  virtual double  correction (const LorentzVector& fJet) const;
  virtual double  correction(const reco::Jet&) const;

  void setParameters(std::string, int);
  /// if correction needs event information
  virtual bool eventRequired () const {return false;}
   
private:

  class ParametrizationTauJet{
      public:
	ParametrizationTauJet(int ptype, std::vector<double> x, double u) {
    		type=ptype;
    		theParam[type] = x;
    		theEtabound[type] = u;
    		//cout<<"ParametrizationTauJet "<<type<<" "<<u<<endl;
  	};

  	double value(double, double) const;

      private:
      	int type;
      	std::map<int, std::vector<double> > theParam;
      	std::map<int,double> theEtabound;
  };

  typedef std::map<double,ParametrizationTauJet *> ParametersMap;
  ParametersMap parametrization;
  int type;
};

#endif
