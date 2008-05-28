#ifndef TauJetCorrector_h
#define TauJetCorrector_h
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include <map>
#include <string>
#include <vector>
namespace edm {
  class ParameterSet;
}

namespace {
class ParametrizationTauJet;
}

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
  typedef std::map<double,ParametrizationTauJet *> ParametersMap;
  ParametersMap parametrization;
  int type;
};

#endif
