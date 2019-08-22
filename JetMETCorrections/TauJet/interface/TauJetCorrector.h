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

class TauJetCorrector : public JetCorrector {
public:
  TauJetCorrector(const edm::ParameterSet& fParameters);
  ~TauJetCorrector() override;
  double correction(const LorentzVector& fJet) const override;
  double correction(const reco::Jet&) const override;

  void setParameters(std::string, int);
  /// if correction needs event information
  bool eventRequired() const override { return false; }

private:
  class ParametrizationTauJet {
  public:
    ParametrizationTauJet(int ptype, const std::vector<double>& x, double u) {
      type = ptype;
      theParam[type] = x;
      theEtabound[type] = u;
      //cout<<"ParametrizationTauJet "<<type<<" "<<u<<endl;
    };

    double value(double, double) const;

  private:
    int type;
    std::map<int, std::vector<double> > theParam;
    std::map<int, double> theEtabound;
  };

  typedef std::map<double, ParametrizationTauJet*> ParametersMap;
  ParametersMap parametrization;
  int type;
};

#endif
