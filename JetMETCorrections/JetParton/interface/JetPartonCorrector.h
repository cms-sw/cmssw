#ifndef JetPartonCorrector_h
#define JetPartonCorrector_h

///
/// jet parton energy corrections
///

#include <map>
#include <string>
#include <vector>
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
namespace edm {
  class ParameterSet;
}

namespace JetPartonNamespace {
class ParametrizationJetParton;
class UserPartonMixture;
}

class JetPartonCorrector : public JetCorrector
{
public:  
  JetPartonCorrector(const edm::ParameterSet& fConfig); 
  virtual ~JetPartonCorrector();
  
  virtual double   correction (const LorentzVector& fJet) const;
   
  void setParameters(std::string aCalibrationType, double aJetFinderRadius, int aPartonMixture);

  /// if correction needs event information
  virtual bool eventRequired () const {return false;}
  
private:

  typedef std::map<double,JetPartonNamespace::ParametrizationJetParton *> ParametersMap;
  ParametersMap parametrization;
  int thePartonMixture;
  double theJetFinderRadius;

};
#endif
