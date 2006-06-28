#ifndef JetCalibratorJetParton_h
#define JetCalibratorJetParton_h

///
/// jet parton energy corrections
///

#include <map>
#include <string>
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CLHEP/Vector/LorentzVector.h"

using namespace std;

class UserPartonMixture;
class ParametrizationJetParton;
class CaloJet;

class JetCalibratorJetParton
{
public:  
  JetCalibratorJetParton(); 
  virtual ~JetCalibratorJetParton();
  
  reco::CaloJet applyCorrection (const reco::CaloJet& fJet);
   
  void setParameters(std::string aCalibrationType, double aJetFinderRadius, int aPartonMixture);
  
private:

  std::map<double,ParametrizationJetParton *> parametrization;

  std::string theJetPartonCalibrationType;
  int thePartonMixture;
  double theJetFinderRadius;
};
#endif
