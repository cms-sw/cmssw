#ifndef JetCalibratorTauJet_h
#define JetCalibratorTauJet_h

#include <map>
#include <string>
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CLHEP/Vector/LorentzVector.h"


class ParametrizationTauJet;
class CaloJet;

///
/// jet energy corrections from Taujet calibration
///

class JetCalibratorTauJet
{
public:  

  JetCalibratorTauJet() : parametrization(),
                         theTauJetCalibrationType() {};
  virtual ~JetCalibratorTauJet();
  reco::CaloJet applyCorrection (const reco::CaloJet& fJet);
  void setParameters(std::string, int);
   
private:
  
  std::map<double,ParametrizationTauJet *> parametrization;

  std::string theTauJetCalibrationType;
  int type;
};

#endif
