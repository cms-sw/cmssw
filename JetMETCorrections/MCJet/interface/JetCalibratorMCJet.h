#ifndef JetCalibratorMCJet_h
#define JetCalibratorMCJet_h

#include <map>
#include <string>
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetObjects/interface/CaloJetCollection.h"
#include "CLHEP/Vector/LorentzVector.h"


class ParametrizationMCJet;

///
/// jet energy corrections from MCjet calibration
///

class JetCalibratorMCJet
{
public:  

  JetCalibratorMCJet() : parametrization(),
                         theCalibrationType() {};
  virtual ~JetCalibratorMCJet();
  void run(const CaloJetCollection*, std::vector<HepLorentzVector> &);    
  void setParameters(std::string );
   
private:
  
  std::map<double,ParametrizationMCJet *> parametrization;

  std::string theCalibrationType;
};

#endif
