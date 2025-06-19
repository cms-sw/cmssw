#ifndef FastHFShowerLibrary_H
#define FastHFShowerLibrary_H
///////////////////////////////////////////////////////////////////////////////
// File: FastHFShowerLibrary.h
// Description: Gets information from a shower library
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FastSimulation/CalorimeterProperties/interface/CalorimetryConsumer.h"
#include "FastSimulation/Utilities/interface/FamosDebug.h"

#include "DataFormats/Math/interface/Vector3D.h"

#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "G4ThreeVector.hh"

//ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include <string>
#include <memory>
#include <map>

class FSimEvent;
class FSimTrack;
class HFShowerLibrary;
class RandomEngineAndDistribution;

class HFHitMaker {
public:
  const std::map<CaloHitID, float>& hitMap() const { return hitMap_; }
  std::map<CaloHitID, float>& hitMap() { return hitMap_; }

private:
  std::map<CaloHitID, float> hitMap_;
};

class FastHFShowerLibrary {
public:
  // Constructor and Destructor
  FastHFShowerLibrary(edm::ParameterSet const&, const edm::EventSetup&, const CalorimetryConsumer&);
  ~FastHFShowerLibrary() {}

public:
  void initHFShowerLibrary(const edm::EventSetup&);
  void recoHFShowerLibrary(const FSimTrack& myTrack, HFHitMaker* hitMaker) const;
  void modifyDepth(HcalNumberingFromDDD::HcalID& id) const;

  static void setRandom(const RandomEngineAndDistribution*);

private:
  const edm::ParameterSet fast;
  std::unique_ptr<HFShowerLibrary> hfshower;
  std::unique_ptr<HcalNumberingFromDDD> numberingFromDDD;
  const HcalDDDSimConstants* hcalConstants;
  HcalNumberingScheme numberingScheme;

  bool applyFidCut;
  std::string name;
};
#endif
