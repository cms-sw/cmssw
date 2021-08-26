#ifndef FastHFShowerLibrary_H
#define FastHFShowerLibrary_H
///////////////////////////////////////////////////////////////////////////////
// File: FastHFShowerLibrary.h
// Description: Gets information from a shower library
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

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

class FastHFShowerLibrary {
public:
  // Constructor and Destructor
  FastHFShowerLibrary(edm::ParameterSet const&, edm::ConsumesCollector&&);
  ~FastHFShowerLibrary() {}

public:
  void const initHFShowerLibrary(const edm::EventSetup&);
  void recoHFShowerLibrary(const FSimTrack& myTrack);
  void modifyDepth(HcalNumberingFromDDD::HcalID& id);
  const std::map<CaloHitID, float>& getHitsMap() { return hitMap; };

  void SetRandom(const RandomEngineAndDistribution*);

private:
  const edm::ParameterSet fast;
  std::unique_ptr<HFShowerLibrary> hfshower;
  std::unique_ptr<HcalNumberingFromDDD> numberingFromDDD;
  const HcalDDDSimConstants* hcalConstants;
  HcalNumberingScheme numberingScheme;

  std::map<CaloHitID, float> hitMap;

  bool applyFidCut;
  std::string name;

  const edm::ESGetToken<HcalDDDSimConstants, HcalSimNumberingRecord> hcalDDDSimConstantsESToken_;
  const edm::ESGetToken<HcalSimulationConstants, HcalSimNumberingRecord> hcalSimulationConstantsESToken_;
};
#endif
