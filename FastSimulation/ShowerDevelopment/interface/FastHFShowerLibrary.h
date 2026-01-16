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
#include "FastSimulation/CaloHitMakers/interface/CaloHitMap.h"

#include "DataFormats/Math/interface/Vector3D.h"

#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
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

class FSimEvent;
class FSimTrack;
class HFShowerLibrary;
class RandomEngineAndDistribution;

class FastHFShowerLibrary {
public:
  // Constructor and Destructor
  FastHFShowerLibrary(edm::ParameterSet const&, const edm::EventSetup&, const CalorimetryConsumer&);
  ~FastHFShowerLibrary() {}

public:
  std::unique_ptr<HFShowerLibrary> initHFShowerLibrary() const;
  void recoHFShowerLibrary(const FSimTrack& myTrack, CaloHitMap& hitMap, HFShowerLibrary* hfshower) const;
  void modifyDepth(HcalNumberingFromDDD::HcalID& id) const;

  static void setRandom(const RandomEngineAndDistribution*);

private:
  const edm::ParameterSet fast;
  std::unique_ptr<HcalNumberingFromDDD> numberingFromDDD;
  const HcalDDDSimConstants* hcalConstants;
  const HcalSimulationConstants* hsps;
  HcalNumberingScheme numberingScheme;

  bool applyFidCut;
  std::string name;
};
#endif
