#ifndef FastHFShowerLibrary_H
#define FastHFShowerLibrary_H
///////////////////////////////////////////////////////////////////////////////
// File: FastHFShowerLibrary.h
// Description: Gets information from a shower library
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/Utilities/interface/FamosDebug.h"

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "G4ThreeVector.hh"

//ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include <string>
#include <memory>
#include <map>

class DDCompactView;    
class FSimEvent;
class FSimTrack;
class HFShowerLibrary;

class FastHFShowerLibrary {
  
public:

// Constructor and Destructor
  FastHFShowerLibrary(edm::ParameterSet const & p);
  ~FastHFShowerLibrary();

public:

  void                initRun();
  void       const    initHFShowerLibrary(const edm::EventSetup& );
  void                recoHFShowerLibrary(const FSimTrack &myTrack);

  std::vector<HFShowerLibrary::Hit> getHits(G4ThreeVector & p, G4ThreeVector & v,
                                            int parCode, double parEnergy, bool &ok, 
                                            double weight, bool onlyLong=false);

  const std::map<CaloHitID,float>& getHitsMap() { return hitMap; };

  bool                isItinFidVolume (G4ThreeVector&);

private:

  HcalNumberingScheme  *numberingScheme;
  HcalNumberingFromDDD *numberingFromDDD;

  std::map<CaloHitID,float> hitMap;

  bool applyFidCut;
  int  emPDG, epPDG, gammaPDG;
  int  pi0PDG, etaPDG, nuePDG, numuPDG, nutauPDG;
  int  anuePDG, anumuPDG, anutauPDG, geantinoPDG;

  HFShowerLibrary* hfshower;
  edm::ParameterSet const fast;
  std::string name;
};
#endif
