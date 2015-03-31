///////////////////////////////////////////////////////////////////////////////
// File: FastHFShowerLibrary.cc
// Description: Shower library for Very forward hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "FastSimulation/ShowerDevelopment/interface/FastHFShowerLibrary.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Randomize.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

// STL headers 
#include <vector>
#include <iostream>

//#define DebugLog

FastHFShowerLibrary::FastHFShowerLibrary(edm::ParameterSet const & p) : fast(p)
{
  edm::ParameterSet m_HS   = p.getParameter<edm::ParameterSet>("HFShowerLibrary");
  applyFidCut              = m_HS.getParameter<bool>("ApplyFiducialCut");
}

FastHFShowerLibrary::~FastHFShowerLibrary() 
{
 if(hfshower)         delete hfshower;
 if(numberingScheme)  delete numberingScheme;
 if(numberingFromDDD) delete numberingFromDDD;
}

void const FastHFShowerLibrary::initHFShowerLibrary(const edm::EventSetup& iSetup) {

  edm::LogInfo("FastCalorimetry") << "initHFShowerLibrary::initialization"; 

  edm::ESTransientHandle<DDCompactView> cpv;
  iSetup.get<IdealGeometryRecord>().get(cpv);

  std::string name = "HcalHits";
  hfshower = new HFShowerLibrary(name,*cpv,fast);
  numberingFromDDD = new HcalNumberingFromDDD(name, *cpv);  
  numberingScheme  = new HcalNumberingScheme();

  initRun();  
}

void FastHFShowerLibrary::initRun() {

  geantinoPDG = 0; gammaPDG = 22;
  emPDG   = 11; epPDG    = -11; nuePDG   = 12; anuePDG   = -12;
  numuPDG = 14; anumuPDG = -14; nutauPDG = 16; anutauPDG = -16;
  pi0PDG = 111; etaPDG   = 221;

#ifdef DebugLog
  edm::LogInfo("FastCalorimetry") << "HFShowerLibrary: Particle codes for e- = " 
			   << emPDG << ", e+ = " << epPDG << ", gamma = " 
			   << gammaPDG << ", pi0 = " << pi0PDG << ", eta = " 
			   << etaPDG << ", geantino = " << geantinoPDG 
			   << "\n        nu_e = " << nuePDG << ", nu_mu = " 
			   << numuPDG << ", nu_tau = " << nutauPDG 
			   << ", anti_nu_e = " << anuePDG << ", anti_nu_mu = " 
			   << anumuPDG << ", anti_nu_tau = " << anutauPDG;
#endif
}

void FastHFShowerLibrary::recoHFShowerLibrary(const FSimTrack& myTrack) {

#ifdef DebugLog
  edm::LogInfo("FastCalorimetry") << "FastHFShowerLibrary: recoHFShowerLibrary ";
#endif 

  if(!myTrack.onVFcal()) {
#ifdef DebugLog
    edm::LogInfo("FastCalorimetry") << "FastHFShowerLibrary: we should not be here ";
#endif
  }

  hitMap.clear();
  double eGen  = 1000.*myTrack.vfcalEntrance().e();                // energy in [MeV]
  double delZv = (myTrack.vfcalEntrance().vertex().Z()>0.0) ? 50.0 : -50.0;
  G4ThreeVector vertex( 10.*myTrack.vfcalEntrance().vertex().X(),
                        10.*myTrack.vfcalEntrance().vertex().Y(),
                        10.*myTrack.vfcalEntrance().vertex().Z()+delZv); // in [mm]

  G4ThreeVector direction(myTrack.vfcalEntrance().Vect().X(),
                          myTrack.vfcalEntrance().Vect().Y(),
                          myTrack.vfcalEntrance().Vect().Z());

  bool ok;
  double weight = 1.0;                     // rad. damage 
  int parCode   = myTrack.type();

  std::vector<HFShowerLibrary::Hit> hits =
              getHits(vertex, direction, parCode, eGen, ok, weight, false);

  for (unsigned int i=0; i<hits.size(); ++i) {
    G4ThreeVector pos = hits[i].position;
    int depth         = hits[i].depth;
    double time       = hits[i].time;

    if (isItinFidVolume (pos)) {     
      int det = 5;
      int lay = 1;
      uint32_t id = 0;
      HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det, pos, depth, lay);
      id = numberingScheme->getUnitID(tmp);

      CaloHitID current_id(id,time,myTrack.id());
      std::map<CaloHitID,float>::iterator cellitr;
      cellitr = hitMap.find(current_id);
      if(cellitr==hitMap.end()) {
         hitMap.insert(std::pair<CaloHitID,float>(current_id,1.0));
      } else {
         cellitr->second += 1.0;
      }
    }  // end of isItinFidVolume check 
  } // end loop over hits

}

bool FastHFShowerLibrary::isItinFidVolume (G4ThreeVector& hitPoint) {
  bool flag = true;
  if (applyFidCut) {
    int npmt = HFFibreFiducial::PMTNumber(hitPoint);
#ifdef DebugLog
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary::isItinFidVolume:#PMT= " 
                                    << npmt << " for hit point " << hitPoint;
#endif
    if (npmt <= 0) flag = false;
  }
#ifdef DebugLog
    edm::LogInfo("FastCalorimetry") << "HFShowerLibrary::isItinFidVolume: point " 
                                    << hitPoint << " return flag " << flag;
#endif
  return flag;
}

std::vector<HFShowerLibrary::Hit> FastHFShowerLibrary::getHits(G4ThreeVector & hitPoint,
                                  G4ThreeVector & momDir, int parCode, double pin, 
                                  bool & ok, double weight, bool onlyLong) {

  std::vector<HFShowerLibrary::Hit> hit;
  ok = false;
  if (parCode == pi0PDG || parCode == etaPDG || parCode == nuePDG ||
      parCode == numuPDG || parCode == nutauPDG || parCode == anuePDG ||
      parCode == anumuPDG || parCode == anutauPDG || parCode == geantinoPDG) 
    return hit;

  ok = true;

  double tSlice = 0.1*hitPoint.mag()/29.98;
  hfshower->fillHits(hitPoint,momDir,hit,parCode,pin,ok,weight,onlyLong,tSlice);
  return hit;
}

