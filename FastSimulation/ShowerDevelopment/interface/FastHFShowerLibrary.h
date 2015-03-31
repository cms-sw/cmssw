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
#include "FastSimulation/ShowerDevelopment/interface/FastHFFibre.h"
#include "FastSimulation/Utilities/interface/FamosDebug.h"
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/Math/interface/Vector3D.h"

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

class FastHFShowerLibrary {
  
public:

// Constructor and Destructor
  FastHFShowerLibrary(edm::ParameterSet const & p);
  ~FastHFShowerLibrary();

public:

  struct Hit {
    Hit() {}
    G4ThreeVector             position;
    int                       depth;
    double                    time;
  };

  void                initRun();
  void       const    initHFShowerLibrary(const edm::EventSetup& );
  void                recoHFShowerLibrary(const FSimTrack &myTrack);
  std::vector<Hit>    getHits(const G4ThreeVector & p, const G4ThreeVector & v,
                              int parCode, double parEnergy, bool &ok, 
                              double weight, bool onlyLong=false);
  const std::map<CaloHitID,float>& getHitsMap() { return hitMap; };
  bool                isItinFidVolume (G4ThreeVector&);

protected:

  bool                rInside(double r);
  void                getRecord(int, int);
  void                loadEventInfo(TBranch *);
  void                interpolate(int, double);
  void                extrapolate(int, double);
  void                storePhoton(int j);
  std::vector<double> getDDDArray(const std::string&, const DDsvalues_type&,
				  int&);

private:

  FastHFFibre *       fibre;
  TFile *             hf;
  TBranch             *emBranch, *hadBranch;

  HcalNumberingScheme  *numberingScheme;
  HcalNumberingFromDDD *numberingFromDDD;
  std::map<CaloHitID,float> hitMap;
  double              cFibre;

  bool                verbose, applyFidCut, newForm;
  int                 nMomBin, totEvents, evtPerBin;
  float               libVers, listVersion; 
  std::vector<double> pmom;

  double              probMax, backProb;
  double              dphi, rMin, rMax;
  std::vector<double> gpar;

  int                 emPDG, epPDG, gammaPDG;
  int                 pi0PDG, etaPDG, nuePDG, numuPDG, nutauPDG;
  int                 anuePDG, anumuPDG, anutauPDG, geantinoPDG;

  int                 npe;
  HFShowerPhotonCollection pe;
  HFShowerPhotonCollection* photo;
  HFShowerPhotonCollection photon;

};
#endif
