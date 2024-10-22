/* 
Functions to give the details of parent track of Generator Tracks

Authors:  Seema Sharma, Sunanda Banerjee
Created: August 2009
*/

#ifndef CalibrationIsolatedParticlesGenSimInfo_h
#define CalibrationIsolatedParticlesGenSimInfo_h

// system include files
#include <memory>
#include <map>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"

namespace spr {

  struct genSimInfo {
    genSimInfo() {
      maxNearP = -1.0;
      cHadronEne = nHadronEne = eleEne = muEne = photonEne = 0.0;
      isChargedIso = true;
      for (int i = 0; i < 3; ++i)
        cHadronEne_[i] = 0.0;
    }
    double maxNearP;
    double cHadronEne, nHadronEne, eleEne, muEne, photonEne;
    bool isChargedIso;
    double cHadronEne_[3];
  };

  void eGenSimInfo(const DetId& coreDet,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   const CaloGeometry* geo,
                   const CaloTopology* caloTopology,
                   int ieta,
                   int iphi,
                   spr::genSimInfo& info,
                   bool debug = false);

  void eGenSimInfo(const DetId& coreDet,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   const CaloGeometry* geo,
                   const CaloTopology* caloTopology,
                   double dR,
                   const GlobalVector& trackMom,
                   spr::genSimInfo& info,
                   bool debug = false);

  void eGenSimInfo(const DetId& coreDet,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   const CaloGeometry* geo,
                   const CaloTopology* caloTopology,
                   int ieta,
                   int iphi,
                   spr::genSimInfo& info,
                   bool debug = false);

  void eGenSimInfo(const DetId& coreDet,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   const CaloGeometry* geo,
                   const CaloTopology* caloTopology,
                   double dR,
                   const GlobalVector& trackMom,
                   spr::genSimInfo& info,
                   bool debug = false);

  void hGenSimInfo(const DetId& coreDet,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   const HcalTopology* topology,
                   int ieta,
                   int iphi,
                   spr::genSimInfo& info,
                   bool includeHO = false,
                   bool debug = false);

  void hGenSimInfo(const DetId& coreDet,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   const CaloGeometry* geo,
                   const HcalTopology* topology,
                   double dR,
                   const GlobalVector& trackMom,
                   spr::genSimInfo& info,
                   bool includeHO = false,
                   bool debug = false);

  void hGenSimInfo(const DetId& coreDet,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   const HcalTopology* topology,
                   int ieta,
                   int iphi,
                   spr::genSimInfo& info,
                   bool includeHO = false,
                   bool debug = false);

  void hGenSimInfo(const DetId& coreDet,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   const CaloGeometry* geo,
                   const HcalTopology* topology,
                   double dR,
                   const GlobalVector& trackMom,
                   spr::genSimInfo& info,
                   bool includeHO = false,
                   bool debug = false);

  void cGenSimInfo(std::vector<DetId>& vdets,
                   HepMC::GenEvent::particle_const_iterator trkItr,
                   std::vector<spr::propagatedGenTrackID>& trackIds,
                   bool ifECAL,
                   spr::genSimInfo& info,
                   bool debug = false);

  void cGenSimInfo(std::vector<DetId>& vdets,
                   reco::GenParticleCollection::const_iterator trkItr,
                   std::vector<spr::propagatedGenParticleID>& trackIds,
                   bool ifECAL,
                   spr::genSimInfo& info,
                   bool debug = false);

  void cGenSimInfo(int charge, int pdgid, double p, spr::genSimInfo& info, bool debug = false);
}  // namespace spr

#endif
