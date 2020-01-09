#include "GeneratorInterface/GenFilters/plugins/doubleEMEnrichingFilterAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CLHEP/Vector/LorentzVector.h"

using namespace edm;
using namespace std;

doubleEMEnrichingFilterAlgo::doubleEMEnrichingFilterAlgo(const edm::ParameterSet& iConfig) {
  //set constants
  FILTER_TKISOCUT_ = 2;
  FILTER_CALOISOCUT_ = 2;
  FILTER_ETA_MIN_ = 0;
  FILTER_ETA_MAX_ = 2.5;
  ECALBARRELMAXETA_ = 1.479;
  ECALBARRELRADIUS_ = 129.0;
  ECALENDCAPZ_ = 304.5;

  // from bctoe
  eTThreshold_ = (float)iConfig.getParameter<double>("eTThreshold");

  isoGenParETMin_ = (float)iConfig.getParameter<double>("isoGenParETMin");
  isoGenParConeSize_ = (float)iConfig.getParameter<double>("isoGenParConeSize");
  clusterThreshold_ = (float)iConfig.getParameter<double>("clusterThreshold");
  seedThreshold_ = (float)iConfig.getParameter<double>("seedThreshold");
  isoConeSize_ = (float)iConfig.getParameter<double>("isoConeSize");
  hOverEMax_ = (float)iConfig.getParameter<double>("hOverEMax");
  tkIsoMax_ = (float)iConfig.getParameter<double>("tkIsoMax");
  caloIsoMax_ = (float)iConfig.getParameter<double>("caloIsoMax");
  requireTrackMatch_ = iConfig.getParameter<bool>("requireTrackMatch");
  genParSource_ = iConfig.getParameter<edm::InputTag>("genParSource");
}

doubleEMEnrichingFilterAlgo::~doubleEMEnrichingFilterAlgo() {}

bool doubleEMEnrichingFilterAlgo::filter(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //----------

  Handle<reco::GenParticleCollection> genParsHandle;
  iEvent.getByLabel(genParSource_, genParsHandle);
  reco::GenParticleCollection genPars = *genParsHandle;

  bool result = false;
  int BCtoEgood = 0;
  int PhotElecgood = 0;
  int IsoGenPargood = 0;

  // cleaning
  sel1seeds.clear();
  sel2seeds.clear();
  selBCtoEseeds.clear();

  //bending of traj. of charged particles under influence of B-field
  std::vector<reco::GenParticle> genParsCurved = applyBFieldCurv(genPars, iSetup);

  PhotElecgood = filterPhotonElectronSeed(clusterThreshold_,
                                          seedThreshold_,
                                          isoConeSize_,
                                          hOverEMax_,
                                          tkIsoMax_,
                                          caloIsoMax_,
                                          requireTrackMatch_,
                                          genPars,
                                          genParsCurved);

  IsoGenPargood = filterIsoGenPar(isoGenParETMin_, isoGenParConeSize_, genPars, genParsCurved);

  /// BCtoE filter modified to store particle seeds
  for (uint32_t ig = 0; ig < genParsCurved.size(); ig++) {
    reco::GenParticle gp = genParsCurved.at(ig);
    if (gp.status() == 1 && abs(gp.pdgId()) == 11 && gp.et() > eTThreshold_ && fabs(gp.eta()) < FILTER_ETA_MAX_) {
      if (hasBCAncestors(gp)) {
        BCtoEgood++;
        selBCtoEseeds.push_back(gp);
      }
    }
  }

  // we want 2 different em candidates
  if (PhotElecgood > 1) {
    result = true;
  } else if (IsoGenPargood > 1) {
    result = true;
  } else if (BCtoEgood > 1) {
    result = true;
  } else if (PhotElecgood == 1 && IsoGenPargood == 1) {
    if ((sel1seeds.at(0).eta() != sel2seeds.at(0).eta()) && (sel1seeds.at(0).phi() != sel2seeds.at(0).phi()) &&
        (sel1seeds.at(0).et() != sel2seeds.at(0).et()))
      result = true;
  } else if (PhotElecgood == 1 && BCtoEgood == 1) {
    if ((sel1seeds.at(0).eta() != selBCtoEseeds.at(0).eta()) && (sel1seeds.at(0).phi() != selBCtoEseeds.at(0).phi()) &&
        (sel1seeds.at(0).et() != selBCtoEseeds.at(0).et()))
      result = true;
  } else if (BCtoEgood == 1 && IsoGenPargood == 1) {
    if ((selBCtoEseeds.at(0).eta() != sel2seeds.at(0).eta()) && (selBCtoEseeds.at(0).phi() != sel2seeds.at(0).phi()) &&
        (selBCtoEseeds.at(0).et() != sel2seeds.at(0).et()))
      result = true;
  }

  return result;
}

//filter that uses clustering around photon and electron seeds
//only electrons, photons, charged pions, and charged kaons are clustered
//additional requirements:

//seed threshold, total threshold, and cone size/shape are specified separately for the barrel and the endcap
int doubleEMEnrichingFilterAlgo::filterPhotonElectronSeed(float clusterthreshold,
                                                          float seedthreshold,
                                                          float isoConeSize,
                                                          float hOverEMax,
                                                          float tkIsoMax,
                                                          float caloIsoMax,
                                                          bool requiretrackmatch,
                                                          const std::vector<reco::GenParticle>& genPars,
                                                          const std::vector<reco::GenParticle>& genParsCurved) {
  float conesizeendcap = 15;

  int retval = 0;

  vector<reco::GenParticle> seeds;
  //find electron and photon seeds - must have E>seedthreshold GeV
  for (uint32_t is = 0; is < genParsCurved.size(); is++) {
    reco::GenParticle gp = genParsCurved.at(is);
    if (gp.status() != 1 || fabs(gp.eta()) > FILTER_ETA_MAX_ || fabs(gp.eta()) < FILTER_ETA_MIN_)
      continue;
    int absid = abs(gp.pdgId());
    if (absid != 11 && absid != 22)
      continue;
    if (gp.et() > seedthreshold)
      seeds.push_back(gp);
  }

  bool matchtrack = false;

  //for every seed, try to cluster stable particles about it in cone
  for (uint32_t is = 0; is < seeds.size(); is++) {
    float eTInCone = 0;   //eT associated to the electron cluster
    float tkIsoET = 0;    //tracker isolation energy
    float caloIsoET = 0;  //calorimeter isolation energy
    float hadET =
        0;  //isolation energy from heavy hadrons that goes in the same area as the "electron" - so contributes to H/E
    bool isBarrel = fabs(seeds.at(is).eta()) < ECALBARRELMAXETA_;
    for (uint32_t ig = 0; ig < genParsCurved.size(); ig++) {
      reco::GenParticle gp = genParsCurved.at(ig);
      reco::GenParticle gpUnCurv = genPars.at(ig);  //for tk isolation, p at vertex
      if (gp.status() != 1)
        continue;
      int gpabsid = abs(gp.pdgId());
      if (gp.et() < 1)
        continue;  //ignore very soft particles
      //BARREL
      if (isBarrel) {
        float dr = deltaR(seeds.at(is), gp);
        float dphi = deltaPhi(seeds.at(is).phi(), gp.phi());
        float deta = fabs(seeds.at(is).eta() - gp.eta());
        if (deta < 0.03 && dphi < 0.2) {
          if (gpabsid == 22 || gpabsid == 11 || gpabsid == 211 || gpabsid == 321) {
            //contributes to electron
            eTInCone += gp.et();
            //check for a matched track with at least 5 GeV
            if ((gpabsid == 11 || gpabsid == 211 || gpabsid == 321) && gp.et() > 5)
              matchtrack = true;
          } else {
            //contributes to H/E
            hadET += gp.et();
          }
        } else {
          float drUnCurv = deltaR(seeds.at(is), gpUnCurv);
          if ((gp.charge() == 0 && dr < isoConeSize && gpabsid != 22) || (gp.charge() != 0 && drUnCurv < isoConeSize)) {
            //contributes to calo isolation energy
            caloIsoET += gp.et();
          }
          if (gp.charge() != 0 && drUnCurv < isoConeSize) {
            //contributes to track isolation energy
            tkIsoET += gp.et();
          }
        }
        //ENDCAP
      } else {
        float drxy = deltaRxyAtEE(seeds.at(is), gp);
        float dr = deltaR(seeds.at(is), gp);  //the isolation is done in dR
        if (drxy < conesizeendcap) {
          if (gpabsid == 22 || gpabsid == 11 || gpabsid == 211 || gpabsid == 321) {
            //contributes to electron
            eTInCone += gp.et();
            //check for a matched track with at least 5 GeV
            if ((gpabsid == 11 || gpabsid == 211 || gpabsid == 321) && gp.et() > 5)
              matchtrack = true;
          } else {
            //contributes to H/E
            hadET += gp.et();
          }
        } else {
          float drUnCurv = deltaR(seeds.at(is), gpUnCurv);
          if ((gp.charge() == 0 && dr < isoConeSize && gpabsid != 22) || (gp.charge() != 0 && drUnCurv < isoConeSize)) {
            //contributes to calo isolation energy
            caloIsoET += gp.et();
          }
          if (gp.charge() != 0 && drUnCurv < isoConeSize) {
            //contributes to track isolation energy
            tkIsoET += gp.et();
          }
        }
      }
    }

    if (eTInCone > clusterthreshold && (!requiretrackmatch || matchtrack)) {
      //       cout <<"isoET: "<<isoET<<endl;
      if (hadET / eTInCone < hOverEMax && tkIsoET < tkIsoMax && caloIsoET < caloIsoMax) {
        retval = retval + 1;
        sel1seeds.push_back(seeds[is]);
        // break;
      }
    }
  }

  return retval;
}

//make new genparticles vector taking into account the bending of charged particles in the b field
//only stable-final-state (status==1) particles, with ET>=1 GeV, have their trajectories bent
std::vector<reco::GenParticle> doubleEMEnrichingFilterAlgo::applyBFieldCurv(
    const std::vector<reco::GenParticle>& genPars, const edm::EventSetup& iSetup) {
  vector<reco::GenParticle> curvedPars;

  edm::ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get(magField);

  Cylinder::CylinderPointer theBarrel =
      Cylinder::build(Surface::PositionType(0, 0, 0), Surface::RotationType(), ECALBARRELRADIUS_);
  Plane::PlanePointer endCapPlus = Plane::build(Surface::PositionType(0, 0, ECALENDCAPZ_), Surface::RotationType());
  Plane::PlanePointer endCapMinus =
      Plane::build(Surface::PositionType(0, 0, -1 * ECALENDCAPZ_), Surface::RotationType());

  AnalyticalPropagator propagator(&(*magField), alongMomentum);

  for (uint32_t ig = 0; ig < genPars.size(); ig++) {
    reco::GenParticle gp = genPars.at(ig);
    //don't bend trajectories of neutral particles, unstable particles, particles with < 1 GeV
    //particles with < ~0.9 GeV don't reach the barrel
    //so just put them as-is into the new vector
    if (gp.charge() == 0 || gp.status() != 1 || gp.et() < 1) {
      curvedPars.push_back(gp);
      continue;
    }
    GlobalPoint vertex(gp.vx(), gp.vy(), gp.vz());
    GlobalVector gvect(gp.px(), gp.py(), gp.pz());
    FreeTrajectoryState fts(vertex, gvect, gp.charge(), &(*magField));
    TrajectoryStateOnSurface impact;
    //choose to propagate to barrel, +Z endcap, or -Z endcap, according to eta
    if (fabs(gp.eta()) < ECALBARRELMAXETA_) {
      impact = propagator.propagate(fts, *theBarrel);
    } else if (gp.eta() > 0) {
      impact = propagator.propagate(fts, *endCapPlus);
    } else {
      impact = propagator.propagate(fts, *endCapMinus);
    }
    //in case the particle doesn't reach the barrel/endcap, just put it as-is into the new vector
    //it should reach though.
    if (!impact.isValid()) {
      curvedPars.push_back(gp);
      continue;
    }
    math::XYZTLorentzVector newp4;

    //the magnitude of p doesn't change, only the direction
    //NB I do get some small change in magnitude by the following,
    //I think it is a numerical precision issue
    float et = gp.et();
    float phinew = impact.globalPosition().phi();
    float pxnew = et * cos(phinew);
    float pynew = et * sin(phinew);
    newp4.SetPx(pxnew);
    newp4.SetPy(pynew);
    newp4.SetPz(gp.pz());
    newp4.SetE(gp.energy());
    reco::GenParticle gpnew = gp;
    gpnew.setP4(newp4);
    curvedPars.push_back(gpnew);
  }
  return curvedPars;
}

//calculate the difference in the xy-plane positions of gp1 and gp1 at the ECAL endcap
//if they go in different z directions returns 9999.
float doubleEMEnrichingFilterAlgo::deltaRxyAtEE(const reco::GenParticle& gp1, const reco::GenParticle& gp2) {
  if (gp1.pz() * gp2.pz() < 0)
    return 9999.;

  float rxy1 = ECALENDCAPZ_ * tan(gp1.theta());
  float x1 = cos(gp1.phi()) * rxy1;
  float y1 = sin(gp1.phi()) * rxy1;

  float rxy2 = ECALENDCAPZ_ * tan(gp2.theta());
  float x2 = cos(gp2.phi()) * rxy2;
  float y2 = sin(gp2.phi()) * rxy2;

  float dxy = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
  return dxy;
}

//filter looking for isolated charged pions, charged kaons, and electrons.
//isolation done in cone of given size, looking at charged particles and neutral hadrons
//photons aren't counted in the isolation requirements

//need to have both the the curved and uncurved genpar collections
//because tracker iso has to be treated differently than calo iso
int doubleEMEnrichingFilterAlgo::filterIsoGenPar(float etmin,
                                                 float conesize,
                                                 const reco::GenParticleCollection& gph,
                                                 const reco::GenParticleCollection& gphCurved) {
  int passed = 0;
  for (uint32_t ip = 0; ip < gph.size(); ip++) {
    reco::GenParticle gp = gph.at(ip);
    reco::GenParticle gpCurved = gphCurved.at(ip);
    int gpabsid = abs(gp.pdgId());
    //find potential candidates
    if (gp.et() <= etmin || gp.status() != 1)
      continue;
    if (gpabsid != 11 && gpabsid != 211 && gpabsid != 321)
      continue;
    if (fabs(gp.eta()) < FILTER_ETA_MIN_)
      continue;
    if (fabs(gp.eta()) > FILTER_ETA_MAX_)
      continue;

    //check isolation
    float tkiso = 0;
    float caloiso = 0;
    for (uint32_t jp = 0; jp < gph.size(); jp++) {
      if (jp == ip)
        continue;
      reco::GenParticle pp = gph.at(jp);
      reco::GenParticle ppCurved = gphCurved.at(jp);
      if (pp.status() != 1)
        continue;
      float dr = deltaR(gp, pp);
      float drCurved = deltaR(gpCurved, ppCurved);
      if (abs(pp.charge()) == 1 && pp.et() > 2 && dr < conesize) {
        tkiso += pp.et();
      }
      if (pp.et() > 2 && abs(pp.pdgId()) != 22 && drCurved < conesize) {
        caloiso += pp.energy();
      }
    }
    if (tkiso < FILTER_TKISOCUT_ && caloiso < FILTER_CALOISOCUT_) {
      sel2seeds.push_back(gpCurved);
      passed++;
    }
  }
  return passed;
}

//does this particle have an ancestor (mother, mother of mother, etc.) that is a b or c hadron?
//attention: the GenParticle argument must have the motherRef correctly filled for this
//to work.  That is, you had better have got it out of the genParticles collection
bool doubleEMEnrichingFilterAlgo::hasBCAncestors(const reco::GenParticle& gp) {
  //stopping condition: this particle is a b or c hadron
  if (isBCHadron(gp))
    return true;
  //stopping condition: this particle has no mothers
  if (gp.numberOfMothers() == 0)
    return false;
  //otherwise continue
  bool retval = false;
  for (uint32_t im = 0; im < gp.numberOfMothers(); im++) {
    retval = retval || hasBCAncestors(*gp.motherRef(im));
  }
  return retval;
}

bool doubleEMEnrichingFilterAlgo::isBCHadron(const reco::GenParticle& gp) { return isBCMeson(gp) || isBCBaryon(gp); }

bool doubleEMEnrichingFilterAlgo::isBCMeson(const reco::GenParticle& gp) {
  uint32_t pdgid = abs(gp.pdgId());
  uint32_t hundreds = pdgid % 1000;
  if (hundreds >= 400 && hundreds < 600) {
    return true;
  } else {
    return false;
  }
}

bool doubleEMEnrichingFilterAlgo::isBCBaryon(const reco::GenParticle& gp) {
  uint32_t pdgid = abs(gp.pdgId());
  uint32_t thousands = pdgid % 10000;
  if (thousands >= 4000 && thousands < 6000) {
    return true;
  } else {
    return false;
  }
}
