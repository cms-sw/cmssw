/// \file CosmicGenFilterHelix.cc 
//
// Original Author:  Gero FLUCKE
//         Created:  Mon Mar  5 16:32:01 CET 2007
// $Id: CosmicGenFilterHelix.cc,v 1.12 2010/01/05 13:49:07 hegner Exp $

#include "GeneratorInterface/GenFilters/interface/CosmicGenFilterHelix.h"

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"


#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TMath.h>
#include <TH1.h> // include checker: don't touch!
#include <TH2.h> // include checker: don't touch!

#include <utility> // for std::pair
#include <string>


//
// constructors and destructor
//

///////////////////////////////////////////////////////////////////////////////////////////////////
CosmicGenFilterHelix::CosmicGenFilterHelix(const edm::ParameterSet& cfg) 
  : theSrc(cfg.getParameter<edm::InputTag>("src")),
    theIds(cfg.getParameter<std::vector<int> >("pdgIds")),
    theCharges(cfg.getParameter<std::vector<int> >("charges")),
    thePropagatorName(cfg.getParameter<std::string>("propagator")),
    theMinP2(cfg.getParameter<double>("minP")*cfg.getParameter<double>("minP")),
    theMinPt2(cfg.getParameter<double>("minPt")*cfg.getParameter<double>("minPt")),
    theDoMonitor(cfg.getUntrackedParameter<bool>("doMonitor"))
{
  if (theIds.size() != theCharges.size()) {
    throw cms::Exception("BadConfig") << "CosmicGenFilterHelix: "
				      << "'pdgIds' and 'charges' need same length.";
  }
  Surface::Scalar radius = cfg.getParameter<double>("radius");
  Surface::Scalar maxZ   = cfg.getParameter<double>("maxZ");
  Surface::Scalar minZ   = cfg.getParameter<double>("minZ");
  
  if (maxZ < minZ) {
    throw cms::Exception("BadConfig") << "CosmicGenFilterHelix: maxZ (" << maxZ
				      << ") smaller than minZ (" << minZ << ").";
  }
  
  const Surface::RotationType dummyRot;
  theTargetCylinder = Cylinder::build(Surface::PositionType(0.,0.,0.), dummyRot, radius);
  theTargetPlaneMin = Plane::build(Surface::PositionType(0.,0.,minZ), dummyRot);
  theTargetPlaneMax = Plane::build(Surface::PositionType(0.,0.,maxZ), dummyRot);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
CosmicGenFilterHelix::~CosmicGenFilterHelix()
{
}


//
// member functions
//

///////////////////////////////////////////////////////////////////////////////////////////////////
bool CosmicGenFilterHelix::filter(edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  edm::Handle<edm::HepMCProduct> hepMCEvt;
  iEvent.getByLabel(theSrc, hepMCEvt);
  const HepMC::GenEvent *mCEvt = hepMCEvt->GetEvent();
  const MagneticField *bField = this->getMagneticField(iSetup); // should be fast (?)
  const Propagator *propagator = this->getPropagator(iSetup);

  ++theNumTotal;
  bool result = false;
  for (HepMC::GenEvent::particle_const_iterator iPart = mCEvt->particles_begin(),
	 endPart = mCEvt->particles_end(); iPart != endPart; ++iPart) {
    int charge = 0; // there is no method providing charge in GenParticle :-(
    if ((*iPart)->status() != 1) continue; // look only at stable particles
    if (!this->charge((*iPart)->pdg_id(), charge)) continue;

    // Get the position and momentum
    const HepMC::ThreeVector hepVertex((*iPart)->production_vertex()->point3d());
    const GlobalPoint vert(hepVertex.x()/10., hepVertex.y()/10., hepVertex.z()/10.); // to cm
    const HepMC::FourVector hepMomentum((*iPart)->momentum());
    const GlobalVector mom(hepMomentum.x(), hepMomentum.y(), hepMomentum.z());

    if (theDoMonitor) this->monitorStart(vert, mom, charge, theHistsBefore);

    if (this->propagateToCutCylinder(vert, mom, charge, bField, propagator)) {
      result = true;
    }
  }

  if (result) ++theNumPass;
  return result;
}

//_________________________________________________________________________________________________
bool CosmicGenFilterHelix::propagateToCutCylinder(const GlobalPoint &vertStart,
						  const GlobalVector &momStart,
						  int charge, const MagneticField *field,
                                                  const Propagator *propagator)
{
  typedef std::pair<TrajectoryStateOnSurface, double> TsosPath;

  const FreeTrajectoryState fts(GlobalTrajectoryParameters(vertStart, momStart, charge, field));

  bool result = true;
  TsosPath aTsosPath(propagator->propagateWithPath(fts, *theTargetCylinder));
  if (!aTsosPath.first.isValid()) {
    result = false;
  } else if (aTsosPath.first.globalPosition().z() < theTargetPlaneMin->position().z()) {
    // If on cylinder, but outside minimum z, try minimum z-plane:
    // (Would it be possible to miss radius on plane, but reach cylinder afterwards in z-range?
    //  No, at least not in B-field parallel to z-axis which is cylinder axis.)
    aTsosPath = propagator->propagateWithPath(fts, *theTargetPlaneMin);
    if (!aTsosPath.first.isValid()
	|| aTsosPath.first.globalPosition().perp() > theTargetCylinder->radius()) {
      result = false;
    }
  } else if (aTsosPath.first.globalPosition().z() > theTargetPlaneMax->position().z()) {
    // Analog for outside maximum z:
    aTsosPath = propagator->propagateWithPath(fts, *theTargetPlaneMax);
    if (!aTsosPath.first.isValid()
	|| aTsosPath.first.globalPosition().perp() > theTargetCylinder->radius()) {
      result = false;
    }
  }

  if (result) {
    const GlobalVector momEnd(aTsosPath.first.globalMomentum());
    if (momEnd.perp2() < theMinPt2 || momEnd.mag2() < theMinP2) {
      result = false;
    } else if (theDoMonitor) {
      const GlobalPoint vertEnd(aTsosPath.first.globalPosition());
      this->monitorStart(vertStart, momStart, charge, theHistsAfter);
      this->monitorEnd(vertEnd, momEnd, vertStart, momStart, aTsosPath.second, theHistsAfter);
    }
  }

  return result;
}


// ------------ method called once each job just before starting event loop  ------------
void CosmicGenFilterHelix::beginJob()
{
  if (theDoMonitor) {
    this->createHistsStart("start", theHistsBefore);
    this->createHistsStart("startAfter", theHistsAfter);
    // must be after the line above: hist indices are static in monitorStart(...)
    this->createHistsEnd("end", theHistsAfter);
  }

  theNumTotal = theNumPass = 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void CosmicGenFilterHelix::createHistsStart(const char *dirName, TObjArray &hists)
{
  edm::Service<TFileService> fs;
  TFileDirectory fd(fs->mkdir(dirName, dirName));
  
  hists.Add(fd.make<TH1F>("momentumP", "|p(#mu^{+})| (start);|p| [GeV]",100, 0., 1000.));
  hists.Add(fd.make<TH1F>("momentumM", "|p(#mu^{-})| (start);|p| [GeV]",100, 0., 1000.));
  hists.Add(fd.make<TH1F>("momentum2", "|p(#mu)| (start);|p| [GeV]",100, 0., 25.));
  const int kNumBins = 50;
  double pBinsLog[kNumBins+1] = {0.}; // fully initialised with 0.
  this->equidistLogBins(pBinsLog, kNumBins, 1., 4000.);
  hists.Add(fd.make<TH1F>("momentumLog", "|p(#mu)| (start);|p| [GeV]", kNumBins, pBinsLog));
  hists.Add(fd.make<TH1F>("phi", "start p_{#phi(#mu)};#phi", 100, -TMath::Pi(), TMath::Pi()));
  hists.Add(fd.make<TH1F>("cosPhi", "cos(p_{#phi(#mu)}) (start);cos(#phi)", 100, -1., 1.));
  hists.Add(fd.make<TH1F>("phiXz", "start p_{#phi_{xz}(#mu)};#phi_{xz}",
                          100, -TMath::Pi(), TMath::Pi()));
  hists.Add(fd.make<TH1F>("theta", "#theta(#mu) (start);#theta", 100, 0., TMath::Pi()));
  hists.Add(fd.make<TH1F>("thetaY", "#theta_{y}(#mu) (start);#theta_{y}", 100,0.,TMath::Pi()/2.));
  
  hists.Add(fd.make<TH2F>("momVsPhi", "|p(#mu)| vs #phi (start);#phi;|p| [GeV]",
                          50, -TMath::Pi(), TMath::Pi(), 50, 1.5, 1000.));
  hists.Add(fd.make<TH2F>("momVsTheta", "|p(#mu)| vs #theta (start);#theta;|p| [GeV]",
                          50, 0., TMath::Pi(), 50, 1.5, 1000.));
  hists.Add(fd.make<TH2F>("momVsThetaY", "|p(#mu)| vs #theta_{y} (start);#theta_{y};|p| [GeV]",
                          50, 0., TMath::Pi()/2., 50, 1.5, 1000.));
  hists.Add(fd.make<TH2F>("momVsZ", "|p(#mu)| vs z (start);z [cm];|p| [GeV]",
                          50, -1600., 1600., 50, 1.5, 1000.));
  hists.Add(fd.make<TH2F>("thetaVsZ", "#theta vs z (start);z [cm];#theta",
                          50, -1600., 1600., 50, 0., TMath::Pi()));
  hists.Add(fd.make<TH2F>("thetaYvsZ", "#theta_{y} vs z (start);z [cm];#theta",
                          50, -1600., 1600., 50, 0., TMath::PiOver2()));
  hists.Add(fd.make<TH2F>("yVsThetaY", "#theta_{y}(#mu) vs y (start);#theta_{y};y [cm]",
                          50, 0., TMath::Pi()/2., 50, -1000., 1000.));
  hists.Add(fd.make<TH2F>("yVsThetaYnoR", "#theta_{y}(#mu) vs y (start, barrel);#theta_{y};y [cm]",
                          50, 0., TMath::Pi()/2., 50, -1000., 1000.));
  
  hists.Add(fd.make<TH1F>("radius", "start radius;r [cm]", 100, 0., 1000.));
  hists.Add(fd.make<TH1F>("z", "start z;z [cm]", 100, -1600., 1600.));
  hists.Add(fd.make<TH2F>("xyPlane", "start xy;x [cm];y [cm]", 50, -1000., 1000.,
                          50, -1000., 1000.));
  hists.Add(fd.make<TH2F>("rzPlane",
                          "start rz (y < 0 #Rightarrow r_{#pm} = -r);z [cm];r_{#pm} [cm]",
                          50, -1600., 1600., 50, -1000., 1000.));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void CosmicGenFilterHelix::createHistsEnd(const char *dirName, TObjArray &hists)
{
  edm::Service<TFileService> fs;
  TFileDirectory fd(fs->mkdir(dirName, dirName));

  const int kNumBins = 50;
  double pBinsLog[kNumBins+1] = {0.}; // fully initialised with 0.
  this->equidistLogBins(pBinsLog, kNumBins, 1., 4000.);

  // take care: hist names must differ from those in createHistsStart!
  hists.Add(fd.make<TH1F>("pathEnd", "path until cylinder;s [cm]", 100, 0., 2000.));
  hists.Add(fd.make<TH1F>("momEnd", "|p_{end}|;p [GeV]", 100, 0., 1000.));
  hists.Add(fd.make<TH1F>("momEndLog", "|p_{end}|;p [GeV]", kNumBins, pBinsLog));
  hists.Add(fd.make<TH1F>("ptEnd", "p_{t} (end);p_{t} [GeV]", 100, 0., 750.));
  hists.Add(fd.make<TH1F>("ptEndLog", "p_{t} (end);p_{t} [GeV]", kNumBins, pBinsLog));
  hists.Add(fd.make<TH1F>("phiXzEnd", "#phi_{xz} (end);#phi_{xz}", 100,-TMath::Pi(),TMath::Pi()));
  hists.Add(fd.make<TH1F>("thetaYEnd","#theta_{y} (end);#theta_{y}", 100, 0., TMath::Pi()));
  
  hists.Add(fd.make<TH1F>("momStartEnd", "|p_{start}|-|p_{end}|;#Deltap [GeV]",100,0.,15.));
  hists.Add(fd.make<TH1F>("momStartEndRel", "(p_{start}-p_{end})/p_{start};#Deltap_{rel}",
                          100,.0,1.));
  hists.Add(fd.make<TH1F>("phiXzStartEnd", "#phi_{xz,start}-#phi_{xz,end};#Delta#phi_{xz}",
                          100,-1.,1.));
  hists.Add(fd.make<TH1F>("thetaYStartEnd","#theta_{y,start}-#theta_{y,end};#Delta#theta_{y}",
                          100,-1.,1.));
  
  hists.Add(fd.make<TH2F>("phiXzStartVsEnd",
                          "#phi_{xz} start vs end;#phi_{xz}^{end};#phi_{xz}^{start}",
                          50, -TMath::Pi(), TMath::Pi(), 50, -TMath::Pi(), TMath::Pi()));
  hists.Add(fd.make<TH2F>("thetaYStartVsEnd",
                          "#theta_{y} start vs end;#theta_{y}^{end};#theta_{y}^{start}",
                          50, 0., TMath::Pi(), 50, 0., TMath::Pi()/2.));
  
  hists.Add(fd.make<TH2F>("momStartEndRelVsZ",
                          "(p_{start}-p_{end})/p_{start} vs z_{start};z [cm];#Deltap_{rel}",
                          50, -1600., 1600., 50,.0,.8));
  hists.Add(fd.make<TH2F>("phiXzStartEndVsZ", 
                          "#phi_{xz,start}-#phi_{xz,end} vs z_{start};z [cm];#Delta#phi_{xz}",
                          50, -1600., 1600., 50,-1., 1.));
  hists.Add(fd.make<TH2F>("thetaYStartEndVsZ",
                          "#theta_{y,start}-#theta_{y,end} vs z_{start};z [cm];#Delta#theta_{y}",
                          50, -1600., 1600., 50,-.5,.5));
  hists.Add(fd.make<TH2F>("momStartEndRelVsP",
                          "(p_{start}-p_{end})/p_{start} vs p_{start};p [GeV];#Deltap_{rel}",
                          kNumBins, pBinsLog, 50, .0, .8));
  hists.Add(fd.make<TH2F>("phiXzStartEndVsP", 
                          "#phi_{xz,start}-#phi_{xz,end} vs |p|_{start};p [GeV];#Delta#phi_{xz}",
                          kNumBins, pBinsLog, 100,-1.5, 1.5));
  hists.Add(fd.make<TH2F>("thetaYStartEndVsP",
                          "#theta_{y,start}-#theta_{y,end} vs |p|_{start};p [GeV];#Delta#theta_{y}",
                          kNumBins, pBinsLog, 100,-1.,1.));
  
  const double maxR = theTargetCylinder->radius() * 1.1;
  hists.Add(fd.make<TH1F>("radiusEnd", "end radius;r [cm]", 100, 0., maxR));
  double minZ = theTargetPlaneMin->position().z();
  minZ -= TMath::Abs(minZ) * 0.1;
  double maxZ = theTargetPlaneMax->position().z();
  maxZ += TMath::Abs(maxZ) * 0.1;
  hists.Add(fd.make<TH1F>("zEnd", "end z;z [cm]", 100, minZ, maxZ));
  hists.Add(fd.make<TH1F>("zDiff", "z_{start}-z_{end};#Deltaz [cm]", 100, -1000., 1000.));
  hists.Add(fd.make<TH2F>("xyPlaneEnd", "end xy;x [cm];y [cm]", 100, -maxR, maxR, 100,-maxR,maxR));
  
  hists.Add(fd.make<TH2F>("rzPlaneEnd", "end rz (y<0 #Rightarrow r_{#pm}=-r);z [cm];r_{#pm} [cm]",
                          50, minZ, maxZ, 50, -maxR, maxR));
  hists.Add(fd.make<TH2F>("thetaVsZend", "#theta vs z (end);z [cm];#theta",
                          50, minZ, maxZ, 50, 0., TMath::Pi()));
}

// ------------ method called once each job just after ending the event loop  ------------
void CosmicGenFilterHelix::endJob()
{
  const char *border = "////////////////////////////////////////////////////////";
  const char *line = "\n// ";
  edm::LogInfo("Filter") << "@SUB=CosmicGenFilterHelix::endJob"
			 << border << line
                         << theNumPass << " events out of " << theNumTotal
                         << ", i.e. " << theNumPass*100./theNumTotal << "%, "
			 << "reached target cylinder," 
			 << line << "defined by r < " 
			 << theTargetCylinder->radius() << " cm and " 
			 << theTargetPlaneMin->position().z() << " < z < " 
			 << theTargetPlaneMax->position().z() << " cm."
			 << line << "Minimal required (transverse) momentum was "
			 << TMath::Sqrt(theMinP2) << " (" << TMath::Sqrt(theMinPt2) << ") GeV."
			 << "\n" << border;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool CosmicGenFilterHelix::charge(int id, int &charge) const
{
  std::vector<int>::const_iterator iC = theCharges.begin();
  for (std::vector<int>::const_iterator i = theIds.begin(), end = theIds.end(); i != end;
       ++i,++iC) {
    if (*i == id) {
      charge = *iC;
      return true;
    }
  }

  return false;
}

//_________________________________________________________________________________________________
const MagneticField* CosmicGenFilterHelix::getMagneticField(const edm::EventSetup &setup) const
{
  edm::ESHandle<MagneticField> fieldHandle;
  setup.get<IdealMagneticFieldRecord>().get(fieldHandle); 

  return fieldHandle.product();
}

//_________________________________________________________________________________________________
const Propagator* CosmicGenFilterHelix::getPropagator(const edm::EventSetup &setup) const
{
  edm::ESHandle<Propagator> propHandle;
  setup.get<TrackingComponentsRecord>().get(thePropagatorName, propHandle);

  const Propagator *prop = propHandle.product();
  if (!dynamic_cast<const SteppingHelixPropagator*>(prop)) {
    edm::LogWarning("BadConfig") << "@SUB=CosmicGenFilterHelix::getPropagator"
                                 << "Not a SteppingHelixPropagator!";

  }
  return prop;
}

//_________________________________________________________________________________________________
void CosmicGenFilterHelix::monitorStart(const GlobalPoint &vert, const GlobalVector &mom,
					int charge, TObjArray &hists)
{
  const double scalarMom = mom.mag();
  const double phi = mom.phi();
  const double phiXz = TMath::ATan2(mom.z(), mom.x());
  const double theta = mom.theta();
  const double thetaY = TMath::ATan2(TMath::Sqrt(mom.x()*mom.x()+mom.z()*mom.z()), -mom.y());

  const double z = vert.z();
  const double r = vert.perp();

  static int iMomP = hists.IndexOf(hists.FindObject("momentumP"));
  static int iMomM = hists.IndexOf(hists.FindObject("momentumM"));
  if (charge > 0) static_cast<TH1*>(hists[iMomP])->Fill(scalarMom);
  else            static_cast<TH1*>(hists[iMomM])->Fill(scalarMom);
  static int iMom2 = hists.IndexOf(hists.FindObject("momentum2"));
  static_cast<TH1*>(hists[iMom2])->Fill(scalarMom);
  static int iMomLog = hists.IndexOf(hists.FindObject("momentumLog"));
  static_cast<TH1*>(hists[iMomLog])->Fill(scalarMom);
  static int iPhi = hists.IndexOf(hists.FindObject("phi"));
  static_cast<TH1*>(hists[iPhi])->Fill(phi);
  static int iCosPhi = hists.IndexOf(hists.FindObject("cosPhi"));
  static_cast<TH1*>(hists[iCosPhi])->Fill(TMath::Cos(phi));
  static int iPhiXz = hists.IndexOf(hists.FindObject("phiXz"));
  static_cast<TH1*>(hists[iPhiXz])->Fill(phiXz);
  static int iTheta = hists.IndexOf(hists.FindObject("theta"));
  static_cast<TH1*>(hists[iTheta])->Fill(theta);
  static int iThetaY = hists.IndexOf(hists.FindObject("thetaY"));
  static_cast<TH1*>(hists[iThetaY])->Fill(thetaY);

  static int iMomVsTheta = hists.IndexOf(hists.FindObject("momVsTheta"));
  static_cast<TH2*>(hists[iMomVsTheta])->Fill(theta, scalarMom);
  static int iMomVsThetaY = hists.IndexOf(hists.FindObject("momVsThetaY"));
  static_cast<TH2*>(hists[iMomVsThetaY])->Fill(thetaY, scalarMom);
  static int iMomVsPhi = hists.IndexOf(hists.FindObject("momVsPhi"));
  static_cast<TH2*>(hists[iMomVsPhi])->Fill(phi, scalarMom);
  static int iMomVsZ = hists.IndexOf(hists.FindObject("momVsZ"));
  static_cast<TH2*>(hists[iMomVsZ])->Fill(z, scalarMom);
  static int iThetaVsZ = hists.IndexOf(hists.FindObject("thetaVsZ"));
  static_cast<TH2*>(hists[iThetaVsZ])->Fill(z, theta);
  static int iThetaYvsZ = hists.IndexOf(hists.FindObject("thetaYvsZ"));
  static_cast<TH2*>(hists[iThetaYvsZ])->Fill(z, thetaY);
  static int iYvsThetaY = hists.IndexOf(hists.FindObject("yVsThetaY"));
  static_cast<TH2*>(hists[iYvsThetaY])->Fill(thetaY, vert.y());
  static int iYvsThetaYnoR = hists.IndexOf(hists.FindObject("yVsThetaYnoR"));
  if (z > -1400. && z < 1400.) {
    static_cast<TH2*>(hists[iYvsThetaYnoR])->Fill(thetaY, vert.y());
  }

  static int iRadius = hists.IndexOf(hists.FindObject("radius"));
  static_cast<TH1*>(hists[iRadius])->Fill(r);
  static int iZ = hists.IndexOf(hists.FindObject("z"));
  static_cast<TH1*>(hists[iZ])->Fill(z);
  static int iXy = hists.IndexOf(hists.FindObject("xyPlane"));
  static_cast<TH1*>(hists[iXy])->Fill(vert.x(), vert.y());
  static int iRz = hists.IndexOf(hists.FindObject("rzPlane"));
  static_cast<TH1*>(hists[iRz])->Fill(z, (vert.y() > 0 ? r : -r));
}

//_________________________________________________________________________________________________
void CosmicGenFilterHelix::monitorEnd(const GlobalPoint &endVert, const GlobalVector &endMom,
				      const GlobalPoint &vert, const GlobalVector &mom,
				      double path, TObjArray &hists)
{
  const double scalarMomStart = mom.mag();
  const double phiXzStart = TMath::ATan2(mom.z(), mom.x());
  const double thetaYStart = TMath::ATan2(TMath::Sqrt(mom.x()*mom.x()+mom.z()*mom.z()), -mom.y());
  const double scalarMomEnd = endMom.mag();
  const double ptEnd = endMom.perp();
  const double phiXzEnd = TMath::ATan2(endMom.z(), endMom.x());
  const double thetaYEnd = TMath::ATan2(TMath::Sqrt(endMom.x()*endMom.x()+endMom.z()*endMom.z()),
					-endMom.y());
  const double thetaEnd = endMom.theta();

  const double diffMomRel = (scalarMomStart-scalarMomEnd)/scalarMomStart;

  const double zEnd = endVert.z();
  const double rEnd = endVert.perp();
  const double diffZ = zEnd - vert.z();

  static int iPathEnd = hists.IndexOf(hists.FindObject("pathEnd"));
  static_cast<TH1*>(hists[iPathEnd])->Fill(path);
  static int iMomEnd = hists.IndexOf(hists.FindObject("momEnd"));
  static_cast<TH1*>(hists[iMomEnd])->Fill(scalarMomEnd);
  static int iMomEndLog = hists.IndexOf(hists.FindObject("momEndLog"));
  static_cast<TH1*>(hists[iMomEndLog])->Fill(scalarMomEnd);
  static int iPtEnd = hists.IndexOf(hists.FindObject("ptEnd"));
  static_cast<TH1*>(hists[iPtEnd])->Fill(ptEnd);
  static int iPtEndLog = hists.IndexOf(hists.FindObject("ptEndLog"));
  static_cast<TH1*>(hists[iPtEndLog])->Fill(ptEnd);
  static int iPhiXzEnd = hists.IndexOf(hists.FindObject("phiXzEnd"));
  static_cast<TH1*>(hists[iPhiXzEnd])->Fill(phiXzEnd);
  static int iThetaYEnd = hists.IndexOf(hists.FindObject("thetaYEnd"));
  static_cast<TH1*>(hists[iThetaYEnd])->Fill(thetaYEnd);

  static int iMomStartEnd = hists.IndexOf(hists.FindObject("momStartEnd"));
  static_cast<TH1*>(hists[iMomStartEnd])->Fill(scalarMomStart-scalarMomEnd);
  static int iMomStartEndRel = hists.IndexOf(hists.FindObject("momStartEndRel"));
  static_cast<TH1*>(hists[iMomStartEndRel])->Fill(diffMomRel);
  static int iPhiStartEnd = hists.IndexOf(hists.FindObject("phiXzStartEnd"));
  static_cast<TH1*>(hists[iPhiStartEnd])->Fill(phiXzStart-phiXzEnd);
  static int iThetaStartEnd = hists.IndexOf(hists.FindObject("thetaYStartEnd"));
  static_cast<TH1*>(hists[iThetaStartEnd])->Fill(thetaYStart-thetaYEnd);

  static int iPhiStartVsEnd = hists.IndexOf(hists.FindObject("phiXzStartVsEnd"));
  static_cast<TH2*>(hists[iPhiStartVsEnd])->Fill(phiXzEnd, phiXzStart);
  static int iThetaStartVsEnd = hists.IndexOf(hists.FindObject("thetaYStartVsEnd"));
  static_cast<TH2*>(hists[iThetaStartVsEnd])->Fill(thetaYEnd, thetaYStart);

  static int iMomStartEndRelVsZ = hists.IndexOf(hists.FindObject("momStartEndRelVsZ"));
  static_cast<TH2*>(hists[iMomStartEndRelVsZ])->Fill(vert.z(), diffMomRel);
  static int iPhiStartEndVsZ = hists.IndexOf(hists.FindObject("phiXzStartEndVsZ"));
  static_cast<TH2*>(hists[iPhiStartEndVsZ])->Fill(vert.z(), phiXzStart-phiXzEnd);
  static int iThetaStartEndVsZ = hists.IndexOf(hists.FindObject("thetaYStartEndVsZ"));
  static_cast<TH2*>(hists[iThetaStartEndVsZ])->Fill(vert.z(), thetaYStart-thetaYEnd);
  static int iMomStartEndRelVsP = hists.IndexOf(hists.FindObject("momStartEndRelVsP"));
  static_cast<TH2*>(hists[iMomStartEndRelVsP])->Fill(scalarMomStart, diffMomRel);
  static int iPhiStartEndVsP = hists.IndexOf(hists.FindObject("phiXzStartEndVsP"));
  static_cast<TH2*>(hists[iPhiStartEndVsP])->Fill(scalarMomStart, phiXzStart-phiXzEnd);
  static int iThetaStartEndVsP = hists.IndexOf(hists.FindObject("thetaYStartEndVsP"));
  static_cast<TH2*>(hists[iThetaStartEndVsP])->Fill(scalarMomStart, thetaYStart-thetaYEnd);

  static int iRadiusEnd = hists.IndexOf(hists.FindObject("radiusEnd"));
  static_cast<TH1*>(hists[iRadiusEnd])->Fill(rEnd);
  static int iZend = hists.IndexOf(hists.FindObject("zEnd"));
  static_cast<TH1*>(hists[iZend])->Fill(zEnd);
  static int iZdiff = hists.IndexOf(hists.FindObject("zDiff"));
  static_cast<TH1*>(hists[iZdiff])->Fill(diffZ);
  static int iXyPlaneEnd = hists.IndexOf(hists.FindObject("xyPlaneEnd"));
  static_cast<TH1*>(hists[iXyPlaneEnd])->Fill(endVert.x(), endVert.y());
  static int iRzPlaneEnd = hists.IndexOf(hists.FindObject("rzPlaneEnd"));
  static_cast<TH1*>(hists[iRzPlaneEnd])->Fill(zEnd, (endVert.y() > 0 ? rEnd : -rEnd));
  static int iThetaVsZend = hists.IndexOf(hists.FindObject("thetaVsZend"));
  static_cast<TH2*>(hists[iThetaVsZend])->Fill(zEnd, thetaEnd);
}

//_________________________________________________________________________________________________
bool CosmicGenFilterHelix::equidistLogBins(double* bins, int nBins, 
					   double first, double last) const
{
  // Filling 'bins' with borders of 'nBins' bins between 'first' and 'last'
  // that are equidistant when viewed in log scale,
  // so 'bins' must have length nBins+1;
  // If 'first', 'last' or 'nBins' are not positive, failure is reported.

  if (nBins < 1 || first <= 0. || last <= 0.) return false;

  bins[0] = first;
  bins[nBins] = last;
  const double firstLog = TMath::Log10(bins[0]);
  const double lastLog  = TMath::Log10(bins[nBins]);
  for (int i = 1; i < nBins; ++i) {
    bins[i] = TMath::Power(10., firstLog + i*(lastLog-firstLog)/(nBins));
  }

  return true;
}
