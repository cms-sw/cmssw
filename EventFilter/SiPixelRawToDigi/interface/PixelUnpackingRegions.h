#ifndef PixelUnpackingRegions_H
#define PixelUnpackingRegions_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <cmath>
#include <vector>
#include <set>


/** \class PixelUnpackingRegions
 *
 * Input: One or several collections of Candidate-based seeds with their objects
 *        defining the directions of unpacking regions; separate deltaPhi and maxZ 
 *        tolerances could be given to each input collection.
 * Output: FED ids and module detIds that need to be unpacked
 *
 */
class PixelUnpackingRegions
{
public:

  /// container to define regions for objects of interest in each event by:
  ///   object direction
  ///   dphi max distance from region direction to center of a pixel module
  ///   maxZ max projected z of a pixel module (when projecting along region direction onto beamline)
  struct Region
  {
    Region(const math::XYZVector &dir, float dphi = 0.5f, float maxz = 24.f):
      v(dir), dPhi(dphi), maxZ(maxz)
    {
      cosphi = v.x()/v.rho();
      sinphi = v.y()/v.rho();
      atantheta = v.z()/v.rho();
    }
    math::XYZVector v;
    float dPhi, maxZ;
    float cosphi, sinphi, atantheta;
  };


  PixelUnpackingRegions(const edm::ParameterSet&, edm::ConsumesCollector &&iC);

  ~PixelUnpackingRegions() {}

  /// has to be run during each event
  void run(const edm::Event& e, const edm::EventSetup& es);

  /// check whether a FED has to be unpacked
  bool mayUnpackFED(unsigned int fed_n) const;

  /// check whether a module has to be unpacked
  bool mayUnpackModule(unsigned int id) const;
  
  /// full set of module ids to unpack
  const std::set<unsigned int> * modulesToUnpack() const {return &modules_;}

  /// various informational accessors:
  unsigned int nFEDs() const { return feds_.size(); }
  unsigned int nBarrelFEDs() const;
  unsigned int nForwardFEDs() const;
  unsigned int nModules() const { return modules_.size(); }
  unsigned int nBarrelModules() const;
  unsigned int nForwardModules() const;
  unsigned int nRegions() const { return nreg_; }

  struct Module
  {
    float phi;
    float x, y, z;
    unsigned int id;
    unsigned int fed;

    Module() {}
    Module(float ph) : phi(ph), x(0.f), y(0.f), z(0.f), id(0), fed(0) {}

    bool operator < (const Module& m) const
    {
      if(phi < m.phi) return true;
      if(phi == m.phi && id < m.id) return true;
      return false;
    }
  };

private:

  // input parameters
  std::vector<edm::InputTag> inputs_;
  std::vector<double> dPhi_;
  std::vector<double> maxZ_;
  edm::InputTag beamSpotTag_;

  edm::EDGetTokenT<reco::BeamSpot> tBeamSpot;
  std::vector<edm::EDGetTokenT<reco::CandidateView>> tCandidateView;

  std::set<unsigned int> feds_;
  std::set<unsigned int> modules_;
  unsigned int nreg_;

  /// run by the run method: (re)initialize the cabling data when it's necessary
  void initialize(const edm::EventSetup& es);

  // add a new direction of a region of interest
  void addRegion(Region &r);

  // gather info into feds_ and modules_ from a range of a Module vector
  void gatherFromRange(Region &r, std::vector<Module>::const_iterator, std::vector<Module>::const_iterator);

  // addRegion for a local (BPIX or +-FPIX) container
  void addRegionLocal(Region &r, std::vector<Module> &container, const Module& lo,const Module& hi);

  // local containers of barrel and endcaps Modules sorted by phi
  std::vector<Module> phiBPIX_;
  std::vector<Module> phiFPIXp_;
  std::vector<Module> phiFPIXm_;

  std::unique_ptr<SiPixelFedCablingTree> cabling_;
  math::XYZPoint beamSpot_;

  edm::ESWatcher<SiPixelFedCablingMapRcd> watcherSiPixelFedCablingMap_;
};

#endif
