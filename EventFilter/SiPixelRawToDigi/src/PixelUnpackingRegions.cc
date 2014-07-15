//
//
#include "EventFilter/SiPixelRawToDigi/interface/PixelUnpackingRegions.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <algorithm>
#include <iterator>

// local convenience functions
namespace 
{
bool isBPIXFED(unsigned int fed) {return fed< 32;}
bool isFPIXFED(unsigned int fed) {return fed>=32;}
bool isBPIXModule(unsigned int id) {return DetId(id).subdetId() == PixelSubdetector::PixelBarrel;}
bool isFPIXModule(unsigned int id) {return DetId(id).subdetId() == PixelSubdetector::PixelEndcap;}

inline std::ostream& operator<<(std::ostream& s, const PixelUnpackingRegions::Module& m)
{
  s<< (isBPIXModule(m.id) ? "BPIX " : "FPIX ") <<m.id<<" "<<m.fed<<"   "<<m.phi<<"   "<<m.x<<" "<<m.y<<" "<<m.z<<"  "<<sqrt(std::pow(m.x,2)+std::pow(m.y,2));
  return s;
}
}



PixelUnpackingRegions::PixelUnpackingRegions(const edm::ParameterSet& conf, edm::ConsumesCollector &&iC)
{
  edm::ParameterSet regPSet = conf.getParameter<edm::ParameterSet>("Regions");
  beamSpotTag_ = regPSet.getParameter<edm::InputTag>("beamSpot");
  inputs_      = regPSet.getParameter<std::vector<edm::InputTag> >("inputs");
  dPhi_ = regPSet.getParameter<std::vector<double> >("deltaPhi");
  maxZ_ = regPSet.getParameter<std::vector<double> >("maxZ");

  tBeamSpot = iC.consumes<reco::BeamSpot>(beamSpotTag_);
  for (unsigned int t=0; t<inputs_.size(); t++ ) tCandidateView.push_back(iC.consumes< reco::CandidateView >(inputs_[t]));

  if (inputs_.size() != dPhi_.size() || dPhi_.size() != maxZ_.size() )
  {
    edm::LogError("PixelUnpackingRegions")<<"Not the same size of config parameters vectors!\n"
        <<"   inputs "<<inputs_.size()<<"  deltaPhi "<<dPhi_.size() <<"  maxZ "<< maxZ_.size();
  }
}


void PixelUnpackingRegions::run(const edm::Event& e, const edm::EventSetup& es)
{
  feds_.clear();
  modules_.clear();
  nreg_ = 0;

  initialize(es);

  edm::Handle<reco::BeamSpot> beamSpot;
  e.getByToken(tBeamSpot, beamSpot);
  beamSpot_ = beamSpot->position();
  //beamSpot_ = math::XYZPoint(0.,0.,0.);

  size_t ninputs = inputs_.size();
  for(size_t input = 0; input < ninputs; ++input)
  {
    edm::Handle< reco::CandidateView > h;
    e.getByToken(tCandidateView[input], h);

    size_t n = h->size();
    for(size_t i = 0; i < n; ++i )
    {
      const reco::Candidate & c = (*h)[i];

      // different input collections can have different dPhi and maxZ
      Region r(c.momentum(), dPhi_[input], maxZ_[input]);
      addRegion(r);
    }
  }
}


void PixelUnpackingRegions::initialize(const edm::EventSetup& es)
{
  // initialize cabling map or update it if necessary
  // and re-cache modules information
  if (watcherSiPixelFedCablingMap_.check( es ))
  {
    edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMap );
    cabling_ = cablingMap->cablingTree();

    edm::ESHandle<TrackerGeometry> geom;
    // get the TrackerGeom
    es.get<TrackerDigiGeometryRecord>().get( geom );

    phiBPIX_.clear();
    phiFPIXp_.clear();
    phiFPIXm_.clear();

    phiBPIX_.reserve(1024);
    phiFPIXp_.reserve(512);
    phiFPIXm_.reserve(512);

    auto it = geom->dets().begin();
    for ( ; it != geom->dets().end(); ++it)
    {
      int subdet = (*it)->geographicalId().subdetId();
      if (! (subdet == PixelSubdetector::PixelBarrel ||
             subdet == PixelSubdetector::PixelEndcap) ) continue;

      Module m;

      m.x = (*it)->position().x();
      m.y = (*it)->position().y();
      m.z = (*it)->position().z();

      m.phi = normalizedPhi( (*it)->position().phi() ); // ensure [-pi,+pi]

      m.id = (*it)->geographicalId().rawId();
      const std::vector<sipixelobjects::CablingPathToDetUnit> path2det = cabling_->pathToDetUnit(m.id);

      m.fed = path2det[0].fed;
      assert(m.fed<40);

      if (subdet == PixelSubdetector::PixelBarrel)
      {
        phiBPIX_.push_back(m);
      }
      else if (subdet == PixelSubdetector::PixelEndcap)
      {
        if (m.z > 0.) phiFPIXp_.push_back(m);
        else phiFPIXm_.push_back(m);
      }
    }

    // pre-sort by phi
    std::sort(phiBPIX_.begin(),  phiBPIX_.end());
    std::sort(phiFPIXp_.begin(), phiFPIXp_.end());
    std::sort(phiFPIXm_.begin(), phiFPIXm_.end());
  }
}


void PixelUnpackingRegions::addRegion(Region &r)
{
  ++nreg_;

  float phi = normalizedPhi(r.v.phi());  // ensure [-pi,+pi]

  Module lo(phi - r.dPhi);
  Module hi(phi + r.dPhi);

  addRegionLocal(r, phiBPIX_, lo, hi);
  if (r.v.eta() >  1.)
  {
    addRegionLocal(r, phiFPIXp_, lo, hi);
  }
  if (r.v.eta() < -1.)
  {
    addRegionLocal(r, phiFPIXm_, lo, hi);
  }
}


void PixelUnpackingRegions::addRegionLocal(Region &r, std::vector<Module> &container,const  Module& _lo,const Module& _hi)
{
  Module lo = _lo;
  Module hi = _hi;
  Module pi_m(-M_PI);
  Module pi_p( M_PI);

  std::vector<Module>::const_iterator a, b;

  if (lo.phi >= -M_PI && hi.phi <= M_PI) // interval doesn't cross the +-pi overlap
  {
    a = lower_bound(container.begin(), container.end(), lo);
    b = upper_bound(container.begin(), container.end(), hi);
    gatherFromRange(r, a, b);
  }
  else // interval is torn by the +-pi overlap
  {
    if (hi.phi >  M_PI) hi.phi -= 2.*M_PI;
    a = lower_bound(container.begin(), container.end(), pi_m);
    b = upper_bound(container.begin(), container.end(), hi);
    gatherFromRange(r, a, b);

    if (lo.phi < -M_PI) lo.phi += 2.*M_PI;
    a = lower_bound(container.begin(), container.end(), lo);
    b = upper_bound(container.begin(), container.end(), pi_p);
    gatherFromRange(r, a, b);
  }
}


void PixelUnpackingRegions::gatherFromRange(Region &r, std::vector<Module>::const_iterator a, std::vector<Module>::const_iterator b)
{
  for(; a != b; ++a)
  {
    // projection in r's direction onto beam's z
    float zmodule = a->z - (  (a->x - beamSpot_.x())*r.cosphi + (a->y - beamSpot_.y())*r.sinphi ) * r.atantheta;

    // do not include modules that project too far in z
    if ( std::abs(zmodule) > r.maxZ ) continue;

    feds_.insert(a->fed);
    modules_.insert(a->id);
  }
}


bool PixelUnpackingRegions::mayUnpackFED(unsigned int fed_n) const
{
  if (feds_.count(fed_n)) return true;
  return false;
}

unsigned int PixelUnpackingRegions::nBarrelFEDs() const
{
  return std::count_if(feds_.begin(), feds_.end(), isBPIXFED );
}

unsigned int PixelUnpackingRegions::nForwardFEDs() const
{
  return std::count_if(feds_.begin(), feds_.end(), isFPIXFED );
}


bool PixelUnpackingRegions::mayUnpackModule(unsigned int id) const
{
  if (modules_.count(id)) return true;
  return false;
}

unsigned int PixelUnpackingRegions::nBarrelModules() const
{
  return std::count_if(modules_.begin(), modules_.end(), isBPIXModule );
}

unsigned int PixelUnpackingRegions::nForwardModules() const
{
  return std::count_if(modules_.begin(), modules_.end(), isFPIXModule );
}
