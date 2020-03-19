#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

//#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
//#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
//#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CaloGeometryTools/interface/CrystalWindowMap.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloDirectionOperations.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
//#include "FastSimulation/Utilities/interface/Histos.h"

//#include "Math/GenVector/Transform3D.h"
#include "FastSimulation/CaloGeometryTools/interface/Transform3DPJ.h"

#include <algorithm>
#include <cmath>

typedef ROOT::Math::Plane3D::Vector Vector;
typedef ROOT::Math::Plane3D::Point Point;
typedef ROOT::Math::Transform3DPJ Transform3DR;

EcalHitMaker::EcalHitMaker(CaloGeometryHelper* theCalo,
                           const XYZPoint& ecalentrance,
                           const DetId& cell,
                           int onEcal,
                           unsigned size,
                           unsigned showertype,
                           const RandomEngineAndDistribution* engine)
    : CaloHitMaker(theCalo, DetId::Ecal, ((onEcal == 1) ? EcalBarrel : EcalEndcap), onEcal, showertype),
      EcalEntrance_(ecalentrance),
      onEcal_(onEcal),
      myTrack_(nullptr),
      random(engine) {
#ifdef FAMOSDEBUG
  myHistos = Histos::instance();
#endif
  //  myHistos->debug("Constructeur EcalHitMaker");
  simulatePreshower_ = true;
  X0depthoffset_ = 0.;
  X0PS1_ = 0.;
  X0PS2_ = 0.;
  X0PS2EE_ = 0.;
  X0ECAL_ = 0.;
  X0EHGAP_ = 0.;
  X0HCAL_ = 0.;
  L0PS1_ = 0.;
  L0PS2_ = 0.;
  L0PS2EE_ = 0.;
  L0ECAL_ = 0.;
  L0EHGAP_ = 0.;
  L0HCAL_ = 0.;
  maxX0_ = 0.;
  totalX0_ = 0;
  totalL0_ = 0.;
  pulledPadProbability_ = 1.;
  outsideWindowEnergy_ = 0.;
  rearleakage_ = 0.;
  bfactor_ = 1.;
  ncrystals_ = 0;

  doreorg_ = !showertype;

  hitmaphasbeencalculated_ = false;

  if (onEcal)
    myCalorimeter->buildCrystal(cell, pivot_);
  else
    pivot_ = Crystal();
  central_ = onEcal == 1;
  ecalFirstSegment_ = -1;

  myCrystalWindowMap_ = nullptr;
  // In some cases, a "dummy" grid, not based on a cell, can be built. The previous variables
  // should however be initialized. In such a case onEcal=0
  if (!onEcal)
    return;

  // Same size in eta-phi
  etasize_ = size;
  phisize_ = size;

  // Build the grid
  // The result is put in CellsWindow and is ordered by distance to the pivot
  myCalorimeter->getWindow(pivot_.getDetId(), size, size, CellsWindow_);

  buildGeometry();
  //  std::cout << " Geometry built " << regionOfInterest_.size() << std::endl;

  truncatedGrid_ = CellsWindow_.size() != (etasize_ * phisize_);

  // A local vector of corners
  mycorners.resize(4);
  corners.resize(4);

#ifdef DEBUGGW
  myHistos->fill("h10", EcalEntrance_.eta(), CellsWindow_.size());
  if (onEcal == 2) {
    myHistos->fill("h20", EcalEntrance_.perp(), CellsWindow_.size());
    if (EcalEntrance_.perp() > 70 && EcalEntrance_.perp() < 80 && CellsWindow_.size() < 35) {
      std::cout << " Truncated grid " << CellsWindow_.size() << " " << EcalEntrance_.perp() << std::endl;
      std::cout << " Pivot "
                << myCalorimeter->getEcalEndcapGeometry()->getGeometry(pivot_.getDetId())->getPosition().perp();
      std::cout << EEDetId(pivot_.getDetId()) << std::endl;

      std::cout << " Test getClosestCell " << EcalEntrance_ << std::endl;
      DetId testcell = myCalorimeter->getClosestCell(EcalEntrance_, true, false);
      std::cout << " Result " << EEDetId(testcell) << std::endl;
      std::cout << " Position " << myCalorimeter->getEcalEndcapGeometry()->getGeometry(testcell)->getPosition()
                << std::endl;
    }
  }

#endif
}

EcalHitMaker::~EcalHitMaker() {
  if (myCrystalWindowMap_ != nullptr) {
    delete myCrystalWindowMap_;
  }
}

bool EcalHitMaker::addHitDepth(double r, double phi, double depth) {
  //  std::cout << " Add hit depth called; Current deph is "  << currentdepth_;
  //  std::cout << " Required depth is " << depth << std::endl;
  depth += X0depthoffset_;
  double sp(1.);
  r *= radiusFactor_;
  CLHEP::Hep2Vector point(r * std::cos(phi), r * std::sin(phi));

  unsigned xtal = fastInsideCell(point, sp);
  //  if(cellid.isZero()) std::cout << " cell is Zero " << std::endl;
  //  if(xtal<1000)
  //    {
  //      std::cout << "Result " << regionOfInterest_[xtal].getX0Back() << " " ;
  //      std::cout << depth << std::endl;
  //    }
  //  myHistos->fill("h5000",depth);
  if (xtal < 1000) {
    //      myHistos->fill("h5002",regionOfInterest_[xtal].getX0Back(),depth);
    //      myHistos->fill("h5003",ecalentrance_.eta(),maxX0_);
    if (regionOfInterest_[xtal].getX0Back() > depth) {
      hits_[xtal] += spotEnergy;
      //	  myHistos->fill("h5005",r);
      return true;
    } else {
      rearleakage_ += spotEnergy;
    }
  } else {
    //      std::cout << " Return false " << std::endl;
    //      std::cout << " Add hit depth called; Current deph is "  << currentdepth_;
    //      std::cout << " Required depth is " << depth << std::endl;
    //      std::cout << " R = " << r << " " << radiusFactor_ << std::endl;
  }

  outsideWindowEnergy_ += spotEnergy;
  return false;
}

//bool EcalHitMaker::addHitDepth(double r,double phi,double depth)
//{
//  std::map<unsigned,float>::iterator itcheck=hitMap_.find(pivot_.getDetId().rawId());
//  if(itcheck==hitMap_.end())
//    {
//      hitMap_.insert(std::pair<uint32_t,float>(pivot_.getDetId().rawId(),spotEnergy));
//    }
//  else
//    {
//      itcheck->second+=spotEnergy;
//    }
//  return true;
//}

bool EcalHitMaker::addHit(double r, double phi, unsigned layer) {
  //  std::cout <<" Addhit " << std::endl;
  //  std::cout << " Before insideCell " << std::endl;
  double sp(1.);
  //  std::cout << " Trying to add " << r << " " << phi << " " << radiusFactor_ << std::endl;
  r *= radiusFactor_;
  CLHEP::Hep2Vector point(r * std::cos(phi), r * std::sin(phi));
  //  std::cout << "point " << point << std::endl;
  //  CellID cellid=insideCell(point,sp);
  unsigned xtal = fastInsideCell(point, sp);
  //  if(cellid.isZero()) std::cout << " cell is Zero " << std::endl;
  if (xtal < 1000) {
    if (sp == 1.)
      hits_[xtal] += spotEnergy;
    else
      hits_[xtal] += (random->flatShoot() < sp) * spotEnergy;
    return true;
  }

  outsideWindowEnergy_ += spotEnergy;
  //  std::cout << " This hit ; r= " << point << " hasn't been added "<<std::endl;
  //  std::cout << " Xtal " << xtal << std::endl;
  //  for(unsigned ip=0;ip<npadsatdepth_;++ip)
  //    {
  //      std::cout << padsatdepth_[ip] << std::endl;
  //    }

  return false;
}

// Temporary solution
//bool EcalHitMaker::addHit(double r,double phi,unsigned layer)
//{
//  std::map<unsigned,float>::iterator itcheck=hitMap_.find(pivot_.getDetId().rawId());
//  if(itcheck==hitMap_.end())
//    {
//      hitMap_.insert(std::pair<uint32_t,float>(pivot_.getDetId().rawId(),spotEnergy));
//    }
//  else
//    {
//      itcheck->second+=spotEnergy;
//    }
//  return true;
//}

unsigned EcalHitMaker::fastInsideCell(const CLHEP::Hep2Vector& point, double& sp, bool debug) {
  //  debug = true;
  bool found = false;
  unsigned niter = 0;
  // something clever has to be implemented here
  unsigned d1, d2;
  convertIntegerCoordinates(point.x(), point.y(), d1, d2);
  //  std::cout << "Fastinside cell " << point.x() << " " <<  point.y() << " " << d1 << " "<< d2 << " " << nx_ << " " << ny_ << std::endl;
  if (d1 >= nx_ || d2 >= ny_) {
    //      std::cout << " Not in the map " <<std::endl;
    return 9999;
  }
  unsigned cell = myCrystalNumberArray_[d1][d2];
  // We are likely to be lucky
  //  std::cout << " Got the cell " << cell << std::endl;
  if (validPads_[cell] && padsatdepth_[cell].inside(point)) {
    //      std::cout << " We are lucky " << cell << std::endl;
    sp = padsatdepth_[cell].survivalProbability();
    return cell;
  }

  //  std::cout << "Starting the loop " << std::endl;
  bool status(true);
  const std::vector<unsigned>& localCellVector(myCrystalWindowMap_->getCrystalWindow(cell, status));
  if (status) {
    unsigned size = localCellVector.size();
    //      std::cout << " Starting from " << EBDetId(regionOfInterest_[cell].getDetId()) << std::endl;
    //      const std::vector<DetId>& neighbours=myCalorimeter->getNeighbours(regionOfInterest_[cell].getDetId());
    //      std::cout << " The neighbours are " << std::endl;
    //      for(unsigned ic=0;ic<neighbours.size(); ++ic)
    //	{
    //	  std::cout << EBDetId(neighbours[ic]) << std::endl;
    //	}
    //      std::cout << " Done " << std::endl;
    for (unsigned ic = 0; ic < 8 && ic < size; ++ic) {
      unsigned iq = localCellVector[ic];
      //	  std::cout << " Testing " << EBDetId(regionOfInterest_[iq].getDetId()) << std::endl; ;
      //	  std::cout << " " << iq << std::endl;
      //	  std::cout << padsatdepth_[iq] ;
      if (validPads_[iq] && padsatdepth_[iq].inside(point)) {
        //	      std::cout << " Yes " << std::endl;
        //	      myHistos->fill("h1000",niter);
        sp = padsatdepth_[iq].survivalProbability();
        //	      std::cout << "Finished the loop " << niter << std::endl;
        //	      std::cout << "Inside " << std::endl;
        return iq;
      }
      //	  std::cout << " Not inside " << std::endl;
      //	  std::cout << "No " << std::endl;
      ++niter;
    }
  }
  if (debug)
    std::cout << " not found in a quad, let's check the " << ncrackpadsatdepth_ << " cracks " << std::endl;
  //  std::cout << "Finished the loop " << niter << std::endl;
  // Let's check the cracks
  //  std::cout << " Let's check the cracks " << ncrackpadsatdepth_ << " " << crackpadsatdepth_.size() << std::endl;
  unsigned iquad = 0;
  unsigned iquadinside = 999;
  while (iquad < ncrackpadsatdepth_ && !found) {
    //      std::cout << " Inside the while " << std::endl;
    if (crackpadsatdepth_[iquad].inside(point)) {
      iquadinside = iquad;
      found = true;
      sp = crackpadsatdepth_[iquad].survivalProbability();
    }
    ++iquad;
    ++niter;
  }
  //  myHistos->fill("h1002",niter);
  if (!found && debug)
    std::cout << " Not found in the cracks " << std::endl;
  return (found) ? crackpadsatdepth_[iquadinside].getNumber() : 9999;
}

void EcalHitMaker::setTrackParameters(const XYZNormal& normal, double X0depthoffset, const FSimTrack& theTrack) {
  //  myHistos->debug("setTrackParameters");
  //  std::cout << " Track " << theTrack << std::endl;
  intersections_.clear();
  // This is certainly enough
  intersections_.reserve(50);
  myTrack_ = &theTrack;
  normal_ = normal.Unit();
  X0depthoffset_ = X0depthoffset;
  cellLine(intersections_);
  buildSegments(intersections_);
  //  std::cout << " Segments " << segments_.size() << std::endl;
  //  for(unsigned ii=0; ii<segments_.size() ; ++ii)
  //    {
  //      std::cout << segments_[ii] << std::endl;
  //    }

  // This is only needed in case of electromagnetic showers
  if (EMSHOWER && onEcal_ && ecalTotalX0() > 0.) {
    //      std::cout << "Total X0  " << ecalTotalX0() << std::endl;
    for (unsigned ic = 0; ic < ncrystals_; ++ic) {
      for (unsigned idir = 0; idir < 4; ++idir) {
        XYZVector norm = regionOfInterest_[ic].exitingNormal(CaloDirectionOperations::Side(idir));
        regionOfInterest_[ic].crystalNeighbour(idir).setToBeGlued((norm.Dot(normal_) < 0.));
      }
      // Now calculate the distance in X0 of the back sides of the crystals
      // (only for EM showers)
      if (EMSHOWER) {
        XYZVector dir = regionOfInterest_[ic].getBackCenter() - segments_[ecalFirstSegment_].entrance();
        double dist = dir.Dot(normal_);
        double absciss = dist + segments_[ecalFirstSegment_].sEntrance();
        std::vector<CaloSegment>::const_iterator segiterator;
        // First identify the correct segment
        //      std::cout << " Crystal " << ic  << regionOfInterest_[ic].getBackCenter() ;
        //      std::cout << " Entrance : " << segments_[ecalFirstSegment_].entrance()<< std::endl;
        //      std::cout << " Looking for the segment " << dist << std::endl;
        segiterator = find_if(segments_.begin(), segments_.end(), CaloSegment::inSegment(absciss));
        //      std::cout << " Done " << std::endl;
        if (segiterator == segments_.end()) {
          // in this case, we won't have any problem. No need to
          // calculate the real depth.
          regionOfInterest_[ic].setX0Back(9999);
        } else {
          DetId::Detector det(segiterator->whichDetector());
          if (det != DetId::Ecal) {
            regionOfInterest_[ic].setX0Back(9999);
          } else {
            double x0 = segiterator->x0FromCm(dist);
            if (x0 < maxX0_)
              maxX0_ = x0;
            regionOfInterest_[ic].setX0Back(x0);
          }
        }
        //  myHistos->fill("h4000",ecalentrance_.eta(), regionOfInterest_[ic].getX0Back());
        //}
        //      else
        //{
        //   // in this case, we won't have any problem. No need to
        //  // calculate the real depth.
        //  regionOfInterest_[ic].setX0Back(9999);
        //}
      }  //EMSHOWER
    }    // ndir
    //      myHistos->fill("h6000",segments_[ecalFirstSegment_].entrance().eta(),maxX0_);
  }
  //  std::cout << "Leaving setTrackParameters" << std::endl
}

void EcalHitMaker::cellLine(std::vector<CaloPoint>& cp) {
  cp.clear();
  //  if(myTrack->onVFcal()!=2)
  //    {
  if (!central_ && onEcal_ && simulatePreshower_)
    preshowerCellLine(cp);
  if (onEcal_)
    ecalCellLine(EcalEntrance_, EcalEntrance_ + normal_, cp);
  //    }

  XYZPoint vertex(myTrack_->vertex().position().Vect());

  //sort the points by distance (in the ECAL they are not necessarily ordered)
  XYZVector dir(0., 0., 0.);
  if (myTrack_->onLayer1()) {
    vertex = (myTrack_->layer1Entrance().vertex()).Vect();
    dir = myTrack_->layer1Entrance().Vect().Unit();
  } else if (myTrack_->onLayer2()) {
    vertex = (myTrack_->layer2Entrance().vertex()).Vect();
    dir = myTrack_->layer2Entrance().Vect().Unit();
  } else if (myTrack_->onEcal()) {
    vertex = (myTrack_->ecalEntrance().vertex()).Vect();
    dir = myTrack_->ecalEntrance().Vect().Unit();
  } else if (myTrack_->onHcal()) {
    vertex = (myTrack_->hcalEntrance().vertex()).Vect();
    dir = myTrack_->hcalEntrance().Vect().Unit();
  } else if (myTrack_->onVFcal() == 2) {
    vertex = (myTrack_->vfcalEntrance().vertex()).Vect();
    dir = myTrack_->vfcalEntrance().Vect().Unit();
  } else {
    std::cout << " Problem with the grid " << std::endl;
  }

  // Move the vertex for distance comparison (5cm)
  vertex -= 5. * dir;
  CaloPoint::DistanceToVertex myDistance(vertex);
  sort(cp.begin(), cp.end(), myDistance);

  // The intersections with the HCAL shouldn't need to be sorted
  // with the N.I it is actually a source of problems
  hcalCellLine(cp);

  //  std::cout << " Intersections ordered by distance to " << vertex << std::endl;
  //
  //  for (unsigned ic=0;ic<cp.size();++ic)
  //    {
  //      XYZVector t=cp[ic]-vertex;
  //      std::cout << cp[ic] << " " << t.mag() << std::endl;
  //    }
}

void EcalHitMaker::preshowerCellLine(std::vector<CaloPoint>& cp) const {
  //  FSimEvent& mySimEvent = myEventMgr->simSignal();
  //  FSimTrack myTrack = mySimEvent.track(fsimtrack_);
  //  std::cout << "FsimTrack " << fsimtrack_<< std::endl;
  //  std::cout << " On layer 1 " << myTrack.onLayer1() << std::endl;
  // std::cout << " preshowerCellLine " << std::endl;
  if (myTrack_->onLayer1()) {
    XYZPoint point1 = (myTrack_->layer1Entrance().vertex()).Vect();
    double phys_eta = myTrack_->layer1Entrance().eta();
    double cmthickness = myCalorimeter->layer1Properties(1)->thickness(phys_eta);

    if (cmthickness > 0) {
      XYZVector dir = myTrack_->layer1Entrance().Vect().Unit();
      XYZPoint point2 = point1 + dir * cmthickness;

      CaloPoint cp1(DetId::Ecal, EcalPreshower, 1, point1);
      CaloPoint cp2(DetId::Ecal, EcalPreshower, 1, point2);
      cp.push_back(cp1);
      cp.push_back(cp2);
    } else {
      //      std::cout << "Track on ECAL " << myTrack.EcalEntrance_().vertex()*0.1<< std::endl;
    }
  }

  //  std::cout << " On layer 2 " << myTrack.onLayer2() << std::endl;
  if (myTrack_->onLayer2()) {
    XYZPoint point1 = (myTrack_->layer2Entrance().vertex()).Vect();
    double phys_eta = myTrack_->layer2Entrance().eta();
    double cmthickness = myCalorimeter->layer2Properties(1)->thickness(phys_eta);
    if (cmthickness > 0) {
      XYZVector dir = myTrack_->layer2Entrance().Vect().Unit();
      XYZPoint point2 = point1 + dir * cmthickness;

      CaloPoint cp1(DetId::Ecal, EcalPreshower, 2, point1);
      CaloPoint cp2(DetId::Ecal, EcalPreshower, 2, point2);

      cp.push_back(cp1);
      cp.push_back(cp2);
    } else {
      //      std::cout << "Track on ECAL " << myTrack.EcalEntrance_().vertex()*0.1 << std::endl;
    }
  }
  //  std::cout << " Exit preshower CellLine " << std::endl;
}

void EcalHitMaker::hcalCellLine(std::vector<CaloPoint>& cp) const {
  //  FSimEvent& mySimEvent = myEventMgr->simSignal();
  //  FSimTrack myTrack = mySimEvent.track(fsimtrack_);
  int onHcal = myTrack_->onHcal();

  if (onHcal <= 2 && onHcal > 0) {
    XYZPoint point1 = (myTrack_->hcalEntrance().vertex()).Vect();

    double eta = point1.eta();
    // HCAL thickness in cm (assuming that the particle is coming from 000)
    double thickness = myCalorimeter->hcalProperties(onHcal)->thickness(eta);
    cp.push_back(CaloPoint(DetId::Hcal, point1));
    XYZVector dir = myTrack_->hcalEntrance().Vect().Unit();
    XYZPoint point2 = point1 + dir * thickness;

    cp.push_back(CaloPoint(DetId::Hcal, point2));
  }
  int onVFcal = myTrack_->onVFcal();
  if (onVFcal == 2) {
    XYZPoint point1 = (myTrack_->vfcalEntrance().vertex()).Vect();
    double eta = point1.eta();
    // HCAL thickness in cm (assuming that the particle is coming from 000)
    double thickness = myCalorimeter->hcalProperties(3)->thickness(eta);
    cp.push_back(CaloPoint(DetId::Hcal, point1));
    XYZVector dir = myTrack_->vfcalEntrance().Vect().Unit();
    if (thickness > 0) {
      XYZPoint point2 = point1 + dir * thickness;
      cp.push_back(CaloPoint(DetId::Hcal, point2));
    }
  }
}

void EcalHitMaker::ecalCellLine(const XYZPoint& a, const XYZPoint& b, std::vector<CaloPoint>& cp) {
  //  std::vector<XYZPoint> corners;
  //  corners.resize(4);
  unsigned ic = 0;
  double t;
  XYZPoint xp;
  DetId c_entrance, c_exit;
  bool entrancefound(false), exitfound(false);
  //  std::cout << " Look for intersections " << ncrystals_ << std::endl;
  //  std::cout << " regionOfInterest_ " << truncatedGrid_ << " " << regionOfInterest_.size() << std::endl;
  // try to determine the number of crystals to test
  // First determine the incident angle
  double angle = std::acos(normal_.Dot(regionOfInterest_[0].getAxis().Unit()));

  //  std::cout << " Normal " << normal_<< " Axis " << regionOfInterest_[0].getAxis().Unit() << std::endl;
  double backdistance = std::sqrt(regionOfInterest_[0].getAxis().mag2()) * std::tan(angle);
  // 1/2.2cm = 0.45
  //   std::cout << " Angle " << angle << std::endl;
  //   std::cout << " Back distance " << backdistance << std::endl;
  unsigned ncrystals = (unsigned)(backdistance * 0.45);
  unsigned highlim = (ncrystals + 4);
  highlim *= highlim;
  if (highlim > ncrystals_)
    highlim = ncrystals_;
  //  unsigned lowlim=(ncrystals>2)? (ncrystals-2):0;
  //  std::cout << " Ncrys " << ncrystals << std::endl;

  while (ic < ncrystals_ && (ic < highlim || !exitfound)) {
    // Check front side
    //      if(!entrancefound)
    {
      const Plane3D& plan = regionOfInterest_[ic].getFrontPlane();
      //	  XYZVector axis1=(plan.Normal());
      //	  XYZVector axis2=regionOfInterest_[ic].getFirstEdge();
      xp = intersect(plan, a, b, t, false);
      regionOfInterest_[ic].getFrontSide(corners);
      //	  CrystalPad pad(9999,onEcal_,corners,regionOfInterest_[ic].getCorner(0),axis1,axis2);
      //	  if(pad.globalinside(xp))
      if (inside3D(corners, xp)) {
        cp.push_back(CaloPoint(regionOfInterest_[ic].getDetId(), UP, xp));
        entrancefound = true;
        c_entrance = regionOfInterest_[ic].getDetId();
        //      myHistos->fill("j12",highlim,ic);
      }
    }

    // check rear side
    //	if(!exitfound)
    {
      const Plane3D& plan = regionOfInterest_[ic].getBackPlane();
      //	  XYZVector axis1=(plan.Normal());
      //	  XYZVector axis2=regionOfInterest_[ic].getFifthEdge();
      xp = intersect(plan, a, b, t, false);
      regionOfInterest_[ic].getBackSide(corners);
      //	  CrystalPad pad(9999,onEcal_,corners,regionOfInterest_[ic].getCorner(4),axis1,axis2);
      //	  if(pad.globalinside(xp))
      if (inside3D(corners, xp)) {
        cp.push_back(CaloPoint(regionOfInterest_[ic].getDetId(), DOWN, xp));
        exitfound = true;
        c_exit = regionOfInterest_[ic].getDetId();
        //	      std::cout << " Crystal : " << ic << std::endl;
        //	      myHistos->fill("j10",highlim,ic);
      }
    }

    if (entrancefound && exitfound && c_entrance == c_exit)
      return;
    // check lateral sides
    for (unsigned iside = 0; iside < 4; ++iside) {
      const Plane3D& plan = regionOfInterest_[ic].getLateralPlane(iside);
      xp = intersect(plan, a, b, t, false);
      //	  XYZVector axis1=(plan.Normal());
      //	  XYZVector axis2=regionOfInterest_[ic].getLateralEdge(iside);
      regionOfInterest_[ic].getLateralSide(iside, corners);
      //	  CrystalPad pad(9999,onEcal_,corners,regionOfInterest_[ic].getCorner(iside),axis1,axis2);
      //	  if(pad.globalinside(xp))
      if (inside3D(corners, xp)) {
        cp.push_back(CaloPoint(regionOfInterest_[ic].getDetId(), CaloDirectionOperations::Side(iside), xp));
        //	      std::cout << cp[cp.size()-1] << std::endl;
      }
    }
    // Go to next crystal
    ++ic;
  }
}

void EcalHitMaker::buildSegments(const std::vector<CaloPoint>& cp) {
  //  myHistos->debug();
  //  TimeMe theT("FamosGrid::buildSegments");
  unsigned size = cp.size();
  if (size % 2 != 0) {
    //      std::cout << " There is a problem " << std::endl;
    return;
  }
  //  myHistos->debug();
  unsigned nsegments = size / 2;
  segments_.reserve(nsegments);
  if (size == 0)
    return;
  // curv abs
  double s = 0.;
  double sX0 = 0.;
  double sL0 = 0.;

  unsigned ncrossedxtals = 0;
  unsigned is = 0;
  while (is < nsegments) {
    if (cp[2 * is].getDetId() != cp[2 * is + 1].getDetId() && cp[2 * is].whichDetector() != DetId::Hcal &&
        cp[2 * is + 1].whichDetector() != DetId::Hcal) {
      //	  std::cout << " Problem with the segments " << std::endl;
      //	  std::cout << cp[2*is].whichDetector() << " " << cp[2*is+1].whichDetector() << std::endl;
      //	  std::cout << is << " " <<cp[2*is].getDetId().rawId() << std::endl;
      //	  std::cout << (2*is+1) << " " <<cp[2*is+1].getDetId().rawId() << std::endl;
      ++is;
      continue;
    }

    // Check if it is a Preshower segment - Layer 1
    // One segment per layer, nothing between
    //      myHistos->debug("Just avant Preshower");
    if (cp[2 * is].whichDetector() == DetId::Ecal && cp[2 * is].whichSubDetector() == EcalPreshower &&
        cp[2 * is].whichLayer() == 1) {
      if (cp[2 * is + 1].whichDetector() == DetId::Ecal && cp[2 * is + 1].whichSubDetector() == EcalPreshower &&
          cp[2 * is + 1].whichLayer() == 1) {
        CaloSegment preshsegment(cp[2 * is], cp[2 * is + 1], s, sX0, sL0, CaloSegment::PS, myCalorimeter);
        segments_.push_back(preshsegment);
        //	      std::cout << " Added (1-1)" << preshsegment << std::endl;
        s += preshsegment.length();
        sX0 += preshsegment.X0length();
        sL0 += preshsegment.L0length();
        X0PS1_ += preshsegment.X0length();
        L0PS1_ += preshsegment.L0length();
      } else {
        std::cout << " Strange segment between Preshower1 and " << cp[2 * is + 1].whichDetector();
        std::cout << std::endl;
      }
      ++is;
      continue;
    }

    // Check if it is a Preshower segment - Layer 2
    // One segment per layer, nothing between
    if (cp[2 * is].whichDetector() == DetId::Ecal && cp[2 * is].whichSubDetector() == EcalPreshower &&
        cp[2 * is].whichLayer() == 2) {
      if (cp[2 * is + 1].whichDetector() == DetId::Ecal && cp[2 * is + 1].whichSubDetector() == EcalPreshower &&
          cp[2 * is + 1].whichLayer() == 2) {
        CaloSegment preshsegment(cp[2 * is], cp[2 * is + 1], s, sX0, sL0, CaloSegment::PS, myCalorimeter);
        segments_.push_back(preshsegment);
        //	      std::cout << " Added (1-2)" << preshsegment << std::endl;
        s += preshsegment.length();
        sX0 += preshsegment.X0length();
        sL0 += preshsegment.L0length();
        X0PS2_ += preshsegment.X0length();
        L0PS2_ += preshsegment.L0length();

        // material between preshower and EE
        if (is < (nsegments - 1) && cp[2 * is + 2].whichDetector() == DetId::Ecal &&
            cp[2 * is + 2].whichSubDetector() == EcalEndcap) {
          CaloSegment gapsef(cp[2 * is + 1], cp[2 * is + 2], s, sX0, sL0, CaloSegment::PSEEGAP, myCalorimeter);
          segments_.push_back(gapsef);
          s += gapsef.length();
          sX0 += gapsef.X0length();
          sL0 += gapsef.L0length();
          X0PS2EE_ += gapsef.X0length();
          L0PS2EE_ += gapsef.L0length();
          //		  std::cout << " Created  a segment " << gapsef.length()<< " " << gapsef.X0length()<< std::endl;
        }
      } else {
        std::cout << " Strange segment between Preshower2 and " << cp[2 * is + 1].whichDetector();
        std::cout << std::endl;
      }
      ++is;
      continue;
    }
    // Now deal with the ECAL
    // One segment in each crystal. Segment corresponding to cracks/gaps are added
    //      myHistos->debug("Just avant ECAL");
    if (cp[2 * is].whichDetector() == DetId::Ecal &&
        (cp[2 * is].whichSubDetector() == EcalBarrel || cp[2 * is].whichSubDetector() == EcalEndcap)) {
      if (cp[2 * is + 1].whichDetector() == DetId::Ecal &&
          (cp[2 * is + 1].whichSubDetector() == EcalBarrel || cp[2 * is + 1].whichSubDetector() == EcalEndcap)) {
        DetId cell2 = cp[2 * is + 1].getDetId();
        // set the real entrance
        if (ecalFirstSegment_ < 0)
          ecalFirstSegment_ = segments_.size();

        // !! Approximatiom : the first segment is always in a crystal
        if (cp[2 * is].getDetId() == cell2) {
          CaloSegment segment(cp[2 * is], cp[2 * is + 1], s, sX0, sL0, CaloSegment::PbWO4, myCalorimeter);
          segments_.push_back(segment);
          // std::cout << " Added (2)" << segment << std::endl;
          s += segment.length();
          sX0 += segment.X0length();
          sL0 += segment.L0length();
          X0ECAL_ += segment.X0length();
          L0ECAL_ += segment.L0length();
          ++ncrossedxtals;
          ++is;
        } else {
          std::cout << " One more bug in the segment " << std::endl;
          ++is;
        }
        // Now check if a gap or crack should be added
        if (is > 0 && is < nsegments) {
          DetId cell3 = cp[2 * is].getDetId();
          if (cp[2 * is].whichDetector() != DetId::Hcal) {
            // Crack inside the ECAL
            bool bordercrossing = myCalorimeter->borderCrossing(cell2, cell3);
            CaloSegment cracksegment(cp[2 * is - 1],
                                     cp[2 * is],
                                     s,
                                     sX0,
                                     sL0,
                                     (bordercrossing) ? CaloSegment::CRACK : CaloSegment::GAP,
                                     myCalorimeter);
            segments_.push_back(cracksegment);
            s += cracksegment.length();
            sX0 += cracksegment.X0length();
            sL0 += cracksegment.L0length();
            X0ECAL_ += cracksegment.X0length();
            L0ECAL_ += cracksegment.L0length();
            //   std::cout <<" Added(3) "<< cracksegment << std::endl;
          } else {
            // a segment corresponding to ECAL/HCAL transition should be
            // added here
            CaloSegment cracksegment(cp[2 * is - 1], cp[2 * is], s, sX0, sL0, CaloSegment::ECALHCALGAP, myCalorimeter);
            segments_.push_back(cracksegment);
            s += cracksegment.length();
            sX0 += cracksegment.X0length();
            sL0 += cracksegment.L0length();
            X0EHGAP_ += cracksegment.X0length();
            L0EHGAP_ += cracksegment.L0length();
          }
        }
        continue;
      } else {
        std::cout << " Strange segment between " << cp[2 * is].whichDetector();
        std::cout << " and " << cp[2 * is + 1].whichDetector() << std::endl;
        ++is;
        continue;
      }
    }
    //      myHistos->debug("Just avant HCAL");
    // HCAL
    if (cp[2 * is].whichDetector() == DetId::Hcal && cp[2 * is + 1].whichDetector() == DetId::Hcal) {
      CaloSegment segment(cp[2 * is], cp[2 * is + 1], s, sX0, sL0, CaloSegment::HCAL, myCalorimeter);
      segments_.push_back(segment);
      s += segment.length();
      sX0 += segment.X0length();
      sL0 += segment.L0length();
      X0HCAL_ += segment.X0length();
      L0HCAL_ += segment.L0length();
      //	  std::cout <<" Added(4) "<< segment << std::endl;
      ++is;
    }
  }
  //  std::cout << " PS1 " << X0PS1_ << " " << L0PS1_ << std::endl;
  //  std::cout << " PS2 " << X0PS2_ << " " << L0PS2_ << std::endl;
  //  std::cout << " ECAL " << X0ECAL_ << " " << L0ECAL_ << std::endl;
  //  std::cout << " HCAL " << X0HCAL_ << " " << L0HCAL_ << std::endl;

  totalX0_ = X0PS1_ + X0PS2_ + X0PS2EE_ + X0ECAL_ + X0EHGAP_ + X0HCAL_;
  totalL0_ = L0PS1_ + L0PS2_ + L0PS2EE_ + L0ECAL_ + L0EHGAP_ + L0HCAL_;
  //  myHistos->debug("Just avant le fill");

#ifdef DEBUGCELLLINE
  myHistos->fill("h200", fabs(EcalEntrance_.eta()), X0ECAL_);
  myHistos->fill("h210", EcalEntrance_.phi(), X0ECAL_);
  if (X0ECAL_ < 20)
    myHistos->fill("h212", EcalEntrance_.phi(), X0ECAL_);
  //  if(X0ECAL_<1.)
  //    {
  //      for(unsigned ii=0; ii<segments_.size() ; ++ii)
  //	{
  //	  std::cout << segments_[ii] << std::endl;
  //	}
  //    }
  myHistos->fillByNumber("h30", ncrossedxtals, EcalEntrance_.eta(), X0ECAL_);

  double zvertex = myTrack_->vertex().position().z();

  myHistos->fill("h310", EcalEntrance_.eta(), X0ECAL_);
  if (X0ECAL_ < 22)
    myHistos->fill("h410", EcalEntrance_.phi());
  myHistos->fill("h400", zvertex, X0ECAL_);
#endif
  //  std::cout << " Finished the segments " << std::endl;
}

void EcalHitMaker::buildGeometry() {
  configuredGeometry_ = false;
  ncrystals_ = CellsWindow_.size();
  // create the vector with of pads with the appropriate size
  padsatdepth_.resize(ncrystals_);

  // This is fully correct in the barrel.
  ny_ = phisize_;
  nx_ = ncrystals_ / ny_;
  std::vector<unsigned> empty;
  empty.resize(ny_, 0);
  myCrystalNumberArray_.reserve((unsigned)nx_);
  for (unsigned inx = 0; inx < (unsigned)nx_; ++inx) {
    myCrystalNumberArray_.push_back(empty);
  }

  hits_.resize(ncrystals_, 0.);
  regionOfInterest_.clear();
  regionOfInterest_.resize(ncrystals_);
  validPads_.resize(ncrystals_);
  for (unsigned ic = 0; ic < ncrystals_; ++ic) {
    myCalorimeter->buildCrystal(CellsWindow_[ic], regionOfInterest_[ic]);
    regionOfInterest_[ic].setNumber(ic);
    DetIdMap_.insert(std::pair<DetId, unsigned>(CellsWindow_[ic], ic));
  }

  // Computes the map of the neighbours
  myCrystalWindowMap_ = new CrystalWindowMap(myCalorimeter, regionOfInterest_);
}

// depth is in X0 , L0 (depending on EMSHOWER/HADSHOWER) or in CM if inCM
bool EcalHitMaker::getPads(double depth, bool inCm) {
  //std::cout << " New depth " << depth << std::endl;
  // The first time, the relationship between crystals must be calculated
  // but only in the case of EM showers

  if (EMSHOWER && !configuredGeometry_)
    configureGeometry();

  radiusFactor_ = (EMSHOWER) ? moliereRadius * radiusCorrectionFactor_ : interactionLength;
  detailedShowerTail_ = false;
  if (EMSHOWER)
    currentdepth_ = depth + X0depthoffset_;
  else
    currentdepth_ = depth;

  //  if(currentdepth_>maxX0_+ps1TotalX0()+ps2TotalX0())
  //    {
  //      currentdepth_=maxX0_+ps1TotalX0()+ps2TotalX0()-1.; // the -1 is for safety
  //      detailedShowerTail_=true;
  //    }

  //  std::cout << " FamosGrid::getQuads " << currentdepth_ << " " << maxX0_ << std::endl;

  ncrackpadsatdepth_ = 0;

  xmin_ = ymin_ = 999;
  xmax_ = ymax_ = -999;
  double locxmin, locxmax, locymin, locymax;

  // Get the depth of the pivot
  std::vector<CaloSegment>::const_iterator segiterator;
  // First identify the correct segment

  if (inCm)  // centimeter
  {
    segiterator = find_if(segments_.begin(), segments_.end(), CaloSegment::inSegment(currentdepth_));
  } else {
    // EM shower
    if (EMSHOWER)
      segiterator = find_if(segments_.begin(), segments_.end(), CaloSegment::inX0Segment(currentdepth_));

    //Hadron shower
    if (HADSHOWER)
      segiterator = find_if(segments_.begin(), segments_.end(), CaloSegment::inL0Segment(currentdepth_));
  }
  if (segiterator == segments_.end()) {
    std::cout << " FamosGrid: Could not go at such depth " << depth << std::endl;
    std::cout << " EMSHOWER " << EMSHOWER << std::endl;
    std::cout << " Track " << *myTrack_ << std::endl;
    std::cout << " Segments " << segments_.size() << std::endl;
    for (unsigned ii = 0; ii < segments_.size(); ++ii) {
      std::cout << segments_[ii] << std::endl;
    }

    return false;
  }
  //  std::cout << *segiterator << std::endl;

  if (segiterator->whichDetector() != DetId::Ecal) {
    std::cout << " In  " << segiterator->whichDetector() << std::endl;
    //      buildPreshower();
    return false;
  }

  //  std::cout << *segiterator << std::endl;
  // get the position of the origin

  XYZPoint origin;
  if (inCm) {
    origin = segiterator->positionAtDepthincm(currentdepth_);
  } else {
    if (EMSHOWER)
      origin = segiterator->positionAtDepthinX0(currentdepth_);
    if (HADSHOWER)
      origin = segiterator->positionAtDepthinL0(currentdepth_);
  }
  //  std::cout << " currentdepth_ " << currentdepth_ << " " << origin << std::endl;
  XYZVector newaxis = pivot_.getFirstEdge().Cross(normal_);

  //  std::cout  << "Normal " << normal_ << std::endl;
  //  std::cout << " New axis " << newaxis << std::endl;

  //  std::cout << " ncrystals  " << ncrystals << std::endl;
  plan_ = Plane3D((Vector)normal_, (Point)origin);

  unsigned nquads = 0;
  double sign = (central_) ? -1. : 1.;
  Transform3DR trans((Point)origin,
                     (Point)(origin + normal_),
                     (Point)(origin + newaxis),
                     Point(0, 0, 0),
                     Point(0., 0., sign),
                     Point(0., 1., 0.));
  for (unsigned ic = 0; ic < ncrystals_; ++ic) {
    //      std::cout << " Building geometry for " << regionOfInterest_[ic].getCellID() << std::endl;
    XYZPoint a, b;

    //      std::cout << " Origin " << origin << std::endl;

    //      std::vector<XYZPoint> corners;
    //      corners.reserve(4);
    double dummyt;
    bool hasbeenpulled = false;
    bool behindback = false;
    for (unsigned il = 0; il < 4; ++il) {
      // a is the il-th front corner of the crystal. b is the corresponding rear corner
      regionOfInterest_[ic].getLateralEdges(il, a, b);

      // pull the surface if necessary (only in the front of the crystals)
      XYZPoint aprime = a;
      if (pulled(origin, normal_, a)) {
        b = aprime;
        hasbeenpulled = true;
      }

      // compute the intersection.
      // Check that the intersection is in the [a,b] segment  if HADSHOWER
      // if EMSHOWER the intersection is calculated as if the crystals were infinite
      XYZPoint xx = (EMSHOWER) ? intersect(plan_, a, b, dummyt, false) : intersect(plan_, a, b, dummyt, true);

      if (dummyt > 1)
        behindback = true;
      //	  std::cout << " Intersect " << il << " " << a << " " << b << " " << plan_ << " " << xx << std::endl;
      // check that the intersection actually exists
      if (xx.mag2() != 0) {
        corners[il] = xx;
      }
    }
    //      std::cout << " ncorners " << corners.size() << std::endl;
    if (behindback && EMSHOWER)
      detailedShowerTail_ = true;
    // If the quad is completly defined. Store it !
    if (corners.size() == 4) {
      padsatdepth_[ic] = CrystalPad(ic, corners, trans, bfactor_, !central_);
      // Parameter to be tuned
      if (hasbeenpulled)
        padsatdepth_[ic].setSurvivalProbability(pulledPadProbability_);
      validPads_[ic] = true;
      ++nquads;
      // In principle, this should be done after the quads reorganization. But it would cost one more loop
      //  quadsatdepth_[ic].extrems(locxmin,locxmax,locymin,locymax);
      //  if(locxmin<xmin_) xmin_=locxmin;
      //  if(locymin<ymin_) ymin_=locymin;
      //  if(locxmax>xmax_) xmax_=locxmax;
      //  if(locymax>ymax_) ymax_=locymax;
    } else {
      padsatdepth_[ic] = CrystalPad();
      validPads_[ic] = false;
    }
  }
  //  std::cout << " Number of quads " << quadsatdepth_.size() << std::endl;
  if (doreorg_)
    reorganizePads();
  //  std::cout << "Finished to reorganize " << std::endl;
  npadsatdepth_ = nquads;
  //  std::cout << " prepareCellIDMap " << std::endl;

  // Resize the Quads to allow for some numerical inaccuracy
  // in the "inside" function
  for (unsigned ic = 0; ic < ncrystals_; ++ic) {
    if (!validPads_[ic])
      continue;

    if (EMSHOWER)
      padsatdepth_[ic].resetCorners();

    padsatdepth_[ic].extrems(locxmin, locxmax, locymin, locymax);
    if (locxmin < xmin_)
      xmin_ = locxmin;
    if (locymin < ymin_)
      ymin_ = locymin;
    if (locxmax > xmax_)
      xmax_ = locxmax;
    if (locymax > ymax_)
      ymax_ = locymax;
  }

  sizex_ = (xmax_ - xmin_) / nx_;
  sizey_ = (ymax_ - ymin_) / ny_;

  // Make sure that sizex_ and sizey_ are set before running prepareCellIDMap
  prepareCrystalNumberArray();

  //  std::cout << " Finished  prepareCellIDMap " << std::endl;
  ncrackpadsatdepth_ = crackpadsatdepth_.size();

  return true;
}

void EcalHitMaker::configureGeometry() {
  configuredGeometry_ = true;
  for (unsigned ic = 0; ic < ncrystals_; ++ic) {
    //      std::cout << " Building " << cellids_[ic] << std::endl;
    for (unsigned idir = 0; idir < 8; ++idir) {
      unsigned oppdir = CaloDirectionOperations::oppositeDirection(idir);
      // Is there something else to do ?
      // The relationship with the neighbour may have been set previously.
      if (regionOfInterest_[ic].crystalNeighbour(idir).status() >= 0) {
        //	      std::cout << " Nothing to do " << std::endl;
        continue;
      }

      const DetId& oldcell(regionOfInterest_[ic].getDetId());
      CaloDirection dir = CaloDirectionOperations::neighbourDirection(idir);
      DetId newcell(oldcell);
      if (!myCalorimeter->move(newcell, dir)) {
        // no neighbour in this direction
        regionOfInterest_[ic].crystalNeighbour(idir).setStatus(-1);
        continue;
      }
      // Determine the number of this neighbour
      //	  std::cout << " The neighbour is " << newcell << std::endl;
      std::map<DetId, unsigned>::const_iterator niter(DetIdMap_.find(newcell));
      if (niter == DetIdMap_.end()) {
        //	      std::cout << " The neighbour is not in the map " << std::endl;
        regionOfInterest_[ic].crystalNeighbour(idir).setStatus(-1);
        continue;
      }
      // Now there is a neighbour
      //	  std::cout << " The neighbour is " << niter->second << " " << cellids_[niter->second] << std::endl;
      regionOfInterest_[ic].crystalNeighbour(idir).setNumber(niter->second);
      //	  std::cout << " Managed to set crystalNeighbour " << ic << " " << idir << std::endl;
      //	  std::cout << " Trying  " << niter->second << " " << oppdir << std::endl;
      regionOfInterest_[niter->second].crystalNeighbour(oppdir).setNumber(ic);
      //	  std::cout << " Crack/gap " << std::endl;
      if (myCalorimeter->borderCrossing(oldcell, newcell)) {
        regionOfInterest_[ic].crystalNeighbour(idir).setStatus(1);
        regionOfInterest_[niter->second].crystalNeighbour(oppdir).setStatus(1);
        //	      std::cout << " Crack ! " << std::endl;
      } else {
        regionOfInterest_[ic].crystalNeighbour(idir).setStatus(0);
        regionOfInterest_[niter->second].crystalNeighbour(oppdir).setStatus(0);
        //	      std::cout << " Gap" << std::endl;
      }
    }
  }
  // Magnetic field a la Charlot
  double theta = EcalEntrance_.theta();
  if (theta > M_PI_2)
    theta = M_PI - theta;
  bfactor_ = 1. / (1. + 0.133 * theta);
  if (myCalorimeter->magneticField() == 0.)
    bfactor_ = 1.;
}

// project fPoint on the plane (original,normal)
bool EcalHitMaker::pulled(const XYZPoint& origin, const XYZNormal& normal, XYZPoint& fPoint) const {
  // check if fPoint is behind the origin
  double dotproduct = normal.Dot(fPoint - origin);
  if (dotproduct <= 0.)
    return false;

  //norm of normal is 1
  fPoint -= (1 + dotproduct) * normal;
  return true;
}

void EcalHitMaker::prepareCrystalNumberArray() {
  for (unsigned iq = 0; iq < npadsatdepth_; ++iq) {
    if (!validPads_[iq])
      continue;
    unsigned d1, d2;
    convertIntegerCoordinates(padsatdepth_[iq].center().x(), padsatdepth_[iq].center().y(), d1, d2);
    myCrystalNumberArray_[d1][d2] = iq;
  }
}

void EcalHitMaker::convertIntegerCoordinates(double x, double y, unsigned& ix, unsigned& iy) const {
  int tix = (int)((x - xmin_) / sizex_);
  int tiy = (int)((y - ymin_) / sizey_);
  ix = iy = 9999;
  if (tix >= 0)
    ix = (unsigned)tix;
  if (tiy >= 0)
    iy = (unsigned)tiy;
}

const std::map<CaloHitID, float>& EcalHitMaker::getHits() {
  if (hitmaphasbeencalculated_)
    return hitMap_;
  for (unsigned ic = 0; ic < ncrystals_; ++ic) {
    //calculate time of flight
    float tof = 0.0;
    if (onEcal_ == 1 || onEcal_ == 2)
      tof =
          (myCalorimeter->getEcalGeometry(onEcal_)->getGeometry(regionOfInterest_[ic].getDetId())->getPosition().mag()) /
          29.98;  //speed of light

    if (onEcal_ == 1) {
      CaloHitID current_id(EBDetId(regionOfInterest_[ic].getDetId().rawId()).hashedIndex(), tof, 0);  //no track yet
      hitMap_.insert(std::pair<CaloHitID, float>(current_id, hits_[ic]));
    } else if (onEcal_ == 2) {
      CaloHitID current_id(EEDetId(regionOfInterest_[ic].getDetId().rawId()).hashedIndex(), tof, 0);  //no track yet
      hitMap_.insert(std::pair<CaloHitID, float>(current_id, hits_[ic]));
    }
  }
  hitmaphasbeencalculated_ = true;
  return hitMap_;
}

///////////////////////////// GAPS/CRACKS TREATMENT////////////

void EcalHitMaker::reorganizePads() {
  // Some cleaning first
  //  std::cout << " Starting reorganize " << std::endl;
  crackpadsatdepth_.clear();
  crackpadsatdepth_.reserve(etasize_ * phisize_);
  ncrackpadsatdepth_ = 0;
  std::vector<neighbour> gaps;
  std::vector<std::vector<neighbour> > cracks;
  //  cracks.clear();
  cracks.resize(ncrystals_);

  for (unsigned iq = 0; iq < ncrystals_; ++iq) {
    if (!validPads_[iq])
      continue;

    gaps.clear();
    //   std::cout << padsatdepth_[iq] << std::endl;
    //check all the directions
    for (unsigned iside = 0; iside < 4; ++iside) {
      //	  std::cout << " To be glued " << iside << " " << regionOfInterest_[iq].crystalNeighbour(iside).toBeGlued() << std::endl;
      CaloDirection thisside = CaloDirectionOperations::Side(iside);
      if (regionOfInterest_[iq].crystalNeighbour(iside).toBeGlued()) {
        // look for the neighbour and check that it exists
        int neighbourstatus = regionOfInterest_[iq].crystalNeighbour(iside).status();
        if (neighbourstatus < 0)
          continue;

        unsigned neighbourNumber = regionOfInterest_[iq].crystalNeighbour(iside).number();
        if (!validPads_[neighbourNumber])
          continue;
        // there is a crack between
        if (neighbourstatus == 1) {
          //		  std::cout << " 1 Crack : " << thisside << " " << cellids_[iq]<< " " << cellids_[neighbourNumber] << std::endl;
          cracks[iq].push_back(neighbour(thisside, neighbourNumber));
        }  // else it is a gap
        else {
          gaps.push_back(neighbour(thisside, neighbourNumber));
        }
      }
    }
    // Now lift the gaps
    gapsLifting(gaps, iq);
  }

  unsigned ncracks = cracks.size();
  //  std::cout << " Cracks treatment : " << cracks.size() << std::endl;
  for (unsigned icrack = 0; icrack < ncracks; ++icrack) {
    //      std::cout << " Crack number " << crackiter->first << std::endl;
    cracksPads(cracks[icrack], icrack);
  }
}

//dir 2 = N,E,W,S
CLHEP::Hep2Vector& EcalHitMaker::correspondingEdge(neighbour& myneighbour, CaloDirection dir2) {
  CaloDirection dir = CaloDirectionOperations::oppositeSide(myneighbour.first);
  CaloDirection corner = CaloDirectionOperations::add2d(dir, dir2);
  //  std::cout << "Corresponding Edge " << dir<< " " << dir2 << " " << corner << std::endl;
  return padsatdepth_[myneighbour.second].edge(corner);
}

bool EcalHitMaker::diagonalEdge(unsigned myPad, CaloDirection dir, CLHEP::Hep2Vector& point) {
  unsigned idir = CaloDirectionOperations::neighbourDirection(dir);
  if (regionOfInterest_[myPad].crystalNeighbour(idir).status() < 0)
    return false;
  unsigned nneighbour = regionOfInterest_[myPad].crystalNeighbour(idir).number();
  if (!validPads_[nneighbour]) {
    //      std::cout << " Wasn't able to move " << std::endl;
    return false;
  }
  point = padsatdepth_[nneighbour].edge(CaloDirectionOperations::oppositeSide(dir));
  return true;
}

bool EcalHitMaker::unbalancedDirection(const std::vector<neighbour>& dirs,
                                       unsigned& unb,
                                       unsigned& dir1,
                                       unsigned& dir2) {
  if (dirs.size() == 1)
    return false;
  if (dirs.size() % 2 == 0)
    return false;
  CaloDirection tmp;
  tmp = CaloDirectionOperations::add2d(dirs[0].first, dirs[1].first);
  if (tmp == NONE) {
    unb = 2;
    dir1 = 0;
    dir2 = 1;
    return true;
  }
  tmp = CaloDirectionOperations::add2d(dirs[0].first, dirs[2].first);
  if (tmp == NONE) {
    unb = 1;
    dir1 = 0;
    dir2 = 2;
    return true;
  }
  unb = 0;
  dir1 = 1;
  dir2 = 2;
  return true;
}

void EcalHitMaker::gapsLifting(std::vector<neighbour>& gaps, unsigned iq) {
  //  std::cout << " Entering gapsLifting "  << std::endl;
  CrystalPad& myPad = padsatdepth_[iq];
  unsigned ngaps = gaps.size();
  constexpr bool debug = false;
  if (ngaps == 1) {
    if (debug) {
      std::cout << " Avant " << ngaps << " " << gaps[0].first << std::endl;
      std::cout << myPad << std::endl;
    }
    if (gaps[0].first == NORTH || gaps[0].first == SOUTH) {
      CaloDirection dir1 = CaloDirectionOperations::add2d(gaps[0].first, EAST);
      CaloDirection dir2 = CaloDirectionOperations::add2d(gaps[0].first, WEST);
      myPad.edge(dir1) = correspondingEdge(gaps[0], EAST);
      myPad.edge(dir2) = correspondingEdge(gaps[0], WEST);
    } else {
      CaloDirection dir1 = CaloDirectionOperations::add2d(gaps[0].first, NORTH);
      CaloDirection dir2 = CaloDirectionOperations::add2d(gaps[0].first, SOUTH);
      myPad.edge(dir1) = correspondingEdge(gaps[0], NORTH);
      myPad.edge(dir2) = correspondingEdge(gaps[0], SOUTH);
    }
    if (debug) {
      std::cout << " Apres " << std::endl;
      std::cout << myPad << std::endl;
    }
  } else if (ngaps == 2) {
    if (debug) {
      std::cout << " Avant " << ngaps << " " << gaps[0].first << " " << gaps[1].first << std::endl;
      std::cout << myPad << std::endl;
      std::cout << " Voisin 1 " << (gaps[0].second) << std::endl;
      std::cout << " Voisin 2 " << (gaps[1].second) << std::endl;
    }
    CaloDirection corner0 = CaloDirectionOperations::add2d(gaps[0].first, gaps[1].first);

    CLHEP::Hep2Vector point(0., 0.);
    if (corner0 != NONE && diagonalEdge(iq, corner0, point)) {
      CaloDirection corner1 =
          CaloDirectionOperations::add2d(CaloDirectionOperations::oppositeSide(gaps[0].first), gaps[1].first);
      CaloDirection corner2 =
          CaloDirectionOperations::add2d(gaps[0].first, CaloDirectionOperations::oppositeSide(gaps[1].first));
      myPad.edge(corner0) = point;
      myPad.edge(corner1) = correspondingEdge(gaps[1], CaloDirectionOperations::oppositeSide(gaps[0].first));
      myPad.edge(corner2) = correspondingEdge(gaps[0], CaloDirectionOperations::oppositeSide(gaps[1].first));
    } else if (corner0 == NONE) {
      if (gaps[0].first == EAST || gaps[0].first == WEST) {
        CaloDirection corner1 = CaloDirectionOperations::add2d(gaps[0].first, NORTH);
        CaloDirection corner2 = CaloDirectionOperations::add2d(gaps[0].first, SOUTH);
        myPad.edge(corner1) = correspondingEdge(gaps[0], NORTH);
        myPad.edge(corner2) = correspondingEdge(gaps[0], SOUTH);

        corner1 = CaloDirectionOperations::add2d(gaps[1].first, NORTH);
        corner2 = CaloDirectionOperations::add2d(gaps[1].first, SOUTH);
        myPad.edge(corner1) = correspondingEdge(gaps[1], NORTH);
        myPad.edge(corner2) = correspondingEdge(gaps[1], SOUTH);
      } else {
        CaloDirection corner1 = CaloDirectionOperations::add2d(gaps[0].first, EAST);
        CaloDirection corner2 = CaloDirectionOperations::add2d(gaps[0].first, WEST);
        myPad.edge(corner1) = correspondingEdge(gaps[0], EAST);
        myPad.edge(corner2) = correspondingEdge(gaps[0], WEST);

        corner1 = CaloDirectionOperations::add2d(gaps[1].first, EAST);
        corner2 = CaloDirectionOperations::add2d(gaps[1].first, WEST);
        myPad.edge(corner1) = correspondingEdge(gaps[1], EAST);
        myPad.edge(corner2) = correspondingEdge(gaps[1], WEST);
      }
    }
    if (debug) {
      std::cout << " Apres " << std::endl;
      std::cout << myPad << std::endl;
    }
  } else if (ngaps == 3) {
    // in this case the four corners have to be changed
    unsigned iubd, idir1, idir2;
    CaloDirection diag;
    CLHEP::Hep2Vector point(0., 0.);
    //	  std::cout << " Yes : 3 gaps" << std::endl;
    if (unbalancedDirection(gaps, iubd, idir1, idir2)) {
      CaloDirection ubd(gaps[iubd].first), dir1(gaps[idir1].first);
      CaloDirection dir2(gaps[idir2].first);

      //	      std::cout << " Avant " << std::endl << myPad << std::endl;
      //	      std::cout << ubd << " " << dir1 << " " << dir2 << std::endl;
      diag = CaloDirectionOperations::add2d(ubd, dir1);
      if (diagonalEdge(iq, diag, point))
        myPad.edge(diag) = point;
      diag = CaloDirectionOperations::add2d(ubd, dir2);
      if (diagonalEdge(iq, diag, point))
        myPad.edge(diag) = point;
      CaloDirection oppside = CaloDirectionOperations::oppositeSide(ubd);
      myPad.edge(CaloDirectionOperations::add2d(oppside, dir1)) = correspondingEdge(gaps[idir1], oppside);
      myPad.edge(CaloDirectionOperations::add2d(oppside, dir2)) = correspondingEdge(gaps[idir2], oppside);
      //	      std::cout << " Apres " << std::endl << myPad << std::endl;
    }
  } else if (ngaps == 4) {
    //	    std::cout << " Waouh :4 gaps" << std::endl;
    //	    std::cout << " Avant " << std::endl;
    //	    std::cout << myPad<< std::endl;
    CLHEP::Hep2Vector point(0., 0.);
    if (diagonalEdge(iq, NORTHEAST, point))
      myPad.edge(NORTHEAST) = point;
    if (diagonalEdge(iq, NORTHWEST, point))
      myPad.edge(NORTHWEST) = point;
    if (diagonalEdge(iq, SOUTHWEST, point))
      myPad.edge(SOUTHWEST) = point;
    if (diagonalEdge(iq, SOUTHEAST, point))
      myPad.edge(SOUTHEAST) = point;
    //	    std::cout << " Apres " << std::endl;
    //	    std::cout << myPad<< std::endl;
  }
}

void EcalHitMaker::cracksPads(std::vector<neighbour>& cracks, unsigned iq) {
  //  std::cout << " myPad " << &myPad << std::endl;
  unsigned ncracks = cracks.size();
  CrystalPad& myPad = padsatdepth_[iq];
  for (unsigned ic = 0; ic < ncracks; ++ic) {
    //      std::vector<CLHEP::Hep2Vector> mycorners;
    //      mycorners.reserve(4);
    switch (cracks[ic].first) {
      case NORTH: {
        mycorners[0] = (padsatdepth_[cracks[ic].second].edge(SOUTHWEST));
        mycorners[1] = (padsatdepth_[cracks[ic].second].edge(SOUTHEAST));
        mycorners[2] = (myPad.edge(NORTHEAST));
        mycorners[3] = (myPad.edge(NORTHWEST));
      } break;
      case SOUTH: {
        mycorners[0] = (myPad.edge(SOUTHWEST));
        mycorners[1] = (myPad.edge(SOUTHEAST));
        mycorners[2] = (padsatdepth_[cracks[ic].second].edge(NORTHEAST));
        mycorners[3] = (padsatdepth_[cracks[ic].second].edge(NORTHWEST));
      } break;
      case EAST: {
        mycorners[0] = (myPad.edge(NORTHEAST));
        mycorners[1] = (padsatdepth_[cracks[ic].second].edge(NORTHWEST));
        mycorners[2] = (padsatdepth_[cracks[ic].second].edge(SOUTHWEST));
        mycorners[3] = (myPad.edge(SOUTHEAST));
      } break;
      case WEST: {
        mycorners[0] = (padsatdepth_[cracks[ic].second].edge(NORTHEAST));
        mycorners[1] = (myPad.edge(NORTHWEST));
        mycorners[2] = (myPad.edge(SOUTHWEST));
        mycorners[3] = (padsatdepth_[cracks[ic].second].edge(SOUTHEAST));
      } break;
      default: {
      }
    }
    CrystalPad crackpad(ic, mycorners);
    // to be tuned. A simpleconfigurable should be used
    crackpad.setSurvivalProbability(crackPadProbability_);
    crackpadsatdepth_.push_back(crackpad);
  }
  //  std::cout << " Finished cracksPads " << std::endl;
}

bool EcalHitMaker::inside3D(const std::vector<XYZPoint>& corners, const XYZPoint& p) const {
  // corners and p are in the same plane
  // p is inside "corners" if the four crossproducts (corners[i]xcorners[i+1])
  // are in the same direction

  XYZVector crossproduct(0., 0., 0.), previouscrossproduct(0., 0., 0.);

  for (unsigned ip = 0; ip < 4; ++ip) {
    crossproduct = (corners[ip] - p).Cross(corners[(ip + 1) % 4] - p);
    if (ip == 0)
      previouscrossproduct = crossproduct;
    else if (crossproduct.Dot(previouscrossproduct) < 0.)
      return false;
  }

  return true;
}
