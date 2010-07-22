// -*- C++ -*-
//
// $Id: ValidateGeometry.cc,v 1.4 2010/07/15 22:21:50 mccauley Exp $
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include <string>
#include <iostream>

#include <TEveGeoNode.h>
#include <TGeoBBox.h>
#include <TGeoArb8.h>
#include <TFile.h>
#include <TH1.h>

class ValidateGeometry : public edm::EDAnalyzer 
{
public:
  explicit ValidateGeometry(const edm::ParameterSet&);
  ~ValidateGeometry();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();  

  void validateRPCGeometry();
  void validateDTGeometry();
  void validateCSCGeometry();

  void validateHBGeometry();
  void validateHEGeometry();
  void validateHOGeometry();
  void validateHFGeometry();

  void validateEBGeometry();
  void validateEEGeometry();

  void validateTrackerGeometry();

  void compareTransform(const GlobalPoint& point, const TGeoHMatrix* matrix);

  void compareShape(const GeomDet* det, TEveGeoShape* shape);
  void compareShape(const DetId& detId);

  double getDistance(const GlobalPoint& point1, const GlobalPoint& point2);

  void fillCorners(std::vector<GlobalPoint>& corners, const GeomDet* det);
  void fillCorners(std::vector<GlobalPoint>& corners, const DetId& detId);

  void makeHistograms(const char* detector);
  void makeHistogram(std::string& name, std::vector<double>& data);
  
  std::string infileName_;
  std::string outfileName_;

  double tolerance_;    // cm
  bool ok_;

  edm::ESHandle<RPCGeometry>     rpcGeometry_;
  edm::ESHandle<DTGeometry>      dtGeometry_;
  edm::ESHandle<CSCGeometry>     cscGeometry_;
  edm::ESHandle<CaloGeometry>    caloGeometry_;
  edm::ESHandle<TrackerGeometry> trackerGeometry_;

  DetIdToMatrix detIdToMatrix_;

  TFile* outFile_;

  std::vector<double> distances_;
  std::vector<double> topWidths_;
  std::vector<double> bottomWidths_;
  std::vector<double> lengths_;
  std::vector<double> thicknesses_;
};


ValidateGeometry::ValidateGeometry(const edm::ParameterSet& iConfig)
  : infileName_(iConfig.getUntrackedParameter<std::string>("infileName")),
    outfileName_(iConfig.getUntrackedParameter<std::string>("outfileName")),
    tolerance_(iConfig.getUntrackedParameter<double>("tolerance")),
    ok_(true)
{
  detIdToMatrix_.loadGeometry(infileName_.c_str());
  detIdToMatrix_.loadMap(infileName_.c_str());

  outFile_ = new TFile(outfileName_.c_str(), "RECREATE");
}


ValidateGeometry::~ValidateGeometry()
{}


void 
ValidateGeometry::analyze(const edm::Event& event, const edm::EventSetup& eventSetup)
{
  eventSetup.get<MuonGeometryRecord>().get(rpcGeometry_);
  
  if ( rpcGeometry_.isValid() )
  {
    std::cout<<"Validating RPC geometry"<<std::endl;
    validateRPCGeometry();
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid RPC geometry"<<std::endl; 


  eventSetup.get<MuonGeometryRecord>().get(dtGeometry_);

  if ( dtGeometry_.isValid() )
  {
    std::cout<<"Validating DT geometry"<<std::endl;
    validateDTGeometry();
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid DT geometry"<<std::endl; 


  eventSetup.get<MuonGeometryRecord>().get(cscGeometry_);
  
  if ( cscGeometry_.isValid() )
  {
    std::cout<<"Validating CSC geometry"<<std::endl;
    validateCSCGeometry();
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid CSC geometry"<<std::endl; 

  
  eventSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry_);

  if ( trackerGeometry_.isValid() )
  {
    std::cout<<"Validating tracker geometry"<<std::endl;
    validateTrackerGeometry();
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid Tracker geometry"<<std::endl;


  eventSetup.get<CaloGeometryRecord>().get(caloGeometry_);

  if ( caloGeometry_.isValid() )
  {
    std::cout<<"Validating EB geometry"<<std::endl;
    validateEBGeometry();

    std::cout<<"Validating EE geometry"<<std::endl;
    validateEEGeometry();

    std::cout<<"Validating HB geometry"<<std::endl;
    validateHBGeometry();
  
    std::cout<<"Validating HE geometry"<<std::endl;
    validateHEGeometry();

    std::cout<<"Validating HO geometry"<<std::endl;
    validateHOGeometry();
    
    std::cout<<"Validating HF geometry"<<std::endl;
    validateHFGeometry();
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid Calo geometry"<<std::endl; 
  

  if ( ok_ )
    std::cout<<"OK"<<std::endl;
}


void
ValidateGeometry::validateRPCGeometry()
{
  std::vector<RPCRoll*> rolls = rpcGeometry_->rolls();
  
  for ( std::vector<RPCRoll*>::const_iterator it = rolls.begin(), 
                                           itEnd = rolls.end();
        it != itEnd; ++it )
  {
    const RPCRoll* roll = *it;

    if ( roll )
    {
      RPCDetId rpcDetId = roll->id();
        
      const GeomDetUnit* det = rpcGeometry_->idToDetUnit(rpcDetId);
      GlobalPoint gp = det->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0)); 
      
      const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(rpcDetId);

      if ( ! matrix )
      {
        fwLog(fwlog::kError)<<"Failed to get geometry of RPC with detid: "
                            << rpcDetId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);


      TEveGeoShape* shape = detIdToMatrix_.getShape(rpcDetId);

      if ( ! shape )
      {
        fwLog(fwlog::kError)<<"Failed to get shape of RPC with detid: "
                            << rpcDetId.rawId() <<std::endl;
        continue;
      }
      
      compareShape(det, shape);
    }
  }

  makeHistograms("RPC");
}


void 
ValidateGeometry::validateDTGeometry()
{
  std::vector<DTChamber*> chambers = dtGeometry_->chambers();
  
  for ( std::vector<DTChamber*>::const_iterator it = chambers.begin(), 
                                             itEnd = chambers.end(); 
        it != itEnd; ++it)
  {
    const DTChamber* chamber = *it;
      
    if ( chamber )
    {
      DTChamberId chId = chamber->id();
      GlobalPoint gp = chamber->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0)); 
     
      const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(chId);
 
      if ( ! matrix )   
      {     
        fwLog(fwlog::kError) << "Failed to get geometry of DT with detid: " 
                             << chId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);

      TEveGeoShape* shape = detIdToMatrix_.getShape(chId);

      if ( ! shape )
      {
        fwLog(fwlog::kError)<<"Failed to get shape of DT with detid: "
                            << chId.rawId() <<std::endl;
        continue;
      }
      
      compareShape(chamber, shape);
    }
  }

  makeHistograms("DT");
}


void 
ValidateGeometry::validateCSCGeometry()
{
  std::vector<CSCChamber *> chambers = cscGeometry_->chambers();
     
  for ( std::vector<CSCChamber*>::const_iterator it = chambers.begin(), 
                                              itEnd = chambers.end(); 
        it != itEnd; ++it )
  {
    const CSCChamber* chamber = *it;
         
    if ( chamber )
    {
      DetId detId = chamber->geographicalId();
      GlobalPoint gp = chamber->surface().toGlobal(LocalPoint(0.0,0.0,0.0));

      const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId);
  
      if ( ! matrix ) 
      {     
        fwLog(fwlog::kError) << "Failed to get geometry of CSC with detid: " 
                             << detId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);


      TEveGeoShape* shape = detIdToMatrix_.getShape(detId);

      if ( ! shape )
      {
        fwLog(fwlog::kError)<<"Failed to get shape of CSC with detid: "
                            << detId.rawId() <<std::endl;
        continue;
      }
      
      compareShape(chamber, shape);
    }
  }

  makeHistograms("CSC");
}

// Q: why does one have to specify subdetector id after already specifying it?
// Check this. If so, then it's a crap interface.

void 
ValidateGeometry::validateEBGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Ecal, EcalBarrel);

  for (std::vector<DetId>::const_iterator it = ids.begin(), 
                                        iEnd = ids.end(); 
       it != iEnd; ++it) 
  {
    
  }
}


void 
ValidateGeometry::validateEEGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Ecal, EcalEndcap);

  for (std::vector<DetId>::const_iterator it = ids.begin(), 
                                        iEnd = ids.end(); 
       it != iEnd; ++it) 
  {
    
  }
}


void 
ValidateGeometry::validateHBGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Hcal, HcalBarrel);

  for (std::vector<DetId>::const_iterator it = ids.begin(), 
                                        iEnd = ids.end(); 
       it != iEnd; ++it) 
  {
    
  }
}


void 
ValidateGeometry::validateHEGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Hcal, HcalEndcap);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Hcal, HcalEndcap);

  for (std::vector<DetId>::const_iterator it = ids.begin(), 
                                        iEnd = ids.end(); 
       it != iEnd; ++it) 
  {
    
  }
}


void 
ValidateGeometry::validateHOGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Hcal, HcalOuter);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Hcal, HcalOuter);

  for (std::vector<DetId>::const_iterator it = ids.begin(), 
                                        iEnd = ids.end(); 
       it != iEnd; ++it) 
  {
    
  }
}


void 
ValidateGeometry::validateHFGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Hcal, HcalForward);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Hcal, HcalForward);

  for (std::vector<DetId>::const_iterator it = ids.begin(), 
                                        iEnd = ids.end(); 
       it != iEnd; ++it) 
  {
    
  }
}


void
ValidateGeometry::validateTrackerGeometry()
{
  for ( TrackerGeometry::DetUnitContainer::const_iterator it = trackerGeometry_->detUnits().begin(),
                                                   itEnd = trackerGeometry_->detUnits().end();
        it != itEnd; ++it )
  {
    
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    DetId detId((*it)->geographicalId());

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId);

    if ( ! matrix )
    { 
      fwLog(fwlog::kError) << "Failed to get geometry of tracker element with detid: "
                           << detId.rawId() <<std::endl;
      continue;
    }
    
    compareTransform(gp, matrix);                                                                
    
    TEveGeoShape* shape = detIdToMatrix_.getShape(detId);

    if ( ! shape )
    {
      fwLog(fwlog::kError)<<"Failed to get shape of tracker element with detid: "
                          << detId.rawId() <<std::endl;
      continue;
    }

    compareShape(*it, shape);
  }

  makeHistograms("Tracker");
}


void    
ValidateGeometry::compareTransform(const GlobalPoint& gp,
                                   const TGeoHMatrix* matrix)
{
  double local[3] = 
    {
      0.0, 0.0, 0.0
    };
      
  double global[3];

  matrix->LocalToMaster(local, global);

  double distance = getDistance(GlobalPoint(global[0], global[1], global[2]), gp);
  distances_.push_back(distance);

  if ( distance > tolerance_ )
  {
    ok_ = false;
  }
}


void 
ValidateGeometry::compareShape(const DetId& detId)
{
  const CaloCellGeometry* cellGeometry = caloGeometry_->getGeometry(detId);
  const CaloCellGeometry::CornersVec& cs = cellGeometry->getCorners();
  assert(cs.size() == 8);
}


void 
ValidateGeometry::compareShape(const GeomDet* det, TEveGeoShape* shape)
{
  TGeoShape* geoShape = shape->GetShape();

  TGeoBBox* box;
  TGeoTrap* trap;

  /*
    X -> width
    Y -> length
    Z -> thickness
  */

  double shape_topWidth;
  double shape_bottomWidth;
  double shape_length;
  double shape_thickness;

  if ( (box = dynamic_cast<TGeoBBox*>(geoShape)) && 
       (trap = dynamic_cast<TGeoTrap*>(box)) )
  {
    shape_topWidth = trap->GetTl2()*2.0;
    shape_bottomWidth = trap->GetBl2()*2.0;
    shape_length = trap->GetH2()*2.0;
    shape_thickness = trap->GetDz()*2.0;
  }

  else
  {
    fwLog(fwlog::kError) << "Failed to get box or trapezoid from shape"<<std::endl;
    return;
  }

  double topWidth, bottomWidth;
  double length, thickness;

  const Bounds* bounds = &(det->surface().bounds());
  const TrapezoidalPlaneBounds* tpbs;

  if ( (tpbs = dynamic_cast<const TrapezoidalPlaneBounds*>(bounds)) )
  {
    std::vector<float> ps = tpbs->parameters();

    assert(ps.size() == 4);
    
    bottomWidth = ps[0]*2.0;
    topWidth = ps[1]*2.0;
    thickness = ps[2]*2.0;
    length = ps[3]*2.0;
  }
  
  // can use parameters above as well
  
  else if ( (dynamic_cast<const RectangularPlaneBounds*>(bounds)) )
  {
    length = det->surface().bounds().length();
    topWidth = det->surface().bounds().width();
    bottomWidth = topWidth;
    thickness = det->surface().bounds().thickness();
  }
  
  else
  {
    fwLog(fwlog::kError) << "Failed to get bounds"<<std::endl;
    return;
  }
  
  topWidths_.push_back(fabs(shape_topWidth - topWidth));
  bottomWidths_.push_back(fabs(shape_bottomWidth - bottomWidth));
  lengths_.push_back(fabs(shape_length - length));
  thicknesses_.push_back(fabs(shape_thickness - thickness));

  return;
}


double 
ValidateGeometry::getDistance(const GlobalPoint& p1, const GlobalPoint& p2)
{
  return sqrt((p1.x()-p2.x())*(p1.x()-p2.x())+
              (p1.y()-p2.y())*(p1.y()-p2.y())+
              (p1.z()-p2.z())*(p1.z()-p2.z()));
}


void
ValidateGeometry::fillCorners(std::vector<GlobalPoint>& corners, const GeomDet* det)
{
  const Bounds* bounds = &(det->surface().bounds());
  const TrapezoidalPlaneBounds* tpbs;

  if ( (tpbs = dynamic_cast<const TrapezoidalPlaneBounds*>(bounds)) )
  {
    std::vector<float> ps = tpbs->parameters();

    assert(ps.size() == 4);
    
    corners.push_back(det->surface().toGlobal(LocalPoint(ps[0],-ps[3],ps[2]))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(-ps[0],-ps[3],ps[2])));                
    corners.push_back(det->surface().toGlobal(LocalPoint(ps[1],ps[3],ps[2]))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(-ps[1],ps[3],ps[2]))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(ps[0],-ps[3],-ps[2]))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(-ps[0],-ps[3],-ps[2]))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(ps[1],ps[3],-ps[2]))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(-ps[1],ps[3],-ps[2])));
  }
  
  else if ( (dynamic_cast<const RectangularPlaneBounds*>(bounds)) )
  {
    float length    = det->surface().bounds().length() / 2;
    float width     = det->surface().bounds().width() / 2 ;
    float thickness = det->surface().bounds().thickness() / 2;

    corners.push_back(det->surface().toGlobal(LocalPoint(width,length,thickness))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(width,-length,thickness))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(-width,length,thickness))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(-width,-length,thickness))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(width,length,-thickness))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(width,-length,-thickness))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(-width,length,-thickness))); 
    corners.push_back(det->surface().toGlobal(LocalPoint(-width,-length,-thickness)));
  }
  
  assert(corners.size() == 8);
  return;
}


void 
ValidateGeometry::fillCorners(std::vector<GlobalPoint>& corners, const DetId& detId)
{
  const CaloCellGeometry* cellGeometry = caloGeometry_->getGeometry(detId);
  const CaloCellGeometry::CornersVec& cs = cellGeometry->getCorners();
  assert(cs.size() == 8);

  if ( detId.det() == DetId::Ecal )
  {
    corners.push_back(cs[3]);
    corners.push_back(cs[2]);
    corners.push_back(cs[1]);
    corners.push_back(cs[0]);

    corners.push_back(cs[7]);
    corners.push_back(cs[6]);
    corners.push_back(cs[5]);
    corners.push_back(cs[4]); 
  }
    
  else if ( detId.det() == DetId::Hcal )
  {
    corners.push_back(cs[0]);
    corners.push_back(cs[1]);
    corners.push_back(cs[2]);
    corners.push_back(cs[3]);

    corners.push_back(cs[4]);
    corners.push_back(cs[5]);
    corners.push_back(cs[6]);
    corners.push_back(cs[7]); 
  }
}


void
ValidateGeometry::makeHistograms(const char* detector)
{
  outFile_->cd();

  std::string d(detector);
  
  std::string dn = d+" distances";
  makeHistogram(dn, distances_);
  
  std::string twn = d + " top widths";
  makeHistogram(twn, topWidths_);
  
  std::string bwn = d + " bottom widths";
  makeHistogram(bwn, bottomWidths_);
  
  std::string ln = d + " lengths";
  makeHistogram(ln, lengths_);

  std::string tn = d + " thicknesses";
  makeHistogram(tn, thicknesses_);

  return;
}


void
ValidateGeometry::makeHistogram(std::string& name, std::vector<double>& data)
{
  if ( data.empty() )
    return;

  std::vector<double>::iterator it = std::max_element(data.begin(), data.end());
  std::vector<double>::iterator itEnd = data.end();

  TH1D hist(name.c_str(), name.c_str(), 100, 0, (*it)*(1+0.10));
  
  for ( it = data.begin(); it != itEnd; ++it )
    hist.Fill(*it);
  
  hist.Write();
  data.clear();
}


void 
ValidateGeometry::beginJob()
{
  outFile_->cd();
}


void 
ValidateGeometry::endJob() 
{
  outFile_->Close();
}

DEFINE_FWK_MODULE(ValidateGeometry);

