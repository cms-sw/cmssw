// -*- C++ -*-
//
// $Id: ValidateGeometry.cc,v 1.3 2010/07/15 21:29:08 mccauley Exp $
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

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include <string>
#include <iostream>

#include <TEveGeoNode.h>
#include <TGeoBBox.h>
#include <TGeoArb8.h>

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

  void validateEBGeometry()
    {};
  void validateEEGeometry()
    {};

  void validateTIBGeometry();
  void validateTOBGeometry();
  void validateTECGeometry();
  void validateTIDGeometry();
  void validatePXBGeometry();
  void validatePXFGeometry();

  void compareTransform(const GlobalPoint& point, const TGeoHMatrix* matrix);
  void compareShape(const double* corners, TGeoShape* shape);

  double getDistance(const double* point1, const GlobalPoint& point2);

  std::string fileName_;
  double tolerance_;    // cm
  bool ok_;

  edm::ESHandle<RPCGeometry>     rpcGeometry_;
  edm::ESHandle<DTGeometry>      dtGeometry_;
  edm::ESHandle<CSCGeometry>     cscGeometry_;
  edm::ESHandle<CaloGeometry>    caloGeometry_;
  edm::ESHandle<TrackerGeometry> trackerGeometry_;

  DetIdToMatrix detIdToMatrix_;
};

ValidateGeometry::ValidateGeometry(const edm::ParameterSet& iConfig)
  : fileName_(iConfig.getUntrackedParameter<std::string>("fileName")),
    tolerance_(iConfig.getUntrackedParameter<double>("tolerance")),
    ok_(true)
{
  detIdToMatrix_.loadGeometry(fileName_.c_str());
  detIdToMatrix_.loadMap(fileName_.c_str());
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
    std::cout<<"Validating TIB geometry"<<std::endl;
    validateTIBGeometry();

    std::cout<<"Validating TOB geometry"<<std::endl;
    validateTOBGeometry();

    std::cout<<"Validating TEC geometry"<<std::endl;
    validateTECGeometry();

    std::cout<<"Validating TID geometry"<<std::endl;
    validateTIDGeometry();

    std::cout<<"Validating PXB geometry"<<std::endl;
    validatePXBGeometry();
    
    std::cout<<"Validating PXF geometry"<<std::endl;
    validatePXFGeometry();
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid Tracker geometry"<<std::endl;



  /*
  eventSetup.get<CaloGeometryRecord>().get(caloGeometry_);

  if ( caloGeometry_.isValid() )
  {
    std::cout<<"Validating HB geometry"<<std::endl;
    validateHBGeometry();
  
    std::cout<<"Validating HE geometry"<<std::endl;
    validateHEGeometry();

    std::cout<<"Validating HO geometry"<<std::endl;
    validateHOGeometry();
    
    std::cout<<"Validating HF geometry"<<std::endl;
    validateHFGeometry();
  }
  */
  
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
    }
  }
}

void 
ValidateGeometry::validateDTGeometry()
{
  std::vector<DTChamber*> chambers = dtGeometry_->chambers();
  
  for ( std::vector<DTChamber*>::const_iterator it = chambers.begin (), 
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
        fwLog(fwlog::kError) << " failed to get geometry of DT with detid: " 
                             << chId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);
    }
  }
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
        fwLog(fwlog::kError) << " failed to get geometry of CSC with detid: " 
                             << detId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);
    }
  }
}

// Q: why does one have to specify subdetector id after already specifying it?
// Check this. If so, then it's a crap interface.

void 
ValidateGeometry::validateHBGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
}

void 
ValidateGeometry::validateHEGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Hcal, HcalEndcap);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
}

void 
ValidateGeometry::validateHOGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Hcal, HcalOuter);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
}

void 
ValidateGeometry::validateHFGeometry()
{
  const CaloSubdetectorGeometry* geometry = (*caloGeometry_).getSubdetectorGeometry(DetId::Hcal, HcalForward);
  std::vector<DetId> ids = geometry->getValidDetIds(DetId::Hcal, HcalForward);
}


void 
ValidateGeometry::validateTIBGeometry()
{       
  int notFound = 0;

  for ( TrackerGeometry::DetContainer::const_iterator it = trackerGeometry_->detsTIB().begin(), 
                                                   itEnd = trackerGeometry_->detsTIB().end(); 
        it != itEnd; ++it )
  {
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    DetId detId((*it)->geographicalId());

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId);

    if ( ! matrix )
    {
      fwLog(fwlog::kError) << " failed to get geometry of TIB element with detid: "
                           << TIBDetId(detId.rawId()) <<std::endl;
      notFound++;
      continue;
    }
    
    compareTransform(gp, matrix);
  }
  
  std::cout<< notFound <<" TIB elements not found out of "<< trackerGeometry_->detsTIB().size() <<" total elements"<<std::endl;
}

void 
ValidateGeometry::validateTOBGeometry()
{
  int notFound = 0;

  for ( TrackerGeometry::DetContainer::const_iterator it = trackerGeometry_->detsTOB().begin(), 
                                                   itEnd = trackerGeometry_->detsTOB().end(); 
        it != itEnd; ++it )
  {
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    DetId detId((*it)->geographicalId());

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId);

    if ( ! matrix )
    {
      fwLog(fwlog::kError) << " failed to get geometry of TOB element with detid: "
                           << TOBDetId(detId.rawId()) <<std::endl;
      notFound++;
      continue;
    }
    
    
    compareTransform(gp, matrix);
  }
  std::cout<< notFound <<" TOB elements not found out of "<< trackerGeometry_->detsTOB().size() <<" total elements"<<std::endl;
}

void 
ValidateGeometry::validateTECGeometry()
{
  int notFound = 0;

  for ( TrackerGeometry::DetContainer::const_iterator it = trackerGeometry_->detsTEC().begin(), 
                                                   itEnd = trackerGeometry_->detsTEC().end(); 
        it != itEnd; ++it )
  {
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    DetId detId((*it)->geographicalId());

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId);

    if ( ! matrix )
    {
      fwLog(fwlog::kError) << " failed to get geometry of TEC element with detid: "
                           << TECDetId(detId.rawId()) <<std::endl;
      notFound++;
      continue;
    }
   
    compareTransform(gp, matrix);
  }
  std::cout<< notFound <<" TEC elements not found out of "<< trackerGeometry_->detsTEC().size() <<" total elements"<<std::endl;
}

void 
ValidateGeometry::validateTIDGeometry()
{
  int notFound = 0;

  for ( TrackerGeometry::DetContainer::const_iterator it = trackerGeometry_->detsTID().begin(), 
                                                   itEnd = trackerGeometry_->detsTID().end(); 
        it != itEnd; ++it )
  {
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    DetId detId((*it)->geographicalId());

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId);

    if ( ! matrix )
    {
      fwLog(fwlog::kError) << " failed to get geometry of TID element with detid: "
                           << TIDDetId(detId.rawId()) <<std::endl;
      notFound++;
      continue;
    }
   
    
    compareTransform(gp, matrix);
  }
  std::cout<< notFound <<" TID elements not found out of "<< trackerGeometry_->detsTID().size() <<" total elements"<<std::endl;
}

void 
ValidateGeometry::validatePXBGeometry()
{
  int notFound = 0;

  for ( TrackerGeometry::DetContainer::const_iterator it = trackerGeometry_->detsPXB().begin(), 
                                                   itEnd = trackerGeometry_->detsPXB().end(); 
        it != itEnd; ++it )
  {
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    DetId detId((*it)->geographicalId());

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId);

    if ( ! matrix )
    {
      fwLog(fwlog::kError) << " failed to get geometry of PXB element with detid: "
                           << PXBDetId(detId.rawId()) <<std::endl;
      notFound++;
      continue;
    }

    
    compareTransform(gp, matrix);
  }
  std::cout<< notFound <<" PXB elements not found out of "<< trackerGeometry_->detsPXB().size() <<" total elements"<<std::endl; 
}

void 
ValidateGeometry::validatePXFGeometry()
{
  int notFound = 0;

  for ( TrackerGeometry::DetContainer::const_iterator it = trackerGeometry_->detsPXF().begin(), 
                                                   itEnd = trackerGeometry_->detsPXF().end(); 
        it != itEnd; ++it )
  {
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    DetId detId((*it)->geographicalId());

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId);

    if ( ! matrix )
    {
      fwLog(fwlog::kError) << " failed to get geometry of PXF element with detid: "
                           << PXFDetId(detId.rawId()) <<std::endl;
      notFound++;
      continue;
    }
    
    
    compareTransform(gp, matrix);
  }
   std::cout<< notFound <<" PXF elements not found out of "<< trackerGeometry_->detsPXF().size() <<" total elements"<<std::endl; 
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

  double distance = getDistance(global, gp);
  
  if ( distance > tolerance_ )
  {
    std::cout<< distance <<" > "<< tolerance_ <<" cm"<<std::endl;
    ok_ = false;
  }
}

void 
ValidateGeometry::compareShape(const double* corners,
                               TGeoShape* shape)
{}

double 
ValidateGeometry::getDistance(const double* p1, const GlobalPoint& p2)
{
  return sqrt((p1[0]-p2.x())*(p1[0]-p2.x())+
              (p1[1]-p2.y())*(p1[1]-p2.y())+
              (p1[2]-p2.z())*(p1[2]-p2.z()));
}
      
void 
ValidateGeometry::beginJob()
{}

void 
ValidateGeometry::endJob() 
{}

DEFINE_FWK_MODULE(ValidateGeometry);

