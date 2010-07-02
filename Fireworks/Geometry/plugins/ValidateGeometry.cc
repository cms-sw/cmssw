#include "FWCore/Framework/interface/Frameworkfwd.h"
// -*- C++ -*-
//
// $Id: ValidateGeometry.cc,v 1.1 2010/07/02 11:37:40 mccauley Exp $
//

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

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

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

  void compareTransform(const GlobalPoint& point, const TGeoHMatrix* matrix);
  void compareShape(const double* corners, TGeoShape* shape);

  double getDistance(const double* point1, const GlobalPoint& point2);

  std::string fileName_;
  double tolerance_;    // cm
  bool ok_;

  edm::ESHandle<RPCGeometry>  rpcGeometry_;
  edm::ESHandle<DTGeometry>   dtGeometry_;
  edm::ESHandle<CSCGeometry>  cscGeometry_;
  edm::ESHandle<CaloGeometry> caloGeometry_;

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
        return;
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
        return;
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
        return;
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

