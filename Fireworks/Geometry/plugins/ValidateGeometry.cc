// -*- C++ -*-
//
// $Id: ValidateGeometry.cc,v 1.13 2010/08/03 14:40:59 mccauley Exp $
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
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include <string>
#include <iostream>

#include <TEveGeoNode.h>
#include <TGeoVolume.h>
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

  void validateRPCGeometry(const int regionNumber, 
                           const char* regionName);

  void validateDTChamberGeometry();
  void validateDTSuperLayerGeometry();
  void validateDTLayerGeometry();

  void validateCSChamberGeometry(const int endcap,
                                 const char* detname);

  void validateCSCLayerGeometry(const int endcap,
                                const char* detname);

  void validateCaloGeometry(DetId::Detector detector, int subdetector,
                            const char* detname);

  void validateTrackerGeometry(const TrackerGeometry::DetContainer& dets, 
                               const char* detname);

  void validateTrackerGeometry(const TrackerGeometry::DetUnitContainer& dets, 
                               const char* detname);

  void compareTransform(const GlobalPoint& point, const TGeoHMatrix* matrix);

  void compareShape(const GeomDet* det, TGeoShape* shape);

  double getDistance(const GlobalPoint& point1, const GlobalPoint& point2);

  void makeHistograms(const char* detector);
  void makeHistogram(const std::string& name, std::vector<double>& data);
  
  std::string infileName_;
  std::string outfileName_;

  edm::ESHandle<RPCGeometry>     rpcGeometry_;
  edm::ESHandle<DTGeometry>      dtGeometry_;
  edm::ESHandle<CSCGeometry>     cscGeometry_;
  edm::ESHandle<CaloGeometry>    caloGeometry_;
  edm::ESHandle<TrackerGeometry> trackerGeometry_;

  DetIdToMatrix detIdToMatrix_;

  TFile* outFile_;

  std::vector<double> globalDistances_;
  std::vector<double> topWidths_;
  std::vector<double> bottomWidths_;
  std::vector<double> lengths_;
  std::vector<double> thicknesses_;

  void clearData()
    {
      globalDistances_.clear();
      topWidths_.clear();
      bottomWidths_.clear();
      lengths_.clear();
      thicknesses_.clear();
    }

  bool dataEmpty()
    {
      return (globalDistances_.empty() && 
              topWidths_.empty() && 
              bottomWidths_.empty() && 
              lengths_.empty() && 
              thicknesses_.empty());
    }
};


ValidateGeometry::ValidateGeometry(const edm::ParameterSet& iConfig)
  : infileName_(iConfig.getUntrackedParameter<std::string>("infileName")),
    outfileName_(iConfig.getUntrackedParameter<std::string>("outfileName"))
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
    std::cout<<"Validating RPC -z endcap geometry"<<std::endl;
    validateRPCGeometry(-1, "RPC -z endcap");

    std::cout<<"Validating RPC +z endcap geometry"<<std::endl;
    validateRPCGeometry(+1, "RPC +z endcap");

    std::cout<<"Validating RPC barrel geometry"<<std::endl;
    validateRPCGeometry(0, "RPC barrel");
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid RPC geometry"<<std::endl; 


  eventSetup.get<MuonGeometryRecord>().get(dtGeometry_);

  if ( dtGeometry_.isValid() )
  {
    std::cout<<"Validating DT chamber geometry"<<std::endl;
    validateDTChamberGeometry();

    std::cout<<"Validating DT superlayer geometry"<<std::endl;
    validateDTSuperLayerGeometry();

    std::cout<<"Validating DT layer geometry"<<std::endl;
    validateDTLayerGeometry();
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid DT geometry"<<std::endl; 


  eventSetup.get<MuonGeometryRecord>().get(cscGeometry_);
  
  if ( cscGeometry_.isValid() )
  {
    std::cout<<"Validating CSC -z geometry"<<std::endl;
    validateCSChamberGeometry(-1, "CSC chamber -z endcap");

    std::cout<<"Validating CSC +z geometry"<<std::endl;
    validateCSChamberGeometry(+1, "CSC chamber +z endcap");

    std::cout<<"Validating CSC layer -z geometry"<<std::endl;
    validateCSCLayerGeometry(-1, "CSC layer -z endcap");

    std::cout<<"Validating CSC layer +z geometry"<<std::endl;
    validateCSCLayerGeometry(+1, "CSC layer +z endcap");
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid CSC geometry"<<std::endl; 

  
  eventSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry_);

  if ( trackerGeometry_.isValid() )
  {
    //std::cout<<"Validating Tracker geometry"<<std::endl;
    //validateTrackerGeometry(trackerGeometry_->detUnits(), "Tracker");

    std::cout<<"Validating TIB geometry"<<std::endl;
    validateTrackerGeometry(trackerGeometry_->detsTIB(), "TIB");

    std::cout<<"Validating TOB geometry"<<std::endl;
    validateTrackerGeometry(trackerGeometry_->detsTIB(), "TOB");

    std::cout<<"Validating TEC geometry"<<std::endl;
    validateTrackerGeometry(trackerGeometry_->detsTEC(), "TEC");
    
    std::cout<<"Validating TID geometry"<<std::endl;
    validateTrackerGeometry(trackerGeometry_->detsTID(), "TID");

    std::cout<<"Validating PXB geometry"<<std::endl;
    validateTrackerGeometry(trackerGeometry_->detsPXB(), "PXB");

    std::cout<<"Validating PXF geometry"<<std::endl;
    validateTrackerGeometry(trackerGeometry_->detsPXF(), "PXF");
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid Tracker geometry"<<std::endl;


  eventSetup.get<CaloGeometryRecord>().get(caloGeometry_);


  if ( caloGeometry_.isValid() )
  {
    std::cout<<"Validating EB geometry"<<std::endl;
    validateCaloGeometry(DetId::Ecal, EcalBarrel, "EB");

    std::cout<<"Validating EE geometry"<<std::endl;
    validateCaloGeometry(DetId::Ecal, EcalEndcap, "EE");

    std::cout<<"Validating HB geometry"<<std::endl;
    validateCaloGeometry(DetId::Hcal, HcalBarrel, "HB");
  
    std::cout<<"Validating HE geometry"<<std::endl;
    validateCaloGeometry(DetId::Hcal, HcalEndcap, "HE");

    std::cout<<"Validating HO geometry"<<std::endl;
    validateCaloGeometry(DetId::Hcal, HcalOuter, "HO");
    
    std::cout<<"Validating HF geometry"<<std::endl;
    validateCaloGeometry(DetId::Hcal, HcalForward, "HF");
  }
  else
    fwLog(fwlog::kWarning)<<"Invalid Calo geometry"<<std::endl; 

}


void
ValidateGeometry::validateRPCGeometry(const int regionNumber, const char* regionName)
{
  clearData();
 
  std::vector<double> centers;
 
  std::vector<RPCRoll*> rolls = rpcGeometry_->rolls();
  
  for ( std::vector<RPCRoll*>::const_iterator it = rolls.begin(), 
                                           itEnd = rolls.end();
        it != itEnd; ++it )
  {
    const RPCRoll* roll = *it;

    if ( roll )
    {
      RPCDetId rpcDetId = roll->id();

      if ( rpcDetId.region() == regionNumber )
      {  
        const GeomDetUnit* det = rpcGeometry_->idToDetUnit(rpcDetId);
        GlobalPoint gp = det->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0)); 
      
        const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(rpcDetId.rawId());

        if ( ! matrix )
        {
          std::cout<<"Failed to get matrix of RPC with detid: "
                   << rpcDetId.rawId() <<std::endl;
          continue;
        }

        compareTransform(gp, matrix);

        //TEveGeoShape* shape = detIdToMatrix_.getShape(rpcDetId.rawId());
        const TGeoVolume* shape = detIdToMatrix_.getVolume(rpcDetId.rawId());

        if ( ! shape )
        {
          std::cout<<"Failed to get shape of RPC with detid: "
                   << rpcDetId.rawId() <<std::endl;
          continue;
        }
      
        compareShape(det, shape->GetShape());
               
        float pitch = roll->specificTopology().pitch();
        int nStrips = roll->nstrips();

        float offset = -0.5*nStrips*pitch;

        float stripLength = roll->specificTopology().stripLength();

        for ( int strip = 1; strip <= roll->nstrips(); ++strip )
        {
          LocalPoint centreOfStrip1 = roll->centreOfStrip(strip);        
          LocalPoint centreOfStrip2 = LocalPoint((strip-0.5)*pitch + offset, 0.0);

          centers.push_back(centreOfStrip1.x()-centreOfStrip2.x());

        }      
      }
    }
  }

  std::string hn(regionName);
  makeHistogram(hn+": centreOfStrip", centers);
  

  makeHistograms(regionName);
}


void 
ValidateGeometry::validateDTChamberGeometry()
{
  clearData();

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
     
      const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(chId.rawId());
 
      if ( ! matrix )   
      {     
        std::cout<<"Failed to get matrix of DT chamber with detid: " 
                 << chId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);

      //TEveGeoShape* shape = detIdToMatrix_.getShape(chId.rawId());
      const TGeoVolume* shape = detIdToMatrix_.getVolume(chId.rawId());

      if ( ! shape )
      {
        std::cout<<"Failed to get shape of DT chamber with detid: "
                 << chId.rawId() <<std::endl;
        continue;
      }
      
      compareShape(chamber, shape->GetShape());
    }
  }

  makeHistograms("DT chamber");
}

void 
ValidateGeometry::validateDTSuperLayerGeometry()
{
  clearData();

  std::vector<DTSuperLayer*> superlayers = dtGeometry_->superLayers();
  
  for ( std::vector<DTSuperLayer*>::const_iterator it = superlayers.begin(), 
                                                itEnd = superlayers.end(); 
        it != itEnd; ++it)
  {
    const DTSuperLayer* superlayer = *it;
      
    if ( superlayer )
    {
      DTSuperLayerId chId = superlayer->id();
      GlobalPoint gp = superlayer->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0)); 
     
      const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(chId.rawId());
 
      if ( ! matrix )   
      {     
        std::cout<<"Failed to get matrix of DT superlayer with detid: " 
                 << chId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);

      //TEveGeoShape* shape = detIdToMatrix_.getShape(chId.rawId());
      const TGeoVolume* shape = detIdToMatrix_.getVolume(chId.rawId());

      if ( ! shape )
      {
        std::cout<<"Failed to get shape of DT superlayer with detid: "
                 << chId.rawId() <<std::endl;
        continue;
      }
      
      compareShape(superlayer, shape->GetShape());
    }
  }

  makeHistograms("DT superlayer");
}

void 
ValidateGeometry::validateDTLayerGeometry()
{
  clearData();
  
  std::vector<double> wire_positions;

  std::vector<DTLayer*> layers = dtGeometry_->layers();
  
  for ( std::vector<DTLayer*>::const_iterator it = layers.begin(), 
                                           itEnd = layers.end(); 
        it != itEnd; ++it)
  {
    const DTLayer* layer = *it;
      
    if ( layer )
    {
      DTLayerId layerId = layer->id();
      GlobalPoint gp = layer->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0)); 
     
      const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(layerId.rawId());
 
      if ( ! matrix )   
      {     
        std::cout<<"Failed to get matrix of DT layer with detid: " 
                 << layerId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);



      //TEveGeoShape* shape = detIdToMatrix_.getShape(layerId.rawId());
      const TGeoVolume* shape = detIdToMatrix_.getVolume(layerId.rawId());

      if ( ! shape )
      {
        std::cout<<"Failed to get shape of DT layer with detid: "
                 << layerId.rawId() <<std::endl;
        continue;
      }
      
      compareShape(layer, shape->GetShape());

      
      std::vector<TEveVector> parameters = detIdToMatrix_.getPoints(layerId.rawId());
      
      if ( parameters.empty() )
      {
        std::cout<<"Parameters empty for DT layer with detid: " 
                 << layerId.rawId() <<std::endl;
        continue;
      }
      
      int firstChannel = layer->specificTopology().firstChannel();
      assert(firstChannel == parameters[1].fX);

      int lastChannel  = layer->specificTopology().lastChannel();
      int nChannels = parameters[1].fZ;
      assert(nChannels == (lastChannel-firstChannel)+1);

      for ( int wireN = firstChannel; wireN <= lastChannel; ++wireN )
      {
        double localX1 = layer->specificTopology().wirePosition(wireN);
        double localX2 = (wireN -(firstChannel-1)-0.5)*parameters[0].fX - nChannels/2.0*parameters[0].fX;

        wire_positions.push_back(localX1-localX2);

        //std::cout<<"wireN, localXpos: "<< wireN <<" "<< localX1 <<" "<< localX2 <<std::endl;

      }
    }
  }

  makeHistogram("DT layer wire localX", wire_positions);

  makeHistograms("DT layer");
}


void 
ValidateGeometry::validateCSChamberGeometry(const int endcap, const char* detname)
{
  clearData();

  std::vector<CSCChamber *> chambers = cscGeometry_->chambers();
     
  for ( std::vector<CSCChamber*>::const_iterator it = chambers.begin(), 
                                              itEnd = chambers.end(); 
        it != itEnd; ++it )
  {
    const CSCChamber* chamber = *it;
         
    if ( chamber && chamber->id().endcap() == endcap )
    {
      DetId detId = chamber->geographicalId();
      GlobalPoint gp = chamber->surface().toGlobal(LocalPoint(0.0,0.0,0.0));

      const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId.rawId());
  
      if ( ! matrix ) 
      {     
        std::cout<<"Failed to get matrix of CSC chamber with detid: " 
                 << detId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);


      //TEveGeoShape* shape = detIdToMatrix_.getShape(detId.rawId());
      const TGeoVolume* shape = detIdToMatrix_.getVolume(detId.rawId());

      if ( ! shape )
      {
        std::cout<<"Failed to get shape of CSC chamber with detid: "
                 << detId.rawId() <<std::endl;
        continue;
      }
      
      compareShape(chamber, shape->GetShape());
    }
  }

  makeHistograms(detname);
}

void 
ValidateGeometry::validateCSCLayerGeometry(const int endcap, const char* detname)
{
  clearData();
  std::vector<double> strip_positions;
  std::vector<double> wire_positions;

  std::vector<CSCLayer*> layers = cscGeometry_->layers();
     
  for ( std::vector<CSCLayer*>::const_iterator it = layers.begin(), 
                                            itEnd = layers.end(); 
        it != itEnd; ++it )
  {
    const CSCLayer* layer = *it;
         
    if ( layer && layer->id().endcap() == endcap )
    {
      DetId detId = layer->geographicalId();
      GlobalPoint gp = layer->surface().toGlobal(LocalPoint(0.0,0.0,0.0));

      const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(detId.rawId());
  
      if ( ! matrix ) 
      {     
        std::cout<<"Failed to get matrix of CSC layer with detid: " 
                 << detId.rawId() <<std::endl;
        continue;
      }

      compareTransform(gp, matrix);


      //TEveGeoShape* shape = detIdToMatrix_.getShape(detId.rawId());
      const TGeoVolume* shape = detIdToMatrix_.getVolume(detId.rawId());

      if ( ! shape )
      {
        std::cout<<"Failed to get shape of CSC layer with detid: "
                 << detId.rawId() <<std::endl;
        continue;
      }
      
      compareShape(layer, shape->GetShape());

      double length;
      double topWidth;
      double bottomWidth;
 
      if ( TGeoTrap* trap = dynamic_cast<TGeoTrap*>(shape->GetShape()) )
      {
        topWidth = trap->GetTl1();
        bottomWidth = trap->GetBl1();
        length = trap->GetH1();
      }
      
      else
      {      
        std::cout<<"Failed to get trapezoid from shape for CSC layer with detid: "
                 << detId.rawId() <<std::endl;
        continue;
      }
      
      int yAxisOrientation = layer->geometry()->topology()->yAxisOrientation();
      float centreToIntersection = layer->geometry()->topology()->centreToIntersection();
      float yCentre = layer->geometry()->topology()->yCentreOfStripPlane();
      float phiOfOneEdge = layer->geometry()->topology()->phiOfOneEdge();
      float stripOffset = layer->geometry()->topology()->stripOffset();
      float angularWidth = layer->geometry()->topology()->angularWidth();

      for ( int nStrip = 1; nStrip <= layer->geometry()->numberOfStrips(); 
            ++nStrip )
      {
        float xOfStrip1 = layer->geometry()->xOfStrip(nStrip);
      
        double stripAngle = phiOfOneEdge + yAxisOrientation*(nStrip-(0.5-stripOffset))*angularWidth;
        double xOfStrip2 = yAxisOrientation*(centreToIntersection-yCentre)*tan(stripAngle);

        strip_positions.push_back(xOfStrip1-xOfStrip2);
      }

      double wireSpacing = layer->geometry()->wireTopology()->wireSpacing();
      float wireAngle = layer->geometry()->wireTopology()->wireAngle();
      float cosWireAngle = cos(wireAngle);

      /* NOTE
         Some parameters don't seem available in a public interface
         so have to perhaps hard-code. This may not be too bad as there
         seems to be a lot of degeneracy. 
      */

      int station = layer->id().station();
      int ring    = layer->id().ring();

      double alignmentPinToFirstWire;
      double yAlignmentFrame;
   
      if ( station == 1 ) 
      {
        yAlignmentFrame = 0.0;
 
        if ( ring == 1 || ring == 4 )
          alignmentPinToFirstWire = 1.065;
        else
          alignmentPinToFirstWire = 2.85;
      }
      
      else if ( station == 4 && ring == 1 )
      {
        alignmentPinToFirstWire = 3.04;
        yAlignmentFrame = 3.49;
      }
      
      else if ( station == 3 && ring == 1 )
      {
        alignmentPinToFirstWire =  2.84;
        yAlignmentFrame = 3.49;
      }
      
      else  
      {
        alignmentPinToFirstWire = 2.87;
        yAlignmentFrame = 3.49;
      }

      double yOfFirstWire = (yAlignmentFrame-length) + alignmentPinToFirstWire;
              
      for ( int nWireGroup = 1; nWireGroup <= layer->geometry()->numberOfWireGroups(); 
            ++nWireGroup )
      {
        //double lengthOfWireGroup = layer->geometry()->lengthOfWireGroup(nWireGroup);

        for ( int nWire = 1; nWire <= layer->geometry()->numberOfWiresPerGroup(nWireGroup);
              ++nWire )
        {    
          float yOfWire1 = layer->geometry()->yOfWire(nWire); 

          double yOfWire2 = yOfFirstWire*cosWireAngle + (nWire-1)*wireSpacing;
          yOfWire2 /= cosWireAngle;

          wire_positions.push_back(yOfWire1-yOfWire2);
        }
      } 
    }
  }
  
  std::string hn(detname);
  makeHistogram(hn+": xOfStrip", strip_positions);

  makeHistogram(hn+": yOfWire", wire_positions);

  makeHistograms(detname);
}

void 
ValidateGeometry::validateCaloGeometry(DetId::Detector detector, 
                                       int subdetector,
                                       const char* detname)
{
  clearData();

  const CaloSubdetectorGeometry* geometry = 
    caloGeometry_->getSubdetectorGeometry(detector, subdetector);

  const std::vector<DetId>& ids = geometry->getValidDetIds(detector, subdetector);

  for (std::vector<DetId>::const_iterator it = ids.begin(), 
                                        iEnd = ids.end(); 
       it != iEnd; ++it) 
  {
    unsigned int rawId = (*it).rawId();

    std::vector<TEveVector> points = detIdToMatrix_.getPoints(rawId);

    if ( points.empty() )
    { 
      std::cout <<"Failed to get points of "<< detname 
                <<" element with detid: "<< rawId <<std::endl;
      continue;
    }

    assert(points.size() == 8);

    const CaloCellGeometry* cellGeometry = geometry->getGeometry(*it);
    const CaloCellGeometry::CornersVec& corners = cellGeometry->getCorners();
    
    assert(corners.size() == 8);

    for ( unsigned int i = 0; i < 8; ++i )
    {
      /*
      std::cout<< points[i][0] <<" "<< points[i][1] <<" "<< points[i][2] <<" | "
               << corners[i].x() <<" "<< corners[i].y() <<" "<< corners[i].z() <<std::endl;

      */

      double distance = getDistance(GlobalPoint(points[i][0], points[i][1], points[i][2]), 
                                    GlobalPoint(corners[i].x(), corners[i].y(), corners[i].z()));
      
      globalDistances_.push_back(distance);
    }
  }

  makeHistograms(detname);
}


void
ValidateGeometry::validateTrackerGeometry(const TrackerGeometry::DetContainer& dets,
                                          const char* detname)
{
  clearData();

  for ( TrackerGeometry::DetContainer::const_iterator it = dets.begin(), 
                                                   itEnd = dets.end(); 
        it != itEnd; ++it )
  {
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    unsigned int rawId = (*it)->geographicalId().rawId();

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(rawId);

    if ( ! matrix )
    {
      std::cout <<"Failed to get matrix of "<< detname 
                <<" element with detid: "<< rawId <<std::endl;
      continue;
    }

    compareTransform(gp, matrix);


    //TEveGeoShape* shape = detIdToMatrix_.getShape(rawId);
    const TGeoVolume* shape = detIdToMatrix_.getVolume(rawId);

    if ( ! shape )
    {
      std::cout<<"Failed to get shape of "<< detname 
               <<" element with detid: "<< rawId <<std::endl;
      continue;
    }

    compareShape(*it, shape->GetShape());
  }
  
  makeHistograms(detname);
}


void
ValidateGeometry::validateTrackerGeometry(const TrackerGeometry::DetUnitContainer& dets,
                                          const char* detname)
{
  clearData();

  for ( TrackerGeometry::DetUnitContainer::const_iterator it = dets.begin(), 
                                                       itEnd = dets.end(); 
        it != itEnd; ++it )
  {
    GlobalPoint gp = (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0,0.0,0.0));
    unsigned int rawId = (*it)->geographicalId().rawId();

    const TGeoHMatrix* matrix = detIdToMatrix_.getMatrix(rawId);

    if ( ! matrix )
    {
      std::cout<< "Failed to get matrix of "<< detname 
               <<" element with detid: "<< rawId <<std::endl;
      continue;
    }

    compareTransform(gp, matrix);


    //TEveGeoShape* shape = detIdToMatrix_.getShape(rawId);
    const TGeoVolume* shape = detIdToMatrix_.getVolume(rawId);

    if ( ! shape )
    {
      std::cout<<"Failed to get shape of "<< detname 
               <<" element with detid: "<< rawId <<std::endl;
      continue;
    }

    compareShape(*it, shape->GetShape());
  }
  
  makeHistograms(detname);
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
  globalDistances_.push_back(distance);
}


void 
ValidateGeometry::compareShape(const GeomDet* det, TGeoShape* shape)
{
  /*
    X -> width
    Y -> length
    Z -> thickness
  */

  double shape_topWidth;
  double shape_bottomWidth;
  double shape_length;
  double shape_thickness;

  bool tgeotrap = false;
  bool tgeobbox = false;

  if ( TGeoTrap* trap = dynamic_cast<TGeoTrap*>(shape) )
  {
    shape_topWidth = trap->GetTl1()*2.0;
    shape_bottomWidth = trap->GetBl1()*2.0;
    shape_length = trap->GetH1()*2.0;
    shape_thickness = trap->GetDz()*2.0;

    tgeotrap = true;
  }

  else if ( TGeoBBox* box = dynamic_cast<TGeoBBox*>(shape) )
  {
    shape_topWidth = box->GetDX()*2.0;
    shape_bottomWidth = shape_topWidth;
    shape_length = box->GetDY()*2.0;
    shape_thickness = box->GetDZ()*2.0;

    tgeobbox = true;
  }
  
  else
  {
    std::cout<<"Failed to get box or trapezoid from shape"<<std::endl;
    return;
  }

  double topWidth, bottomWidth;
  double length, thickness;

  bool trapezoid = false;
  bool rectangle = false;

  const Bounds* bounds = &(det->surface().bounds());
 
  if ( const TrapezoidalPlaneBounds* tpbs = dynamic_cast<const TrapezoidalPlaneBounds*>(bounds) )
  {
    std::vector<float> ps = tpbs->parameters();

    assert(ps.size() == 4);
    
    bottomWidth = ps[0]*2.0;
    topWidth = ps[1]*2.0;
    thickness = ps[2]*2.0;
    length = ps[3]*2.0;

    trapezoid = true;
  }

  else if ( (dynamic_cast<const RectangularPlaneBounds*>(bounds)) )
  {
    length = det->surface().bounds().length();
    topWidth = det->surface().bounds().width();
    bottomWidth = topWidth;
    thickness = det->surface().bounds().thickness();

    rectangle = true;
  }
  
  else
  {
    std::cout<<"Failed to get bounds"<<std::endl;
    return;
  }
   
  //assert((tgeotrap && trapezoid) || (tgeobbox && rectangle)); 

  /*
  std::cout<<"topWidth: "<< shape_topWidth <<" "<< topWidth <<std::endl;
  std::cout<<"bottomWidth: "<< shape_bottomWidth <<" "<< bottomWidth <<std::endl;
  std::cout<<"length: "<< shape_length <<" "<< length <<std::endl;
  std::cout<<"thickness: "<< shape_thickness <<" "<< thickness <<std::endl;
  */

  topWidths_.push_back(fabs(shape_topWidth - topWidth));
  bottomWidths_.push_back(fabs(shape_bottomWidth - bottomWidth));
  lengths_.push_back(fabs(shape_length - length));
  thicknesses_.push_back(fabs(shape_thickness - thickness));

  return;
}


double 
ValidateGeometry::getDistance(const GlobalPoint& p1, const GlobalPoint& p2)
{
  /*
  std::cout<<"X: "<< p1.x() <<" "<< p2.x() <<std::endl;
  std::cout<<"Y: "<< p1.y() <<" "<< p2.y() <<std::endl;
  std::cout<<"Z: "<< p1.z() <<" "<< p2.z() <<std::endl;
  */

  return sqrt((p1.x()-p2.x())*(p1.x()-p2.x())+
              (p1.y()-p2.y())*(p1.y()-p2.y())+
              (p1.z()-p2.z())*(p1.z()-p2.z()));
}


void
ValidateGeometry::makeHistograms(const char* detector)
{
  outFile_->cd();

  std::string d(detector);
  
  std::string gdn = d+": distance between points in global coordinates";
  makeHistogram(gdn, globalDistances_);

  std::string twn = d + ": absolute difference between top widths (along X)";
  makeHistogram(twn, topWidths_);

  std::string bwn = d + ": absolute difference between bottom widths (along X)";
  makeHistogram(bwn, bottomWidths_);

  std::string ln = d + ": absolute difference between lengths (along Y)";
  makeHistogram(ln, lengths_);

  std::string tn = d + ": absolute difference between thicknesses (along Z)";
  makeHistogram(tn, thicknesses_);

  return;
}


void
ValidateGeometry::makeHistogram(const std::string& name, std::vector<double>& data)
{
  if ( data.empty() )
    return;

  std::vector<double>::iterator it;
  
  it = std::min_element(data.begin(), data.end());
  double minE = *it;

  it = std::max_element(data.begin(), data.end());
  double maxE = *it;

  std::vector<double>::iterator itEnd = data.end();

  TH1D hist(name.c_str(), name.c_str(), 100, minE*(1+0.10), maxE*(1+0.10));
  
  for ( it = data.begin(); it != itEnd; ++it )
    hist.Fill(*it);
 
  hist.GetXaxis()->SetTitle("[cm]");
  hist.Write();
}

void 
ValidateGeometry::beginJob()
{
  outFile_->cd();
}


void 
ValidateGeometry::endJob() 
{
  std::cout<<"Done. "<<std::endl;
  std::cout<<"Results written to "<< outfileName_ <<std::endl;
  outFile_->Close();
}

DEFINE_FWK_MODULE(ValidateGeometry);

