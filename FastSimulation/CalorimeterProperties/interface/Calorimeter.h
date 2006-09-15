#ifndef Calorimeter_h
#define Calorimeter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//#include "Geometry/Vector/interface/GlobalPoint.h"
#include "CLHEP/Geometry/Point3D.h"
class PreshowerLayer1Properties;
class PreshowerLayer2Properties;
class ECALProperties;
class ECALBarrelProperties;
class ECALEndcapProperties;
class HCALProperties;
class HCALBarrelProperties;
class HCALEndcapProperties;
class HCALForwardProperties;
class CaloSubdetectorGeometry;
class EcalBarrelGeometry;
class EcalEndcapGeometry;

class Calorimeter{
 public:
  Calorimeter();
  Calorimeter(const edm::ParameterSet& caloParameters);
  ~Calorimeter();

    // Setup the geometry
  void setupGeometry(const edm::ESHandle<CaloGeometry>& pG);
  
 /// ECAL properties
  const ECALProperties* ecalProperties(int onEcal) const;

  /// HCAL properties
  const HCALProperties* hcalProperties(int onHcal) const;

  /// Preshower Layer1 properties
  const PreshowerLayer1Properties* layer1Properties(int onLayer1) const;

  /// Preshower Layer2 properties
  const PreshowerLayer2Properties* layer2Properties(int onLayer2) const;

  double preshowerZPosition(int layer) const
  {
    return (layer==1) ? psLayer1Z_: psLayer2Z_ ; 
  }

  // more user friendly getClosestCell  
  DetId getClosestCell(const HepPoint3D& point, bool ecal, bool central) const;

 private:

  //Calorimeter properties
  PreshowerLayer1Properties*     myPreshowerLayer1Properties_  ;
  PreshowerLayer2Properties*     myPreshowerLayer2Properties_  ;
  ECALBarrelProperties*          myECALBarrelProperties_       ;
  ECALEndcapProperties*	         myECALEndcapProperties_       ;
  HCALBarrelProperties*	         myHCALBarrelProperties_       ;
  HCALEndcapProperties*          myHCALEndcapProperties_       ;
  HCALForwardProperties*         myHCALForwardProperties_      ;

  // Preshower layer positions
  double psLayer1Z_,psLayer2Z_;

  // The subdetectors geometry
  const EcalBarrelGeometry* EcalBarrelGeometry_;
  const EcalEndcapGeometry* EcalEndcapGeometry_;
  const CaloSubdetectorGeometry* HcalGeometry_;
  const CaloSubdetectorGeometry* PreshowerGeometry_;

};

#endif
