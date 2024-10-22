#ifndef FastSimulation_CalorimeterProperties_Calorimeter_h
#define FastSimulation_CalorimeterProperties_Calorimeter_h

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

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
class CaloTopology;
class CaloSubdetectorTopology;
class EcalBarrelGeometry;
class EcalEndcapGeometry;
class EcalPreshowerGeometry;

namespace edm {
  class ParameterSet;
}

class Calorimeter {
public:
  Calorimeter();
  Calorimeter(const edm::ParameterSet& caloParameters);
  ~Calorimeter();

  // Setup the geometry
  void setupGeometry(const CaloGeometry& pG);

  // Setup the topology
  void setupTopology(const CaloTopology&);

  /// ECAL properties
  const ECALProperties* ecalProperties(int onEcal) const;

  /// HCAL properties
  const HCALProperties* hcalProperties(int onHcal) const;

  /// Preshower Layer1 properties
  const PreshowerLayer1Properties* layer1Properties(int onLayer1) const;

  /// Preshower Layer2 properties
  const PreshowerLayer2Properties* layer2Properties(int onLayer2) const;

  inline const EcalBarrelGeometry* getEcalBarrelGeometry() const { return EcalBarrelGeometry_; }

  inline const EcalEndcapGeometry* getEcalEndcapGeometry() const { return EcalEndcapGeometry_; }

  inline const EcalPreshowerGeometry* getEcalPreshowerGeometry() const { return PreshowerGeometry_; }

  inline const CaloSubdetectorGeometry* getHcalGeometry() const { return HcalGeometry_; }

  const CaloSubdetectorGeometry* getEcalGeometry(int subdetn) const;

  const CaloSubdetectorTopology* getEcalTopology(int subdetn) const;

protected:
  //Calorimeter properties
  PreshowerLayer1Properties* myPreshowerLayer1Properties_;
  PreshowerLayer2Properties* myPreshowerLayer2Properties_;
  ECALBarrelProperties* myECALBarrelProperties_;
  ECALEndcapProperties* myECALEndcapProperties_;
  HCALBarrelProperties* myHCALBarrelProperties_;
  HCALEndcapProperties* myHCALEndcapProperties_;
  HCALForwardProperties* myHCALForwardProperties_;

  // The subdetectors geometry
  const EcalBarrelGeometry* EcalBarrelGeometry_;
  const EcalEndcapGeometry* EcalEndcapGeometry_;
  const CaloSubdetectorGeometry* HcalGeometry_;
  const EcalPreshowerGeometry* PreshowerGeometry_;

  // The ECAL topologies
  const CaloSubdetectorTopology* EcalBarrelTopology_;
  const CaloSubdetectorTopology* EcalEndcapTopology_;
};

#endif
