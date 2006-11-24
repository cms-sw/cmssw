#ifndef CALORIMETRYMANAGER_H
#define CALORIMETRYMANAGER_H

//CMSSW headerders 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"


// FastSimulation headers
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/Calorimetry/interface/HCALResponse.h"

#include "FastSimulation/Calorimetry/interface/FamosDebug.h"

// For the uint32_t
#include <boost/cstdint.hpp>
#include <map>

class FSimEvent;
class FSimTrack;
class RawParticle;
class CaloGeometryHelper;
class Histos;
class HSParameters;
class RandomEngine;

class CalorimetryManager{

 public:
  CalorimetryManager();
  CalorimetryManager(FSimEvent* aSimEvent, const edm::ParameterSet& fastCalo);
  ~CalorimetryManager();

  // Does the real job
  void reconstruct();

   // access method to calorimeter info
  inline std::map<uint32_t,float>& getESMapping() { return ESMapping_;}
  inline std::map<uint32_t,float>& getEBMapping() { return EBMapping_;}
  inline std::map<uint32_t,float>& getEEMapping() { return EEMapping_;}
  inline std::map<uint32_t,float>& getHMapping() { return HMapping_;}

    // Return the address of the Calorimeter 
  CaloGeometryHelper * getCalorimeter() const {return myCalorimeter_;}

  // load container from edm::Event
  void loadFromEcalBarrel(edm::PCaloHitContainer & c) const;

  void loadFromEcalEndcap(edm::PCaloHitContainer & c) const;

  void loadFromHcal(edm::PCaloHitContainer & c) const;


 private:
  // Simulation of electromagnetic showers in PS, ECAL, HCAL
  void EMShowerSimulation(const FSimTrack& myTrack);
  
  // Simulation of electromagnetic showers in VFCAL
  void reconstructECAL(const FSimTrack& track) ;

  void reconstructHCAL(const FSimTrack& myTrack);

  /// Hadronic Shower Simulation
  void HDShowerSimulation(const FSimTrack& myTrack);

  // Read the parameters 
  void readParameters(const edm::ParameterSet& fastCalo);

  void updateMap(uint32_t cellid,float energy,std::map<uint32_t,float>& mymap);

 private:
  FSimEvent* mySimEvent;
  CaloGeometryHelper* myCalorimeter_;

  Histos * myHistos;

  HCALResponse* myHDResponse_;
  HSParameters * myHSParameters_;

  std::map<unsigned,float> EBMapping_;
  std::map<unsigned,float> EEMapping_;
  std::map<unsigned,float> HMapping_;
  std::map<unsigned,float> ESMapping_;
  
  bool debug_;

  /// A few pointers to save time
  RawParticle myElec;
  RawParticle myPosi;
  RawParticle myPart;

  // Parameters 
  double pulledPadSurvivalProbability_;
  double crackPadSurvivalProbability_;
  double spotFraction_;
  double radiusFactor_;
  int gridSize_;
  std::vector<double> theCoreIntervals_,theTailIntervals_;
  //FR
  int optionHDSim_, hdGridSize_, hdSimMethod_;
  //RF

  // Famos Random Engine
  RandomEngine* random;
};
#endif
