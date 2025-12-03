#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

//inputs
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThDigi.h"

//outputs
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhiThetaPair.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhiThetaPair.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhiThetaPairContainer.h"

#include <fstream>
#include <iostream>
#include <queue>
#include <cmath>

using namespace edm;
using namespace std;
using namespace cmsdt;

namespace {
  struct {
    bool operator()(const L1Phase2MuDTExtPhDigi &mp1, const L1Phase2MuDTExtPhDigi &mp2) const {

      int sector1 = mp1.scNum();
      int wheel1 = mp1.whNum();
      int station1 = mp1.stNum();

      int sector2 = mp2.scNum();
      int wheel2 = mp2.whNum();
      int station2 = mp2.stNum();

      // First, compare by chamber
      if (sector1 != sector2)
        return sector1 < sector2;
      if (wheel1 != wheel2)
        return wheel1 < wheel2;
      if (station1 != station2)
        return station1 < station2;

      // If they are in the same category, sort by the value (4th index)
      return mp1.quality() > mp2.quality();
    }
  } const comparePhiDigis;

  struct {
    bool operator()(const L1Phase2MuDTThDigi &mp1, const L1Phase2MuDTThDigi &mp2) const {

      int sector1 = mp1.scNum();
      int wheel1 = mp1.whNum();
      int station1 = mp1.stNum();

      int sector2 = mp2.scNum();
      int wheel2 = mp2.whNum();
      int station2 = mp2.stNum();

      // First, compare by chamber
      if (sector1 != sector2)
        return sector1 < sector2;
      if (wheel1 != wheel2)
        return wheel1 < wheel2;
      if (station1 != station2)
        return station1 < station2;

      // If they are in the same category, sort by the value (4th index)
      return mp1.quality() > mp2.quality();
    }
  } const compareThetaDigis;

  struct {
    bool operator()(const L1Phase2MuDTExtPhiThetaPair &mp1, const L1Phase2MuDTExtPhiThetaPair &mp2) const {
      return mp1.quality() > mp2.quality();
    }
  } const comparePairs;

}  //namespace


class DTTrigPhase2PairsProd : public edm::stream::EDProducer<> {

public:
  //! Constructor
  DTTrigPhase2PairsProd(const edm::ParameterSet& pset);

  //! Destructor
  ~DTTrigPhase2PairsProd() override;

  //! Create Trigger Units before starting event processing
  void beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  //! Producer: process every event and generates trigger data
  void produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup) override;

  //! endRun: finish things
  void endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  // Methods
  void sortPhiDigis(const edm::Handle<L1Phase2MuDTExtPhContainer> &thePhiDigis);
  void sortThetaDigis(const edm::Handle<L1Phase2MuDTExtThContainer> &theThetaDigis);
  std::vector<L1Phase2MuDTExtPhiThetaPair> bestPairsPerChamber(
    const std::vector<L1Phase2MuDTExtPhDigi>& phiDigis,
    const std::vector<L1Phase2MuDTExtThDigi>& thetaDigis,
    unsigned int maxPairs, int wheel, int sector, int station);
  float computeTimePosDistance(
   const L1Phase2MuDTExtPhDigi phiDigi,
   const L1Phase2MuDTExtThDigi thetaDigi,
   int sector, int wheel, int station);
  // Getter-methods

  // Setter-methods

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // data-members

  double shift_back;

private:

  // Debug Flag
  bool debug_;
  int scenario_;
  int max_index_;

  // ParameterSet
  edm::EDGetTokenT<L1Phase2MuDTExtPhContainer> digiPhToken_;
  edm::EDGetTokenT<L1Phase2MuDTExtThContainer> digiThToken_;

};


DTTrigPhase2PairsProd::DTTrigPhase2PairsProd(const ParameterSet& pset){
  produces<L1Phase2MuDTExtPhiThetaPairContainer>();

  debug_    = pset.getUntrackedParameter<bool>("debug");
  scenario_ = pset.getParameter<int>("scenario");
//  max_index_ = pset.getParameter<int>("max_primitives") - 1;
  max_index_ = 4;

  digiPhToken_ = consumes<L1Phase2MuDTExtPhContainer>(pset.getParameter<edm::InputTag>("digiPhTag"));
  digiThToken_ = consumes<L1Phase2MuDTExtThContainer>(pset.getParameter<edm::InputTag>("digiThTag"));

  double temp_shift = 0;
  if (scenario_ == MC)
    temp_shift = 400;  // value used in standard CMSSW simulation
  else if (scenario_ == DATA)
    temp_shift = 0;
  else if (scenario_ == SLICE_TEST)
    temp_shift = 400;  // slice test mimics simulation

  shift_back = temp_shift; 

}

DTTrigPhase2PairsProd::~DTTrigPhase2PairsProd() {
  if (debug_)
    LogDebug("DTTrigPhase2PairsProd") << "calling destructor" << std::endl;
}

void DTTrigPhase2PairsProd::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if (debug_)
    LogDebug("DTTrigPhase2PairsProd") << "beginRun " << iRun.id().run();
}

void DTTrigPhase2PairsProd::produce(Event& iEvent, const EventSetup& iEventSetup) {
  if (debug_)
    LogDebug("DTTrigPhase2PairsProd") << "produce";

  edm::Handle<L1Phase2MuDTExtPhContainer> thePhiDigis;
  iEvent.getByToken(digiPhToken_, thePhiDigis);

  edm::Handle<L1Phase2MuDTExtThContainer> theThetaDigis;
  iEvent.getByToken(digiThToken_, theThetaDigis);

  if (!thePhiDigis || !theThetaDigis) {
    throw cms::Exception("NullPointer") << "Phi or Theta container is null!";
  }

  cout<<"DTTrigPhase2PairsProd" << " produced"<<endl;

  //Order Theta Digis by quality in the same chamber
  //sortThetaDigis(theThetaDigis);
  //Order Phi Digis by quality in the same chamber
  //sortPhiDigis(thePhiDigis);
  //Not needed, we sort later based on quality once we have keys (chambers)

  using ChamberKey = std::tuple<int, int, int>; // (wheel, sector, station)

  std::map<ChamberKey, std::vector<L1Phase2MuDTExtPhDigi>> phiByChamber;
  std::map<ChamberKey, std::vector<L1Phase2MuDTExtThDigi>> thetaByChamber;

  if (debug_)
    LogDebug("DTTrigPhase2PairsProd") << "Variables declaration";

  // Group phi digis
  for (auto phiIte = thePhiDigis->getContainer()->begin();
     phiIte != thePhiDigis->getContainer()->end(); ++phiIte) {
    ChamberKey key(phiIte->whNum(), phiIte->scNum(), phiIte->stNum());
    phiByChamber[key].push_back(*phiIte);
  }

  // Group theta digis
  for (auto thetaIte = theThetaDigis->getContainer()->begin();
     thetaIte != theThetaDigis->getContainer()->end(); ++thetaIte) {
    ChamberKey key(thetaIte->whNum(), thetaIte->scNum(), thetaIte->stNum());
    thetaByChamber[key].push_back(*thetaIte);
  }

  if (debug_)
    LogDebug("DTTrigPhase2PairsProd") << "Grouping per chamber";
  
   cout<<"DTTrigPhase2PairsProd: Grouping per chamber"<<endl;
  std::vector<L1Phase2MuDTExtPhiThetaPair> allPairs;

  // Process each chamber key from phi digis
  for (const auto& [key, phiList] : phiByChamber) {
    std::vector<L1Phase2MuDTExtPhiThetaPair> chamberPairs;
    std::vector<L1Phase2MuDTExtPhDigi> phiDigis;
    std::vector<L1Phase2MuDTExtThDigi> thetaDigis;

    auto thetaListIt = thetaByChamber.find(key);
//    if (thetaListIt != thetaByChamber.end()) {
//    Both phi and theta exist
//
     for (const auto& phi : phiList) 
            phiDigis.emplace_back(phi);
   
     for (const auto& theta : thetaListIt->second) 
            thetaDigis.emplace_back(theta);     

     if (debug_)
         LogDebug("DTTrigPhase2PairsProd") << "Working on chamber:";
      cout<<"DTTrigPhase2PairsProd: Working on chamber"<<endl;
     std::sort(phiDigis.begin(), phiDigis.end(), comparePhiDigis);
     std::sort(thetaDigis.begin(), thetaDigis.end(), compareThetaDigis);
 
     if (debug_)
       LogDebug("DTTrigPhase2PairsProd") << "Sorting";
      cout<<"DTTrigPhase2PairsProd: Sorting"<<endl;
     auto [wheel, sector, station] = key; // unpack tuple
     chamberPairs = std::move(bestPairsPerChamber(phiDigis,thetaDigis,max_index_,wheel,sector,station));
     if (debug_)
         LogDebug("DTTrigPhase2PairsProd") << "Saving top 4";
       cout<<"DTTrigPhase2PairsProd: Saving top 4"<<endl;
    for (const auto& pair : chamberPairs) 
          allPairs.emplace_back(pair);
        }
     if (debug_)
       LogDebug("DTTrigPhase2PairsProd") << "Saved";
        cout<<"DTTrigPhase2PairsProd: Saved"<<endl;

  // Storing results in the event
    std::unique_ptr<L1Phase2MuDTExtPhiThetaPairContainer> resultPhiThetaPair(new L1Phase2MuDTExtPhiThetaPairContainer);
    resultPhiThetaPair->setContainer(allPairs);
  iEvent.put(std::move(resultPhiThetaPair));
    if (debug_)
    LogDebug("DTTrigPhase2PairsProd") << "Saved in the event";
    cout<<"DTTrigPhase2PairsProd: Saved in the event"<<endl;
  allPairs.clear();
  allPairs.erase(allPairs.begin(), allPairs.end());
}

void DTTrigPhase2PairsProd::endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
};

void DTTrigPhase2PairsProd::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // dtTriggerPhase2PrimitivePairDigis
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiPhTag", edm::InputTag("dtTriggerPhase2PrimitiveDigis"));
  desc.add<edm::InputTag>("digiThTag", edm::InputTag("dtTriggerPhase2PrimitiveDigis"));
  desc.add<int>("scenario", 0);
  desc.addUntracked<bool>("debug", false);
  descriptions.add("dtTriggerPhase2PrimitivePairDigis", desc);
}

DEFINE_FWK_MODULE(DTTrigPhase2PairsProd);


std::vector<L1Phase2MuDTExtPhiThetaPair> DTTrigPhase2PairsProd::bestPairsPerChamber(
    const std::vector<L1Phase2MuDTExtPhDigi>& phiDigis,
    const std::vector<L1Phase2MuDTExtThDigi>& thetaDigis,
    unsigned int maxPairs, int wheel, int sector, int station) {

  std::vector<L1Phase2MuDTExtPhiThetaPair> pairs;

  if (station == 4 || thetaDigis.empty()) { // no theta digis in chamber

  L1Phase2MuDTExtThDigi emptyTheta; 
  for (const auto& phi : phiDigis){
    int phiQuality = phi.quality();
    pairs.emplace_back(phi, emptyTheta, phiQuality);
  }
 
  }

  else if ( phiDigis.empty() && !(thetaDigis.empty())){ // no phi digis in chamber

  L1Phase2MuDTExtPhDigi emptyPhi;
  for (const auto& theta : thetaDigis){
    int thetaQuality = theta.quality();
    pairs.emplace_back(emptyPhi, theta, thetaQuality);

  } 

}

  else{  // phi and theta digis in chamber

  for (const auto& phi : phiDigis) {
    float closestDistance = 9999.;
    const L1Phase2MuDTExtThDigi *closestTheta=nullptr;

    for (const auto& theta : thetaDigis) {
        float currentDistance = computeTimePosDistance(phi,theta,sector,wheel,station);
        if(closestDistance > currentDistance){
          closestDistance = currentDistance;
          closestTheta = &theta;
        }             
      
	if (!closestTheta) {
         cout << "[ERROR] closestTheta is null for phi digi with quality " << phi.quality() << " Current distance is:" << currentDistance<< "and closest distance is: "<< closestDistance<<endl;
          continue; // or throw an exception
         }
           
      int phiQuality = phi.quality();
      if(closestTheta)
      pairs.emplace_back(phi, *closestTheta, phiQuality);
    }
  }
}
  // Sort by quality descending
  std::sort(pairs.begin(), pairs.end(),comparePairs);

  // Keep only top-N
  if (pairs.size() > maxPairs)
    pairs.resize(maxPairs);

  return pairs;
}


float DTTrigPhase2PairsProd::computeTimePosDistance(
   const L1Phase2MuDTExtPhDigi phiDigi,
   const L1Phase2MuDTExtThDigi thetaDigi,
   int sector, int wheel, int station) {

   float t01 = ((int)round(thetaDigi.t0() / (float)LHC_CLK_FREQ)) - shift_back;
   float posRefZ = zFE[wheel + 2];
 
   if (wheel == 0 && (sector == 1 || sector == 4 || sector == 5 || sector == 8 || sector == 9 || sector == 12))
      posRefZ = -posRefZ;

   float posZ = abs(thetaDigi.z());

   float t02 = ((int)round(phiDigi.t0() / (float)LHC_CLK_FREQ)) - shift_back;

   float tphi = t02 - abs(posZ / ZRES_CONV - posRefZ) / vwire;

   int LR = -1;
   if (wheel == 0 && (sector == 3 || sector == 4 || sector == 7 || sector == 8 || sector == 11 || sector == 12))
        LR = +1;
   else if (wheel > 0)
        LR = pow(-1, wheel + sector + 1);
   else if (wheel < 0)
        LR = pow(-1, -wheel + sector);

   float posRefX = LR * xFE[station - 1];
   float ttheta = t01 - (phiDigi.xLocal() / 1000 - posRefX) / vwire;

   return abs(tphi - ttheta);
}
   
/*void DTTrigPhase2PairsProd::sortThetaDigis(const edm::Handle<L1Phase2MuDTExtThContainer> &theThetaDigis) {
  // Copy to a vector for sorting
  std::vector<L1Phase2MuDTThDigi> sortedThetaDigis(theThetaDigis->begin(), theThetaDigis->end());

  // Sort using comparator
  std::sort(sortedThetaDigis.begin(), sortedThetaDigis.end(), compareThetaDigis);

  for (const auto &theta : sortedThetaDigis) {
    edm::LogInfo("ThetaSorting") << "Wheel=" << theta.whNum()
                                 << " Sector=" << theta.scNum()
                                 << " Station=" << theta.stNum();
  }
}

void DTTrigPhase2PairsProd::sortPhiDigis(const edm::Handle<L1Phase2MuDTExtPhContainer> &thePhiDigis) {
  // Copy to a vector for sorting
  std::vector<L1Phase2MuDTExtPhDigi> sortedThetaDigis(thePhiDigis->begin(), thePhiDigis->end());

  // Sort using comparator
  std::sort(sortedPhiDigis.begin(), sortedPhiDigis.end(), comparePhiDigis());

  for (const auto &theta : sortedPhiDigis) {
    edm::LogInfo("PhiSorting") << "Wheel=" << theta.whNum()
                                 << " Sector=" << theta.scNum()
                                 << " Station=" << theta.stNum();
  }
}
*/
