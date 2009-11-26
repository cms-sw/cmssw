#ifndef L1RCTProducer_h
#define L1RCTProducer_h 

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

class L1RCT;
class L1RCTLookupTables;

class L1RCTProducer : public edm::EDProducer
{
 public:
  explicit L1RCTProducer(const edm::ParameterSet& ps);
  virtual ~L1RCTProducer();
  virtual void beginJob(const edm::EventSetup& c);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
 private:
  L1RCTLookupTables* rctLookupTables;
  L1RCT* rct;
  bool useEcal;
  bool useHcal;
  std::vector<edm::InputTag> ecalDigis;
  std::vector<edm::InputTag> hcalDigis;
  std::vector<int> bunchCrossings; 
  bool useDebugTpgScales;

  enum crateSection{
    c_min,
    ebOddFed = c_min,
    ebEvenFed,
    eeFed,
    hbheFed,
    hfFed,
    c_max = hfFed
  };



  static const int crateFED[][5];
  /*
    {{613, 614, 603, 702, 718},
    {611, 612, 602, 700, 718},
    {627, 610, 601,716,   722},
    {625, 626, 609, 714, 722},
    {623, 624, 608, 712, 722},
    {621, 622, 607, 710, 720},
    {619, 620, 606, 708, 720},
    {617, 618, 605, 706, 720},
    {615, 616, 604, 704, 718},
    {631, 632, 648, 703, 719},
    {629, 630, 647, 701, 719},
    {645, 628, 646, 717, 723},
    {643, 644, 654, 715, 723},
    {641, 642, 653, 713, 723},
    {639, 640, 652, 711, 721},
    {637, 638, 651, 709, 721},
    {635, 636, 650, 707, 721},
    {633, 634, 649, 705, 719}};
  */
  
  static const int minBarrel = 1;
  static const int maxBarrel = 17;
  static const int minEndcap = 17;
  static const int maxEndcap = 28;
  static const int minHF = 29;
  static const int maxHF =32;



};


#endif
