#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/ProcessMatch.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include <string>
#include <iostream>
#include <bitset>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

using namespace std;

namespace cms {

  /** \class HcalRecHitDump
      
  \author J. Mans - Minnesota
  */
  class HcalRecHitDump : public edm::EDAnalyzer
  {
  public:
    explicit HcalRecHitDump(edm::ParameterSet const& conf);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

  private:
    edm::GetterOfProducts<HcalSourcePositionData> getHcalSourcePositionData_;
    edm::GetterOfProducts<HBHERecHitCollection> getHBHERecHitCollection_;
    edm::GetterOfProducts<HFRecHitCollection> getHFRecHitCollection_;
    string hbhePrefix_;
    string hoPrefix_;
    string hfPrefix_;
  };

  HcalRecHitDump::HcalRecHitDump(edm::ParameterSet const& conf) :
    getHcalSourcePositionData_(edm::ProcessMatch("*"), this),
    getHBHERecHitCollection_(edm::ProcessMatch("*"), this),
    getHFRecHitCollection_(edm::ProcessMatch("*"), this),
    hbhePrefix_(conf.getUntrackedParameter<string>("hbhePrefix", "")),
    hoPrefix_(conf.getUntrackedParameter<string>("hoPrefix", "")),
    hfPrefix_(conf.getUntrackedParameter<string>("hfPrefix", ""))
  {
    callWhenNewProductsRegistered([this](edm::BranchDescription const& bd) {
       getHcalSourcePositionData_(bd);
       getHBHERecHitCollection_(bd);
       getHFRecHitCollection_(bd);
    });
  }

  template<typename COLL>
  static void analyzeT(typename std::vector<edm::Handle<COLL> > const& handles , const char* name=0, const char* prefix=0)
  {
    bool printAllBits = true;
    const string marker(prefix ? prefix : "");
//    try {
      //vector<edm::Handle<COLL> > handles;
      //e.getManyByType(colls);
      typename std::vector<edm::Handle<COLL> >::const_iterator i;
      cout << "New event (" << handles.size() << " handles)" << endl;
      for (i=handles.begin(); i!=handles.end(); i++) {
        for (typename COLL::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++){
          cout << marker << *j
               << "; stb: ";
          if (printAllBits) cout << std::bitset<32>(j->flags());

          if(j->id().subdet() == HcalBarrel || j->id().subdet() == HcalEndcap){
            if(j->flagField(HcalCaloFlagLabels::HBHEHpdHitMultiplicity) == 1) cout << " HBHEHpdHitMultiplicity";
            if(j->flagField(HcalCaloFlagLabels::HBHEPulseShape) == 1) cout << " HBHEPulseShape";
            if(j->flagField(HcalCaloFlagLabels::HSCP_R1R2) == 1) cout << " HSCP_R1R2";
            if(j->flagField(HcalCaloFlagLabels::HSCP_FracLeader) == 1) cout << " HSCP_FracLeader";
            if(j->flagField(HcalCaloFlagLabels::HSCP_OuterEnergy) == 1) cout << " HSCP_OuterEnergy";
            if(j->flagField(HcalCaloFlagLabels::HSCP_ExpFit) == 1) cout << " HSCP_ExpFit";
            //if(j->flagField(HcalCaloFlagLabels::HBHETimingTrustBits=6, // 2-bit counter; not yet in use
            int status = j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits, 3);
            cout << " HBHETimingShapedCutsBits " << status;
            if(j->flagField(HcalCaloFlagLabels::HBHEIsolatedNoise) == 1) cout << " HBHEIsolatedNoise";
            if(j->flagField(HcalCaloFlagLabels::HBHEFlatNoise) == 1) cout << " HBHEFlatNoise";
            if(j->flagField(HcalCaloFlagLabels::HBHESpikeNoise) == 1) cout << " HBHESpikeNoise";
            if(j->flagField(HcalCaloFlagLabels::HBHETriangleNoise) == 1) cout << " HBHETriangleNoise";
            if(j->flagField(HcalCaloFlagLabels::HBHETS4TS5Noise) == 1) cout << " HBHETS4TS5Noise";
            if(j->flagField(HcalCaloFlagLabels::HBHENegativeNoise) == 1) cout << " HBHENegativeNoise";
            if(j->flagField(HcalCaloFlagLabels::HBHEPulseFitBit) == 1) cout << " HBHEPulseFitBit";
            if(j->flagField(HcalCaloFlagLabels::HBHEOOTPU) == 1) cout << " HBHEOOTPU";
          }else if(j->id().subdet() == HcalForward){
            if(j->flagField(HcalCaloFlagLabels::HFDigiTime) == 1) cout << " HFDigiTime";
            if(j->flagField(HcalCaloFlagLabels::HFInTimeWindow) == 1) cout << " HFInTimeWindow";
            if(j->flagField(HcalCaloFlagLabels::HFS8S1Ratio) == 1) cout << " HFS8S1Ratio";
            if(j->flagField(HcalCaloFlagLabels::HFPET) == 1) cout << " HFPET";
          }

          cout << "; auxb: ";
          if (printAllBits) cout << std::bitset<32>(j->aux()); 
          cout << endl;
        }
      }
//    } catch (...) {
//      if(name) cout << "No " << name << " RecHits." << endl;
//    }
  }

  void HcalRecHitDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
    
    std::vector<edm::Handle<HBHERecHitCollection> > handles0;
    getHBHERecHitCollection_.fillHandles(e, handles0);    
    analyzeT<HBHERecHitCollection>(handles0, "HB/HE", hbhePrefix_.c_str()); 

    std::vector<edm::Handle<HFRecHitCollection> > handles1;
    getHFRecHitCollection_.fillHandles(e, handles1);    
    analyzeT<HFRecHitCollection>(handles1, "HF", hfPrefix_.c_str());
    //analyzeT<HORecHitCollection>(e, "HO", hoPrefix_.c_str());
    //analyzeT<HcalCalibRecHitCollection>(e);
    //analyzeT<ZDCRecHitCollection>(e);
    //analyzeT<CastorRecHitCollection>(e);

    std::vector<edm::Handle<HcalSourcePositionData> > handles;
    getHcalSourcePositionData_.fillHandles(e, handles);
    for (auto const& spd : handles){
      cout << *spd << endl;
    }
    cout << endl;    
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;


DEFINE_FWK_MODULE(HcalRecHitDump);

