#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/ProcessMatch.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include <string>
#include <vector>
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
    template<typename COLL>
    void analyzeT(typename std::vector<edm::Handle<COLL> > const& , const char* =0, const char* =0);
  
  private:
    edm::GetterOfProducts<HcalSourcePositionData> getHcalSourcePositionData_;
    edm::GetterOfProducts<HBHERecHitCollection> getHBHERecHitCollection_;
    edm::GetterOfProducts<HFRecHitCollection> getHFRecHitCollection_;
    string hbhePrefix_;
    string hoPrefix_;
    string hfPrefix_;
    std::vector<int> flagsb_;
    std::vector<int> auxb_;
    std::vector<int> auxHBHEb_;
    std::vector<int> auxPhase1b_;
  };

  HcalRecHitDump::HcalRecHitDump(edm::ParameterSet const& conf) :
    getHcalSourcePositionData_(edm::ProcessMatch("*"), this),
    getHBHERecHitCollection_(edm::ProcessMatch("*"), this),
    getHFRecHitCollection_(edm::ProcessMatch("*"), this),
    hbhePrefix_(conf.getUntrackedParameter<string>("hbhePrefix", "")),
    hoPrefix_(conf.getUntrackedParameter<string>("hoPrefix", "")),
    hfPrefix_(conf.getUntrackedParameter<string>("hfPrefix", "")),
    flagsb_ ( conf.getUntrackedParameter<std::vector<int>>( "flagsb" )) ,
    auxb_ ( conf.getUntrackedParameter<std::vector<int>>( "auxb" )) ,
    auxHBHEb_ ( conf.getUntrackedParameter<std::vector<int>>( "auxHBHEb" )) ,
    auxPhase1b_ ( conf.getUntrackedParameter<std::vector<int>>( "auxPhase1b" )) 
  {
    callWhenNewProductsRegistered([this](edm::BranchDescription const& bd) {
       getHcalSourcePositionData_(bd);
       getHBHERecHitCollection_(bd);
       getHFRecHitCollection_(bd);
    });
  }

  template<typename COLL>
  void HcalRecHitDump::analyzeT(typename std::vector<edm::Handle<COLL> > const& handles , const char* name, const char* prefix)
  {
    //bool printAllBits = true;
    const string marker(prefix ? prefix : "");
//    try {
      //vector<edm::Handle<COLL> > handles;
      //e.getManyByType(colls);
      typename std::vector<edm::Handle<COLL> >::const_iterator i;
      cout << "New event (" << handles.size() << " handles)" << endl;
      bool dbit; 
      for (i=handles.begin(); i!=handles.end(); i++) {
        for (typename COLL::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++){ // loop over rechits
          cout << marker << *j
               << "; flagsb: ";
          //cout << std::bitset<32>(j->flags()) << "; ";
          if (flagsb_.size()>0) cout << "; flag bits: ";
          for (std::vector<int>::iterator it = flagsb_.begin() ; it != flagsb_.end(); ++it)
              if (*it == -1){ // print separator
                cout << '-';
              } else {
                dbit=(j->flags() & ( 1 << *it )) >> *it;
                cout << dbit;
              }
          if (auxb_.size()>0) cout << "; aux bits: ";
          for (std::vector<int>::iterator it = auxb_.begin() ; it != auxb_.end(); ++it)
              if (*it == -1){ // print separator
                cout << '-';
              } else {
                dbit=(j->aux() & ( 1 << *it )) >> *it;
                cout << dbit;
              }
          //cout << endl;
          if(j->id().subdet() == HcalBarrel || j->id().subdet() == HcalEndcap){
              if (auxHBHEb_.size()>0) cout << "; aux bits: ";
              for (std::vector<int>::iterator it = auxHBHEb_.begin() ; it != auxHBHEb_.end(); ++it)
                if (*it == -1){ // print separator
                  cout << '-';
                } else {
                  dbit=(j->auxHBHE() & ( 1 << *it )) >> *it;
                  cout << dbit;
                }
          }else if(j->id().subdet() == HcalForward){
            ;
          }

          if (auxPhase1b_.size()>0) cout << "; aux_phase1 bits: ";
              for (std::vector<int>::iterator it = auxPhase1b_.begin() ; it != auxPhase1b_.end(); ++it)
                if (*it == -1){ // print separator
                  cout << '-';
                } else {
                  dbit=(j->auxPhase1() & ( 1 << *it )) >> *it;
                  cout << dbit;
                }
          
          cout << endl;
          throw;

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

