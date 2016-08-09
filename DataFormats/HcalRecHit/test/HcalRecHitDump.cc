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

template<typename T>
void getauxb( T & j, uint32_t allbits[4]){
    allbits[0] = 0;
    allbits[1] = 0;
    //std::cout << std::endl << " [not HBHE type] " << std::bitset<32>(allbits[3]) << " " << std::bitset<32>(allbits[2]) << endl;
}

template<>
void getauxb( edm::SortedCollection<HBHERecHit>::const_iterator & j, uint32_t allbits[4]){
    allbits[0] = j->auxPhase1();
    allbits[1] = j->auxHBHE();
    //std::cout << std::endl << " [HBHE type] " << std::bitset<32>(allbits[3]) << " " << std::bitset<32>(allbits[2]) << " " << std::bitset<32>(allbits[1]) << " " << std::bitset<32>(allbits[0]) << endl;
}


namespace cms {

  /** \class HcalRecHitDump
      
  \author J. Mans - Minnesota
  Heavily modified by Halil Gamsizkan (Anadolu U.)
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
    std::vector<int> bits_;
  };

  HcalRecHitDump::HcalRecHitDump(edm::ParameterSet const& conf) :
    getHcalSourcePositionData_(edm::ProcessMatch("*"), this),
    getHBHERecHitCollection_(edm::ProcessMatch("*"), this),
    getHFRecHitCollection_(edm::ProcessMatch("*"), this),
    hbhePrefix_(conf.getUntrackedParameter<string>("hbhePrefix", "")),
    hoPrefix_(conf.getUntrackedParameter<string>("hoPrefix", "")),
    hfPrefix_(conf.getUntrackedParameter<string>("hfPrefix", "")),
    bits_ ( conf.getUntrackedParameter<std::vector<int>>( "bits" )) 
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
    uint32_t allbits[4];
    const string marker(prefix ? prefix : "");
    typename std::vector<edm::Handle<COLL> >::const_iterator i;
    cout << "New event (" << handles.size() << " handles)" << endl;
    bool dbit;
    int ibit;
    for (i=handles.begin(); i!=handles.end(); i++) {  // loop over rechit collections
        for (typename COLL::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++){ // loop over rechits
            cout << marker << *j << "; ";
            allbits[0]=0;  // auxPhase1
            allbits[1]=0;  //auxhbhe
            allbits[2]=j->aux();
            allbits[3]=j->flags();
         
            getauxb< typename COLL::const_iterator >(j, allbits);            
            if (bits_.size()>0) cout << "bits: ";
            for (std::vector<int>::iterator it = bits_.begin() ; it != bits_.end(); ++it){
                ibit=*it % 32;
                if (*it != -1){ // print the bit
                    dbit=(allbits[*it / 32] & ( 1 << ibit )) >> ibit;
                    cout << dbit;
                } else { // print the seperator
                    cout << '-';
                };
            }
            cout << endl;
        }
    }
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

