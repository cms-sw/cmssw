/* AS */

// system include files
#include <ap_int.h>
#include <array>
#include <cmath>
#include <typeinfo>
// #include <cstdint>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <TLorentzVector.h>
#ifdef __MAKECINT__
#pragma link C++ class vector<TLorentzVector>+;
#endif

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloToCorrelatorTMI18.h"

//#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterCorrelatorTMI18.h"
//#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedPFClusterCorrelatorTMI18.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedCaloToCorrelatorTMI18.h"


class Phase2L1CaloToCorrelatorTMI18 : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1CaloToCorrelatorTMI18(const edm::ParameterSet&);
  ~Phase2L1CaloToCorrelatorTMI18() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<l1tp2::GCTEmDigiClusterCollection> gctEmDigiClustersSrc_;
  edm::EDGetTokenT<l1tp2::GCTHadDigiClusterCollection> gctHadDigiClustersSrc_;
};


Phase2L1CaloToCorrelatorTMI18::Phase2L1CaloToCorrelatorTMI18( const edm::ParameterSet & cfg ) :
  gctEmDigiClustersSrc_(consumes<l1tp2::GCTEmDigiClusterCollection>(cfg.getParameter<edm::InputTag>("gctEmDigiClusters"))),
  gctHadDigiClustersSrc_(consumes<l1tp2::GCTHadDigiClusterCollection>(cfg.getParameter<edm::InputTag>("gctHadDigiClusters")))
{
  produces<l1tp2::DigitizedCaloToCorrelatorCollectionTMI18>("DigitizedCaloToCorrelatorTMI18");
}

void Phase2L1CaloToCorrelatorTMI18::produce( edm::Event& evt, const edm::EventSetup& es )
 {
  using namespace edm;
  std::unique_ptr<l1tp2::DigitizedCaloToCorrelatorCollectionTMI18> caloCandsTMI18(std::make_unique<l1tp2::DigitizedCaloToCorrelatorCollectionTMI18>());

  int cntr03pos = 0 ;
  int cntr03neg = 0 ;
  int cntr01pos = 0 ;
  int cntr01neg = 0 ;

  int cntr13pos = 0 ;
  int cntr13neg = 0 ;
  int cntr11pos = 0 ;
  int cntr11neg = 0 ;

  int cntr23pos = 0 ;
  int cntr23neg = 0 ;
  int cntr21pos = 0 ;
  int cntr21neg = 0 ;

  ap_uint<64> mydata = 0 ;
  ap_uint<64> dataToCL1Card0[162] = {0} ;
  ap_uint<64> dataToCL1Card1[162] = {0} ;
  ap_uint<64> dataToCL1Card2[162] = {0} ;
  l1tp2::GCTDigiClusterLink clusterCollCard0(162);
  l1tp2::GCTDigiClusterLink clusterCollCard1(162);
  l1tp2::GCTDigiClusterLink clusterCollCard2(162);

  // SLR1 and SLR3 both send 24 PFclusters each from +ve and -ve eta, total 48 words: 4x12(x64b)

  edm::Handle<l1tp2::GCTHadDigiClusterCollection> gctHadDigiClusters;
  if(evt.getByToken(gctHadDigiClustersSrc_, gctHadDigiClusters)) {
    for (int iLink = 0; iLink < 12; iLink++) {
      // in order:
      // GCT1 SLR3 +ve, GCT1 SLR3 -ve, GCT1 SLR1 +ve, GCT1 SLR1 -ve
      // GCT2 SLR3 +ve, GCT2 SLR3 -ve, GCT2 SLR1 +ve, GCT2 SLR1 -ve
      // GCT3 SLR3 +ve, GCT3 SLR3 -ve, GCT3 SLR1 +ve, GCT3 SLR1 -ve
      int iGCT = (iLink/4);
      // 4 RCT regions per GCT
      int iRCT = iLink - iGCT*4;
      // Eta: positive or negative depends on the link. If iLink is even, it is in positive eta
      bool isNegativeEta = (iRCT % 2 == 1);
      // SLR alternates every two links
      bool isSLR3 = ((iLink % 4) < 2);
      bool isSLR1 = !isSLR3;
      for (const auto & cluster : (*gctHadDigiClusters).at(iLink)) {
	mydata = cluster.data() ;
	if (isSLR1 && isNegativeEta) goto slr1negp;
	if (isSLR1 && !isNegativeEta) goto slr1posp;
	if (isSLR3 && isNegativeEta) goto slr3negp;
	if (isSLR3 && !isNegativeEta) goto slr3posp;

slr3posp:  ;

        if(iGCT == 0 && cntr03pos < 24){
          dataToCL1Card0[17+cntr03pos] = mydata;
	  clusterCollCard0[17+cntr03pos] = cluster;
          cntr03pos++;  
          goto fillendp;
        } 
        if(iGCT == 1 && cntr13pos < 24){
          dataToCL1Card1[17+cntr13pos] = mydata;
	  clusterCollCard1[17+cntr13pos] = cluster;
          cntr13pos++;
          goto fillendp;
        } 
        if(iGCT == 2 && cntr23pos < 24){
          dataToCL1Card2[17+cntr23pos] = mydata;
	  clusterCollCard2[17+cntr23pos] = cluster;
          cntr23pos++;
          goto fillendp;
        } 
        goto fillendp;

slr3negp:  ;

        if(iGCT == 0 && cntr03neg < 24){
          dataToCL1Card0[57+cntr03neg] = mydata;
	  clusterCollCard0[57+cntr03neg] = cluster;
          cntr03neg++;
          goto fillendp;
        } 
        if(iGCT == 1 && cntr13neg < 24){
          dataToCL1Card1[57+cntr13neg] = mydata;
	  clusterCollCard1[57+cntr13neg] = cluster;
          cntr13neg++;
          goto fillendp;
        } 
        if(iGCT == 2 && cntr23neg < 24){
          dataToCL1Card2[57+cntr23neg] = mydata;
	  clusterCollCard2[57+cntr23neg] = cluster;
          cntr23neg++;
          goto fillendp;
        } 
        goto fillendp;

slr1posp:  ;

        if(iGCT == 0 && cntr01pos < 24){
          dataToCL1Card0[81+17+cntr01pos] = mydata;
	  clusterCollCard0[81+17+cntr01pos] = cluster;
          cntr01pos++;
          goto fillendp;
        } 
        if(iGCT == 1 && cntr11pos < 24){
          dataToCL1Card1[81+17+cntr11pos] = mydata;
	  clusterCollCard1[81+17+cntr11pos] = cluster;
          cntr11pos++;
          goto fillendp;
        } 
        if(iGCT == 2 && cntr21pos < 24){
          dataToCL1Card2[81+17+cntr21pos] = mydata;
	  clusterCollCard2[81+17+cntr21pos] = cluster;
          cntr21pos++;
          goto fillendp;
        } 
        goto fillendp;

slr1negp:  ;

        if(iGCT == 0 && cntr01neg < 24){
          dataToCL1Card0[81+57+cntr01neg] = mydata;
	  clusterCollCard0[81+57+cntr01neg] = cluster;
          cntr01neg++;
          goto fillendp;
        } 
        if(iGCT == 1 && cntr11neg < 24){
          dataToCL1Card1[81+57+cntr11neg] = mydata;
	  clusterCollCard1[81+57+cntr11neg] = cluster;
          cntr11neg++;
          goto fillendp;
        } 
        if(iGCT == 2 && cntr21neg < 24){
          dataToCL1Card2[81+57+cntr21neg] = mydata;
	  clusterCollCard2[81+57+cntr21neg] = cluster;
          cntr21neg++;
          goto fillendp; 
        } 

fillendp:  ;
      }
    }
  }

  cntr03pos = 0 ;
  cntr03neg = 0 ;
  cntr01pos = 0 ;
  cntr01neg = 0 ;

  cntr13pos = 0 ;
  cntr13neg = 0 ;
  cntr11pos = 0 ;
  cntr11neg = 0 ;

  cntr23pos = 0 ;
  cntr23neg = 0 ;
  cntr21pos = 0 ;
  cntr21neg = 0 ;

  edm::Handle<l1tp2::GCTEmDigiClusterCollection> gctEmDigiClusters;
  if(evt.getByToken(gctEmDigiClustersSrc_, gctEmDigiClusters)){
    for (int iLink = 0; iLink < 12; iLink++) {
      // in order:
      // GCT1 SLR3 +ve, GCT1 SLR3 -ve, GCT1 SLR1 +ve, GCT1 SLR1 -ve
      // GCT2 SLR3 +ve, GCT2 SLR3 -ve, GCT2 SLR1 +ve, GCT2 SLR1 -ve
      // GCT3 SLR3 +ve, GCT3 SLR3 -ve, GCT3 SLR1 +ve, GCT3 SLR1 -ve
      int iGCT = (iLink/4);
      // 4 RCT regions per GCT
      int iRCT = iLink - iGCT*4;
      // Eta: positive or negative depends on the link. If iLink is even, it is in positive eta
      bool isNegativeEta = (iRCT % 2 == 1);
      // SLR alternates every two links
      bool isSLR3 = ((iLink % 4) < 2);
      bool isSLR1 = !isSLR3;
      for (const auto & cluster : (*gctEmDigiClusters).at(iLink)) {
        mydata = cluster.data() ;
	if (isSLR1 && isNegativeEta) goto slr1neg;
	if (isSLR1 && !isNegativeEta) goto slr1pos;
	if (isSLR3 && isNegativeEta) goto slr3neg;
	if (isSLR3 && !isNegativeEta) goto slr3pos;

slr3pos:  ;

        if(iGCT == 0 && cntr03pos < 16){
          dataToCL1Card0[1+cntr03pos] = mydata;
	  clusterCollCard0[1+cntr03pos] = cluster;
          cntr03pos++;
          goto fillend;
        } 
        if(iGCT == 1 && cntr13pos < 16){
          dataToCL1Card1[1+cntr13pos] = mydata;
	  clusterCollCard1[1+cntr13pos] = cluster;
          cntr13pos++;
          goto fillend;
        } 
        if(iGCT == 2 && cntr23pos < 16){
          dataToCL1Card2[1+cntr23pos] = mydata;
	  clusterCollCard2[1+cntr23pos] = cluster;
          cntr23pos++;
          goto fillend;
        }
	goto fillend;

slr3neg:  ;

        if(iGCT == 0 && cntr03neg < 16){
          dataToCL1Card0[41+cntr03neg] = mydata;
	  clusterCollCard0[41+cntr03neg] = cluster;
          cntr03neg++;
          goto fillend;
        } 
        if(iGCT == 1 && cntr13neg < 16){
          dataToCL1Card1[41+cntr13neg] = mydata;
	  clusterCollCard1[41+cntr13neg] = cluster;
          cntr13neg++;
          goto fillend;
        } 
        if(iGCT == 2 && cntr23neg < 16){
          dataToCL1Card2[41+cntr23neg] = mydata;
	  clusterCollCard2[41+cntr23neg] = cluster;
          cntr23neg++;
          goto fillend;
        }
	goto fillend;

slr1pos:  ;

        if(iGCT == 0 && cntr01pos < 16){
          dataToCL1Card0[81+1+cntr01pos] = mydata;
	  clusterCollCard0[81+1+cntr01pos] = cluster;
          cntr01pos++;
          goto fillend;
        } 
        if(iGCT == 1 && cntr11pos < 16){
          dataToCL1Card1[81+1+cntr11pos] = mydata;
	  clusterCollCard1[81+1+cntr11pos] = cluster;
          cntr11pos++;
          goto fillend;
        } 
        if(iGCT == 2 && cntr21pos < 16){
          dataToCL1Card2[81+1+cntr21pos] = mydata;
	  clusterCollCard2[81+1+cntr21pos] = cluster;
          cntr21pos++;
          goto fillend;
        }
	goto fillend;

slr1neg:  ;

        if(iGCT == 0 && cntr01neg < 16){
          dataToCL1Card0[81+41+cntr01neg] = mydata;
	  clusterCollCard0[81+41+cntr01neg] = cluster;
          cntr01neg++;
          goto fillend;
        } 
        if(iGCT == 1 && cntr11neg < 16){
          dataToCL1Card1[81+41+cntr11neg] = mydata;
	  clusterCollCard1[81+41+cntr11neg] = cluster;
          cntr11neg++ ;
          goto fillend ;
        } 
        if(iGCT == 2 && cntr21neg < 16){
          dataToCL1Card2[81+41+cntr21neg] = mydata;
	  clusterCollCard2[81+41+cntr21neg] = cluster;
          cntr21neg++ ;
          goto fillend ;
        }

fillend:  ;
      }
    }
  }

  l1tp2::DigitizedCaloToCorrelatorTMI18 l1CaloTMI18 = l1tp2::DigitizedCaloToCorrelatorTMI18(dataToCL1Card0, dataToCL1Card1, dataToCL1Card2, clusterCollCard0, clusterCollCard1, clusterCollCard2) ;
  caloCandsTMI18->push_back(l1CaloTMI18) ;
  evt.put(std::move(caloCandsTMI18), "DigitizedCaloToCorrelatorTMI18");
 
 }

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void Phase2L1CaloToCorrelatorTMI18::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gctEmDigiClusters", edm::InputTag("l1tPhase2GCTBarrelToCorrelatorLayer1Emulator", "GCTEmDigiClusters"));
  desc.add<edm::InputTag>("gctHadDigiClusters", edm::InputTag("l1tPhase2GCTBarrelToCorrelatorLayer1Emulator", "GCTHadDigiClusters"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Phase2L1CaloToCorrelatorTMI18);
