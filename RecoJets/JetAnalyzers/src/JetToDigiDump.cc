// JetToDigiDump.cc
// Description:  Prints out Jets, consituent CaloTowers, constituent RecHits and associated Digis (the digis for HCAL only).
//               The user can specify which level in the config file:
//               DumpLevel="Jets":    Printout of jets and their kinematic quantities.
//               DumpLevel="Towers":  Nested Printout of jets and their constituent CaloTowers
//               DumpLevel="RecHits": Nested Printout of jets, constituent CaloTowers and constituent RecHits
//               DumpLevel="Digis":   Nested Printout of jets, constituent CaloTowers, RecHits and all the HCAL digis 
//                                    associated with the RecHit channel (no links exist to go back to actual digis used).
//               Does simple sanity checks on energy sums at each level: jets=sum of towers, tower=sum of RecHits.
//               Does quick and dirty estimate of the fC/GeV factor that was applied to make the RecHit from the Digis.
//               
// Author: Robert M. Harris
// Date:  19 - October - 2006
// 
#include "RecoJets/JetAnalyzers/interface/JetToDigiDump.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
//in CaloJet: #include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
// just for the CaloTowerPtr declaration:
// in CaloTowerCollection: #include "DataFormats/CaloTowers/interface/CaloTower.h"
// in CaloTowerCollection: #include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TCanvas.h>
#include <cmath>
using namespace edm;
using namespace reco;
using namespace std;

JetToDigiDump::JetToDigiDump( const ParameterSet & cfg ) :
  DumpLevel( cfg.getParameter<string>( "DumpLevel" ) ),
  CaloJetAlg( cfg.getParameter<string>( "CaloJetAlg" ) ),
  DebugLevel( cfg.getParameter<int>( "DebugLevel" ) ),
  ShowECal( cfg.getParameter<bool>( "ShowECal" ) )
  {
}

void JetToDigiDump::beginJob( ) {
  if(DumpLevel=="Jets")
  {
    cout << "Dump of Jets" << endl;
    Dump=1;
  }
  else if(DumpLevel=="Towers")
  {
    cout << "Dump of Jets and constituent CaloTowers" << endl;    
    Dump=2;
  }
  else if(DumpLevel=="RecHits")
  {
    cout << "Dump of Jets, constituent CaloTowers, and constituent RecHits" << endl;    
    Dump=3;
  }
  else if(DumpLevel=="Digis")
  {
    cout << "Dump of Jets, constituent CaloTowers, constituent RecHits and associated Digis" << endl;    
    Dump=4;
  }
  cout << "Jet Algorithm being dumped is " << CaloJetAlg << endl;
  cout<<"Debug level is " << DebugLevel << endl;
  //Initialize some stuff
  evtCount = 0;
}

void JetToDigiDump::analyze( const Event& evt, const EventSetup& es ) {

  int jetInd;
  Handle<CaloJetCollection> caloJets;
  Handle<CaloTowerCollection> caloTowers;
  Handle<HBHERecHitCollection> HBHERecHits;
  Handle<HORecHitCollection> HORecHits;
  Handle<HFRecHitCollection> HFRecHits;
  Handle<EBRecHitCollection> EBRecHits;
  Handle<EERecHitCollection> EERecHits;
  Handle<HBHEDigiCollection> HBHEDigis;
  Handle<HODigiCollection> HODigis;
  Handle<HFDigiCollection> HFDigis;
  Handle<EEDigiCollection> EEDigis;
  Handle<EBDigiCollection> EBDigis;
  // Old:
  //Handle<edm::SortedCollection<EBDataFrame> > EBDigis;
  //Handle<edm::SortedCollection<EBDataFrame> > EEDigis;
   
  //Find the CaloTowers in leading CaloJets
  if (DebugLevel) cout<<"Getting caloJets"<<endl;

  evt.getByLabel( CaloJetAlg, caloJets );
  if (Dump >= 2) evt.getByLabel( "towerMaker", caloTowers );
  if (Dump >= 3) {
    if (DebugLevel) cout<<"Getting recHits"<<endl;
    evt.getByLabel( "hbhereco", HBHERecHits );
    evt.getByLabel( "horeco", HORecHits );
    evt.getByLabel( "hfreco", HFRecHits );
    evt.getByLabel( "ecalRecHit", "EcalRecHitsEB", EBRecHits );
    evt.getByLabel( "ecalRecHit", "EcalRecHitsEE", EERecHits );
    if (DebugLevel) cout<<"# of hits gotten - HBHE: "<<HBHERecHits->size()<<endl;
    evt.getByLabel( "hcalDigis", HBHEDigis );
    evt.getByLabel( "hcalDigis", HODigis );
    evt.getByLabel( "hcalDigis", HFDigis );
    if (DebugLevel) cout<<"# of digis gotten - HBHE: "<<HBHEDigis->size()<<endl;
    if (ShowECal) {
      evt.getByLabel( "ecalDigis", "ebDigis", EBDigis );
      evt.getByLabel( "ecalDigis", "eeDigis", EEDigis );
    }
  }
    
  cout << endl << "Evt: "<<evtCount <<", Num Jets=" <<caloJets->end() - caloJets->begin() << endl;
  if(Dump>=1)cout <<"   *********************************************************" <<endl;
  jetInd = 0;
  if(Dump>=1)for( CaloJetCollection::const_iterator jet = caloJets->begin(); jet != caloJets->end(); ++ jet ) {
    //2_1_?    std::vector <CaloTowerPtr> towers = jet->getCaloConstituents ();
    //2_0_7"
    std::vector <CaloTowerPtr> towers = jet->getCaloConstituents ();
    int nConstituents= towers.size();
    cout <<"   Jet: "<<jetInd<<", eta="<<jet->eta()<<", phi="<<jet->phi()<<", pt="<<jet->pt()<<\
    ",E="<<jet->energy()<<", EB E="<<jet->emEnergyInEB()<<" ,HB E="<<jet->hadEnergyInHB()<<\
    ", HO E="<<jet->hadEnergyInHO()<<" ,EE E="<< jet->emEnergyInEE()\
     <<", HE E="<<jet->hadEnergyInHE()<<", HF E="<<jet->hadEnergyInHF()+jet->emEnergyInHF()<<", Num Towers="<<nConstituents<<endl;
    if(Dump>=2)cout <<"      ====================================================="<<endl;
    float sumTowerE = 0.0;
    if(Dump>=2)for (int i = 0; i <nConstituents ; i++) {
       CaloTowerCollection::const_iterator theTower=caloTowers->find(towers[i]->id());  //Find the tower from its CaloTowerDetID	
       if (theTower == caloTowers->end()) {cerr<<"Bug? Can't find the tower"<<endl; return;}
       int ietaTower = towers[i]->id().ieta();
       int iphiTower = towers[i]->id().iphi();
       sumTowerE += theTower->energy();
       size_t numRecHits = theTower->constituentsSize();
       cout << "      Tower " << i <<": ieta=" << ietaTower <<  ", eta=" << theTower->eta() <<", iphi=" << iphiTower << ", phi=" << theTower->phi() << \
       ", energy=" << theTower->energy() << ", EM=" << theTower->emEnergy()<< ", HAD=" << theTower->hadEnergy()\
       << ", HO=" << theTower->outerEnergy() <<",  Num Rec Hits =" << numRecHits << endl;
       if(Dump>=3)cout << "         ------------------------------------------------"<<endl;
       float sumRecHitE = 0.0;
       if(Dump>=3)for(size_t j = 0; j <numRecHits ; j++) {
          DetId RecHitDetID=theTower->constituent(j);
          DetId::Detector DetNum=RecHitDetID.det();
          if( DetNum == DetId::Hcal ){
	    //cout << "RecHit " << j << ": Detector = " << DetNum << ": Hcal " << endl;
	    HcalDetId HcalID = RecHitDetID;
	    HcalSubdetector HcalNum = HcalID.subdet();
	    if(  HcalNum == HcalBarrel ){
              HBHERecHitCollection::const_iterator theRecHit=HBHERecHits->find(HcalID);	    
 	      sumRecHitE += theRecHit->energy();
              HBHEDigiCollection::const_iterator theDigis=HBHEDigis->find(HcalID);
	      cout << "         RecHit: " << j << ": HB, ieta=" << HcalID.ieta() << ", iphi=" << HcalID.iphi()<<      
	      ", depth=" << HcalID.depth() << ", energy=" << theRecHit->energy() << ", time=" <<\
	      theRecHit->time() <<", All Digis=" << theDigis->size() << ", presamples =" <<\
	      theDigis->presamples() <<endl;              
              /* const HcalElectronicsId HW_ID = theDigis->elecId();
                cout << "Digi: Index=" << HW_ID.linearIndex() << ", raw ID=" <<  HW_ID.rawId() << ", fiberChan=" << HW_ID.fiberChanId() <<  ", fiberInd=" <<  HW_ID.fiberIndex() \
	             << ", HTR chan=" <<  HW_ID.htrChanId() << ", HTR Slot=" <<   HW_ID.htrSlot() << ", HDR top/bot=" << HW_ID.htrTopBottom() \
	        << ", VME crate=" <<  HW_ID.readoutVMECrateId() << ", DCC=" << HW_ID.dccid() << ", DCC spigot=" <<  HW_ID.spigot() << endl;
              */
              float SumDigiCharge = 0.0;
	      float EstimatedPedestal=0.0;
	      int SamplesToAdd = 4;
              if(Dump>=4)cout << "            ......................................"<<endl;
              if(Dump>=4)for(int k=0; k<theDigis->size(); k++){
                const HcalQIESample QIE = theDigis->sample(k);
                if(k>=theDigis->presamples()&&k<theDigis->presamples()+SamplesToAdd)SumDigiCharge+=QIE.nominal_fC();
                if(k<theDigis->presamples()-1)EstimatedPedestal+=QIE.nominal_fC()*SamplesToAdd/(theDigis->presamples()-1);
		cout << "            Digi: " << k <<  ", cap ID = " << QIE.capid() << ": ADC Counts = " << QIE.adc() <<  ", nominal fC = " << QIE.nominal_fC() <<endl;
              }
              if(Dump>=4)cout << "            4 Digi fC ="<<SumDigiCharge<<", est. ped. fC="<<EstimatedPedestal<<", est. GeV/fc="<<theRecHit->energy()/(SumDigiCharge-EstimatedPedestal) << endl;
              if(Dump>=4)cout << "            ......................................"<<endl;
	    }
            else if(  HcalNum == HcalEndcap  ){
              HBHERecHitCollection::const_iterator theRecHit=HBHERecHits->find(HcalID);	    
              if( (abs(HcalID.ieta())==28||abs(HcalID.ieta())==29)&&HcalID.depth()==3){
	        sumRecHitE += theRecHit->energy()/2;  //Depth 3 split over tower 28 & 29
              }
	      else{
 	        sumRecHitE += theRecHit->energy();
              }
              HBHEDigiCollection::const_iterator theDigis=HBHEDigis->find(HcalID);
	      cout << "         RecHit: " << j << ": HE, ieta=" << HcalID.ieta() << ", iphi=" << HcalID.iphi()<<      
	      ", depth=" << HcalID.depth() << ", energy=" << theRecHit->energy() << ", time=" <<\
	      theRecHit->time() <<", All Digis=" << theDigis->size() << ", presamples =" <<\
	      theDigis->presamples() <<endl;              
              float SumDigiCharge = 0.0;
	      float EstimatedPedestal=0.0;
	      int SamplesToAdd = 4;
              if(Dump>=4)cout << "            ......................................"<<endl;
              if(Dump>=4)for(int k=0; k<theDigis->size(); k++){
                const HcalQIESample QIE = theDigis->sample(k);
                if(k>=theDigis->presamples()&&k<theDigis->presamples()+SamplesToAdd)SumDigiCharge+=QIE.nominal_fC();
                if(k<theDigis->presamples()-1)EstimatedPedestal+=QIE.nominal_fC()*SamplesToAdd/(theDigis->presamples()-1);
		cout << "            Digi: " << k <<  ", cap ID = " << QIE.capid() << ": ADC Counts = " << QIE.adc() <<  ", nominal fC = " << QIE.nominal_fC() <<endl;
              }
              if(Dump>=4)cout << "            4 Digi fC ="<<SumDigiCharge<<", est. ped. fC="<<EstimatedPedestal<<", est. GeV/fc="<<theRecHit->energy()/(SumDigiCharge-EstimatedPedestal) << endl;
              if(Dump>=4)cout << "            ......................................"<<endl;
            }	     
            else if(  HcalNum == HcalOuter  ){
              HORecHitCollection::const_iterator theRecHit=HORecHits->find(HcalID);	    
	      sumRecHitE += theRecHit->energy();
              HODigiCollection::const_iterator theDigis=HODigis->find(HcalID);
	      cout << "         RecHit: " << j << ": HO, ieta=" << HcalID.ieta() << ", iphi=" << HcalID.iphi()<<      
	      ", depth=" << HcalID.depth() << ", energy=" << theRecHit->energy() << ", time=" <<\
	      theRecHit->time() <<", All Digis=" << theDigis->size() << ", presamples =" <<\
	      theDigis->presamples() <<endl;              
              float SumDigiCharge = 0.0;
	      float EstimatedPedestal=0.0;
	      int SamplesToAdd = 4;
              if(Dump>=4)cout << "            ......................................"<<endl;
              if(Dump>=4)for(int k=0; k<theDigis->size(); k++){
                const HcalQIESample QIE = theDigis->sample(k);
                if(k>=theDigis->presamples()&&k<theDigis->presamples()+SamplesToAdd)SumDigiCharge+=QIE.nominal_fC();
                if(k<theDigis->presamples()-1)EstimatedPedestal+=QIE.nominal_fC()*SamplesToAdd/(theDigis->presamples()-1);
		cout << "            Digi: " << k <<  ", cap ID = " << QIE.capid() << ": ADC Counts = " << QIE.adc() <<  ", nominal fC = " << QIE.nominal_fC() <<endl;
              }
              if(Dump>=4)cout << "            4 Digi fC ="<<SumDigiCharge<<", est. ped. fC="<<EstimatedPedestal<<", est. GeV/fc="<<theRecHit->energy()/(SumDigiCharge-EstimatedPedestal) << endl;
              if(Dump>=4)cout << "            ......................................"<<endl;
            }	     
            else if(  HcalNum == HcalForward  ){
              HFRecHitCollection::const_iterator theRecHit=HFRecHits->find(HcalID);	    
	      sumRecHitE += theRecHit->energy();
              HFDigiCollection::const_iterator theDigis=HFDigis->find(HcalID);
	      cout << "         RecHit: " << j << ": HF, ieta=" << HcalID.ieta() << ", iphi=" << HcalID.iphi()<<      
	      ", depth=" << HcalID.depth() << ", energy=" << theRecHit->energy() << ", time=" <<\
	      theRecHit->time() <<", All Digis=" << theDigis->size() << ", presamples =" <<\
	      theDigis->presamples() <<endl;              
              float SumDigiCharge = 0.0;
	      float EstimatedPedestal=0.0;	      
              int SamplesToAdd = 1;
              if(Dump>=4)cout << "            ......................................"<<endl;
              if(Dump>=4)for(int k=0; k<theDigis->size(); k++){
                const HcalQIESample QIE = theDigis->sample(k);
                if(k>=theDigis->presamples()&&k<theDigis->presamples()+SamplesToAdd)SumDigiCharge+=QIE.nominal_fC();
                if(k<theDigis->presamples()-1)EstimatedPedestal+=QIE.nominal_fC()*SamplesToAdd/(theDigis->presamples()-1);
		cout << "            Digi: " << k <<  ", cap ID = " << QIE.capid() << ": ADC Counts = " << QIE.adc() <<  ", nominal fC = " << QIE.nominal_fC() <<endl;
              }
              if(Dump>=4)cout << "            1 Digi fC ="<<SumDigiCharge<<", est. ped. fC="<<EstimatedPedestal<<", est. GeV/fc="<<theRecHit->energy()/(SumDigiCharge-EstimatedPedestal) << endl;
              if(Dump>=4)cout << "            ......................................"<<endl;
           }	                 	      
          }
          if( ShowECal && DetNum == DetId::Ecal ){
            int EcalNum =  RecHitDetID.subdetId();
            if( EcalNum == 1 ){
	      EBDetId EcalID = RecHitDetID;
              EBRecHitCollection::const_iterator theRecHit=EBRecHits->find(EcalID);	    
              EBDigiCollection::const_iterator theDigis=EBDigis->find(EcalID);
	      sumRecHitE += theRecHit->energy();
	      cout << "         RecHit " << j << ": EB, ieta=" << EcalID.ieta() <<  ", iphi=" << EcalID.iphi() <<  ", SM=" << EcalID.ism() << ", energy=" << theRecHit->energy() <<", All Digis=" << theDigis->size()<< endl;
              if(Dump>=4)cout << "            ......................................"<<endl;
              if(Dump>=4)for(unsigned int k=0; k<theDigis->size(); k++){
		EBDataFrame frame (*theDigis);
                const EcalMGPASample MGPA = frame.sample(k);
		cout << "            Digi: " << k <<   ": ADC Sample = " << MGPA.adc() << ", Gain ID = "<< MGPA.gainId() <<endl;
	      }
              if(Dump>=4)cout << "            ......................................"<<endl;
	    }
            else if(  EcalNum == 2 ){
	      EEDetId EcalID = RecHitDetID;
              EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);	    
	      EEDigiCollection::const_iterator theDigis=EEDigis->find(EcalID);
	      sumRecHitE += theRecHit->energy();
	      cout << "         RecHit " << j << ": EE, ix=" << EcalID.ix() <<  ", iy=" << EcalID.iy() << ", energy=" << theRecHit->energy() << ", All Digis=" << theDigis->size()<< endl;
              if(Dump>=4)cout << "            ......................................"<<endl;
              if(Dump>=4)for(unsigned int k=0; k<theDigis->size(); k++){
		EEDataFrame frame (*theDigis);
                const EcalMGPASample MGPA = frame.sample(k);
		cout << "            Digi: " << k <<   ": ADC Sample = " << MGPA.adc() << ", Gain ID = "<< MGPA.gainId() <<endl;
	      }
              if(Dump>=4)cout << "            ......................................"<<endl;
	    }
          }
       }
       if(Dump>=3){
         if( abs(ietaTower)==28||abs(ietaTower)==29){
             cout << "         Splitted Sum of RecHit Energies=" << sumRecHitE <<", CaloTower energy=" << theTower->energy() <<  endl;
         }
	 else{
             cout << "         Sum of RecHit Energies=" << sumRecHitE <<", CaloTower energy=" << theTower->energy() <<  endl;
         }
       }
       if(Dump>=3)cout << "         ------------------------------------------------"<<endl;
    }
    if(Dump>=2)cout << "      Sum of tower energies=" << sumTowerE << ", CaloJet energy=" << jet->energy() <<  endl;
    jetInd++;
    if(Dump>=2)cout <<"      ====================================================="<<endl;
  }
  evtCount++;    
  if(Dump>=1)cout <<"   *********************************************************" <<endl;

}

void JetToDigiDump::endJob() {


}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetToDigiDump);
