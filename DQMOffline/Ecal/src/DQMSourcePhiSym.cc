/*
 * \file DQMSourcePhiSym.cc
 *
 * \author Andrea Gozzelino - Università e INFN Torino
 *         
 * $Date: 2008/04/28  $
 * $Revision: 1.1 $
 *
 *
 * Description: Creating and filling monitoring elements using in Phi Symmetry  
*/


// **********************************************************
// ---------------- include files ---------------------------
// **********************************************************

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

// DQM include files

#include "DQMServices/Core/interface/MonitorElement.h"

// work on collections
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMOffline/Ecal/src/DQMSourcePhiSym.h"


using namespace std;
using namespace edm;


// ******************************************
// constructors
// *****************************************

DQMSourcePhiSym::DQMSourcePhiSym( const edm::ParameterSet& ps ) :
counterEvt_(0)
{
     dbe_ = Service<DQMStore>().operator->();
     parameters_ = ps;
     monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
     //cout << "Monitor name = " << monitorName_ << endl;
     if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
     prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
     //cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;


  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"C1");

  // book some histograms 1D

  hphidistr = dbe_->book1D("histo phi distribution", "phi distribution", 360, -3.14, 3.14);
  hphidistr->setAxisTitle("phi (rad)", 1);
  hphidistr->setAxisTitle(" # uncalib rec hits", 2);

  hiphidistr = dbe_->book1D("histo iphi distribution", "iphi distribution", 360, 1., 360.);
  hiphidistr->setAxisTitle(" iphi ", 1);
  hiphidistr->setAxisTitle(" # uncalib rec hits", 2);

  hetadistr = dbe_->book1D("histo eta distribution", "eta distribution", 170, -1.48, 1.48);
  hetadistr->setAxisTitle("eta", 1);
  hetadistr->setAxisTitle(" # uncalib rec hits", 2);

  hietadistr = dbe_->book1D("histo ieta distribution", "ieta distribution", 170,-85.,85.);
  hietadistr->setAxisTitle(" ieta ", 1);
  hietadistr->setAxisTitle(" # uncalib rec hits", 2);

  hweightamplitude = dbe_->book1D("weight amplitude","weight amplitude",50,0.,50.);
  hweightamplitude->setAxisTitle("uncalib rechits amplitude (ADC) ",1);
  hweightamplitude->setAxisTitle(" ",2);

  henergyEB = dbe_->book1D("rechits energy above cut","rechits energy above cut",160,0.,1.6);
  henergyEB->setAxisTitle("rechits energy (GeV) ",1);
  henergyEB->setAxisTitle("rechits above thresold ",2);

  hEventEnergy = dbe_->book1D("event energy","event energy",100,0.,1.4);
  hEventEnergy->setAxisTitle("event energy (GeV) ",1);
  hEventEnergy->setAxisTitle(" ",2);

  hEventRh = dbe_->book1D("rechits in event","rechits in event",100,0.,250.);
  hEventRh->setAxisTitle("rechits ",1);
  hEventRh->setAxisTitle(" ",2);

  hEventSumE = dbe_->book1D("energy sum in event","energy sum in event",50,0.,150.);
  hEventSumE->setAxisTitle("energy sum (GeV) ",1);
  hEventSumE->setAxisTitle("entries ",2);

}


// ******************************************
// destructor
// ******************************************

DQMSourcePhiSym::~DQMSourcePhiSym()
{
   
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void DQMSourcePhiSym::beginJob(const EventSetup& context){

}

//--------------------------------------------------------
void DQMSourcePhiSym::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void DQMSourcePhiSym::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  
}


// ****************************************************************
// ----------- implementation method analyze -----------------------
// ****************************************************************

void DQMSourcePhiSym::analyze(const Event& iEvent, 
			       const EventSetup& iSetup )
{  
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  // cout << " processing conterEvt_: " << counterEvt_ <<endl;

   if(counterEvt_%50 == 0)
    cout << " # of events = " << counterEvt_ << endl;


   // --------------- Geometry ---------------------------------------
   // ******* include new libraries in Build file ********************
  
   edm::ESHandle<CaloGeometry> cGeom;
   iSetup.get<IdealGeometryRecord>().get(cGeom);
   geo = cGeom.product();
  
   // ------------- weight uncalib rechit ----------------

   edm::Handle<EcalUncalibratedRecHitCollection> UWrh;
   iEvent.getByLabel("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB", UWrh);

   EcalUncalibratedRecHitCollection::const_iterator UWhit;

   double AcutEB = 5.0;//amplitude cut in EB = 5 ADC

   for(UWhit = UWrh->begin(); UWhit != UWrh->end(); UWhit ++)
     {
       if (UWhit->amplitude() > AcutEB)
	 {
	   //cout << "weight uncalib amplitude above cut" << UWhit->amplitude()<<endl;
	   hweightamplitude-> Fill(UWhit->amplitude());

	   //cout << "    phi  " << (geo->getPosition(UWhit->id())).phi() << endl;
	   hphidistr -> Fill((geo->getPosition(UWhit->id())).phi());

	   hetadistr -> Fill((geo->getPosition(UWhit->id())).eta());

	   //convert in  EBDetId
	   //cout << "  iphi (integer)  " << EBDetId(UWhit->id()).iphi() << endl; 
	   hiphidistr -> Fill(EBDetId(UWhit->id()).iphi());

	   hietadistr -> Fill(EBDetId(UWhit->id()).ieta());

	 }
     } // end for on uncalib rechit


 //---------------------- work on rechit-----------------------

   edm::Handle<EcalRecHitCollection> rhEB;
   iEvent.getByLabel("ecalRecHit","EcalRecHitsEB", rhEB);

   EcalRecHitCollection::const_iterator rechitEB;

   double EcutEB = 0.19; // energy cut in EB = 190 MeV

   int rh_counter = 0;
   double sum_energy = 0.;
   double ratio = 0.;
   vector <double> energy_event;

   for(rechitEB = rhEB->begin(); rechitEB != rhEB->end(); rechitEB ++)
     {
       if (rechitEB->energy() > EcutEB) 
	 {
	   rh_counter = rh_counter + 1;
	   sum_energy = sum_energy + (rechitEB->energy());
 
	   // cout << " rechit energy in EB " << rechitEB->energy() << endl;
	   // cout << " above fixed thresold Ecut = " << EcutEB << endl;
	   henergyEB -> Fill(rechitEB->energy());
	 }

     }// end for on rechits

   ratio = sum_energy/rh_counter;

   // few report lines
   /*
   cout << " ************************************************** " << endl;
   cout << " event number is  " << counterEvt_ << endl;
   cout << "rechits energy sum in event =  "  << sum_energy << endl;
   cout << "rechits above thresold number in event  =  " << rh_counter << endl;
   cout << "ratio =  "<< ratio << endl << endl;
   cout << " ************************************************** " << endl;
   */

   hEventEnergy -> Fill (ratio);
   hEventRh -> Fill(rh_counter);
   hEventSumE -> Fill(sum_energy);

  usleep(100);

}




//--------------------------------------------------------
void DQMSourcePhiSym::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
}
//--------------------------------------------------------
void DQMSourcePhiSym::endRun(const Run& r, const EventSetup& context){

   dbe_->setCurrentFolder(monitorName_+"C1");
  // dbe_->clone1D("cloneh1",rooth1);

}
//--------------------------------------------------------
void DQMSourcePhiSym::endJob()
{
  
}


