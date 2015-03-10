/**\class EcalTimingCorrection

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  J. Haupt
//
// 
#include "CalibCalorimetry/EcalTiming/interface/EcalTimingCorrection.h"
#include "CalibCalorimetry/EcalTiming/interface/EcalTimingAnalysis.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalPnDiodeDigi.h"
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


#include <fstream>
#include <iomanip>
#include <iostream>
#include "TFile.h"
#include <string>
#include <vector>

#include "TProfile2D.h"
#include "TProfile.h"


//
// constants, enums and
//
// static data member definitions
//

//
// constructors and destructor
//


//========================================================================
EcalTimingCorrection::EcalTimingCorrection( const edm::ParameterSet& iConfig )
//========================================================================
{
   //now do what ever initialization is needed
   rootfile_           = iConfig.getUntrackedParameter<std::string>("rootfile","LaserTiming.root");
   txtFileName_        = iConfig.getUntrackedParameter<std::string>("TTPeakTime","TTPeakPositionFile.txt");
   txtFileForChGroups_ = iConfig.getUntrackedParameter<std::string>("ChPeakTime","ChPeakTime.txt");
   EBradius_           = iConfig.getUntrackedParameter<double>("EBRadius",1.4);
   corrtimeEcal        = iConfig.getUntrackedParameter<bool>("CorrectEcalReadout",false);
   corrtimeBH          = iConfig.getUntrackedParameter<bool>("CorrectBH",false);
   bhplus_             = iConfig.getUntrackedParameter<bool>("BeamHaloPlus",true);
   allave_             = iConfig.getUntrackedParameter<double>("AllAverage",5.7);

   std::vector<double> listDefaults;
   for (int ji = 0; ji<54; ++ji)
     {
        listDefaults.push_back(0.);
     }
   sMAves_ = iConfig.getUntrackedParameter<std::vector<double> >("SMAverages", listDefaults);
   sMCorr_ = iConfig.getUntrackedParameter<std::vector<double> >("SMCorrections", listDefaults);

   writetxtfiles_      = iConfig.getUntrackedParameter<bool>("WriteTxtFiles",false);
}


//========================================================================
EcalTimingCorrection::~EcalTimingCorrection()
//========================================================================
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}
//========================================================================
void
EcalTimingCorrection::beginRun(edm::EventSetup const& eventSetup ) {
//========================================================================
  // edm::ESHandle< EcalElectronicsMapping > handle;
  // eventSetup.get< EcalMappingRcd >().get(handle);
  // ecalElectronicsMap_ = handle.product();
}
//========================================================================


//========================================================================
void
EcalTimingCorrection::beginJob( ) {
//========================================================================
  //char profName[150];char profTit[150];
 
  ievt_ = 0; //Just a simple stupid event counter

}

//========================================================================
void EcalTimingCorrection::endJob() {
//========================================================================
   

  TFile f(rootfile_.c_str(),"RECREATE");
  
  f.Close();

}

//
// member functions
//

//========================================================================
void
EcalTimingCorrection::analyze(  edm::Event const& iEvent,  edm::EventSetup const& iSetup ) {
//========================================================================

   using namespace edm;
   using namespace cms;
   //using namespace std;
   ievt_++;
   if (ievt_ > 1 ) return;

   //Geometry information
   edm::ESHandle<CaloGeometry> geoHandle;
   iSetup.get<CaloGeometryRecord>().get(geoHandle);
   
   const CaloSubdetectorGeometry *geometry_pEB = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
   const CaloSubdetectorGeometry *geometry_pEE = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

   EEDetId detIdM = EEDetId(50,98,1,EEDetId::XYMODE);
   const CaloCellGeometry *thisCell = geometry_pEE->getGeometry(detIdM);
   GlobalPoint position = thisCell->getPosition();

   double speedlight = 0.299792458; //in meters/ns
   double z = position.z()/100.;
   double x = position.x()/100.;
   double y = position.y()/100.;

   double r = pow(x*x+y*y+z*z,0.5);
   
   double diffrz = r-z;
   double difft = diffrz/speedlight;
   double difftr = r/speedlight;
   double difftz = z/speedlight;

   std::cout << " ix=50, iy=98 " << std::endl;
   std::cout << " X:Y:Z:R in m " << x << ":" << y << ":" << z << ":" << r << std::endl;
   std::cout << " R Time: " << difftr << "   Z Time: " << difftz << "  Difference: " << -difft << std::endl; 

   EEDetId NdetIdM = EEDetId(50,65,1,EEDetId::XYMODE);
   const CaloCellGeometry *NthisCell = geometry_pEE->getGeometry(NdetIdM);
   GlobalPoint Nposition = NthisCell->getPosition();

   //double speedlight = 0.299792458; //in meters/ns
   z = Nposition.z()/100.;
   x = Nposition.x()/100.;
   y = Nposition.y()/100.;

   r = pow(x*x+y*y+z*z,0.5);

   diffrz = r-z;
   difft = diffrz/speedlight;
   difftr = r/speedlight;
   difftz = z/speedlight;

   std::cout << " ix=50, iy=65 " << std::endl;
   std::cout << " X:Y:Z:R in m " << x << ":" << y << ":" << z << ":" << r << std::endl;
   std::cout << " R Time: " << difftr << "   Z Time: " << difftz << "  Difference: " << -difft << std::endl;



   return;

   //First I need to loop over the DetId's and then get the timing values for teach TT
   for (int ieta = -83, inum=0; ieta < 84; ieta += 5,++inum)
   {
     if ( ieta == 2 ) ieta++; //Just because there is no crystal at zero
     int iphi = 10; //This was just picked at random and should have no bearing on the results TOCHECK
     EBDetId detId = EBDetId(ieta,iphi,EBDetId::ETAPHIMODE); //This gets the detId
     double extrajit = timecorr(geometry_pEB,detId);   //This returns the corrected time
     EBTTVals_[inum] = extrajit;
     std::cout << " inum " << inum << " ieta " << ieta << std::endl;
   }

    for (int iz = -1, inum=0; iz < 2; iz += 2,++inum)
   {
     int ix = 50, iy = 81; //This was just picked at random and should have no bearing on the results TOCHECK
     EEDetId detId = EEDetId(ix,iy,iz,EEDetId::XYMODE); //This gets the detId
     double extrajit = timecorr(geometry_pEE,detId);   //This returns the corrected time
     EETTVals_[inum] = extrajit;
   }
  // std::cout << " ok 0.00 " << std::endl;
   for (int ii = 0; ii < 34 ; ++ii)
   {
     int TT = ii - 17;
     if ( TT >= 0 ) TT++;
     std::cout << " TT " << TT << " value is " << EBTTVals_[ii] << " ns; when ii is " << ii << std::endl;
   }
  // std::cout << " ok 0.05 " << std::endl;
   for (int iz = -1, inum=0; iz < 2; iz += 2,++inum)
   {
     std::cout << " EE " << iz << " value is " << EETTVals_[inum] << " ns" << std::endl;
   }

  // std::cout << " ok 0.010 " << std::endl;
   for (int idcc = 0; idcc < 54; ++idcc)
   {
     for (int itt = 0; itt < 68; ++itt)
     {
       ETT_[idcc][itt]=-100;
     }
   }
   //std::cout << " ok 0.015 " << std::endl;
   //New I am filling the TT values
   for (int ieta = -83, inum=0; ieta < 84; ieta += 5,++inum)
   {
     if ( ieta == 2 ) ieta++; //Just because there is no crystal at zero
     for (int iphi = 2; iphi < 360; iphi+=5)
     {
     EBDetId detId = EBDetId(ieta,iphi,EBDetId::ETAPHIMODE); //This gets the detId
     EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);
     int TT = elecId.towerId();
     int DCCid = elecId.dccId();
     ETT_[DCCid-1][TT-1] = EBTTVals_[inum];
     }
   }
   //std::cout << " ok 0.020 " << std::endl;
   //endcaps are not as easy to do
   //double eepn = 0;
   //double eemn = 0;
   //EEDetId detIde = EEDetId(50,10,-1);
   //EEDetId detIde = EEDetId(50,10,1);
   //std::cout << " ix " << detIde.ix() << " iy " << detIde.iy() << " iz " << detIde.zside() << std::endl; 
  
   //NEW EE stuff.... Jason TEST TEST TEST
   //_+_+_+_+_+_+__+_+_+_+_+_+_+_+_+_+_+_++_+_+_+_+_+__+_+_+_+_+_+_+_+_+_+_+_++_+_+_+_+_+__+_+_+_+_+_+_+_+_+_+_+_+
   for (int iy = 65; iy < 96; iy += 5)
   {
      int iz = 1;
      int ix = 50;
      EEDetId detId = EEDetId(ix,iy,iz,EEDetId::XYMODE); 
      double extrajit = timecorr(geometry_pEE,detId);
      const CaloCellGeometry *thisCell = geometry_pEE->getGeometry(detId);
      GlobalPoint position = thisCell->getPosition();
      std::cout << " ix: " << ix << " iy: " << iy << " iz: " << iz << std::endl;
      std::cout << " x: " << position.x() << " y: " << position.y() << " z: " << position.z() << " TimeCorr: " << extrajit << std::endl; 
   }
   for (int ix = 65; ix < 96; ix += 5)
   {
      int iz = 1;
      int iy = 50;
      EEDetId detId = EEDetId(ix,iy,iz,EEDetId::XYMODE); 
      double extrajit = timecorr(geometry_pEE,detId);
      const CaloCellGeometry *thisCell = geometry_pEE->getGeometry(detId);
      GlobalPoint position = thisCell->getPosition();
      std::cout << " ix: " << ix << " iy: " << iy << " iz: " << iz << std::endl;
      std::cout << " x: " << position.x() << " y: " << position.y() << " z: " << position.z() << " TimeCorr: " << extrajit << std::endl; 
   }
    //_+_+_+_+_+_+__+_+_+_+_+_+_+_+_+_+_+_++_+_+_+_+_+__+_+_+_+_+_+_+_+_+_+_+_++_+_+_+_+_+__+_+_+_+_+_+_+_+_+_+_+_+
   //------------__END OF TEST TEST TEST
 
   for(int cry=1;cry<numXtals;cry++)
   {
     //std::cout << " ok 0.021 " << std::endl;
     EEDetId detId = EEDetId::unhashIndex(cry);
     //std::cout << " ok 0.022 " << std::endl;
     //std::cout << " ix " << detId.ix() << " iy " << detId.iy() << " iz " << detId.zside() << std::endl;
     //EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(DetId(detId));
     //std::cout << " ok 0.023 " << std::endl;
     //int TT = elecId.towerId();
     //std::cout << " ok 0.024 " << std::endl;
     //int DCCid = elecId.dccId();
     //std::cout << " ok 0.0245 " << std::endl;
    // ETT_[DCCid-1][TT-1]= ((detId.positiveZ()) ? (EETTVals_[1]) : (EETTVals_[0]));
     //std::cout << " ok 0.0250 " << std::endl;
   }
   //std::cout << " ok 0.025 " << std::endl;
   //New I am writing out the txt file.
   for (int idcc = 0; idcc < 54; ++idcc)
   {
     int FED = idcc + 601;
     //Open the text file
     std::cout << " ok 0.0 " << std::endl;
     ofstream txt_file;
     txt_file.open(Form("SM_%d.txt",FED),std::ios::out);
     //std::cout << " ok 0.1 " << std::endl;
     for (int itt = 0; itt < 68; ++itt)
     {
       //std::cout << " ok 0 " << std::endl;
       if (ETT_[idcc][itt] > -100)
       {
         //std::cout << " ok 1 dcc " << idcc << " itt " << itt << std::endl;
         // ETT_[idcc][itt]=-100;
         txt_file << std::setw(8)<< std::setprecision(4) << itt+1 << "   " << std::setw(8)<<std::setprecision(0) << std::fixed << ETT_[idcc][itt]<< std::endl;
       } 
     }
     //close the text file
   }
   std::cout << " The EE+ number is " << EETTVals_[1] << " The EE- Number is " << EETTVals_[0] <<  std::endl;
}

double EcalTimingCorrection::timecorr(const CaloSubdetectorGeometry *geometry_p, DetId id)
{
   double time = 0.0;

   if (!(corrtimeEcal || corrtimeBH) ) { return time;}
    
   bool inEB = true;
   if ((id.det() == DetId::Ecal) && (id.subdetId() == EcalEndcap)) {
      inEB = false;
   }
   
   const CaloCellGeometry *thisCell = geometry_p->getGeometry(id);
   GlobalPoint position = thisCell->getPosition();
   
   double speedlight = 0.299792458; //in meters/ns
   
   
   double z = position.z()/100.;
   //Correct Ecal IP readout time assumption
   if (corrtimeEcal && inEB){
   
     //int ieta = (EBDetId(id)).ieta() ;
     //  double zz=0.0;
	  /*
      if (ieta > 65 )  zz=5.188213395;
	  else if (ieta > 45 )  zz=2.192428069;
	  else if (ieta > 25 )  zz=0.756752107;
	  else if (ieta > 1 ) zz=0.088368264;
	  else if (ieta > -26 )  zz=0.088368264;
	  else if (ieta > -45 )  zz=0.756752107;
	  else if (ieta > -65 ) zz=2.192428069;
	  else zz=5.188213395;
	  */
	  /*
	  if (ieta > 65 )  zz=5.06880196;
	  else if (ieta > 45 )  zz=2.08167184;
	  else if (ieta > 25 )  zz=0.86397025;
	  else if (ieta > 1 ) zz=0.088368264;
	  else if (ieta > -26 )  zz=0.088368264;
	  else if (ieta > -45 )  zz=0.86397025;
	  else if (ieta > -65 ) zz=2.08167184;
	  else zz=5.06880196;
          */
      double change = (pow(EBradius_*EBradius_+z*z,0.5)-EBradius_)/speedlight;
      ///double change = (pow(EBradius_*EBradius_+zz,0.5)-EBradius_)/speedlight;
	  time += change;
	  
	  //std::cout << " Woohoo... z is " << z << " ieta is " << (EBDetId(id)).ieta() << std::endl;
	  //std::cout << " Subtracting " << change << std::endl;
   }
   
   if (corrtimeEcal && !inEB){
      double x = position.x()/100.;
      double y = position.y()/100.;
	  double change = (pow(x*x+y*y+z*z,0.5)-EBradius_)/speedlight;
	  //double change = (pow(z*z,0.5)-EBradius_)/speedlight; //Assuming they are all the same length...
	  time += change; //Take this out for the time being
	  
	  //std::cout << " Woohoo... z is " << z << " ieta is " << (EBDetId(id)).ieta() << std::endl;
	  //std::cout << " Subtracting " << change << std::endl;
   }
   ///speedlight = (0.299792458*(1.0-.08));
   //Correct out the BH or Beam-shot assumption
   if (corrtimeBH){
      time += ((bhplus_) ? (z/speedlight) :  (-z/speedlight) );
	  //std::cout << " Adding " << z/speedlight << std::endl;

   }

   return (time);
}


