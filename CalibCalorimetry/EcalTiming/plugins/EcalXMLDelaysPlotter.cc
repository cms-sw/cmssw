/**\class EcalXMLDelaysPlotter

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  J. Haupt
//
// 
#include "CalibCalorimetry/EcalTiming/interface/EcalXMLDelaysPlotter.h"
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
EcalXMLDelaysPlotter::EcalXMLDelaysPlotter( const edm::ParameterSet& iConfig )
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
EcalXMLDelaysPlotter::~EcalXMLDelaysPlotter()
//========================================================================
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}
//========================================================================
void
EcalXMLDelaysPlotter::beginRun(edm::EventSetup const& eventSetup ) {
//========================================================================

}
//========================================================================


//========================================================================
void
EcalXMLDelaysPlotter::beginJob( ) {
//========================================================================
  //char profName[150];char profTit[150];
 
  ievt_ = 0; //Just a simple stupid event counter
  //Now for the 3D timing plots.
  double ttEtaBins[36] = {-85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86 };
  double ttPhiBins[73];
  //double timingBins[126];
  // double timingtBins[501];
  // double ttEtaEEBins[21];
  for (int i = 0; i < 501; ++i)
    {
      // timingtBins[i]=2.5+double(i)*5./500.;
	  
      // if ( i < 126) {timingBins[i]=2.5+double(i)*5./125.;}
      // if ( i < 21) {ttEtaEEBins[i]=0.0+double(i)*100./20.;}
      if (i<73) 
	   {
          ttPhiBins[i]=1+5*i;
	   }
    }

  EBXMLProfile_ =new TProfile2D("timeTTAllFEDs","(Phi,Eta,time) for all FEDs (SM,TT binning);i#phi;i#eta;Relative Time (1 clock = 25ns)",72,ttPhiBins,35,ttEtaBins);
  EEPXMLProfile_ =new TProfile2D("EEPtimeTTAllFEDs","(ix,iy,time) for all FEDs EE+ (SM,tt binning);ix;iy;Relative Time (1 clock = 25ns)",100/5,1.,101.,100/5,1.0,101.);
  EEMXMLProfile_ =new TProfile2D("EEMtimeTTAllFEDs","(ix,iy,time) for all FEDs EE- (SM,tt binning);ix;iy;Relative Time (1 clock = 25ns)",100/5,1.,101.,100/5,1.0,101.);
  EEPXMLProfileCh_ =new TProfile2D("EEPtimeCHAllFEDs","(ix,iy,time) for all FEDs EE+ (SM,ch binning);ix;iy;Relative Time (1 clock = 25ns)",100,1.,101.,100,1.0,101.);
  EEMXMLProfileCh_ =new TProfile2D("EEMtimeCHAllFEDs","(ix,iy,time) for all FEDs EE- (SM,ch binning);ix;iy;Relative Time (1 clock = 25ns)",100,1.,101.,100,1.0,101.);

}

//========================================================================
void EcalXMLDelaysPlotter::endJob() {
//========================================================================
   

  TFile f(rootfile_.c_str(),"RECREATE");
  EBXMLProfile_->Write();
  EEPXMLProfile_->Write();
  EEMXMLProfile_->Write();
  EEPXMLProfileCh_->Write();
  EEMXMLProfileCh_->Write();
  f.Close();

}

//
// member functions
//

//========================================================================
void
EcalXMLDelaysPlotter::analyze(  edm::Event const& iEvent,  edm::EventSetup const& iSetup ) {
//========================================================================

   using namespace edm;
   using namespace cms;
   ievt_++;
   if (ievt_ > 1 ) return;
   edm::ESHandle< EcalElectronicsMapping > handle;
   iSetup.get< EcalMappingRcd >().get(handle);
   ecalElectronicsMap_ = handle.product();
   //Geometry information
   edm::ESHandle<CaloGeometry> geoHandle;
   iSetup.get<CaloGeometryRecord>().get(geoHandle);
   
   // const CaloSubdetectorGeometry *geometry_pEB = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
   //const CaloSubdetectorGeometry *geometry_pEE = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  // std::cout << " ok 0.010 " << std::endl;
   for (int idcc = 0; idcc < 54; ++idcc)
   {
     for (int itt = 0; itt < 68; ++itt)
     {
       ETT_[idcc][itt]=-100;
       XMLTTVals_[idcc][itt]=-100;
     }
   }
   readinXMLs();


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
     EBXMLProfile_->Fill(iphi,ieta,XMLTTVals_[DCCid-1][TT-1]);
     }
   }

   // int iz = -1;
   for (int idcc = 1; idcc < 55; ++idcc)
   {
     if ( idcc == 10 ) {idcc = 46; /*iz=1;*/} //Just because I only care about EE
     for (int iTT = 1; iTT < 44; ++iTT)
     {
    // EEDetId detId = EEDetId(ix,iy,iz,EEDetId::XYMODE); //This gets the detId
     //EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);
     std::cout << " idd " << idcc << " iTT " << iTT << std::endl;
     for (int istp = 1; istp < 6; ++istp)
       {
       for (int ich = 1; ich < 6; ++ich)
         {
         EcalElectronicsId elecId =  EcalElectronicsId(idcc, iTT, istp, ich);
         EEDetId detId = ecalElectronicsMap_->getDetId(elecId);
         std::cout << " idd " << idcc << " iTT " << iTT << " detId " << detId << std::endl;
           //if (!detId.isValid()) continue;
         int ix = detId.ix();
         int iy = detId.iy();
         if ( ix == 0 || iy == 0 ) continue;
         if ( idcc > 45) {EEPXMLProfile_->Fill(ix,iy,XMLTTVals_[idcc-1][iTT-1]);EEPXMLProfileCh_->Fill(ix,iy,XMLTTVals_[idcc-1][iTT-1]);}
         else { EEMXMLProfile_->Fill(ix,iy,XMLTTVals_[idcc-1][iTT-1]); EEMXMLProfileCh_->Fill(ix,iy,XMLTTVals_[idcc-1][iTT-1]);}
         }
       }
     }
   }
   
}

void EcalXMLDelaysPlotter::readinXMLs(void)
{

  for ( int fed = 601; fed < 655 ; ++fed)
  {
  ifstream FileXML(Form("sm_%d.xml",fed));
  if( !(FileXML.is_open()) ){std::cout<<"Error: file"<< Form("sm_%d.xml",fed) <<" not found!!"<<std::endl; return; }
  char Buffer[5000];
  //int TimeOffset[71];
  //for(int i=0;i<71;i++){TimeOffset[i]=-100;}
  int SMn =0;
  bool find_SMnum = true;
  std::vector<int> ttVector;
   while( !(FileXML.eof()) ){
    FileXML.getline(Buffer,5000);
    //    if (!strstr(Buffer,"#") && !(strspn(Buffer," ") == strlen(Buffer)))
    std::string initial(Buffer);
    if( find_SMnum && initial.find("<SUPERMODULE>") != std::string::npos){
      char stSM[100];
      sscanf(Buffer,"%s",stSM);
      sscanf(stSM,"<SUPERMODULE>%d</SUPERMODULE>",&SMn);
      find_SMnum = false;
    }

    if( initial.find("<DELAY_OFFSET>") != std::string::npos ){
      FileXML.getline(Buffer,5000);// get the line with SM id
      FileXML.getline(Buffer,5000);// get the line with TT id
      char st1[200];
      int TT = -1;
      sscanf(Buffer,"%s",st1);
      sscanf(st1,"<TRIGGERTOWER>%d</TRIGGERTOWER>",&TT);
      //std::cout<<"TT: "<<TT<<std::endl;
      //std::cout<<"Buffer: "<<Buffer<<"  []TT: "<<TT<<std::endl;
      if(TT< 1 || TT >68){std::cout<<"Ignoring TT "<<TT<<std::endl;}
      else{
	ttVector.push_back(TT); 
	int time_off = -10;
	char st2[200];
	FileXML.getline(Buffer,5000);// line for the delay
	sscanf(Buffer,"%s",st2);
	sscanf(st2,"<TIME_OFFSET>%d</TIME_OFFSET>",&time_off);
	XMLTTVals_[fed-601][TT-1]=time_off;
	//else{std::cout<<"Error for delays in TT: "<<TT<<" Offsets: "<<TimeOffset[TT] <<std::endl;}
      }

    }//end of detecting offset of a TT

  }//end of file
  FileXML.close();

     
  }



}

double EcalXMLDelaysPlotter::timecorr(const CaloSubdetectorGeometry *geometry_p, DetId id)
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
   
     // int ieta = (EBDetId(id)).ieta() ;
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


