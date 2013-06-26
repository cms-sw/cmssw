
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/EcalCalibAlgos/interface/ElectronCalibration.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

ElectronCalibration::ElectronCalibration(const edm::ParameterSet& iConfig)
{

   rootfile_ = iConfig.getParameter<std::string>("rootfile");
   recHitLabel_ = iConfig.getParameter< edm::InputTag > ("ebRecHitsLabel");
   electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");
   trackLabel_ = iConfig.getParameter< edm::InputTag > ("trackLabel");
   calibAlgo_       = iConfig.getParameter<std::string>("CALIBRATION_ALGO");
   std::cout << " The used Algorithm is  " << calibAlgo_ << std::endl;
   keventweight_ = iConfig.getParameter<int>("keventweight");
   ClusterSize_ = iConfig.getParameter<int>("Clustersize");
   ElePt_ = iConfig.getParameter<double>("ElePt");
   maxeta_ = iConfig.getParameter<int>("maxeta");
   mineta_ = iConfig.getParameter<int>("mineta");
   maxphi_ = iConfig.getParameter<int>("maxphi");
   minphi_ = iConfig.getParameter<int>("minphi");
   cut1_ = iConfig.getParameter<double>("cut1");
   cut2_ = iConfig.getParameter<double>("cut2");
   cut3_ = iConfig.getParameter<double>("cut3");
   elecclass_ = iConfig.getParameter<int>("elecclass");
   std::cout << " The electronclass is " << elecclass_ <<std::endl;
   numevent_ = iConfig.getParameter<int>("numevent");
   miscalibfile_ = iConfig.getParameter<std::string>("miscalibfile");

   cutEPCalo1_ = iConfig.getParameter<double>("cutEPCaloMin");
   cutEPCalo2_ = iConfig.getParameter<double>("cutEPCaloMax");
   cutEPin1_ = iConfig.getParameter<double>("cutEPinMin");
   cutEPin2_ = iConfig.getParameter<double>("cutEPinMax");
   cutCalo1_ = iConfig.getParameter<double>("cutCaloMin");
   cutCalo2_ = iConfig.getParameter<double>("cutCaloMax");

   cutESeed_ = iConfig.getParameter<double>("cutESeed");
}


ElectronCalibration::~ElectronCalibration()
{
  
  
}

//========================================================================
void ElectronCalibration::beginJob() {
  //========================================================================
   f = new TFile(rootfile_.c_str(),"RECREATE");

  // Book histograms 
   e9 = new TH1F("e9","E9 energy", 300, 0., 150.);
  e25 = new TH1F("e25","E25 energy", 300, 0., 150.);
  scE = new TH1F("scE","SC energy", 300, 0., 150.);
  trP = new TH1F("trP","Trk momentum", 300, 0., 150.);
  EoP = new TH1F("EoP","EoP", 600, 0., 3.);
  EoP_all = new TH1F("EoP_all","EoP_all",600, 0., 3.);

  if (elecclass_ ==0 || elecclass_ == -1) { 
    calibs = new TH1F("calib","Calibration constants", 4000, 0.5, 2.);
  }else{
    calibs = new TH1F("calib","Calibration constants", 800, 0.5, 2.);
  }

  e25OverScE = new TH1F("e25OverscE","E25 / SC energy", 400, 0., 2.);
  E25oP = new TH1F("E25oP","E25 / P", 1200, 0., 1.5);

  Map = new TH2F("Map","Nb Events in Crystal",85,1, 85,70 ,5, 75);
  e9Overe25 = new TH1F("e9Overe25","E9 / E25", 400, 0., 2.);
  Map3Dcalib = new TH2F("3Dcalib", "3Dcalib",85 ,1 ,85,70, 5, 75 );

  MapCor1 = new TH2F ("MapCor1", "Correlation E25/Pcalo versus E25/Pin",100 ,0. ,5. ,100,0.,5. );
  MapCor2 = new TH2F ("MapCor2", "Correlation E25/Pcalo versus E/P",100 ,0. ,5. ,100,0.,5. );
  MapCor3 = new TH2F ("MapCor3", "Correlation E25/Pcalo versus Pout/Pin",100 ,0. ,5. ,100,0.,5. );
  MapCor4 = new TH2F ("MapCor4", "Correlation E25/Pcalo versus E25/highestP",100 ,0. ,5. ,100,0.,5. );
  MapCor5 = new TH2F ("MapCor5", "Correlation E25/Pcalo versus Pcalo/Pout",100 ,0. ,5. ,100,0.,5. );
  MapCor6 = new TH2F ("MapCor6", "Correlation Pout/Pin versus E25/Pin",100 ,0. ,5. ,100,0.,5. );
  MapCor7 = new TH2F ("MapCor7", "Correlation Pout/Pin versus Pcalo/Pout",100 ,0. ,5. ,100,0.,5. );
  MapCor8 = new TH2F ("MapCor8", "Correlation E25/Pin versus Pcalo/Pout",100 ,0. ,5. ,100,0.,5. );
  MapCor9 = new TH2F ("MapCor9", "Correlation  E25/Pcalo versus Eseed/Pout",100 ,0. ,5. ,100,0.,5. );
  MapCor10 = new TH2F ("MapCor10", "Correlation Eseed/Pout versus Pout/Pin",100 ,0. ,5. ,100,0.,5. );
  MapCor11 = new TH2F ("MapCor11", "Correlation Eseed/Pout versus E25/Pin",100 ,0. ,5. ,100,0.,5. );
  MapCorCalib = new TH2F ("MapCorCalib", "Correlation Miscalibration versus Calibration constants", 100, 0.5,1.5, 100, 0.5, 1.5);

  PinMinPout = new TH1F("PinMinPout","(Pin - Pout)/Pin",600,-2.0,2.0);

  if(elecclass_ == 0 || elecclass_ == -1) { 
    calibinter = new TH1F("calibinter", "internal calibration constants", 2000 , 0.5,2.);
    PinOverPout= new TH1F("PinOverPout", "pinOverpout", 600,0., 3.);
    eSeedOverPout= new TH1F("eSeedOverPout", "eSeedOverpout ", 600, 0., 3.);
    MisCalibs = new TH1F("MisCalibs","Miscalibration constants",4000,0.5,2.);
    RatioCalibs = new TH1F("RatioCalibs","Ratio in Calibration Constants", 4000, 0.5, 2.0);
    DiffCalibs = new TH1F("DiffCalibs", "Difference in Calibration constants", 4000, -1.0,1.0);
  }else { 
    calibinter = new TH1F("calibinter", "internal calibration constants",400 , 0.5,2.);
    PinOverPout= new TH1F("PinOverPout", "pinOverpout", 600,0., 3.);
    eSeedOverPout= new TH1F("eSeedOverPout", "eSeedOverpout ", 600, 0., 3.);
    MisCalibs = new TH1F("MisCalibs","Miscalibration constants",800,0.5,2.);
    RatioCalibs = new TH1F("RatioCalibs","Ratio in Calibration Constants", 800, 0.5, 2.0);
    DiffCalibs = new TH1F("DiffCalibs", "Difference in Calibration constants", 800, -1.0,1.0);
  }
  Error1 = new TH1F ("Error1","DeltaP/Pin",800 ,-1.0,1.0 );
  Error2 = new TH1F ("Error2","DeltaP/Pout",800 ,-1.0,1.0 );
  Error3 = new TH1F ("Error3","DeltaP/Pcalo",800 ,-1.0,1.0 );
  eSeedOverPout2= new TH1F("eSeedOverPout2", "eSeedOverpout (No Supercluster)", 600, 0., 4.);
  hadOverEm= new TH1F("hadOverEm", "Had/EM distribution", 600, -2., 2.);
  
  // Book histograms  
  Map3DcalibNoCuts = new TH2F("3DcalibNoCuts", "3Dcalib (Before Cuts)",85 ,1 ,85,70, 5, 75 );
  e9NoCuts = new TH1F("e9NoCuts","E9 energy (Before Cuts)",300, 0., 150.);
  e25NoCuts = new TH1F("e25NoCuts","E25 energy (Before Cuts)", 300, 0., 150.);
  scENoCuts = new TH1F("scENoCuts","SC energy (Before Cuts)", 300, 0., 150.);
  trPNoCuts = new TH1F("trPNoCuts","Trk momentum (Before Cuts)", 300, 0., 150.);
  EoPNoCuts = new TH1F("EoPNoCuts","EoP (Before Cuts)", 600, 0., 3.);
  if (elecclass_ ==0 || elecclass_ == -1){ 
    calibsNoCuts = new TH1F("calibNoCuts","Calibration constants (Before Cuts)", 4000, 0., 2.);
  }else{
    calibsNoCuts = new TH1F("calibNoCuts","Calibration constants (Before Cuts)", 800, 0., 2.);
  }
  e25OverScENoCuts = new TH1F("e25OverscENoCuts","E25 / SC energy (Before Cuts)", 400, 0., 2.);
  E25oPNoCuts = new TH1F("E25oPNoCuts","E25 / P (Before Cuts)", 1200, 0., 1.5);
  MapNoCuts = new TH2F("MapNoCuts","Nb Events in Crystal (Before Cuts)",85,1, 85,70 ,5, 75);
  e9Overe25NoCuts = new TH1F("e9Overe25NoCuts","E9 / E25 (Before Cuts)", 400, 0., 2.);
  PinOverPoutNoCuts = new TH1F("PinOverPoutNoCuts", "pinOverpout (Before Cuts)", 600,0., 3.);
  eSeedOverPoutNoCuts = new TH1F(" eSeedOverPoutNoCuts", "eSeedOverpout (Before Cuts) ", 600, 0., 4.);
  PinMinPoutNoCuts = new TH1F("PinMinPoutNoCuts","(Pin - Pout)/Pin (Before Cuts)",600,-2.0,2.0);

  RatioCalibsNoCuts = new TH1F("RatioCalibsNoCuts","Ratio in Calibration Constants (Before Cuts)", 4000, 0.5, 2.0);
  DiffCalibsNoCuts = new TH1F("DiffCalibsNoCuts", "Difference in Calibration constants (Before Cuts)", 4000, -1.0,1.0);
  calibinterNoCuts = new TH1F("calibinterNoCuts", "internal calibration constants", 2000 , 0.5,2.);
 
  MapCor1NoCuts = new TH2F ("MapCor1NoCuts", "Correlation E25/PatCalo versus E25/Pin (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor2NoCuts = new TH2F ("MapCor2NoCuts", "Correlation E25/PatCalo versus E/P (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor3NoCuts = new TH2F ("MapCor3NoCuts", "Correlation E25/PatCalo versus Pout/Pin (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor4NoCuts = new TH2F ("MapCor4NoCuts", "Correlation E25/PatCalo versus E25/highestP (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor5NoCuts = new TH2F ("MapCor5NoCuts", "Correlation E25/Pcalo versus Pcalo/Pout (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor6NoCuts = new TH2F ("MapCor6NoCuts", "Correlation Pout/Pin versus E25/Pin (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor7NoCuts = new TH2F ("MapCor7NoCuts", "Correlation Pout/Pin versus Pcalo/Pout (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor8NoCuts = new TH2F ("MapCor8NoCuts", "Correlation E25/Pin versus Pcalo/Pout (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor9NoCuts = new TH2F ("MapCor9NoCuts", "Correlation  E25/Pcalo versus Eseed/Pout (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor10NoCuts = new TH2F ("MapCor10NoCuts", "Correlation Eseed/Pout versus Pout/Pin (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCor11NoCuts = new TH2F ("MapCor11NoCuts", "Correlation Eseed/Pout versus E25/Pin (Before Cuts)",100 ,0. ,5. ,100,0.,5. );
  MapCorCalibNoCuts = new TH2F ("MapCorCalibNoCuts", "Correlation Miscalibration versus Calibration constants (Before Cuts)", 100, 0., 3., 100, 0., 3.);

  Error1NoCuts = new TH1F ("Eror1NoCuts","DeltaP/Pin (Before Cuts)",800 ,-1.0,1.0 );
  Error2NoCuts = new TH1F ("Error2NoCuts","DeltaP/Pout (Before Cuts)",800 ,-1.0,1.0 );
  Error3NoCuts = new TH1F ("Error3NoCuts","DeltaP/Pcalo (Before Cuts)",800 ,-1.0, 1.0);
  eSeedOverPout2NoCuts= new TH1F("eSeedOverPout2NoCuts", "eSeedOverpout (No Supercluster, Before Cuts)", 600, 0., 4.);
  hadOverEmNoCuts= new TH1F("hadOverEmNoCuts", "Had/EM distribution (Before Cuts)", 600, -2., 2.);

  //Book histograms after ESeed cut
  MapCor1ESeed = new TH2F ("MapCor1ESeed", "Correlation E25/Pcalo versus E25/Pin (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor2ESeed = new TH2F ("MapCor2ESeed", "Correlation E25/Pcalo versus E/P (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor3ESeed = new TH2F ("MapCor3ESeed", "Correlation E25/Pcalo versus Pout/Pin (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor4ESeed = new TH2F ("MapCor4ESeed", "Correlation E25/Pcalo versus E25/highestP (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor5ESeed = new TH2F ("MapCor5ESeed", "Correlation E25/Pcalo versus Pcalo/Pout (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor6ESeed = new TH2F ("MapCor6ESeed", "Correlation Pout/Pin versus E25/Pin (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor7ESeed = new TH2F ("MapCor7ESeed", "Correlation Pout/Pin versus Pcalo/Pout (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor8ESeed = new TH2F ("MapCor8ESeed", "Correlation E25/Pin versus Pcalo/Pout (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor9ESeed = new TH2F ("MapCor9ESeed", "Correlation  E25/Pcalo versus Eseed/Pout (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor10ESeed = new TH2F ("MapCor10ESeed", "Correlation Eseed/Pout versus Pout/Pin (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
  MapCor11ESeed = new TH2F ("MapCor11ESeed", "Correlation Eseed/Pout versus E25/Pin (after Eseed/Pout cut)",100 ,0. ,5. ,100,0.,5. );
 
  eSeedOverPout2ESeed= new TH1F("eSeedOverPout2ESeed", "eSeedOverpout (No Supercluster, after Eseed/Pout cut)", 600, 0., 4.);

  hadOverEmESeed= new TH1F("hadOverEmESeed", "Had/EM distribution (after Eseed/Pout cut)", 600, -2., 2.);
 
 //Book histograms without any cut
  GeneralMap = new TH2F("GeneralMap","Map without any cuts",85,1,85,70,5,75);

  calibClusterSize=ClusterSize_; 
  etaMin = mineta_;
  etaMax = maxeta_;
  phiMin = minphi_;
  phiMax = maxphi_;
  if(calibAlgo_=="L3") {
    MyL3Algo1 = new MinL3Algorithm(keventweight_,calibClusterSize, etaMin, etaMax, phiMin, phiMax);
  }else{ 
    if(calibAlgo_=="HH" || calibAlgo_=="HHReg"){
      MyHH = new HouseholderDecomposition(calibClusterSize, etaMin,etaMax, phiMin, phiMax); 
    }else{ 
      std::cout<<" Name of Algorithm is not recognize "<<calibAlgo_<<" Should be either L3, HH or HHReg. Abort! "<<std::endl;
    }
  }
  read_events=0;
  
  // get Region to be calibrated  
  ReducedMap = calibCluster.getMap(etaMin, etaMax, phiMin, phiMax);
  
  oldCalibs.resize(ReducedMap.size(),0.);

   // table is set to zero
  for (int phi=0; phi<360; phi++){for (int eta=0; eta<171; eta++){eventcrystal[eta][phi]=0;}}
 

  std::cout<<" Begin JOB "<<std::endl;
}


//========================================================================

void
ElectronCalibration::endJob() {
//========================================================================

int nIterations =10;
if(calibAlgo_=="L3"){ 
  solution = MyL3Algo1->iterate(EventMatrix, MaxCCeta, MaxCCphi, EnergyVector, nIterations);
 }else{
  if(calibAlgo_=="HH"){
    solution = MyHH->iterate(EventMatrix, MaxCCeta, MaxCCphi,EnergyVector, 1,false);
  }else{
    if(calibAlgo_=="HHReg"){solution = MyHH->runRegional(EventMatrix, MaxCCeta, MaxCCphi,EnergyVector, 2);
    }else{ 
      std::cout<<" Calibration not run due to problem in Algo Choice..."<<std::endl;return;
    }
  }
 }
 for (int ii=0;ii<(int)solution.size();ii++)
   {
     std::cout << "solution[" << ii << "] = " << solution[ii] << std::endl;
     calibs->Fill(solution[ii]); 
   }
 
 newCalibs.resize(ReducedMap.size(),0.);
 
 calibXMLwriter write_calibrations;
 
 FILE* MisCalib;
 MisCalib = fopen(miscalibfile_.c_str(),"r");
 int fileStatus=1;
 int eta=-1;
 int phi=-1;
 float coeff=-1;
 
 std::map<EBDetId,float> OldCoeff;
 
 while(fileStatus != EOF) {
   fileStatus = fscanf(MisCalib,"%d %d %f\n",  &eta,&phi,&coeff);
   if(eta!=-1&&phi!=-1&& coeff!=-1){
     //      std::cout<<" We have read correctly the coefficient " << coeff << " corresponding to eta "<<eta<<" and  phi "<<phi<<std::endl;
     OldCoeff.insert(std::make_pair(EBDetId(eta,phi,EBDetId::ETAPHIMODE),coeff )); 
   }
 } 
 
 fclose(MisCalib);
 
 int icry=0;
 CalibrationCluster::CalibMap::iterator itmap;
 for (itmap=ReducedMap.begin(); itmap != ReducedMap.end();itmap++){
   
   newCalibs[icry] = solution[icry];
   
   write_calibrations.writeLine(itmap->first,newCalibs[icry]);
   float Compare =1.;   
   std::map<EBDetId,float>::iterator iter = OldCoeff.find(itmap->first);
   if( iter != OldCoeff.end() )Compare = iter->second;

   if((itmap->first).ieta()>mineta_ && (itmap->first).ieta()<maxeta_ && (itmap->first).iphi()>minphi_ && (itmap->first).iphi()<maxphi_){
     Map3Dcalib->Fill((itmap->first).ieta(),(itmap->first).iphi(),newCalibs[icry]*Compare ) ;
     MisCalibs->Fill(Compare);

}
   if((itmap->first).ieta()< mineta_+2){icry++; continue;}
   if((itmap->first).ieta()> maxeta_-2){icry++; continue;}
   if((itmap->first).iphi()< minphi_+2){icry++; continue;} 
   if((itmap->first).iphi()> maxphi_-2){icry++; continue;}

   calibinter->Fill(newCalibs[icry]);
   DiffCalibs->Fill(newCalibs[icry]-1./Compare);
   RatioCalibs->Fill(newCalibs[icry]*Compare);
   MapCorCalib->Fill(1./Compare, newCalibs[icry]);
   icry++;
 }
 
 if(calibAlgo_=="L3"){
   solutionNoCuts = MyL3Algo1->iterate(EventMatrixNoCuts, MaxCCetaNoCuts, MaxCCphiNoCuts,EnergyVectorNoCuts,nIterations);
 }else{
   if(calibAlgo_=="HH"){
     solutionNoCuts = MyHH->iterate(EventMatrixNoCuts, MaxCCetaNoCuts, MaxCCphiNoCuts, EnergyVectorNoCuts, 1,false);
   }else{ 
     if(calibAlgo_=="HHReg"){
       solutionNoCuts = MyHH->runRegional(EventMatrixNoCuts, MaxCCetaNoCuts, MaxCCphiNoCuts,EnergyVectorNoCuts, 2);
     }else{
       std::cout<<" Calibration not run due to problem in AlgoChoice..."<<std::endl;return;
     }
   }
 }
 for (int ii=0;ii<(int)solutionNoCuts.size();ii++){
   calibsNoCuts->Fill(solutionNoCuts[ii]); 
 }
 int icryp=0;
 CalibrationCluster::CalibMap::iterator itmapp;
 for (itmapp=ReducedMap.begin(); itmapp != ReducedMap.end();itmapp++){
   
   newCalibs[icryp] = solutionNoCuts[icryp];
   float Compare2 =1.;   
   std::map<EBDetId,float>::iterator iter2 = OldCoeff.find(itmapp->first);
   if( iter2 != OldCoeff.end() )Compare2 = iter2->second;
   
   if((itmapp->first).ieta()>mineta_ && (itmapp->first).ieta()<maxeta_ && (itmapp->first).iphi()>minphi_ && (itmapp->first).iphi()<maxphi_)Map3DcalibNoCuts->Fill((itmapp->first).ieta(),(itmapp->first).iphi(),newCalibs[icryp]*Compare2) ;
   if ((itmapp->first).ieta()< mineta_+2){icryp++; continue;}
   if ((itmapp->first).ieta()> maxeta_-2){icryp++; continue;}
   if ((itmapp->first).iphi()< minphi_+2){icryp++; continue;} 
   if ((itmapp->first).iphi()> maxphi_-2){icryp++; continue;}
   calibinterNoCuts->Fill(newCalibs[icryp]);
   DiffCalibsNoCuts->Fill(newCalibs[icryp]-1./(Compare2));
   RatioCalibsNoCuts->Fill(newCalibs[icryp]*Compare2);
   MapCorCalibNoCuts->Fill(1./Compare2 ,newCalibs[icryp]);
   icryp++;
 }
 
 
 
 ////////////////////////       FINAL STATISTICS           ////////////////////
 
 std::cout << " " << std::endl;
 std::cout << "************* STATISTICS **************" << std::endl;
 std::cout << " Events Studied "<<read_events<< std::endl;
 
 /////////////////////////////////////////////////////////////////////////////
   
    f->Write();
   
    f->Close();
}


//=================================================================================
EBDetId ElectronCalibration::findMaxHit(edm::Handle<EBRecHitCollection> &  phits) {
//=================================================================================

     EcalRecHitCollection ecrh = *phits;
     EcalRecHitCollection::iterator it;
     int count=0;
     EBDetId save;
     float en_save=0;
     for (it = ecrh.begin(); it != ecrh.end(); it++)
     {
       EBDetId p = EBDetId(it->id().rawId());
        if(it->energy()> en_save){
	  en_save=it->energy();
	  save=p;
	}
	count++;
     }
     return save;

}

//=================================================================================
EBDetId  ElectronCalibration::findMaxHit2(const std::vector<DetId> & v1,const EBRecHitCollection* hits) {
//=================================================================================

  double currEnergy = 0.;
  EBDetId maxHit;
  
  for( std::vector<DetId>::const_iterator idsIt = v1.begin(); idsIt != v1.end(); ++idsIt) {
    if(idsIt->subdetId()!=1) continue;
    EBRecHitCollection::const_iterator itrechit;
    itrechit = hits->find(*idsIt);
	   
    if(itrechit == hits->end()){
      std::cout << "ElectronCalibration::findMaxHit2: rechit not found! " << std::endl;
      continue;
    }
    if(itrechit->energy() > currEnergy) {
      currEnergy=itrechit->energy();
      maxHit= *idsIt;
    }
  }
  
      return maxHit;
}


//=================================================================================
void ElectronCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
//=================================================================================
   using namespace edm;
   
   // Get EBRecHits
   Handle<EBRecHitCollection> phits;
   iEvent.getByLabel( recHitLabel_, phits);
   if (!phits.isValid()) {
     std::cerr << "Error! can't get the product EBRecHitCollection: " << std::endl;
   }

   const EBRecHitCollection* hits = phits.product(); // get a ptr to the product

   // Get pixelElectrons
   Handle<reco::GsfElectronCollection> pElectrons;

   iEvent.getByLabel(electronLabel_, pElectrons);
   if (!pElectrons.isValid()) {
     std::cerr << "Error! can't get the product ElectronCollection: " << std::endl;
   }

  const reco::GsfElectronCollection* electronCollection = pElectrons.product();
  read_events++;
  if(read_events%1000 ==0)std::cout << "read_events = " << read_events << std::endl;
  
  if(!hits)return;
  if(hits->size() == 0)return;
  if(!electronCollection)return;
  if(electronCollection->size() == 0)return;
  
  
  ////////////////////////////////////////////////////////////////////////////////////////
  //                          START HERE....
  ///////////////////////////////////////////////////////////////////////////////////////
  reco::GsfElectronCollection::const_iterator eleIt = electronCollection->begin();

  reco::GsfElectron highPtElectron;

  float highestElePt=0.;
  bool found=false;
  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {
    //Comments
    if(fabs(eleIt->eta())>(maxeta_+3) * 0.0175) continue;
    if(eleIt->eta()<(mineta_-3) * 0.0175) continue;

     if(eleIt->pt()>highestElePt) {
       highestElePt=eleIt->pt();
       highPtElectron = *eleIt;
       found =true;
     }

  }
  if(highestElePt<ElePt_)return;
      if(!found) return;
      const reco::SuperCluster & sc = *(highPtElectron.superCluster()) ;
      if(fabs(sc.eta())>(maxeta_+3) * 0.0175){
	std::cout<<"++++ Problem with electron, electron eta is "<< highPtElectron.eta()<<" while SC is "<<sc.eta()<<std::endl;return;
      }
//      std::cout << "track eta = " << highPtElectron.eta() << std::endl;
//      std::cout << "track phi = " << highPtElectron.phi() << std::endl;
    
      std::vector<DetId> v1;
      //Loop to fill the vector of DetIds
for (std::vector<std::pair<DetId,float> >::const_iterator idsIt = sc.hitsAndFractions().begin();
       idsIt != sc.hitsAndFractions().end ();++idsIt)
  {v1.push_back(idsIt->first);
 }

      //getHitsByDetId(); //Change function name
      EBDetId maxHitId;
      
      maxHitId = findMaxHit2(v1,hits); 
      
      if(maxHitId.null()){std::cout<<" Null "<<std::endl; return;}
      
      int maxCC_Eta = maxHitId.ieta();
      int maxCC_Phi = maxHitId.iphi();
      
      if(maxCC_Eta>maxeta_ )return;
      if(maxCC_Eta<mineta_ )return;
      if(maxCC_Phi>maxphi_ ) return;
      if(maxCC_Phi<minphi_ )  return;

      // number of events per crystal is set
      if(numevent_>0){
	eventcrystal[maxCC_Eta+85][maxCC_Phi-1]+=1;
	if (eventcrystal[maxCC_Eta+85][maxCC_Phi-1] > numevent_) return;
      }
      
      std::vector<EBDetId> Xtals5x5 = calibCluster.get5x5Id(maxHitId);
      
      if((int)Xtals5x5.size()!=ClusterSize_*ClusterSize_)return;
 
      // fill cluster energy
      std::vector<float> energy;
      float energy3x3=0.;  
      float energy5x5=0.;  
      
      for (int icry=0;icry<ClusterSize_*ClusterSize_;icry++){
	
	   EBRecHitCollection::const_iterator itrechit;
	   if(Xtals5x5[icry].subdetId()!=1) continue;
	   itrechit = hits->find(Xtals5x5[icry]);
	   if(itrechit==hits->end())
	     { std::cout << "DetId not is e25" << std::endl;
	       continue;
	     }
	   
	   if (edm::isNotFinite(itrechit->energy())) return;	  
	   energy.push_back(itrechit->energy());
	   energy5x5 += energy[icry];
	   
	   if ( icry == 6  || icry == 7  || icry == 8 ||
		icry == 11 || icry == 12 || icry ==13 ||
		icry == 16 || icry == 17 || icry ==18   )
	     {
	       energy3x3+=energy[icry];
	     }
	   
      }
      if((int)energy.size()!=ClusterSize_*ClusterSize_) return;
      //Once we have the matrix 5x5, we have to correct for gaps/cracks/umbrella and maincontainement  
      
      GeneralMap->Fill(maxCC_Eta,maxCC_Phi);
      
      EoP_all->Fill(highPtElectron.eSuperClusterOverP()); 
      
      if(highPtElectron.classification()==elecclass_ || elecclass_== -1 ){
	
	float Ptrack_in=sqrt( pow(highPtElectron.trackMomentumAtVtx().X(),2) +pow(highPtElectron.trackMomentumAtVtx().Y(),2) + pow(highPtElectron.trackMomentumAtVtx().Z(),2) );
	
	float UncorrectedPatCalo = sqrt(pow(highPtElectron.trackMomentumAtCalo().X(),2)+pow(highPtElectron.trackMomentumAtCalo().Y(),2)+pow(highPtElectron.trackMomentumAtCalo().Z(),2));
	
	float Ptrack_out = sqrt( pow(highPtElectron.trackMomentumOut().X(),2)+ pow(highPtElectron.trackMomentumOut().Y(),2)+ pow(highPtElectron.trackMomentumOut().Z(),2) );
	
	
	EventMatrixNoCuts.push_back(energy);
	EnergyVectorNoCuts.push_back(UncorrectedPatCalo);
	
	MaxCCetaNoCuts.push_back(maxCC_Eta);
	MaxCCphiNoCuts.push_back(maxCC_Phi);
	
	WeightVectorNoCuts.push_back(energy5x5/UncorrectedPatCalo);
	
	//---------------------------------------------------No Cuts-------------------------------------------------------
	e9NoCuts->Fill(energy3x3); 
	e25NoCuts->Fill(energy5x5); 
	e9Overe25NoCuts->Fill(energy3x3/energy5x5);
	scENoCuts->Fill(sc.energy()); 
	
	trPNoCuts->Fill(UncorrectedPatCalo); 
	
	EoPNoCuts->Fill(highPtElectron.eSuperClusterOverP()); 
	e25OverScENoCuts->Fill(energy5x5/sc.energy());
	
	E25oPNoCuts->Fill(energy5x5/UncorrectedPatCalo);
	
	MapNoCuts->Fill(maxCC_Eta,maxCC_Phi);
	PinOverPoutNoCuts->Fill( sqrt( pow(highPtElectron.trackMomentumAtVtx().X(),2) +pow(highPtElectron.trackMomentumAtVtx().Y(),2) + pow(highPtElectron.trackMomentumAtVtx().Z(),2) )/sqrt( pow(highPtElectron.trackMomentumOut().X(),2)+ pow(highPtElectron.trackMomentumOut().Y(),2)+ pow(highPtElectron.trackMomentumOut().Z(),2) ) );
	eSeedOverPoutNoCuts->Fill(highPtElectron.eSuperClusterOverP());
	
	MapCor1NoCuts->Fill(energy5x5/UncorrectedPatCalo,energy5x5/Ptrack_in);
	MapCor2NoCuts->Fill(energy5x5/UncorrectedPatCalo,highPtElectron.eSuperClusterOverP());
	MapCor3NoCuts->Fill(energy5x5/UncorrectedPatCalo,Ptrack_out/Ptrack_in);
	MapCor4NoCuts->Fill(energy5x5/UncorrectedPatCalo,energy5x5/highPtElectron.p());
	MapCor5NoCuts->Fill(energy5x5/UncorrectedPatCalo,UncorrectedPatCalo/Ptrack_out);
	MapCor6NoCuts->Fill(Ptrack_out/Ptrack_in,energy5x5/Ptrack_in);
	MapCor7NoCuts->Fill(Ptrack_out/Ptrack_in,UncorrectedPatCalo/Ptrack_out);
	MapCor8NoCuts->Fill(energy5x5/Ptrack_in,UncorrectedPatCalo/Ptrack_out);
	MapCor9NoCuts->Fill(energy5x5/UncorrectedPatCalo,highPtElectron.eSeedClusterOverPout());
	MapCor10NoCuts->Fill(highPtElectron.eSeedClusterOverPout(),Ptrack_out/Ptrack_in);
	MapCor11NoCuts->Fill(highPtElectron.eSeedClusterOverPout(),energy5x5/Ptrack_in);
	
	PinMinPoutNoCuts->Fill((Ptrack_in-Ptrack_out)/Ptrack_in);
	
	Error1NoCuts-> Fill(highPtElectron.trackMomentumError()/Ptrack_in);
	Error2NoCuts->Fill(highPtElectron.trackMomentumError()/Ptrack_out);
	Error3NoCuts->Fill(highPtElectron.trackMomentumError()/UncorrectedPatCalo);
	eSeedOverPout2NoCuts->Fill(highPtElectron.eSeedClusterOverPout());
	
	hadOverEmNoCuts->Fill(highPtElectron.hadronicOverEm());
	
	//------------------------------------------------Cuts-----------------------------------------------------
	//Cuts!
	if((energy3x3/energy5x5)<cut1_)return;
	
	if((Ptrack_out/Ptrack_in)< cut2_  || (Ptrack_out/Ptrack_in)> cut3_ )return;
	if((energy5x5/Ptrack_in)< cutEPin1_  || (energy5x5/Ptrack_in)> cutEPin2_ )return;
	
	e9->Fill(energy3x3); 
	e25->Fill(energy5x5); 
	e9Overe25->Fill(energy3x3/energy5x5);
	scE->Fill(sc.energy()); 
	trP->Fill(UncorrectedPatCalo);
	
	EoP->Fill(highPtElectron.eSuperClusterOverP()); 
	e25OverScE->Fill(energy5x5/sc.energy());
	
	E25oP->Fill(energy5x5/UncorrectedPatCalo);
	
	Map->Fill(maxCC_Eta,maxCC_Phi);
	PinOverPout->Fill( sqrt( pow(highPtElectron.trackMomentumAtVtx().X(),2) +pow(highPtElectron.trackMomentumAtVtx().Y(),2) + pow(highPtElectron.trackMomentumAtVtx().Z(),2) )/sqrt( pow(highPtElectron.trackMomentumOut().X(),2)+ pow(highPtElectron.trackMomentumOut().Y(),2)+ pow(highPtElectron.trackMomentumOut().Z(),2) ) );
	eSeedOverPout->Fill(highPtElectron.eSuperClusterOverP());
	
	MapCor1->Fill(energy5x5/UncorrectedPatCalo,energy5x5/Ptrack_in);
	MapCor2->Fill(energy5x5/UncorrectedPatCalo,highPtElectron.eSuperClusterOverP());
	MapCor3->Fill(energy5x5/UncorrectedPatCalo,Ptrack_out/Ptrack_in);
	MapCor4->Fill(energy5x5/UncorrectedPatCalo,energy5x5/highPtElectron.p());
	MapCor5->Fill(energy5x5/UncorrectedPatCalo,UncorrectedPatCalo/Ptrack_out);
	MapCor6->Fill(Ptrack_out/Ptrack_in,energy5x5/Ptrack_in);
	MapCor7->Fill(Ptrack_out/Ptrack_in,UncorrectedPatCalo/Ptrack_out);
	MapCor8->Fill(energy5x5/Ptrack_in,UncorrectedPatCalo/Ptrack_out);
	MapCor9->Fill(energy5x5/UncorrectedPatCalo,highPtElectron.eSeedClusterOverPout());
	MapCor10->Fill(highPtElectron.eSeedClusterOverPout(),Ptrack_out/Ptrack_in);
	MapCor11->Fill(highPtElectron.eSeedClusterOverPout(),energy5x5/Ptrack_in);
	
	PinMinPout->Fill((Ptrack_in-Ptrack_out)/Ptrack_in);
	
	Error1-> Fill(highPtElectron.trackMomentumError()/Ptrack_in);
	Error2->Fill(highPtElectron.trackMomentumError()/Ptrack_out);
	Error3->Fill(highPtElectron.trackMomentumError()/UncorrectedPatCalo);
	
	eSeedOverPout2->Fill(highPtElectron.eSeedClusterOverPout());
	hadOverEm->Fill(highPtElectron.hadronicOverEm());
	
	EventMatrix.push_back(energy);
	EnergyVector.push_back(UncorrectedPatCalo);
	MaxCCeta.push_back(maxCC_Eta);
	MaxCCphi.push_back(maxCC_Phi);
	
	WeightVector.push_back(energy5x5/UncorrectedPatCalo);
	
	//-------------------------------------------------------Extra Cut-----------------------------------------------------
	if(highPtElectron.eSeedClusterOverPout()< cutESeed_ ) return;

	MapCor1ESeed->Fill(energy5x5/UncorrectedPatCalo,energy5x5/Ptrack_in);
	MapCor2ESeed->Fill(energy5x5/UncorrectedPatCalo,highPtElectron.eSuperClusterOverP());
	MapCor3ESeed->Fill(energy5x5/UncorrectedPatCalo,Ptrack_out/Ptrack_in);
	MapCor4ESeed->Fill(energy5x5/UncorrectedPatCalo,energy5x5/highPtElectron.p());
	MapCor5ESeed->Fill(energy5x5/UncorrectedPatCalo,UncorrectedPatCalo/Ptrack_out);
	MapCor6ESeed->Fill(Ptrack_out/Ptrack_in,energy5x5/Ptrack_in);
	MapCor7ESeed->Fill(Ptrack_out/Ptrack_in,UncorrectedPatCalo/Ptrack_out);
	MapCor8ESeed->Fill(energy5x5/Ptrack_in,UncorrectedPatCalo/Ptrack_out);
	MapCor9ESeed->Fill(energy5x5/UncorrectedPatCalo,highPtElectron.eSeedClusterOverPout());
	MapCor10ESeed->Fill(highPtElectron.eSeedClusterOverPout(),Ptrack_out/Ptrack_in);
	MapCor11ESeed->Fill(highPtElectron.eSeedClusterOverPout(),energy5x5/Ptrack_in);
	
	eSeedOverPout2ESeed->Fill(highPtElectron.eSeedClusterOverPout());
	
	hadOverEmESeed->Fill(highPtElectron.hadronicOverEm());
	
      }else{return;}
}

