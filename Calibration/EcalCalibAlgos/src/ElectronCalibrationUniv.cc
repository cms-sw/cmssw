
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
#include "Calibration/EcalCalibAlgos/interface/ElectronCalibrationUniv.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
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

ElectronCalibrationUniv::ElectronCalibrationUniv(const edm::ParameterSet& iConfig)
{

  rootfile_ = iConfig.getParameter<std::string>("rootfile");
  EBrecHitLabel_ = iConfig.getParameter< edm::InputTag > ("ebRecHitsLabel");
  EErecHitLabel_ = iConfig.getParameter< edm::InputTag > ("eeRecHitsLabel");
  electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");
  trackLabel_ = iConfig.getParameter< edm::InputTag > ("trackLabel");
  calibAlgo_       = iConfig.getParameter<std::string>("CALIBRATION_ALGO");
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
  numevent_ = iConfig.getParameter<int>("numevent");
  miscalibfile_ = iConfig.getParameter<std::string>("miscalibfile");
  miscalibfileEndCap_ = iConfig.getParameter<std::string>("miscalibfileEndCap");

  cutEPCalo1_ = iConfig.getParameter<double>("cutEPCaloMin");
  cutEPCalo2_ = iConfig.getParameter<double>("cutEPCaloMax");
  cutEPin1_ = iConfig.getParameter<double>("cutEPinMin");
  cutEPin2_ = iConfig.getParameter<double>("cutEPinMax");
  cutCalo1_ = iConfig.getParameter<double>("cutCaloMin");
  cutCalo2_ = iConfig.getParameter<double>("cutCaloMax");
  
  cutESeed_ = iConfig.getParameter<double>("cutESeed");
  
   
}


ElectronCalibrationUniv::~ElectronCalibrationUniv()
{
}

//========================================================================
void ElectronCalibrationUniv::beginJob() {
  //========================================================================
  f = new TFile(rootfile_.c_str(),"RECREATE");
  f->cd();
  EventsAfterCuts = new TH1F("EventsAfterCuts","Events After Cuts",30,0,30);
  
  // Book histograms 
  e9 = new TH1F("e9","E9 energy", 300, 0., 150.);
  e25 = new TH1F("e25","E25 energy", 300, 0., 150.);
  scE = new TH1F("scE","SC energy", 300, 0., 150.);
  trP = new TH1F("trP","Trk momentum", 300, 0., 150.);
  EoP = new TH1F("EoP","EoP", 600, 0., 3.);
  EoP_all = new TH1F("EoP_all","EoP_all",600, 0., 3.);

  calibs = new TH1F("calib","Calibration constants", 800, 0.5, 2.);
  calibsEndCapMinus = new TH1F("calibEndCapMinus","Calibration constants EE-", 800, 0.5, 2.);
  calibsEndCapPlus = new TH1F("calibEndCapPlus","Calibration constants EE+", 800, 0.5, 2.);
  
  e25OverScE = new TH1F("e25OverscE","E25 / SC energy", 400, 0., 2.);
  E25oP = new TH1F("E25oP","E25 / P", 750, 0., 1.5);

  Map = new TH2F("Map","Nb Events in Crystal",173 ,-86 ,86,362, 0, 361 );
  e9Overe25 = new TH1F("e9Overe25","E9 / E25", 400, 0., 2.);
  Map3Dcalib = new TH2F("3Dcalib", "3Dcalib",173 ,-86 ,86,362, 0, 361 );
  Map3DcalibEndCapMinus = new TH2F("3DcalibEndCapMinus", "3Dcalib EE-",100 ,0 ,100,100, 0, 100 );
  Map3DcalibEndCapPlus = new TH2F("3DcalibEndCapPlus", "3Dcalib EE+",100 ,0 ,100,100, 0, 100 );

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
//   MapCorCalib = new TH2F ("MapCorCalib", "Correlation Miscalibration versus Calibration constants", 500, 0.5,1.5, 500, 0.5, 1.5);

  E25oPvsEta = new TH2F ("E25oPvsEta", "E/P vs Eta", 173, -86, 86, 600, 0.7,1.3);
  E25oPvsEtaEndCapMinus = new TH2F ("E25oPvsEtaEndCapMinus", "E/P vs R EE-", 100, 0, 100, 600, 0.7,1.3);
  E25oPvsEtaEndCapPlus = new TH2F ("E25oPvsEtaEndCapPlus", "E/P vs R EE+", 100, 0, 100, 600, 0.7,1.3);

  PinMinPout = new TH1F("PinMinPout","(Pin - Pout)/Pin",600,-2.0,2.0);

  calibinter = new TH1F("calibinter", "internal calibration constants", 800 , 0.5,2.);
  PinOverPout= new TH1F("PinOverPout", "pinOverpout", 600,0., 3.);
  eSeedOverPout= new TH1F("eSeedOverPout", "eSeedOverpout ", 600, 0., 3.);
//   MisCalibs = new TH1F("MisCalibs","Miscalibration constants",800,0.5,2.);
//   RatioCalibs = new TH1F("RatioCalibs","Ratio in Calibration Constants", 800, 0.5, 2.0);
//   DiffCalibs = new TH1F("DiffCalibs", "Difference in Calibration constants", 800, -1.0,1.0);
  calibinterEndCapMinus = new TH1F("calibinterEndCapMinus", "internal calibration constants", 800 , 0.5,2.);
  calibinterEndCapPlus = new TH1F("calibinterEndCapPlus", "internal calibration constants", 800 , 0.5,2.);
//   MisCalibsEndCapMinus = new TH1F("MisCalibsEndCapMinus","Miscalibration constants",800,0.5,2.);
//   MisCalibsEndCapPlus = new TH1F("MisCalibsEndCapPlus","Miscalibration constants",800,0.5,2.);
//   RatioCalibsEndCapMinus = new TH1F("RatioCalibsEndCapMinus","Ratio in Calibration Constants", 800, 0.5, 2.0);
//   RatioCalibsEndCapPlus = new TH1F("RatioCalibsEndCapPlus","Ratio in Calibration Constants", 800, 0.5, 2.0);
//   DiffCalibsEndCapMinus = new TH1F("DiffCalibsEndCapMinus", "Difference in Calibration constants", 800, -1.0,1.0);
//   DiffCalibsEndCapPlus = new TH1F("DiffCalibsEndCapPlus", "Difference in Calibration constants", 800, -1.0,1.0);
  Error1 = new TH1F ("Error1","DeltaP/Pin",800 ,-1.0,1.0 );
  Error2 = new TH1F ("Error2","DeltaP/Pout",800 ,-1.0,1.0 );
  Error3 = new TH1F ("Error3","DeltaP/Pcalo",800 ,-1.0,1.0 );
  eSeedOverPout2= new TH1F("eSeedOverPout2", "eSeedOverpout (No Supercluster)", 600, 0., 4.);
  hadOverEm= new TH1F("hadOverEm", "Had/EM distribution", 600, -2., 2.);
  
  // Book histograms  
  Map3DcalibNoCuts = new TH2F("3DcalibNoCuts", "3Dcalib (Before Cuts)",173 ,-86 ,86,362, 0, 361 );
  e9NoCuts = new TH1F("e9NoCuts","E9 energy (Before Cuts)",300, 0., 150.);
  e25NoCuts = new TH1F("e25NoCuts","E25 energy (Before Cuts)", 300, 0., 150.);
  scENoCuts = new TH1F("scENoCuts","SC energy (Before Cuts)", 300, 0., 150.);
  trPNoCuts = new TH1F("trPNoCuts","Trk momentum (Before Cuts)", 300, 0., 150.);
  EoPNoCuts = new TH1F("EoPNoCuts","EoP (Before Cuts)", 600, 0., 3.);
  calibsNoCuts = new TH1F("calibNoCuts","Calibration constants (Before Cuts)", 800, 0., 2.);
  e25OverScENoCuts = new TH1F("e25OverscENoCuts","E25 / SC energy (Before Cuts)", 400, 0., 2.);
  E25oPNoCuts = new TH1F("E25oPNoCuts","E25 / P (Before Cuts)", 750, 0., 1.5);
  MapEndCapMinus = new TH2F("MapEndCapMinus","Nb Events in Crystal (EndCap)",100 ,0 ,100,100, 0, 100 );
  MapEndCapPlus = new TH2F("MapEndCapPlus","Nb Events in Crystal (EndCap)",100 ,0 ,100,100, 0, 100 );
  e9Overe25NoCuts = new TH1F("e9Overe25NoCuts","E9 / E25 (Before Cuts)", 400, 0., 2.);
  PinOverPoutNoCuts = new TH1F("PinOverPoutNoCuts", "pinOverpout (Before Cuts)", 600,0., 3.);
  eSeedOverPoutNoCuts = new TH1F(" eSeedOverPoutNoCuts", "eSeedOverpout (Before Cuts) ", 600, 0., 4.);
  PinMinPoutNoCuts = new TH1F("PinMinPoutNoCuts","(Pin - Pout)/Pin (Before Cuts)",600,-2.0,2.0);

//   RatioCalibsNoCuts = new TH1F("RatioCalibsNoCuts","Ratio in Calibration Constants (Before Cuts)", 800, 0.5, 2.0);
//   DiffCalibsNoCuts = new TH1F("DiffCalibsNoCuts", "Difference in Calibration constants (Before Cuts)", 800, -1.0,1.0);
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
//   MapCorCalibEndCapMinus = new TH2F ("MapCorCalibEndCapMinus", "Correlation Miscalibration versus Calibration constants (EndCap)",  500, 0.5,1.5, 500, 0.5, 1.5);
//   MapCorCalibEndCapPlus = new TH2F ("MapCorCalibEndCapPlus", "Correlation Miscalibration versus Calibration constants (EndCap)",  500, 0.5,1.5, 500, 0.5, 1.5);

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
  GeneralMap = new TH2F("GeneralMap","Map without any cuts",173 ,-86 ,86,362, 0, 361 );
  GeneralMapEndCapMinus = new TH2F("GeneralMapEndCapMinus","Map without any cuts",100 ,0 ,100,100, 0, 100 );
  GeneralMapEndCapPlus = new TH2F("GeneralMapEndCapPlus","Map without any cuts",100 ,0 ,100,100, 0, 100 );
  GeneralMapBeforePt = new TH2F("GeneralMapBeforePt","Map without any cuts",173 ,-86 ,86,362, 0, 361 );
  GeneralMapEndCapMinusBeforePt = new TH2F("GeneralMapEndCapMinusBeforePt","Map without any cuts",100 ,0 ,100,100, 0, 100 );
  GeneralMapEndCapPlusBeforePt = new TH2F("GeneralMapEndCapPlusBeforePt","Map without any cuts",100 ,0 ,100,100, 0, 100 );
  
  calibClusterSize=ClusterSize_; 
  etaMin = int(mineta_);
  etaMax = int(maxeta_);
  phiMin = int(minphi_);
  phiMax = int(maxphi_);
  if(calibAlgo_=="L3"){
    MyL3Algo1 = new MinL3Algorithm(keventweight_,calibClusterSize, etaMin, etaMax, phiMin, phiMax);
  }else{
    if(calibAlgo_=="L3Univ"){
      UnivL3 = new MinL3AlgoUniv<DetId>(keventweight_);
    }else{
      if(calibAlgo_=="HH" || calibAlgo_=="HHReg"){
	MyHH = new HouseholderDecomposition(calibClusterSize, etaMin,etaMax, phiMin, phiMax); 
      }else{
	std::cout<<" Name of Algorithm is not recognize "<<calibAlgo_<<" Should be either L3, HH or HHReg. Abort! "<<std::endl;
      }
    }
  }
  read_events=0;
}

//========================================================================
void ElectronCalibrationUniv::beginRun(edm::Run const &, edm::EventSetup const& iSetup) {
  //========================================================================
  

  //To Deal with Geometry:
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  

}


//========================================================================

void
ElectronCalibrationUniv::endJob() {
//========================================================================

  f->cd();
  time_t start, end;
  time_t cpu_time_used;
  start = time(NULL);

  //In order to do only one loop to use properly looper properties, ask only for 1 iterations!
  int nIterations =10;
 if(calibAlgo_=="L3"){ 
   solution = MyL3Algo1->iterate(EventMatrix, MaxCCeta, MaxCCphi, EnergyVector,nIterations);
 }else{
   if(calibAlgo_=="L3Univ"){ 
     //Univsolution= UnivL3->getSolution();
     //     std::cout<<" Should derive solution "<<EnergyVector.size()<<std::endl;
     Univsolution= UnivL3->iterate(EventMatrix, UnivEventIds, EnergyVector, nIterations);
     //std::cout<<" solution size "<<Univsolution.size()<<std::endl;
  }else {
     if(calibAlgo_=="HH"){
       solution = MyHH->iterate(EventMatrix, MaxCCeta, MaxCCphi,EnergyVector,1,false);
     }else{
       if(calibAlgo_=="HHReg"){
	 solution = MyHH->runRegional(EventMatrix, MaxCCeta, MaxCCphi,EnergyVector, 2);
       }else{ 
	 std::cout<<" Calibration not run due to problem in Algo Choice..."<<std::endl;
	 return ;
       }
     }
   }
 }
   end = time(NULL);
   cpu_time_used = end - start;
   //     std::cout<<"222 solution size "<<Univsolution.size()<<std::endl;


  calibXMLwriter write_calibrations;
  
//   FILE* MisCalib;
//   //char* calibfile="miscalibfile";
//   MisCalib = fopen(miscalibfile_.c_str(),"r");
  
//   int fileStatus=0;
//   int eta=-1;
//   int phi=-1;
//   float coeff=-1;
  
  
   std::map<EBDetId,float> OldCoeff;
 
//  while(fileStatus != EOF) {
//    fileStatus = fscanf(MisCalib,"%d %d %f\n",  &eta,&phi,&coeff);
//    if(eta!=-1&&phi!=-1&& coeff!=-1){
//      //     std::cout<<" We have read correctly the coefficient " << coeff << " corresponding to eta "<<eta<<" and  phi "<<phi<<std::endl;
//      OldCoeff.insert(std::make_pair(EBDetId(eta,phi,EBDetId::ETAPHIMODE),coeff )); 
//    }
//  } 
 
//  fclose(MisCalib);
//   FILE* MisCalibEndCap;
//   //char* calibfile="miscalibfile";
//   MisCalibEndCap = fopen(miscalibfileEndCap_.c_str(),"r");
  
//   int fileStatus2=0;
//   int X=-1;
//   int Y=-1;
//   float coeff2=-1;
   std::map<EEDetId,float> OldCoeffEndCap;
 
//  while(fileStatus2 != EOF) {
//    fileStatus2 = fscanf(MisCalibEndCap,"%d %d %f\n",  &X,&Y,&coeff2);
//    if(X!=-1&&Y!=-1&& coeff2!=-1){
//      //     std::cout<<" We have read correctly the coefficient " << coeff << " corresponding to eta "<<eta<<" and  phi "<<phi<<std::endl;
//      if(TestEEvalidDetId(X,Y,1)){
//        OldCoeffEndCap.insert(std::make_pair(EEDetId(X,Y,1,EEDetId::XYMODE),coeff2 )); 
//      }
//    }
//  } 
 
// fclose(MisCalibEndCap);
  std::map<DetId,float>::const_iterator itmap;
  for (itmap = Univsolution.begin(); itmap != Univsolution.end(); itmap++){
    const DetId Id(itmap->first);
     if(Id.subdetId()==1){
      const EBDetId IChannelDetId(itmap->first);
      if (IChannelDetId.ieta()< mineta_){continue;}
      if (IChannelDetId.ieta()> maxeta_){continue;}
      if (IChannelDetId.iphi()< minphi_){continue;} 
      if (IChannelDetId.iphi()> maxphi_){continue;}
//      float Compare=1;
//      std::map<EBDetId,float>::iterator iter = OldCoeff.find(itmap->first);
//      if( iter != OldCoeff.end() )Compare = iter->second;
      Map3Dcalib->Fill(IChannelDetId.ieta(),IChannelDetId.iphi(),itmap->second) ;
      calibs->Fill(itmap->second);
      //DiffCalibs->Fill(newCalibs[icry]-miscalib[IChannelDetId.ieta()-1][IChannelDetId.iphi()-21]);
      //RatioCalibs->Fill(newCalibs[icry]/miscalib[IChannelDetId.ieta()-1][IChannelDetId.iphi()-21]);
      if (IChannelDetId.ieta()< mineta_+2){continue;}
      if (IChannelDetId.ieta()> maxeta_-2){continue;}
      if (IChannelDetId.iphi()< minphi_+2){continue;} 
      if (IChannelDetId.iphi()> maxphi_-2){continue;}
      write_calibrations.writeLine(IChannelDetId,itmap->second);
         calibinter->Fill(itmap->second);
//       MapCorCalib->Fill(itmap->second,Compare);
//       DiffCalibs->Fill(itmap->second-Compare);
//       RatioCalibs->Fill(itmap->second*Compare);
    }else{
      const EEDetId IChannelDetId(itmap->first);
//       if (IChannelDetId.ix()<0 ){continue;}
//       if (IChannelDetId.ix()>100 ){continue;}
//       if (IChannelDetId.iy()<0 ){continue;} 
//       if (IChannelDetId.iy()>100 ){continue;}
//     std::map<EEDetId,float>::iterator iter = OldCoeffEndCap.find(itmap->first);
//      float Compare=1;
//      if( iter != OldCoeffEndCap.end() )Compare = iter->second;
      if(IChannelDetId.zside()<0){
 	Map3DcalibEndCapMinus->Fill(IChannelDetId.ix(),IChannelDetId.iy(),itmap->second) ;
 	calibsEndCapMinus->Fill(itmap->second);
 	calibinterEndCapMinus->Fill(itmap->second);
// 	DiffCalibsEndCapMinus->Fill(itmap->second-Compare);
// 	RatioCalibsEndCapMinus->Fill(itmap->second*Compare);
// 	MapCorCalibEndCapMinus->Fill(itmap->second,Compare);
      }else{
 	Map3DcalibEndCapPlus->Fill(IChannelDetId.ix(),IChannelDetId.iy(),itmap->second) ;
 	calibsEndCapPlus->Fill(itmap->second);
 	calibinterEndCapPlus->Fill(itmap->second);
// 	DiffCalibsEndCapPlus->Fill(itmap->second-Compare);
// 	RatioCalibsEndCapPlus->Fill(itmap->second*Compare);
// 	MapCorCalibEndCapPlus->Fill(itmap->second,Compare);
      }
      write_calibrations.writeLine(IChannelDetId,itmap->second);
    }
  }
  EventsAfterCuts->Write();

  // Book histograms 
  e25->Write();
  e9->Write();
  scE->Write();
  trP->Write();
  EoP->Write();
  EoP_all->Write();
  calibs->Write();
  calibsEndCapMinus->Write();
  calibsEndCapPlus->Write();
  e9Overe25->Write();
  e25OverScE->Write();
  Map->Write();
  E25oP->Write();

  PinOverPout->Write();
  eSeedOverPout->Write();
//   MisCalibs->Write();
//   RatioCalibs->Write();
//   DiffCalibs->Write();
//   RatioCalibsNoCuts->Write();
//   DiffCalibsNoCuts->Write();
//   MisCalibsEndCapMinus->Write();
//   MisCalibsEndCapPlus->Write();
//   RatioCalibsEndCapMinus->Write();
//   RatioCalibsEndCapPlus->Write();
//   DiffCalibsEndCapMinus->Write();
//   DiffCalibsEndCapPlus->Write();

  e25NoCuts->Write();
  e9NoCuts->Write();
  scENoCuts->Write();
  trPNoCuts->Write();
  EoPNoCuts->Write();
  calibsNoCuts->Write();
  e9Overe25NoCuts->Write();
  e25OverScENoCuts->Write();
  MapEndCapMinus->Write();
  MapEndCapPlus->Write();
  E25oPNoCuts->Write();
  Map3Dcalib->Write();
  Map3DcalibEndCapMinus->Write();
  Map3DcalibEndCapPlus->Write();
  Map3DcalibNoCuts->Write();
  calibinter->Write();
  calibinterEndCapMinus->Write();
  calibinterEndCapPlus->Write();
  calibinterNoCuts->Write();
  PinOverPoutNoCuts->Write();
  eSeedOverPoutNoCuts->Write();

  GeneralMap->Write();
  GeneralMapEndCapMinus->Write();
  GeneralMapEndCapPlus->Write();
  GeneralMapBeforePt->Write();
  GeneralMapEndCapMinusBeforePt->Write();
  GeneralMapEndCapPlusBeforePt->Write();

  MapCor1->Write();
  MapCor2->Write();
  MapCor3->Write();
  MapCor4->Write();
  MapCor5->Write();
  MapCor6->Write();
  MapCor7->Write();
  MapCor8->Write();
  MapCor9->Write();
  MapCor10->Write();
  MapCor11->Write();
  //  MapCorCalib->Write();

  MapCor1NoCuts->Write();
  MapCor2NoCuts->Write();
  MapCor3NoCuts->Write();
  MapCor4NoCuts->Write();
  MapCor5NoCuts->Write();
  MapCor6NoCuts->Write();
  MapCor7NoCuts->Write();
  MapCor8NoCuts->Write();
  MapCor9NoCuts->Write();
  MapCor10NoCuts->Write();
  MapCor11NoCuts->Write();
//   MapCorCalibEndCapMinus->Write();
//   MapCorCalibEndCapPlus->Write();

  MapCor1ESeed->Write();
  MapCor2ESeed->Write();
  MapCor3ESeed->Write();
  MapCor4ESeed->Write();
  MapCor5ESeed->Write();
  MapCor6ESeed->Write();
  MapCor7ESeed->Write();
  MapCor8ESeed->Write();
  MapCor9ESeed->Write();
  MapCor10ESeed->Write();
  MapCor11ESeed->Write();

  E25oPvsEta->Write();
  E25oPvsEtaEndCapMinus->Write();
  E25oPvsEtaEndCapPlus->Write();

  PinMinPout->Write(); 
  PinMinPoutNoCuts->Write();

  Error1->Write();
  Error2->Write();
  Error3->Write();
  Error1NoCuts->Write();
  Error2NoCuts->Write();
  Error3NoCuts->Write();

  eSeedOverPout2->Write();
  eSeedOverPout2NoCuts->Write();
  eSeedOverPout2ESeed->Write();

  hadOverEm->Write();
  hadOverEmNoCuts->Write();
  hadOverEmESeed->Write();

  f->Write();

  f->Close();
//   if(MyL3Algo1)delete MyL3Algo1; 
//   if(UnivL3)delete UnivL3; 
//   if(MyHH)delete MyHH; 
//  delete f;
  ////////////////////////       FINAL STATISTICS           ////////////////////

  std::cout << " " << std::endl;
  std::cout << "************* STATISTICS **************" << std::endl;
  std::cout << " Events Studied "<<read_events << std::endl;
  std::cout << "Timing info:" << std::endl;
  std::cout << "CPU time usage  -- calibrating: " << cpu_time_used << " sec." << std::endl;
 
  /////////////////////////////////////////////////////////////////////////////
}


DetId  ElectronCalibrationUniv::findMaxHit(const std::vector<DetId> & v1,const EBRecHitCollection *EBhits,const EERecHitCollection *EEhits) {
  //=================================================================================
  
  double currEnergy = 0.;
  DetId maxHit;
  
  for( std::vector<DetId>::const_iterator idsIt = v1.begin(); idsIt != v1.end(); ++idsIt) {
    if(idsIt->subdetId()==1){
     EBRecHitCollection::const_iterator itrechit;
      itrechit = EBhits->find(*idsIt);
      if(itrechit==EBhits->end()){
	std::cout << "ElectronCalibration::findMaxHit: rechit not found! " << (EBDetId)(*idsIt)<<std::endl;
	continue;
      }
      if(itrechit->energy() > currEnergy) {
	currEnergy=itrechit->energy();
	maxHit= *idsIt;
      }
    }else{
      EERecHitCollection::const_iterator itrechit;
      itrechit = EEhits->find(*idsIt);
      if(itrechit==EEhits->end()){
      	std::cout << "ElectronCalibration::findMaxHit: rechit not found! idsIt = " << (EEDetId)(*idsIt)<< std::endl;
	continue;
      }
      
      if(itrechit->energy() > currEnergy) {
	currEnergy=itrechit->energy();
	maxHit= *idsIt;
      }
    }
  }
  
  return maxHit;
  
}

bool ElectronCalibrationUniv::TestEEvalidDetId(int crystal_ix, int crystal_iy, int iz) {

  bool valid = false;
  if (crystal_ix < 1 ||  crystal_ix > 100 ||
      crystal_iy < 1 || crystal_iy > 100 || abs(iz) != 1 ) 
    { return valid; }
  if ( (crystal_ix >= 1 && crystal_ix <= 3 && (crystal_iy <= 40 || crystal_iy > 60) ) ||
       (crystal_ix >= 4 && crystal_ix <= 5 && (crystal_iy <= 35 || crystal_iy > 65) ) || 
       (crystal_ix >= 6 && crystal_ix <= 8 && (crystal_iy <= 25 || crystal_iy > 75) ) || 
       (crystal_ix >= 9 && crystal_ix <= 13 && (crystal_iy <= 20 || crystal_iy > 80) ) || 
       (crystal_ix >= 14 && crystal_ix <= 15 && (crystal_iy <= 15 || crystal_iy > 85) ) || 
       (crystal_ix >= 16 && crystal_ix <= 20 && (crystal_iy <= 13 || crystal_iy > 87) ) || 
       (crystal_ix >= 21 && crystal_ix <= 25 && (crystal_iy <= 8 || crystal_iy > 92) ) || 
       (crystal_ix >= 26 && crystal_ix <= 35 && (crystal_iy <= 5 || crystal_iy > 95) ) || 
       (crystal_ix >= 36 && crystal_ix <= 39 && (crystal_iy <= 3 || crystal_iy > 97) ) || 
       (crystal_ix >= 98 && crystal_ix <= 100 && (crystal_iy <= 40 || crystal_iy > 60) ) ||
       (crystal_ix >= 96 && crystal_ix <= 97 && (crystal_iy <= 35 || crystal_iy > 65) ) || 
       (crystal_ix >= 93 && crystal_ix <= 95 && (crystal_iy <= 25 || crystal_iy > 75) ) || 
       (crystal_ix >= 88 && crystal_ix <= 92 && (crystal_iy <= 20 || crystal_iy > 80) ) || 
       (crystal_ix >= 86 && crystal_ix <= 87 && (crystal_iy <= 15 || crystal_iy > 85) ) || 
       (crystal_ix >= 81 && crystal_ix <= 85 && (crystal_iy <= 13 || crystal_iy > 87) ) || 
       (crystal_ix >= 76 && crystal_ix <= 80 && (crystal_iy <= 8 || crystal_iy > 92) ) || 
       (crystal_ix >= 66 && crystal_ix <= 75 && (crystal_iy <= 5 || crystal_iy > 95) ) || 
       (crystal_ix >= 62 && crystal_ix <= 65 && (crystal_iy <= 3 || crystal_iy > 97) ) ||
       ( (crystal_ix == 40 || crystal_ix == 61) && ( (crystal_iy >= 46 && crystal_iy <= 55 ) || crystal_iy <= 3 || crystal_iy > 97 )) ||
       ( (crystal_ix == 41 || crystal_ix == 60) && crystal_iy >= 44 && crystal_iy <= 57 ) ||
       ( (crystal_ix == 42 || crystal_ix == 59) && crystal_iy >= 43 && crystal_iy <= 58 ) ||
       ( (crystal_ix == 43 || crystal_ix == 58) && crystal_iy >= 42 && crystal_iy <= 59 ) ||
       ( (crystal_ix == 44 || crystal_ix == 45 || crystal_ix == 57 || crystal_ix == 56) && crystal_iy >= 41 && crystal_iy <= 60 ) ||
       ( crystal_ix >= 46 && crystal_ix <= 55 && crystal_iy >= 40 && crystal_iy <= 61 ) 
       )
    { return valid; }
  valid = true;
  return valid;
}


//=================================================================================
void
ElectronCalibrationUniv::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//=================================================================================
   using namespace edm;

  // Get EBRecHits
  edm::Handle<EBRecHitCollection> EBphits;
  iEvent.getByLabel( EBrecHitLabel_, EBphits);
  if (!EBphits.isValid()) {
     std::cerr << "Error! can't get the product EBRecHitCollection: " << std::endl;
  }
   const EBRecHitCollection* EBhits = EBphits.product(); // get a ptr to the product

   // Get EERecHits
   edm::Handle<EERecHitCollection> EEphits;

   iEvent.getByLabel( EErecHitLabel_, EEphits);
   if (!EEphits.isValid()) {
     std::cerr << "Error! can't get the product EERecHitCollection: " << std::endl;
   }
   const EERecHitCollection* EEhits = EEphits.product(); // get a ptr to the product

  // Get pixelElectrons
   edm::Handle<reco::GsfElectronCollection> pElectrons;
   iEvent.getByLabel(electronLabel_, pElectrons);
   if (!pElectrons.isValid()) {
     std::cerr << "Error! can't get the product ElectronCollection: " << std::endl;
   }
  const reco::GsfElectronCollection* electronCollection = pElectrons.product();
  read_events++;
  if(read_events%1000 ==0)std::cout << "read_events = " << read_events << std::endl;

  EventsAfterCuts->Fill(1);
  if (!EBhits || !EEhits)return;
  EventsAfterCuts->Fill(2);
  if (EBhits->size() == 0 && EEhits->size() == 0 )     return ;
  EventsAfterCuts->Fill(3);
  if (!electronCollection)     return ;
   EventsAfterCuts->Fill(4); 
  if (electronCollection->size() == 0)     return;

//    ////////////////Need to recalibrate the events (copy code from EcalRecHitRecalib):

////////////////////////////////////////////////////////////////////////////////////////
///                          START HERE....
///////////////////////////////////////////////////////////////////////////////////////

  reco::GsfElectronCollection::const_iterator eleIt = electronCollection->begin();

  reco::GsfElectron highPtElectron;

  float highestElePt=0.;
  bool found=false;
  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {

     if(fabs(eleIt->eta())>2.4) continue;
     //     if(eleIt->eta()<0.0) continue;
      
     if(eleIt->pt()>highestElePt) {
       highestElePt=eleIt->pt();
       highPtElectron = *eleIt;
       found =true;
       //       std::cout<<" eleIt->pt( "<<eleIt->pt()<<" eleIt->eta() "<<eleIt->eta()<<std::endl;
    }

  }
  EventsAfterCuts->Fill(5); 
  if(!found) return ;
  
  const reco::SuperCluster & sc = *(highPtElectron.superCluster()) ;
  //  if(fabs(sc.eta())>1.479){std::cout<<" SC not in Barrel "<<sc.eta()<<std::endl;;}
  //  const std::vector<DetId> & v1 = sc.getHitsByDetId();

      std::vector<DetId> v1;
      //Loop to fill the vector of DetIds
for (std::vector<std::pair<DetId,float> >::const_iterator idsIt = sc.hitsAndFractions().begin();
       idsIt != sc.hitsAndFractions().end ();++idsIt)
  {v1.push_back(idsIt->first);
 }

  DetId maxHitId;
  
  maxHitId = findMaxHit(v1,(EBhits),(EEhits)); 
  //maxHitId = findMaxHit(v1,EBhits,EEhits); 
  
  EventsAfterCuts->Fill(6);
  if(maxHitId.null()){std::cout<<" Null "<<std::endl; return ;}

  int maxCC_Eta = 0;
  int maxCC_Phi = 0;
  int Zside =0 ;
  if(maxHitId.subdetId()!=1) {
    maxCC_Eta = ((EEDetId)maxHitId).ix();
    maxCC_Phi = ((EEDetId)maxHitId).iy();
    Zside = ((EEDetId)maxHitId).zside();
    //    std::cout<<" ++++++++ Zside "<<Zside<<std::endl;
  }else{
    maxCC_Eta = ((EBDetId)maxHitId).ieta();
    maxCC_Phi = ((EBDetId)maxHitId).iphi();
  }




//   if(maxCC_Eta>maxeta_ ) ;
//   if(maxCC_Eta<mineta_ )  ;

  // number of events per crystal is set
//   eventcrystal[maxCC_Eta][maxCC_Phi]+=1;
//   if(eventcrystal[maxCC_Eta][maxCC_Phi] > numevent_) ;
  
  
  // fill cluster energy
  std::vector<float> energy;
  float energy3x3=0.;  
  float energy5x5=0.;  
  //Should be moved to cfg file!
  int ClusterSize = ClusterSize_; 
  
  const CaloSubdetectorTopology* topology=theCaloTopology->getSubdetectorTopology(DetId::Ecal,maxHitId.subdetId());
  std::vector<DetId> NxNaroundMax = topology->getWindow(maxHitId,ClusterSize,ClusterSize);
  //ToCompute 3x3
  std::vector<DetId> S9aroundMax = topology->getWindow(maxHitId,3,3);
  
  EventsAfterCuts->Fill(7);
  if((int)NxNaroundMax.size()!=ClusterSize*ClusterSize)return;
  EventsAfterCuts->Fill(8);
   if(S9aroundMax.size()!=9)return;
 
   //   std::cout<<" ******** New Event "<<std::endl;

  EventsAfterCuts->Fill(9);
   for (int icry=0;icry<ClusterSize*ClusterSize;icry++){
    if (NxNaroundMax[icry].subdetId() == EcalBarrel) {
      EBRecHitCollection::const_iterator itrechit;
      itrechit = EBhits->find(NxNaroundMax[icry]);
      if(itrechit==EBhits->end()){ 
	//	std::cout << "EB DetId not in e25" << std::endl;
	energy.push_back(0.);
	energy5x5 += 0.;
	continue;
      }
      
      if (edm::isNotFinite(itrechit->energy())){std::cout<<" nan energy "<<std::endl; return;} 	  
      energy.push_back(itrechit->energy());
      energy5x5 += itrechit->energy();
      
      //Summing in 3x3 to cut later on:	   
      for (int tt=0;tt<9;tt++){
	if(NxNaroundMax[icry]==S9aroundMax[tt])energy3x3+=itrechit->energy();
      }
    }else{
      EERecHitCollection::const_iterator itrechit;
      
      itrechit = EEhits->find(NxNaroundMax[icry]);
      
      if(itrechit==EEhits->end()){ 
	//	std::cout << "EE DetId not in e25" << std::endl;
	//	std::cout<<" ******** putting 0 "<<std::endl;
	energy.push_back(0.);
	energy5x5 += 0.;
 	continue;
      }
      
      if (edm::isNotFinite(itrechit->energy())){std::cout<<" nan energy "<<std::endl; return;}
      energy.push_back(itrechit->energy());
      energy5x5 += itrechit->energy();
      
      //Summing in 3x3 to cut later on:	   
      for (int tt=0;tt<9;tt++){
	if(NxNaroundMax[icry]==S9aroundMax[tt])energy3x3+=itrechit->energy();
      }
    }
  }
  //  if((read_events-50)%10000 ==0)cout << "++++++++++++ENERGY 5x5 " <<  energy5x5 << std::endl;
  EventsAfterCuts->Fill(10);
  //  std::cout<<" ******** NxNaroundMax.size() "<<NxNaroundMax.size()<<std::endl;
  //  std::cout<<" ******** energy.size() "<<energy.size()<<std::endl;
  if((int)energy.size()!=ClusterSize*ClusterSize) return ;

  if(maxHitId.subdetId() == EcalBarrel){
    GeneralMapBeforePt->Fill(maxCC_Eta,maxCC_Phi);
  }else{
    if(Zside<0){
      GeneralMapEndCapMinusBeforePt->Fill(maxCC_Eta,maxCC_Phi);
    }else{
      GeneralMapEndCapPlusBeforePt->Fill(maxCC_Eta,maxCC_Phi);
    }
  }

  EventsAfterCuts->Fill(11);
  if(highestElePt<ElePt_)return ;

  if(maxHitId.subdetId() == EcalBarrel){
    GeneralMap->Fill(maxCC_Eta,maxCC_Phi);
  }else{
    if(Zside<0){
    GeneralMapEndCapMinus->Fill(maxCC_Eta,maxCC_Phi);
    }else{
    GeneralMapEndCapPlus->Fill(maxCC_Eta,maxCC_Phi);
    }
  }

  EventsAfterCuts->Fill(12);
   if(highPtElectron.classification()!=elecclass_ && elecclass_!= -1 )   return;

 	float Ptrack_in=sqrt( pow(highPtElectron.trackMomentumAtVtx().X(),2) +pow(highPtElectron.trackMomentumAtVtx().Y(),2) + pow(highPtElectron.trackMomentumAtVtx().Z(),2) );
	
	float UncorrectedPatCalo = sqrt(pow(highPtElectron.trackMomentumAtCalo().X(),2)+pow(highPtElectron.trackMomentumAtCalo().Y(),2)+pow(highPtElectron.trackMomentumAtCalo().Z(),2));
	
	float Ptrack_out = sqrt( pow(highPtElectron.trackMomentumOut().X(),2)+ pow(highPtElectron.trackMomentumOut().Y(),2)+ pow(highPtElectron.trackMomentumOut().Z(),2) );

	e9NoCuts->Fill(energy3x3); 
	e25NoCuts->Fill(energy5x5); 
	e9Overe25NoCuts->Fill(energy3x3/energy5x5);
	scENoCuts->Fill(sc.energy()); 
	
	trPNoCuts->Fill(UncorrectedPatCalo); 
	
	EoPNoCuts->Fill(highPtElectron.eSuperClusterOverP()); 
	e25OverScENoCuts->Fill(energy5x5/sc.energy());
	
	E25oPNoCuts->Fill(energy5x5/UncorrectedPatCalo);
	
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


   //Cuts!
   if((energy3x3/energy5x5)<cut1_)return ;
   if((Ptrack_out/Ptrack_in)< cut2_  || (Ptrack_out/Ptrack_in)> cut3_ )return;
   if((energy5x5/Ptrack_in)< cutEPin1_  || (energy5x5/Ptrack_in)> cutEPin2_ )return;
//    if(!highPtElectron.ecalDriven())return;
//    if(!highPtElectron.passingCutBasedPreselection())return;


// //  Apply Pietro cuts:   
// 	EventsAfterCuts->Fill(13);
// 	//Module 1
// 	if(maxHitId.subdetId() == EcalBarrel){
// 	  //Module 1
// 	  if(maxCC_Eta <= 25){
// 	    if(highPtElectron.eSuperClusterOverP()>1.05 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 	    if(highPtElectron.eSeedClusterOverPout()>1.4 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 	    if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.2)return ;
// 	  }else{
// 	    //Module 2
// 	    if( maxCC_Eta > 25&& maxCC_Eta <= 45){
// 	      if(highPtElectron.eSuperClusterOverP()>1.05 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 	      if(highPtElectron.eSeedClusterOverPout()>1.25 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 	      if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.2)return ;
// 	    }else{
// 	    //Module 3
// 	      if( maxCC_Eta > 45&& maxCC_Eta <= 65){
// 		if(highPtElectron.eSuperClusterOverP()>1.05 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 		if(highPtElectron.eSeedClusterOverPout()>1.15 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 		if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.15)return ;
// 	      }else{
// 	      if( maxCC_Eta > 65&& maxCC_Eta <= 85){
// 		if(highPtElectron.eSuperClusterOverP()>1.05 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 		if(highPtElectron.eSeedClusterOverPout()>1.15 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 		if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.15)return ;
// 	      }else{
// 		return;
// 	      }
// 	      }
// 	    }
// 	  }
// 	}else{
// 	  //EndCapMinus Side:
// 	  //EndCapPlus Side:
// 	  int iR = sqrt((maxCC_Eta-50)*(maxCC_Eta-50) + (maxCC_Phi-50)*(maxCC_Phi-50));
// 	  if( iR >= 22&& iR < 27){
// 	    if(highPtElectron.eSuperClusterOverP()>1.05 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 	    if(highPtElectron.eSeedClusterOverPout()>1.15 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 	    if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.2)return ;
// 	  }else{
// 	    if( iR >= 27&& iR < 32){
// 	      if(highPtElectron.eSuperClusterOverP()>1.1 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 	      if(highPtElectron.eSeedClusterOverPout()>1.25 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 	      if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.2)return ;
// 	    }else{
// 	      if( iR >= 32&& iR < 37){
// 		if(highPtElectron.eSuperClusterOverP()>1.05 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 		if(highPtElectron.eSeedClusterOverPout()>1.15 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 		if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.2)return ;
// 	      }else{
// 		if( iR >= 37&& iR < 42){
// 		  if(highPtElectron.eSuperClusterOverP()>1.1 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 		  if(highPtElectron.eSeedClusterOverPout()>1.15 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 		  if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.15)return ;
// 		}else{
// 		  if( iR >= 42){
// 		    if(highPtElectron.eSuperClusterOverP()>1.05 || highPtElectron.eSuperClusterOverP()<0.95)return ;
// 		    if(highPtElectron.eSeedClusterOverPout()>1.15 || highPtElectron.eSeedClusterOverPout()<0.90)return ;
// 		  if((Ptrack_in- Ptrack_out) / Ptrack_in <-0.05 || (Ptrack_in- Ptrack_out) / Ptrack_in >0.15)return ;
// 		  }
// 		}
// 	      }
// 	    }
// 	  }
//	}
	
	
	if(maxHitId.subdetId() == EcalBarrel){
	  E25oPvsEta->Fill(maxCC_Eta,energy5x5/UncorrectedPatCalo);
	}else{
	  float Radius = sqrt((maxCC_Eta)*(maxCC_Eta) + (maxCC_Phi)*(maxCC_Phi));
	  if(Zside<0){
	    E25oPvsEtaEndCapMinus->Fill(Radius,energy5x5/UncorrectedPatCalo);
	  }else{
	    E25oPvsEtaEndCapPlus->Fill(Radius,energy5x5/UncorrectedPatCalo);
	  }
	}
	e9->Fill(energy3x3); 
	e25->Fill(energy5x5); 
	e9Overe25->Fill(energy3x3/energy5x5);
	scE->Fill(sc.energy()); 
	trP->Fill(UncorrectedPatCalo);
	
	EoP->Fill(highPtElectron.eSuperClusterOverP()); 
	e25OverScE->Fill(energy5x5/sc.energy());
	
	E25oP->Fill(energy5x5/UncorrectedPatCalo);
	
	if(maxHitId.subdetId() == EcalBarrel){
	  Map->Fill(maxCC_Eta,maxCC_Phi);
	}else{
	  if(Zside<0){
	    MapEndCapMinus->Fill(maxCC_Eta,maxCC_Phi);
	  }else{
	    MapEndCapPlus->Fill(maxCC_Eta,maxCC_Phi);
	  }
	}
	

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
  
    
	UnivEventIds.push_back(NxNaroundMax);
	EventMatrix.push_back(energy);
	EnergyVector.push_back(UncorrectedPatCalo);
   
	EventsAfterCuts->Fill(14);
    
	if(!highPtElectron.ecalDrivenSeed())EventsAfterCuts->Fill(15);


	return;	
}

DEFINE_FWK_MODULE(ElectronCalibrationUniv);
