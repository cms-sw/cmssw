#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"

#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"
#include "Calibration/Tools/interface/EcalIndexingTools.h"

#include "CLHEP/Vector/LorentzVector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "Calibration/EcalCalibAlgos/interface/ZeeCalibration.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeKinematicTools.h"

#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

/////////
#include "HLTrigger/HLTanalyzers/interface/HLTrigReport.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "TTree.h"
#include "TBranch.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TProfile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TRandom.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <map>
#include <fstream>

#define MZ 91.1876

#define DEBUG 1

ZeeCalibration::ZeeCalibration(const edm::ParameterSet& iConfig)
{

#ifdef DEBUG
  std::cout<<"[ZeeCalibration] Starting the ctor"<<std::endl;
#endif

  theMaxLoops =  iConfig.getUntrackedParameter<unsigned int>("maxLoops",0);

  wantEtaCorrection_ = iConfig.getUntrackedParameter<bool>("wantEtaCorrection",true);   

  outputFileName_  = iConfig.getParameter<std::string>("outputFile");

  minInvMassCut_ = iConfig.getUntrackedParameter<double>("minInvMassCut", 70.);   
  maxInvMassCut_ = iConfig.getUntrackedParameter<double>("maxInvMassCut", 110.);   

  rechitProducer_ = iConfig.getParameter<std::string>("rechitProducer");
  rechitCollection_ = iConfig.getParameter<std::string>("rechitCollection");

  erechitProducer_ = iConfig.getParameter<std::string>("erechitProducer");
  erechitCollection_ = iConfig.getParameter<std::string>("erechitCollection");

  scProducer_ = iConfig.getParameter<std::string>("scProducer");
  scCollection_ = iConfig.getParameter<std::string>("scCollection");

  scIslandProducer_ = iConfig.getParameter<std::string>("scIslandProducer");
  scIslandCollection_ = iConfig.getParameter<std::string>("scIslandCollection");

  calibMode_ = iConfig.getUntrackedParameter<std::string>("ZCalib_CalibType");

  mcProducer_ = iConfig.getUntrackedParameter<std::string>("mcProducer","");


  electronProducer_ = iConfig.getParameter<std::string > ("electronProducer");
  electronCollection_ = iConfig.getParameter<std::string > ("electronCollection");

  outputFile_ = TFile::Open(outputFileName_.c_str(),"RECREATE"); // open output file to store histograms  
  
  myTree = new TTree("myTree","myTree");
  //  myTree->Branch("zMass","zMass", &mass);
  myTree->Branch("zMass",&mass4tree,"mass/F");
  myTree->Branch("zMassDiff",&massDiff4tree,"massDiff/F");

  barrelfile_=iConfig.getUntrackedParameter<std::string> ("initialMiscalibrationBarrel","");
  endcapfile_=iConfig.getUntrackedParameter<std::string> ("initialMiscalibrationEndcap","");

  electronSelection_=iConfig.getUntrackedParameter<unsigned int> ("electronSelection",0);//option for electron selection

  etaBins_ = iConfig.getUntrackedParameter<unsigned int>("etaBins", 10);   
  etBins_ = iConfig.getUntrackedParameter<unsigned int>("etBins", 10);   

  etaMin_ = iConfig.getUntrackedParameter<double>("etaMin", 0.);   
  etMin_ = iConfig.getUntrackedParameter<double>("etMin", 0.);   
  etaMax_ = iConfig.getUntrackedParameter<double>("etaMax", 3.);   
  etMax_ = iConfig.getUntrackedParameter<double>("etMax", 100.);   

 
  //  new ZeePlots("zeePlots.root");
  //  ZeePlots->bookHistos();

  //ZeeCalibrationPLots("zeeCalibPlots");
  //ZeecaPlots->bookHistos(maxsIter); 
  
  hlTriggerResults_ = iConfig.getParameter<edm::InputTag> ("HLTriggerResults");

  theParameterSet=iConfig;
  EcalIndexingTools* myIndexTool=0;

  
  myIndexTool = EcalIndexingTools::getInstance();
  
  myIndexTool->setBinRange( etaBins_, etaMin_, etaMax_, etBins_, etMin_, etMax_ );
  
  //creating the algorithm
  theAlgorithm_ = new ZIterativeAlgorithmWithFit(iConfig);
  
  // Tell the framework what data is being produced
  //setWhatProduced(this);
  setWhatProduced (this, &ZeeCalibration::produceEcalIntercalibConstants ) ;
  findingRecord<EcalIntercalibConstantsRcd> () ;

  for(int i = 0; i<50; i++){

    coefficientDistanceAtIteration[i] = -1.;
    loopArray[i] = -1.;
    sigmaArray[i] = -1.;
    sigmaErrorArray[i] = -1.;

  }

#ifdef DEBUG
  std::cout<<"[ZeeCalibration] Done with the ctor"<<std::endl;
#endif

}


ZeeCalibration::~ZeeCalibration()
{
//   if (theAlgorithm_)
//     delete theAlgorithm_;
}

//_____________________________________________________________________________
// Produce EcalIntercalibConstants
boost::shared_ptr<EcalIntercalibConstants>
ZeeCalibration::produceEcalIntercalibConstants( const EcalIntercalibConstantsRcd& iRecord )
{
  std::cout << "@SUB=ZeeCalibration::produceEcalIntercalibConstants" << std::endl;
  return ical;
}

void ZeeCalibration::beginOfJob(){isfirstcall_=true;}




//========================================================================
void
ZeeCalibration::endOfJob() {


  printStatistics();

  if(calibMode_ != "ETA_ET_MODE"){

  ///if not ETA_ET MODE - begin

  //Writing out calibration coefficients
  calibXMLwriter* barrelWriter = new calibXMLwriter(EcalBarrel);
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  barrelWriter->writeLine(ebid,* (ical->getMap().find(ebid.rawId()) ));
	}
    }
  }
  

  
  calibXMLwriter* endcapWriter = new calibXMLwriter(EcalEndcap);
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  EEDetId eeid(iX,iY,1);
          endcapWriter->writeLine(eeid,*(ical->getMap().find(eeid.rawId())  ) );
	}
      if (EEDetId::validDetId(iX,iY,-1))
	{
	  EEDetId eeid(iX,iY,-1);
          endcapWriter->writeLine(eeid, *(ical->getMap().find(eeid.rawId())) );
	}
      
    }
  }
  

  } ///if not ETA_ET MODE - end

  std::cout<<"Writing  histos..."<<std::endl;
  outputFile_->cd();

  //  zeeplts->Write();

  h1_eventsBeforeEWKSelection_ ->Write();
  h1_eventsAfterEWKSelection_ ->Write();

  h1_eventsBeforeBorderSelection_ ->Write();
  h1_eventsAfterBorderSelection_ ->Write();

  h1_borderElectronClassification_->Write();
 
  h2_xtalMiscalibCoeffBarrel_->Write();
  h2_xtalMiscalibCoeffEndcapMinus_->Write();
  h2_xtalMiscalibCoeffEndcapPlus_->Write();

  h1_electronCosTheta_SC_->Write();
  h1_electronCosTheta_TK_->Write();
  h1_electronCosTheta_SC_TK_->Write();

  h1_zMassResol_->Write();
  h1_zEtaResol_->Write();
  h1_zPhiResol_->Write();
  h1_eleEtaResol_->Write();
  h1_elePhiResol_->Write();
  h1_seedOverSC_ ->Write();
  h1_preshowerOverSC_ ->Write();
   
  for(unsigned int i =0; i<25; i++){
    if( i < theMaxLoops ){
      
      h_ESCEtrueVsEta_[i]->Write();
      h_ESCEtrue_[i]->Write();
      
      h_ESCcorrEtrueVsEta_[i]->Write();
      h_ESCcorrEtrue_[i]->Write();
      
      h2_chi2_[i]->Write();
      h2_iterations_[i]->Write();
      
      //      h_DiffZMassDistr_[i]->Write();
      
      //h_ZMassDistr_[i]->Write();
    }
  }

  h2_fEtaBarrelGood_->Write();
  h2_fEtaBarrelBad_->Write();
  h2_fEtaEndcapGood_->Write();
  h2_fEtaEndcapBad_->Write();
  h1_eleClasses_->Write();

  h_eleEffEta_[0]->Write();
  h_eleEffPhi_[0]->Write();
  h_eleEffPt_[0]->Write();
  
  h_eleEffEta_[1]->Write();
  h_eleEffPhi_[1]->Write();
  h_eleEffPt_[1]->Write();

  
  int j = 0;

  int flag=0;
  
  Double_t mean[25] = {0.};
  Double_t num[25] = {0.};
  Double_t meanErr[25] = {0.};
  Float_t rms[25] = {0.};
  Float_t tempRms[10][25];
  
  for(int ia = 0; ia<10; ia++){
    for(int ib = 0; ib<25; ib++){
  
      tempRms[ia][ib] = 0.;

    }
  }
    
  int aa = 0;
      
  for( int k = 0; k<theAlgorithm_->getNumberOfChannels(); k++ )
    {
 

      
      //////grouped
      bool isNearCrack = false;
      
      if( calibMode_ == "RING"){
	
	isNearCrack = ( abs( ringNumberCorrector(k) ) == 1 || abs( ringNumberCorrector(k) ) == 25 ||
			abs( ringNumberCorrector(k) ) == 26 || abs( ringNumberCorrector(k) ) == 45 ||
			abs( ringNumberCorrector(k) ) == 46 || abs( ringNumberCorrector(k) ) == 65 ||
			abs( ringNumberCorrector(k) ) == 66 || abs( ringNumberCorrector(k) ) == 85 ||
			abs( ringNumberCorrector(k) ) == 86 || abs( ringNumberCorrector(k) ) == 124 );
      }
      
      if(k<85)
	{
	  
	  if((k+1)%5!=0)
	    {
	      
	      if(!isNearCrack){
		mean[j]+=calibCoeff[k];
		mean[j]+=calibCoeff[169 - k];
		
		num[j] += 2.;
		
		//meanErr[j]+= calibCoeffError[k];
		//meanErr[j]+= calibCoeffError[169 - k];
		
		meanErr[j]+= 1./ pow ( calibCoeffError[k], 2 );
		meanErr[j]+= 1./ pow ( calibCoeffError[169 - k], 2);


	      tempRms[aa][j]+=calibCoeff[k];
		aa++;
		tempRms[aa][j]+=calibCoeff[169 - k];
		aa++;
	      }
	    }
	  
	  else 	{
	    if(!isNearCrack){
	      mean[j]+=calibCoeff[k];
	      mean[j]+=calibCoeff[169 - k];
	      
	      num[j] += 2.;
	      
	      //meanErr[j]+= calibCoeffError[k];
	      //meanErr[j]+= calibCoeffError[169 - k];
	      
	      meanErr[j]+= 1./ pow ( calibCoeffError[k], 2 );
	      meanErr[j]+= 1./ pow ( calibCoeffError[169 - k], 2);

	      tempRms[aa][j]+=calibCoeff[k];
	      aa++;
	      tempRms[aa][j]+=calibCoeff[169 - k];
	      aa++;

	    }
	    j++;
	    aa = 0;
	    
	  }
	  
	}
      //EE begin
      
      
      if(k>=170 && k<=204){
	
	if(flag<4){
	  //make groups of 5 Xtals in #eta
	  mean[j]+=calibCoeff[k]/10.;
	  mean[j]+=calibCoeff[k+39]/10.;
	
	  meanErr[j]+= calibCoeffError[k]/30.;
          meanErr[j]+= calibCoeffError[k + 39]/30.;

	  
	  tempRms[aa][j]+=calibCoeff[k];
	  aa++;
	  tempRms[aa][j]+=calibCoeff[k + 39];
	  aa++;
	  
	  flag++;
	}

	else if(flag==4){
	  //make groups of 5 Xtals in #eta
	  mean[j]+=calibCoeff[k]/10.;
	  mean[j]+=calibCoeff[k+39]/10.;

	  meanErr[j]+= calibCoeffError[k]/30.;
	  meanErr[j]+= calibCoeffError[k + 39]/30.;
	  
	  tempRms[aa][j]+=calibCoeff[k];
	  aa++;
	  tempRms[aa][j]+=calibCoeff[k + 39];
	  aa++;
	  
	  flag=0;
	  //	  std::cout<<" index(>85) "<<k<<" j is "<<j<<" mean[j] is "<<mean[j]<<std::endl;
	  j++;
	  aa = 0;

	}
	
      }
      if(k>=205 && k<=208){
	mean[j]+=calibCoeff[k]/8.;
	mean[j]+=calibCoeff[k+39]/8.;

	meanErr[j]+= calibCoeffError[k]/30.;
	meanErr[j]+= calibCoeffError[k + 39]/30.;
	
	
	tempRms[aa][j]+=calibCoeff[k];
	aa++;
	tempRms[aa][j]+=calibCoeff[k + 39];
	aa++;
      }
      //EE end

      /*
      for(int jj =0; jj< 25; jj++){ 
      if(meanErr[jj] > 0.)
	std::cout<<" meanErr[jj] before sqrt: "<<meanErr[jj]<<std::endl;

	meanErr[jj] = 1./sqrt( meanErr[jj] );

	std::cout<<" meanErr[jj] after sqrt: "<<meanErr[jj]<<std::endl;
      }
      */
      
      
      
      if(!isNearCrack){
	h2_coeffVsEta_->Fill( ringNumberCorrector(k), calibCoeff[k] );
	h2_miscalRecal_->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
	h1_mc_->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
	
	
	
	if(k<170){
	  h2_miscalRecalEB_->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
	  h1_mcEB_->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
	}
	
	if(k>=170){
	  h2_miscalRecalEE_->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
	  h1_mcEE_->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
	}    
	
      }
    }
  
  for(int ic = 0; ic< 17; ic++){

    mean[ic] = mean[ic] / num[ic]; //find mean of recalib coeff on group of rings
    //meanErr[ic] = meanErr[ic] / ( sqrt( num[ic] ) * num[ic] ); //find mean of recalib coeff on group of rings
    meanErr[ic] = 1. / sqrt(meanErr[ic]); //find mean of recalib coeff on group of rings
    
  }


  //build array of RMS
  for(int ic = 0; ic< 25; ic++){
    for(int id = 0; id< 10; id++){

      if(tempRms[id][ic] > 0.){
	
	rms[ic] += (tempRms[id][ic] - mean[j])*(tempRms[id][ic] - mean[j]);
	
      }
    }
    rms[ic]/= 10.;//this is approximate
    rms[ic] = sqrt(rms[ic]);
  }
  
  //build array of RMS
  
  
  
  Double_t xtalEta[25] = {1.4425, 1.3567,1.2711,1.1855,
			 1.10,1.01,0.92,0.83,
			 0.7468,0.6612,0.5756,0.4897,0.3985,0.3117,0.2250,0.1384,0.0487,
			 1.546, 1.651, 1.771, 1.908, 2.071, 2.267, 2.516, 2.8};
  
  Double_t zero[25] = {0.026};//interval/sqrt(12)

  for(int j = 0; j <25; j++)
    h2_coeffVsEtaGrouped_->Fill( xtalEta[j],mean[j]);
  
  //  for(int sho = 0; sho <25; sho++)
  //cout<<"xtalEta[j] "<< xtalEta[sho]<<" mean[j]  "<<mean[sho]<<"  err[j] "<<meanErr[sho]<<std::endl;

  TProfile *px = h2_coeffVsEta_->ProfileX("coeffVsEtaProfile");
  px->SetXTitle("Eta channel");
  px->SetYTitle("recalibCoeff");
  px->Write();

  h2_coeffVsEta_->Write();
  h2_coeffVsEtaGrouped_->Write();
  h2_zMassVsLoop_->Write();
  h2_zMassDiffVsLoop_->Write();
  h2_zWidthVsLoop_->Write();
  h2_coeffVsLoop_->Write();
  h2_miscalRecal_->Write();
  h1_mc_->Write();
  h2_miscalRecalEB_->Write();
  h1_mcEB_->Write();
  h2_miscalRecalEE_->Write();
  h1_mcEE_->Write();

  h2_residualSigma_->Write();

  const ZIterativeAlgorithmWithFit::ZIterativeAlgorithmWithFitPlots* algoHistos=theAlgorithm_->getHistos();

  double weightSumMeanBarrel = 0.;
  double weightSumMeanEndcap = 0.;

  for (int iIteration=0;iIteration<theAlgorithm_->getNumberOfIterations();iIteration++)
    for (int iChannel=0;iChannel<theAlgorithm_->getNumberOfChannels();iChannel++)
      {

	if( iIteration==(theAlgorithm_->getNumberOfIterations()-1) ){
	  
	  if(iChannel < 170)
	    weightSumMeanBarrel += algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral()/170.; 

	  if(iChannel >= 170)
	    weightSumMeanEndcap += algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral()/78.; 
	  
	  h1_occupancyVsEta_->Fill((Double_t)ringNumberCorrector(iChannel), algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral() );
	  
	  
	  h1_occupancy_->Fill( algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral() );
	  
	  if(iChannel < 170)
	    h1_occupancyBarrel_->Fill( algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral() );

	  if(iChannel >= 170)
	    h1_occupancyEndcap_->Fill( algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral() );

#ifdef DEBUG
	  std::cout<<"Writing weighted integral for channel "<<ringNumberCorrector(iChannel)<<" ,value "<<algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral()<<std::endl;
#endif

	}
	
      }
  
  //  std::cout<<"Done! Closing output file... "<<std::endl;

  h1_weightSumMeanBarrel_ ->Fill(weightSumMeanBarrel);
  h1_weightSumMeanEndcap_ ->Fill(weightSumMeanEndcap);

  std::cout<<"Weight sum mean on channels in Barrel is :"<<weightSumMeanBarrel<<std::endl;
  std::cout<<"Weight sum mean on channels in Endcap is :"<<weightSumMeanEndcap<<std::endl;

  h1_weightSumMeanBarrel_ ->Write();
  h1_weightSumMeanEndcap_ ->Write();

  h1_occupancyVsEta_->Write();
   h1_occupancy_->Write();
  h1_occupancyBarrel_->Write();
  h1_occupancyEndcap_->Write();

  myTree->Write();

  TGraphErrors* graph = new TGraphErrors(25,xtalEta,mean,zero,meanErr);
  graph->Draw("APL");
  graph->Write();

  double zero50[50] = { 0. };

  TGraphErrors* residualSigmaGraph = new TGraphErrors(50,loopArray,sigmaArray,zero50,sigmaErrorArray);
  residualSigmaGraph->SetName("residualSigmaGraph");
  residualSigmaGraph->Draw("APL");
  residualSigmaGraph->Write();

  TGraphErrors* coefficientDistanceAtIterationGraph = new TGraphErrors(50,loopArray,coefficientDistanceAtIteration,zero50,zero50);
  coefficientDistanceAtIterationGraph->SetName("coefficientDistanceAtIterationGraph");
  coefficientDistanceAtIterationGraph->Draw("APL");
  coefficientDistanceAtIterationGraph->Write();

  Float_t noError[250] = {0.};

  Float_t ringInd[250];
  for(int i =0; i<250; i++)
    ringInd[i]=ringNumberCorrector(i);

  TGraphErrors* graphCoeff = new TGraphErrors(theAlgorithm_->getNumberOfChannels(),ringInd,calibCoeff,noError,calibCoeffError);
  graphCoeff->SetName("graphCoeff");
  graphCoeff->Draw("APL");
  graphCoeff->Write();
  
  //   outputFile_->Write();//this automatically writes all histos on file
 

  h1_ZCandMult_->Write();
  h1_reco_ZMass_->Write();
  
  h1_reco_ZMassCorr_->Write();
  h1_reco_ZMassCorrBB_->Write();
  h1_reco_ZMassCorrEE_->Write();

  outputFile_->Close();
  
  myZeePlots_ ->writeEleHistograms();
  myZeePlots_ ->writeMCEleHistograms();
  myZeePlots_ ->writeZHistograms();
  myZeePlots_ ->writeMCZHistograms();
  
  // myZeeRescaleFactorPlots_ = new ZeeRescaleFactorPlots("zeeRescaleFactorPlots.root");
  //myZeeRescaleFactorPlots_->writeHistograms( theAlgorithm_ );
  
  //  delete myZeeRescaleFactorPlots_;
  
  
  
}

//_____________________________________________________________________________
// Called at each event
//________________________________________

edm::EDLooper::Status
ZeeCalibration::duringLoop( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;

#ifdef DEBUG
  std::cout<<"[ZeeCalibration] Entering duringLoop"<<std::endl;
#endif
 
  
  // code that used to be in beginJob
  if (isfirstcall_){

    //inizializzare la geometria di ecal
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);     
    EcalRingCalibrationTools::setCaloGeometry(&(*pG));  
     
    myZeePlots_ = new ZeePlots( "zeePlots.root" );
    //  myZeeRescaleFactorPlots_ = new ZeeRescaleFactorPlots("zeeRescaleFactorPlots.root");

    // go to *OUR* rootfile and book histograms                                                                                                                
    outputFile_->cd();
    bookHistograms();

    std::cout<<"[ZeeCalibration::beginOfJob] Histograms booked "<<std::endl;

    loopFlag_ = 0;

    //Read miscalibration map if requested
    CaloMiscalibMapEcal* miscalibMap=0;
    if (!barrelfile_.empty() || !barrelfile_.empty())
      {
	miscalibMap=new CaloMiscalibMapEcal();
	miscalibMap->prefillMap();
      }


    if(!barrelfile_.empty())
      {
	MiscalibReaderFromXMLEcalBarrel barrelreader_(*miscalibMap);
	barrelreader_.parseXMLMiscalibFile(barrelfile_);
#ifdef DEBUG
	std::cout<<"[ZeeCalibration::beginOfJob] Parsed EB miscal file"<<std::endl;
#endif
      }

    if(!endcapfile_.empty())
      {
	MiscalibReaderFromXMLEcalEndcap endcapreader_(*miscalibMap);
	endcapreader_.parseXMLMiscalibFile(endcapfile_);
#ifdef DEBUG
	std::cout<<"[ZeeCalibration::beginOfJob] Parsed EE miscal file"<<std::endl;
#endif
      }

    std::cout << "  theAlgorithm_->getNumberOfChannels() "
	      << theAlgorithm_->getNumberOfChannels() << std::endl;


    ////////////////////set miscalibration  
    for(int k = 0; k < theAlgorithm_->getNumberOfChannels(); k++)
      {
	calibCoeff[k]=1.;
	calibCoeffError[k]=0.;
     
	std::vector<DetId> ringIds;

	if(calibMode_ == "RING")
	  ringIds = EcalRingCalibrationTools::getDetIdsInRing(k);

	if(calibMode_ == "MODULE")
	  ringIds = EcalRingCalibrationTools::getDetIdsInModule(k);

	if(calibMode_ == "ABS_SCALE" || calibMode_ == "ETA_ET_MODE" )
	  ringIds = EcalRingCalibrationTools::getDetIdsInECAL();
      
	if (miscalibMap)
	  {
	    initCalibCoeff[k]=0.;	      
	    for (unsigned int iid=0; iid<ringIds.size();++iid)
	      {
		float miscalib=* (miscalibMap->get().getMap().find(ringIds[iid])  );
		//	      float miscalib=miscalibMap->get().getMap().find(ringIds[iid])->second; ////////AP
		initCalibCoeff[k]+=miscalib;
	      }
	    initCalibCoeff[k]/=(float)ringIds.size();
	    std::cout << k << " " << initCalibCoeff[k] << " " << ringIds.size() << std::endl;
	  }
	else
	  {
	    initCalibCoeff[k]=1.;
	  }
      }

    ical = boost::shared_ptr<EcalIntercalibConstants>( new EcalIntercalibConstants() );
  
    for(int k = 0; k < theAlgorithm_->getNumberOfChannels(); k++)
      {
	//      std::vector<DetId> ringIds = EcalRingCalibrationTools::getDetIdsInRing(k);

	std::vector<DetId> ringIds;

	if(calibMode_ == "RING")
	  ringIds = EcalRingCalibrationTools::getDetIdsInRing(k);

	if(calibMode_ == "MODULE")
	  ringIds = EcalRingCalibrationTools::getDetIdsInModule(k);

	if(calibMode_ == "ABS_SCALE" || calibMode_ == "ETA_ET_MODE")
	  ringIds = EcalRingCalibrationTools::getDetIdsInECAL();
      
      
	for (unsigned int iid=0; iid<ringIds.size();++iid){
	  //	ical->setValue( ringIds[iid], 1. * initCalibCoeff[k] );
	
	  if(ringIds[iid].subdetId() == EcalBarrel){
	    EBDetId myEBDetId(ringIds[iid]);  
	    h2_xtalMiscalibCoeffBarrel_->SetBinContent( myEBDetId.ieta() + 86, myEBDetId.iphi(), * (miscalibMap->get().getMap().find(ringIds[iid]) ) );//fill TH2 with miscalibCoeff
	 
	  }

	  if(ringIds[iid].subdetId() == EcalEndcap){
	    EEDetId myEEDetId(ringIds[iid]);
	    if(myEEDetId.zside() < 0)
	      h2_xtalMiscalibCoeffEndcapMinus_->SetBinContent( myEEDetId.ix(), myEEDetId.iy(), * ( miscalibMap->get().getMap().find(ringIds[iid]) ) );//fill TH2 with miscalibCoeff

	    if(myEEDetId.zside() > 0)
	      h2_xtalMiscalibCoeffEndcapPlus_->SetBinContent( myEEDetId.ix(), myEEDetId.iy(), * (miscalibMap->get().getMap().find(ringIds[iid]) ) );//fill TH2 with miscalibCoeff
	  
	  }
	
	  ical->setValue( ringIds[iid], *(miscalibMap->get().getMap().find(ringIds[iid])  ) );

	}

  
	read_events = 0;
	init_ = false;


      }
    isfirstcall_=false;
  }// if isfirstcall

  







  ////////////////////////////////////////////////////////////////////////////HLT begin
  
  for(unsigned int iHLT=0; iHLT<200; ++iHLT) {
    aHLTResults[iHLT] = false;
  }

#ifdef DEBUG
  std::cout<<"[ZeeCalibration::duringLoop] Done with initializing aHLTresults[] "<<std::endl;
#endif

  edm::Handle<edm::TriggerResults> hltTriggerResultHandle;
  iEvent.getByLabel(hlTriggerResults_, hltTriggerResultHandle);
  
  if(!hltTriggerResultHandle.isValid()) {
    //std::cout << "invalid handle for HLT TriggerResults" << std::endl;
  } else {

    hltCount = hltTriggerResultHandle->size();

    if (loopFlag_ == 0)
      myZeePlots_->fillHLTInfo(hltTriggerResultHandle);
    
#ifdef DEBUG
  std::cout<<"[ZeeCalibration::duringLoop] Done with myZeePlots_->fillHLTInfo(hltTriggerResultHandle); "<<std::endl;
#endif

    for(int i = 0 ; i < hltCount ; i++) {
      aHLTResults[i] = hltTriggerResultHandle->accept(i);
    
      //HLT bit 32 = HLT1Electron
    //HLT bit 34 = HLT2Electron
    //HLT bit 35 = HLT2ElectronRelaxed

     }

      if(!aHLTResults[32] && !aHLTResults[34] && !aHLTResults[35])
	return kContinue;
      
  }
  
#ifdef DEBUG
  std::cout<<"[ZeeCalibration::duringLoop] End HLT section"<<std::endl;
#endif
  
  //////////////////////////////////////////////////////////////////////HLT end
  

  std::vector<HepMC::GenParticle*> mcEle;

  float myGenZMass(-1);
      
  if (!mcProducer_.empty())
    {

      //DUMP GENERATED Z MASS - BEGIN
      Handle< HepMCProduct > hepProd ;
      //   iEvent.getByLabel( "source", hepProd ) ;
      iEvent.getByLabel( mcProducer_.c_str(), hepProd ) ;
                                                                                                                             
      const HepMC::GenEvent * myGenEvent = hepProd->GetEvent();
      
      if (loopFlag_ == 0)
	myZeePlots_->fillZMCInfo( & (*myGenEvent) );
      
#ifdef DEBUG
  std::cout<<"[ZeeCalibration::duringLoop] Done with myZeePlots_->fillZMCInfo( & (*myGenEvent) ); "<<std::endl;
#endif
  
      
      for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
            p != myGenEvent->particles_end(); ++p ) {
        //return a pointer to MC Z in the event
        if ( (*p)->pdg_id() == 23 && (*p)->status()==2){

	  myGenZMass = (*p)->momentum().m();
        }
      }
      //DUMP GENERATED Z MASS - END
     

      if (loopFlag_ == 0)
	myZeePlots_ ->fillEleMCInfo( & (*myGenEvent) );
      
            
      //loop over MC positrons and find the closest MC positron in (eta,phi) phace space - begin
      HepMC::GenParticle MCele;
      
      for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	    p != myGenEvent->particles_end(); ++p ) {
	
	if (  abs( (*p)->pdg_id() ) == 11 )
	  {
	    mcEle.push_back( (*p) );
	    MCele=*(*p);
	    
	  }
      }
      
      
      if(mcEle.size()==2 && fabs(mcEle[0]->momentum().eta())<2.4 &&  fabs(mcEle[1]->momentum().eta())<2.4 ){
	NEVT++;
	
	if( fabs(mcEle[0]->momentum().eta())<1.479 && fabs(mcEle[1]->momentum().eta())<1.479 )MCZBB++;
	
	if( (fabs(mcEle[0]->momentum().eta())>1.479 && fabs(mcEle[1]->momentum().eta())<1.479) || (fabs(mcEle[0]->momentum().eta())<1.479 && fabs(mcEle[1]->momentum().eta())>1.479) )MCZEB++;
	
	if( fabs(mcEle[0]->momentum().eta())>1.479 && fabs(mcEle[1]->momentum().eta())>1.479 )MCZEE++;
	
	
      }
      
    }    
  
  
  
  // Get EBRecHits
  Handle<EBRecHitCollection> phits;
  try {
    iEvent.getByLabel( rechitProducer_, rechitCollection_, phits);
  } catch (std::exception& ex) {
    std::cerr << "Error! can't get the product EBRecHitCollection " << std::endl;
  }
  const EBRecHitCollection* hits = phits.product(); // get a ptr to the product

  // Get EERecHits
  Handle<EERecHitCollection> ephits;
  try {
    iEvent.getByLabel( erechitProducer_, erechitCollection_, ephits);
  } catch (std::exception& ex) {
    std::cerr << "Error! can't get the product EERecHitCollection " << std::endl;
  }
  const EERecHitCollection* ehits = ephits.product(); // get a ptr to the product

  
  //Get Hybrid SuperClusters
  Handle<reco::SuperClusterCollection> pSuperClusters;
  try {
    iEvent.getByLabel(scProducer_, scCollection_, pSuperClusters);
  } catch (std::exception& ex ) {
    std::cerr << "Error! can't get the product SuperClusterCollection "<< std::endl;
  }
  const reco::SuperClusterCollection* scCollection = pSuperClusters.product();

#ifdef DEBUG
  std::cout<<"scCollection->size()"<<scCollection->size()<<std::endl;
  for(reco::SuperClusterCollection::const_iterator scIt = scCollection->begin();   scIt != scCollection->end(); scIt++)
    {
      std::cout<<scIt->energy()<<std::endl;
    }
#endif
  
  //Get Island SuperClusters
  Handle<reco::SuperClusterCollection> pIslandSuperClusters;
  try {
    iEvent.getByLabel(scIslandProducer_, scIslandCollection_, pIslandSuperClusters);
  } catch (std::exception& ex ) {
    std::cerr << "Error! can't get the product IslandSuperClusterCollection "<< std::endl;
  }
  const reco::SuperClusterCollection* scIslandCollection = pIslandSuperClusters.product();

#ifdef DEBUG
  std::cout<<"scCollection->size()"<<scIslandCollection->size()<<std::endl;
#endif

  if(  ( scCollection->size()+scIslandCollection->size() ) < 2) 
    return kContinue;

  // Get Electrons
  Handle<reco::GsfElectronCollection> pElectrons;
  try {
    iEvent.getByLabel(electronProducer_, electronCollection_, pElectrons);
  } catch (std::exception& ex ) {
    std::cerr << "Error! can't get the product ElectronCollection "<< std::endl;
  }
  const reco::GsfElectronCollection* electronCollection = pElectrons.product();

  /*
  //reco-mc association map
  std::map<HepMC::GenParticle*,const reco::PixelMatchGsfElectron*> myMCmap;
  
    fillMCmap(&(*electronCollection),mcEle,myMCmap);
    
    fillEleInfo(mcEle,myMCmap);
  */
  
  if(electronCollection->size() < 2) 
    return kContinue;
  
  if ( !hits && !ehits){
    std::cout << "!hits" << std::endl;   
    return kContinue;
  }
  
  if (hits->size() == 0 && ehits->size() == 0){
    std::cout << "hits->size() == 0" << std::endl;   
    return kContinue;
  }  
  
  if (!electronCollection){
    std::cout << "!electronCollection" << std::endl;
    return kContinue;
  }
  
  if (electronCollection->size() == 0){
    std::cout << "electronCollection->size() == 0" << std::endl;
    return kContinue;
  }


  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///                          START HERE....
  ///////////////////////////////////////////////////////////////////////////////////////
  
  read_events++;
  
  //  std::cout << "read_events = " << read_events << std::endl;
  
  ////////////////////////CHOOSE EVENTS WITH 2 OR MORE ELECTRONS ////////////////////
  
#ifdef DEBUG
  std::cout <<" Starting with myZeePlots_->fillEleInfo(electronCollection); " << std::endl; 
#endif

  if (loopFlag_ == 0)
    myZeePlots_->fillEleInfo(electronCollection);
  
#ifdef DEBUG
  std::cout <<" Done with myZeePlots_->fillEleInfo(electronCollection); " << std::endl; 
#endif

  //FILL an electron vector - end
  //###################################Electron-SC association: begin#####################################################
  //Filling new ElectronCollection with new SC ref and calibElectron container
  std::vector<calib::CalibElectron> calibElectrons;
  //std::map< const calib::CalibElectron* , const reco::SuperCluster* > eleScMap;
  
  

  //#####################################Electron-SC association map: end#####################################################  
  for(unsigned int e_it = 0 ; e_it != electronCollection->size() ; e_it++)
    {
      calibElectrons.push_back(calib::CalibElectron(&((*electronCollection)[e_it]),hits,ehits));
#ifdef DEBUG
      std::cout << calibElectrons.back().getRecoElectron()->superCluster()->energy() << " " << calibElectrons.back().getRecoElectron()->energy() << std::endl;
#endif
      //      h1_recoEleEnergy_->Fill(calibElectrons.back().getRecoElectron()->superCluster()->energy());
    }
  //  if (iLoop == 0)
  //fillCalibElectrons(calibElectrons);

#ifdef DEBUG
  std::cout << "Filled histos" << std::endl;
#endif  
  
  //COMBINATORY FOR Z MASS - begin                                                                                                                           
  std::vector<std::pair<calib::CalibElectron*,calib::CalibElectron*> > zeeCandidates;
  int  myBestZ=-1;
  
  mass = -1.;
  double DeltaMinvMin(5000.);
  
  if (calibElectrons.size() < 2)
    return kContinue;

  for(unsigned int e_it = 0 ; e_it != calibElectrons.size() - 1 ; e_it++){
    for(unsigned int p_it = e_it + 1 ; p_it != calibElectrons.size() ; p_it++)
      {
#ifdef DEBUG
	std::cout << e_it << " " << calibElectrons[e_it].getRecoElectron()->charge() << " " << p_it << " " << calibElectrons[p_it].getRecoElectron()->charge() << std::endl;
#endif		
	if (calibElectrons[e_it].getRecoElectron()->charge() * calibElectrons[p_it].getRecoElectron()->charge() != -1)
	  continue;
	
	mass =  ZeeKinematicTools::calculateZMass_withTK(std::pair<calib::CalibElectron*,calib::CalibElectron*>(&(calibElectrons[e_it]),&(calibElectrons[p_it])));
	
	if (mass<0)
	  continue;
	
#ifdef DEBUG
	std::cout << "#######################mass "<<mass << std::endl;
#endif
	
	zeeCandidates.push_back(std::pair<calib::CalibElectron*,calib::CalibElectron*>(&(calibElectrons[e_it]),&(calibElectrons[p_it])));
	double DeltaMinv = fabs(mass - MZ); 
	
	if( DeltaMinv < DeltaMinvMin)
	  {
	    DeltaMinvMin = DeltaMinv;
	    myBestZ=zeeCandidates.size()-1;
	  }
      }
  }      
  
  //  h_DeltaZMassDistr_[loopFlag_]->Fill( (mass-MZ) / MZ );

  //  zeeCa->Fill(zeeCandidates);
  //
  h1_ZCandMult_->Fill(zeeCandidates.size());
  
  if(zeeCandidates.size()==0 || myBestZ==-1 )
    return kContinue;
      
  if (loopFlag_ == 0)
    myZeePlots_->fillZInfo( zeeCandidates[myBestZ] );
  
#ifdef DEBUG  
  std::cout << "Found ZCandidates " << myBestZ << std::endl;
#endif  

  //  h1_zMassResol_ ->Fill(mass-myGenZMass);

  /////////////////////////////DUMP ELECTRON CLASS
  
  
  h1_eleClasses_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->classification());
  h1_eleClasses_->Fill(zeeCandidates[myBestZ].second->getRecoElectron()->classification());
  
  int class1 = zeeCandidates[myBestZ].first->getRecoElectron()->classification();
  int class2 = zeeCandidates[myBestZ].second->getRecoElectron()->classification();

  std::cout << "BEFORE "<<std::endl;

  //  myZeePlots_->fillEleClassesPlots( zeeCandidates[myBestZ].first );
  //myZeePlots_->fillEleClassesPlots( zeeCandidates[myBestZ].second );
  
  std::cout << "AFTER "<<std::endl;

  ///////////////////////ELECTRON CLASS STATISTICS

  if(class1 < 100)
    //    h1_Elec_->Fill(1);
    TOTAL_ELECTRONS_IN_BARREL++;
  if(class1 >= 100)
    TOTAL_ELECTRONS_IN_ENDCAP++;

  if(class2 < 100)
    TOTAL_ELECTRONS_IN_BARREL++;
  if(class2 >= 100)
    TOTAL_ELECTRONS_IN_ENDCAP++;


  if( class1==0)
    GOLDEN_ELECTRONS_IN_BARREL++;
  if( class1==100)
    GOLDEN_ELECTRONS_IN_ENDCAP++;
  if( class1==10 || class1 ==20)
    SILVER_ELECTRONS_IN_BARREL++;
  if( class1==110 || class1 ==120)
    SILVER_ELECTRONS_IN_ENDCAP++;
  if( class1>=30 && class1 <=34)
    SHOWER_ELECTRONS_IN_BARREL++;
  if( class1>=130 && class1 <=134)
    SHOWER_ELECTRONS_IN_ENDCAP++;
  if( class1==40)
    CRACK_ELECTRONS_IN_BARREL++;
  if( class1==140)
    CRACK_ELECTRONS_IN_ENDCAP++;

  if( class2==0)
    GOLDEN_ELECTRONS_IN_BARREL++;
  if( class2==100)
    GOLDEN_ELECTRONS_IN_ENDCAP++;
  if( class2==10 || class2 ==20)
    SILVER_ELECTRONS_IN_BARREL++;
  if( class2==110 || class2 ==120)
    SILVER_ELECTRONS_IN_ENDCAP++;
  if( class2>=30 && class2 <=34)
    SHOWER_ELECTRONS_IN_BARREL++;
  if( class2>=130 && class2 <=134)
    SHOWER_ELECTRONS_IN_ENDCAP++;
  if( class2==40)
    CRACK_ELECTRONS_IN_BARREL++;
  if( class2==140)
    CRACK_ELECTRONS_IN_ENDCAP++;
  
  /////////////////////////////////

 ///////////////////////////////////////EXCLUDE ELECTRONS HAVING HOTTEST XTAL WHICH IS A BORDER XTAL - begin

  
  DetId firstElehottestDetId = getHottestDetId( zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->seed()->hitsAndFractions() , hits, ehits ).first;
  DetId secondElehottestDetId = getHottestDetId( zeeCandidates[myBestZ].second->getRecoElectron()->superCluster()->seed()->hitsAndFractions()  , hits, ehits ).first;
  
  bool firstElectronIsOnModuleBorder(false);
  bool secondElectronIsOnModuleBorder(false);
  
  h1_eventsBeforeBorderSelection_->Fill(1);

  if(class1<100){

    if( firstElehottestDetId.subdetId() == EcalBarrel)
      firstElectronIsOnModuleBorder = xtalIsOnModuleBorder( firstElehottestDetId  );
    
    BARREL_ELECTRONS_BEFORE_BORDER_CUT++;
    
    if( firstElehottestDetId.subdetId() == EcalBarrel &&  !firstElectronIsOnModuleBorder )
      BARREL_ELECTRONS_AFTER_BORDER_CUT++;
    
  }
  
  if(class2<100){
    
    if( secondElehottestDetId.subdetId() == EcalBarrel)
      secondElectronIsOnModuleBorder = xtalIsOnModuleBorder( secondElehottestDetId  );
    
    BARREL_ELECTRONS_BEFORE_BORDER_CUT++;
    
    if( secondElehottestDetId.subdetId() == EcalBarrel &&  !secondElectronIsOnModuleBorder )
      BARREL_ELECTRONS_AFTER_BORDER_CUT++;
    
  }
  
  
  if(class1<100){
    if ( firstElehottestDetId.subdetId() == EcalBarrel &&  firstElectronIsOnModuleBorder ){
      h1_borderElectronClassification_ -> Fill( zeeCandidates[myBestZ].first->getRecoElectron()->classification() );
      return kContinue;
    }  
  }
  
  if(class2<100){
    if ( secondElehottestDetId.subdetId() == EcalBarrel &&  secondElectronIsOnModuleBorder ){ 
      h1_borderElectronClassification_ -> Fill( zeeCandidates[myBestZ].second->getRecoElectron()->classification() );
      return kContinue;
    }
  }
  
  h1_eventsAfterBorderSelection_->Fill(1);
  ///////////////////////////////////////EXCLUDE ELECTRONS HAVING HOTTEST XTAL WHICH IS A BORDER XTAL - end
  
  
  if(class1<100 && class2<100){
    BBZN++;
    if(class1==0 && class2==0)BBZN_gg++;
    if(class1<21 && class2<21)BBZN_tt++;
    if(class1<21 || class2<21)BBZN_t0++;
    
  }

  if(class1>=100 && class2>=100){
    EEZN++;
    if(class1==100 && class2==100)EEZN_gg++;
    if(class1<121 && class2<121)EEZN_tt++;
    if(class1<121 || class2<121)EEZN_t0++;

  }

  if( (class1<100 && class2>=100) || (class2<100 && class1>=100)){
    EBZN++;
    if( (class1==0 && class2==100)||(class2==0 && class1==100) )EBZN_gg++;
    if( ( class1<21 && class2<121) ||(class2<21 && class1<121) )EBZN_tt++;
    if(   class2<21 || class1<21 ||  class2<121 || class1<121 )EBZN_t0++;
  }


  ///////////////////////////ELECTRON SELECTION///////////////////////////////
  
  if(myBestZ == -1)
    return kContinue;


  bool invMassBool = ( (mass > minInvMassCut_) && (mass < maxInvMassCut_) );

  bool selectionBool=false;  
  //0 = all electrons (but no crack)
  
  ////////EWK selection - begin

  float theta1 = 2. * atan( exp(- zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta()) );
  bool ET_1 = (  (zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->energy() * sin( theta1) ) > 20.);

  float theta2 = 2. * atan( exp(- zeeCandidates[myBestZ].second->getRecoElectron()->superCluster()->eta()) );
  bool ET_2 = (  (zeeCandidates[myBestZ].second->getRecoElectron()->superCluster()->energy() * sin( theta2) ) > 20.);


  bool HoE_1 = (zeeCandidates[myBestZ].first->getRecoElectron()->hadronicOverEm() < 0.115);
  bool HoE_2 = (zeeCandidates[myBestZ].second->getRecoElectron()->hadronicOverEm() < 0.115);

  bool DeltaPhiIn_1 = ( zeeCandidates[myBestZ].first->getRecoElectron()->deltaPhiSuperClusterTrackAtVtx() < 0.090);
  bool DeltaPhiIn_2 = ( zeeCandidates[myBestZ].second->getRecoElectron()->deltaPhiSuperClusterTrackAtVtx() < 0.090);

  bool DeltaEtaIn_1 = ( zeeCandidates[myBestZ].first->getRecoElectron()->deltaEtaSuperClusterTrackAtVtx() < 0.0090);
  bool DeltaEtaIn_2 = ( zeeCandidates[myBestZ].second->getRecoElectron()->deltaEtaSuperClusterTrackAtVtx() < 0.0090);
   
  h1_eventsBeforeEWKSelection_->Fill(1);

  if(! (invMassBool &&
	ET_1 && ET_2 &&
	HoE_1 && HoE_2 &&
	DeltaPhiIn_1 && DeltaPhiIn_2 &&
	DeltaEtaIn_1 && DeltaEtaIn_2
	) )
    return kContinue;
  ////////EWK selection - end

  h1_eventsAfterEWKSelection_->Fill(1);

   ///////////////////////////////////////EXCLUDE ELECTRONS HAVING HOTTEST XTAL WHICH IS A BORDER XTAL
  


  if(electronSelection_==0)selectionBool=( myBestZ != -1 && 
					   zeeCandidates[myBestZ].first->getRecoElectron()->classification()!= 40 && 
					   zeeCandidates[myBestZ].first->getRecoElectron()->classification()!= 40 && 
					   zeeCandidates[myBestZ].second->getRecoElectron()->classification()!= 40 && 
					   zeeCandidates[myBestZ].second->getRecoElectron()->classification()!= 140);
  
  //1 = all electrons are Golden, BB or Narrow
  
  if(electronSelection_==1)selectionBool=( myBestZ != -1 &&
					   (zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==0 || 
					    zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==10 || 
					    zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==20 ||
					    zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==100 ||
                                            zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==110 ||
                                            zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==120
					    ) &&
					   (zeeCandidates[myBestZ].second->getRecoElectron()->classification() == 0 || 
					    zeeCandidates[myBestZ].second->getRecoElectron()->classification() == 10 ||
					    zeeCandidates[myBestZ].second->getRecoElectron()->classification() == 20 ||
					    zeeCandidates[myBestZ].second->getRecoElectron()->classification() == 100 ||
                                            zeeCandidates[myBestZ].second->getRecoElectron()->classification() == 110 ||
                                            zeeCandidates[myBestZ].second->getRecoElectron()->classification() == 120
					    )
					   );
  
  //2 = all electrons are Golden
  if(electronSelection_==2)selectionBool=( myBestZ != -1 &&
					   (zeeCandidates[myBestZ].first->getRecoElectron()->classification() == 0 ||
					    zeeCandidates[myBestZ].first->getRecoElectron()->classification() == 100
					    ) &&
					   (zeeCandidates[myBestZ].second->getRecoElectron()->classification() == 0 ||
					    zeeCandidates[myBestZ].second->getRecoElectron()->classification() == 100
					    ) 
					   );
  //3 = all electrons are showering
  if(electronSelection_==3)selectionBool=( myBestZ != -1 &&
					  (
					   (zeeCandidates[myBestZ].first->getRecoElectron()->classification() >=30  &&
					   zeeCandidates[myBestZ].first->getRecoElectron()->classification() <=34)  
					   ||
					   ((zeeCandidates[myBestZ].first->getRecoElectron()->classification() >=130  &&
					     zeeCandidates[myBestZ].first->getRecoElectron()->classification() <=134))
					   )
					   &&
					   ( (zeeCandidates[myBestZ].second->getRecoElectron()->classification() >=30  &&
					      zeeCandidates[myBestZ].second->getRecoElectron()->classification() <=34)
					     ||
					     ((zeeCandidates[myBestZ].second->getRecoElectron()->classification() >=130  &&
					       zeeCandidates[myBestZ].second->getRecoElectron()->classification() <=134))
					     )
					   
					   );
  
  //4 = all Barrel electrons are Golden, BB or Narrow; take all Endcap electrons
                                                                                                                             
  if(electronSelection_==4)selectionBool=( myBestZ != -1 && 
					   (

					   (
					    (zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==0 ||
					      zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==10 ||
					      zeeCandidates[myBestZ].first->getRecoElectron()->classification() ==20 
					      ) && zeeCandidates[myBestZ].second->getRecoElectron()->classification()>=100 
					    && zeeCandidates[myBestZ].second->getRecoElectron()->classification()!=140
					    )

					   ||
					   
					   (
                                            (zeeCandidates[myBestZ].second->getRecoElectron()->classification() ==0 ||
					     zeeCandidates[myBestZ].second->getRecoElectron()->classification() ==10 ||
					     zeeCandidates[myBestZ].second->getRecoElectron()->classification() ==20
					     ) && zeeCandidates[myBestZ].first->getRecoElectron()->classification()>=100
                                            && zeeCandidates[myBestZ].first->getRecoElectron()->classification()!=140
                                            )


					   )
					   ); 
  
  //5 = all Endcap electrons (but no crack)
  
  if(electronSelection_==5)selectionBool=( myBestZ != -1 && 
					   zeeCandidates[myBestZ].first->getRecoElectron()->classification()>=100 && 
					   zeeCandidates[myBestZ].second->getRecoElectron()->classification()>= 100 && 
					   zeeCandidates[myBestZ].first->getRecoElectron()->classification()!= 140 &&
					   zeeCandidates[myBestZ].second->getRecoElectron()->classification()!= 140);

  //6 = all Barrel electrons (but no crack)
  
  if(electronSelection_==6)selectionBool=( myBestZ != -1 && 
					   zeeCandidates[myBestZ].first->getRecoElectron()->classification()<100 && 
					   zeeCandidates[myBestZ].second->getRecoElectron()->classification()< 100 && 
					   zeeCandidates[myBestZ].first->getRecoElectron()->classification()!= 40 &&
					   zeeCandidates[myBestZ].second->getRecoElectron()->classification()!= 40);

  //7 = this eliminates the events which have 1 ele in the Barrel and 1 in the Endcap
  
  if(electronSelection_==7)selectionBool=( myBestZ != -1 && 
					   !(zeeCandidates[myBestZ].first->getRecoElectron()->classification()<100 && 
					   zeeCandidates[myBestZ].second->getRecoElectron()->classification()>=100) &&
					   !(zeeCandidates[myBestZ].first->getRecoElectron()->classification()>=100 &&
					     zeeCandidates[myBestZ].second->getRecoElectron()->classification()<100) );


      float ele1EnergyCorrection(1.);
      float ele2EnergyCorrection(1.);

	if(invMassBool && selectionBool && wantEtaCorrection_){
	  
	  ele1EnergyCorrection=getEtaCorrection(zeeCandidates[myBestZ].first->getRecoElectron());
	  ele2EnergyCorrection=getEtaCorrection(zeeCandidates[myBestZ].second->getRecoElectron());

	}

  if (invMassBool && selectionBool)  
    {

      h1_electronCosTheta_SC_ -> Fill( ZeeKinematicTools::cosThetaElectrons_SC(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection)  );
      h1_electronCosTheta_TK_ -> Fill( ZeeKinematicTools::cosThetaElectrons_TK(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection)  );
      h1_electronCosTheta_SC_TK_ -> Fill( ZeeKinematicTools::cosThetaElectrons_SC(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection)/ZeeKinematicTools::cosThetaElectrons_TK(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection) - 1. );

        if (!mcProducer_.empty())
	  {
	    h1_zMassResol_ ->Fill(mass-myGenZMass);
	    
	    //reco-mc association map - begin
	    
	    std::map<HepMC::GenParticle*,const reco::GsfElectron*> myMCmap;
	    
	    std::vector<const reco::GsfElectron*> dauElectronCollection;
	    
	    dauElectronCollection.push_back(zeeCandidates[myBestZ].first->getRecoElectron()  );
	    dauElectronCollection.push_back(zeeCandidates[myBestZ].second->getRecoElectron()  );
	    
	    fillMCmap(&dauElectronCollection,mcEle,myMCmap);
	    fillEleInfo(mcEle,myMCmap);
	    //h_DiffZMassDistr_[loopFlag_]->Fill( (mass-myGenZMass) );
	  }

      //PUT f(eta) IN OUR Zee ALGORITHM
      theAlgorithm_->addEvent(zeeCandidates[myBestZ].first, zeeCandidates[myBestZ].second,MZ*sqrt(ele1EnergyCorrection*ele2EnergyCorrection) );
     
      h1_reco_ZMass_->Fill(ZeeKinematicTools::calculateZMass_withTK(zeeCandidates[myBestZ]));

      h1_reco_ZMassCorr_->Fill(ZeeKinematicTools::calculateZMassWithCorrectedElectrons_withTK(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection));

      if(zeeCandidates[myBestZ].first->getRecoElectron()->classification()<100 && zeeCandidates[myBestZ].second->getRecoElectron()->classification()<100 )
	h1_reco_ZMassCorrBB_->Fill(ZeeKinematicTools::calculateZMassWithCorrectedElectrons_withTK(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection));


      if(zeeCandidates[myBestZ].first->getRecoElectron()->classification()>=100 && zeeCandidates[myBestZ].second->getRecoElectron()->classification()>=100 )
	h1_reco_ZMassCorrEE_->Fill(ZeeKinematicTools::calculateZMassWithCorrectedElectrons_withTK(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection));


      mass4tree = ZeeKinematicTools::calculateZMassWithCorrectedElectrons_withTK(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection);

      massDiff4tree = ZeeKinematicTools::calculateZMassWithCorrectedElectrons_withTK(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection) - myGenZMass;

      //      h_ZMassDistr_[loopFlag_]->Fill(ZeeKinematicTools::calculateZMassWithCorrectedElectrons_withTK(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection));

      myTree->Fill();
    
    }



#ifdef DEBUG
  std::cout << "Added event to algorithm" << std::endl;  
#endif

  return kContinue;
    }
//end of ZeeCalibration::duringLoop


// Called at beginning of loop
void ZeeCalibration::startingNewLoop ( unsigned int iLoop )
{

std::cout<< "[ZeeCalibration] Starting loop number " << iLoop<<std::endl;
 
 theAlgorithm_->resetIteration();
 
 resetVariables();
 
 resetHistograms();

#ifdef DEBUG
 std::cout<< "[ZeeCalibration] exiting from startingNewLoop" << std::endl;
#endif

}



// Called at end of loop
edm::EDLooper::Status
ZeeCalibration::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop)
{



  double par[3];
  double errpar[3];
  double zChi2;
  int zIters;

  ZIterativeAlgorithmWithFit::gausfit(h1_reco_ZMass_,par,errpar,2.,2., &zChi2, &zIters );

  h2_zMassVsLoop_ -> Fill(loopFlag_,  par[1] );

  h2_zMassDiffVsLoop_ -> Fill(loopFlag_,  (par[1]-MZ)/MZ );

  h2_zWidthVsLoop_ -> Fill(loopFlag_, par[2] );
 

  //////////////////FIT Z PEAK

  std::cout<< "[ZeeCalibration] Ending loop " << iLoop<<std::endl;
  //RUN the algorithm
  theAlgorithm_->iterate();

  const std::vector<float>& optimizedCoefficients = theAlgorithm_->getOptimizedCoefficients();
  const std::vector<float>& optimizedCoefficientsError = theAlgorithm_->getOptimizedCoefficientsError();
  //const std::vector<float>& weightSum = theAlgorithm_->getWeightSum();
  const std::vector<float>& optimizedChi2 = theAlgorithm_->getOptimizedChiSquare();
  const std::vector<int>& optimizedIterations = theAlgorithm_->getOptimizedIterations();


  //#ifdef DEBUG
  std::cout<< "Optimized coefficients " << optimizedCoefficients.size() <<std::endl;
  //#endif

  //  h2_coeffVsLoop_->Fill(loopFlag_, optimizedCoefficients[75]); //show the evolution of just 1 ring coefficient (well chosen...)

  //////////////define NewCalibCoeff - begin
  for (unsigned int ieta=0;ieta<optimizedCoefficients.size();ieta++)
    {
      NewCalibCoeff[ieta] = calibCoeff[ieta] * optimizedCoefficients[ieta];

      h2_chi2_[loopFlag_]->Fill( ringNumberCorrector( ieta ), optimizedChi2[ieta] );
      h2_iterations_[loopFlag_]->Fill( ringNumberCorrector( ieta ), optimizedIterations[ieta] );
 
    }
  //////////////define NewCalibCoeff - end
  
  
  coefficientDistanceAtIteration[loopFlag_]= computeCoefficientDistanceAtIteration(calibCoeff, NewCalibCoeff, optimizedCoefficients.size() );

  std::cout<<"Iteration # : "<< loopFlag_ << " CoefficientDistanceAtIteration "<< coefficientDistanceAtIteration[loopFlag_] <<std::endl;
  std::cout<<"size "<<optimizedCoefficients.size()<<std::endl;

  for (unsigned int ieta=0;ieta<optimizedCoefficients.size();ieta++)
    {
      calibCoeff[ieta] *= optimizedCoefficients[ieta];
      calibCoeffError[ieta] = calibCoeff[ieta] * sqrt ( pow( optimizedCoefficientsError[ieta]/optimizedCoefficients[ieta], 2 ) + pow( calibCoeffError[ieta]/calibCoeff[ieta] , 2 )  );
      //calibCoeffError[ieta] = optimizedCoefficientsError[ieta];


#ifdef DEBUG
      std::cout<< ieta << " " << optimizedCoefficients[ieta] <<std::endl;  
#endif

      std::vector<DetId> ringIds;

      if(calibMode_ == "RING")
	ringIds = EcalRingCalibrationTools::getDetIdsInRing(ieta);

      if(calibMode_ == "MODULE")
	ringIds = EcalRingCalibrationTools::getDetIdsInModule(ieta);

      if(calibMode_ == "ABS_SCALE" || calibMode_ == "ETA_ET_MODE" )
	ringIds = EcalRingCalibrationTools::getDetIdsInECAL();

      
      for (unsigned int iid=0; iid<ringIds.size();++iid){
	
	if(ringIds[iid].subdetId() == EcalBarrel){
	  EBDetId myEBDetId(ringIds[iid]);  
	  h2_xtalRecalibCoeffBarrel_[loopFlag_]->SetBinContent( myEBDetId.ieta() + 86, myEBDetId.iphi(), 100 * (calibCoeff[ieta]*initCalibCoeff[ieta] - 1.) );//fill TH2 with recalibCoeff

	}

	if(ringIds[iid].subdetId() == EcalEndcap){
	  EEDetId myEEDetId(ringIds[iid]);
	  if(myEEDetId.zside() < 0)
	    h2_xtalRecalibCoeffEndcapMinus_[loopFlag_]->SetBinContent( myEEDetId.ix(), myEEDetId.iy(), 100 * (calibCoeff[ieta]*initCalibCoeff[ieta] - 1.) );//fill TH2 with recalibCoeff

	  if(myEEDetId.zside() > 0)
	    h2_xtalRecalibCoeffEndcapPlus_[loopFlag_]->SetBinContent( myEEDetId.ix(), myEEDetId.iy(), 100 * (calibCoeff[ieta]*initCalibCoeff[ieta] - 1.) );//fill TH2 with recalibCoeff
	  
	}
	
	
	ical->setValue( ringIds[iid], *(ical->getMap().find(ringIds[iid])  ) * optimizedCoefficients[ieta] );
      }    

    }
  
  
  /////////dump residual miscalibration at each loop

  for( int k = 0; k<theAlgorithm_->getNumberOfChannels(); k++ )
    {
      bool isNearCrack = ( abs( ringNumberCorrector(k) ) == 1 || abs( ringNumberCorrector(k) ) == 25 ||
                           abs( ringNumberCorrector(k) ) == 26 || abs( ringNumberCorrector(k) ) == 45 ||
                           abs( ringNumberCorrector(k) ) == 46 || abs( ringNumberCorrector(k) ) == 65 ||
                           abs( ringNumberCorrector(k) ) == 66 || abs( ringNumberCorrector(k) ) == 85 ||
                           abs( ringNumberCorrector(k) ) == 86 || abs( ringNumberCorrector(k) ) == 124 );

      if(!isNearCrack){

	//	h2_miscalRecalParz_[iLoop]->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
	h1_mcParz_[iLoop]->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
	
	if(k<170){
	  //h2_miscalRecalEBParz_[iLoop]->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
	  h1_mcEBParz_[iLoop]->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
	  
	}
	
	if(k>=170){
	  //h2_miscalRecalEEParz_[iLoop]->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
	  h1_mcEEParz_[iLoop]->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
	}
     
      }
    }
  
  
  /////////////////////////
  double parResidual[3];
  double errparResidual[3];
  double zResChi2;
  int zResIters;
  
  ZIterativeAlgorithmWithFit::gausfit(h1_mcParz_[iLoop],parResidual,errparResidual,3.,3., &zResChi2, &zResIters);
  //h1_mcParz_[iLoop]->Fit("gaus");
  
  h2_residualSigma_ -> Fill(loopFlag_ + 1,  parResidual[2]);
  loopArray[loopFlag_] = loopFlag_ + 1;
  sigmaArray[loopFlag_] = parResidual[2];
  sigmaErrorArray[loopFlag_] = errparResidual[2];

  std::cout<<"Fit on residuals, sigma is "<<parResidual[2]<<" +/- "<<errparResidual[2]<<std::endl;

  /////////////////////
  outputFile_->cd();


  //  h2_miscalRecalParz_[iLoop]->Write();
  h1_mcParz_[iLoop]->Write();

  //h2_miscalRecalEBParz_[iLoop]->Write();
  h1_mcEBParz_[iLoop]->Write();

  //h2_miscalRecalEEParz_[iLoop]->Write();
  h1_mcEEParz_[iLoop]->Write();
  h2_xtalRecalibCoeffBarrel_[loopFlag_] -> Write();
  h2_xtalRecalibCoeffEndcapPlus_[loopFlag_] -> Write();
  h2_xtalRecalibCoeffEndcapMinus_[loopFlag_] -> Write();

  /////////dump residual miscalibration at each loop
  
  loopFlag_++;
  
#ifdef DEBUG  
  std::cout<<" loopFlag_ is "<<loopFlag_<<std::endl;
#endif  
  
  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;
}

void ZeeCalibration::bookHistograms()
{

  h1_eventsBeforeEWKSelection_=  new TH1F("h1_eventsBeforeEWKSelection", "h1_eventsBeforeEWKSelection", 5,0,5); 
  h1_eventsAfterEWKSelection_ =  new TH1F("h1_eventsAfterEWKSelection", "h1_eventsAfterEWKSelection", 5,0,5);

  h1_eventsBeforeBorderSelection_=  new TH1F("h1_eventsBeforeBorderSelection", "h1_eventsBeforeBorderSelection", 5,0,5); 
  h1_eventsAfterBorderSelection_ =  new TH1F("h1_eventsAfterBorderSelection", "h1_eventsAfterBorderSelection", 5,0,5);

  h1_seedOverSC_= new TH1F("h1_seedOverSC", "h1_seedOverSC", 400, 0., 2.);

  myZeePlots_ -> bookHLTHistograms();
  
  h1_borderElectronClassification_ = new TH1F("h1_borderElectronClassification", "h1_borderElectronClassification", 55, -5 , 50);
  h1_preshowerOverSC_= new TH1F("h1_preshowerOverSC", "h1_preshowerOverSC", 400, 0., 1.);
  
  h2_fEtaBarrelGood_ = new TH2F("fEtaBarrelGood","fEtaBarrelGood",800,-4.,4.,800,0.8,1.2);
  h2_fEtaBarrelGood_->SetXTitle("Eta");
  h2_fEtaBarrelGood_->SetYTitle("1/fEtaBarrelGood");
  
  h2_fEtaBarrelBad_ = new TH2F("fEtaBarrelBad","fEtaBarrelBad",800,-4.,4.,800,0.8,1.2);
  h2_fEtaBarrelBad_->SetXTitle("Eta");
  h2_fEtaBarrelBad_->SetYTitle("1/fEtaBarrelBad");
  
  h2_fEtaEndcapGood_ = new TH2F("fEtaEndcapGood","fEtaEndcapGood",800,-4.,4.,800,0.8,1.2);
  h2_fEtaEndcapGood_->SetXTitle("Eta");
  h2_fEtaEndcapGood_->SetYTitle("1/fEtaEndcapGood");
  
  h2_fEtaEndcapBad_ = new TH2F("fEtaEndcapBad","fEtaEndcapBad",800,-4.,4.,800,0.8,1.2);
  h2_fEtaEndcapBad_->SetXTitle("Eta");
  h2_fEtaEndcapBad_->SetYTitle("1/fEtaEndcapBad");
  
  for (int i=0;i<2;i++)
    {
      char histoName[50];

      sprintf(histoName,"h_eleEffEta_%d",i);
      h_eleEffEta_[i] = new TH1F(histoName,histoName, 150, 0., 2.7);
      h_eleEffEta_[i]->SetXTitle("|#eta|");

      sprintf(histoName,"h_eleEffPhi_%d",i);
      h_eleEffPhi_[i] = new TH1F(histoName,histoName, 400, -4., 4.);
      h_eleEffPhi_[i]->SetXTitle("Phi");

      sprintf(histoName,"h_eleEffPt_%d",i);
      h_eleEffPt_[i] = new TH1F(histoName,histoName, 200, 0., 200.);
      h_eleEffPt_[i]->SetXTitle("p_{T}(GeV/c)");
   }
  
  h2_xtalMiscalibCoeffBarrel_ = new TH2F("h2_xtalMiscalibCoeffBarrel","h2_xtalMiscalibCoeffBarrel", 171, -85, 85, 360, 0, 360);
  h2_xtalMiscalibCoeffEndcapMinus_ = new TH2F("h2_xtalMiscalibCoeffEndcapMinus", "h2_xtalMiscalibCoeffEndcapMinus", 100, 0,100, 100, 0, 100);
  h2_xtalMiscalibCoeffEndcapPlus_ = new TH2F("h2_xtalMiscalibCoeffEndcapPlus", "h2_xtalMiscalibCoeffEndcapPlus", 100, 0,100, 100, 0, 100);

  h2_xtalMiscalibCoeffBarrel_ ->SetXTitle("ieta");
  h2_xtalMiscalibCoeffBarrel_ ->SetYTitle("iphi");

  h2_xtalMiscalibCoeffEndcapMinus_->SetXTitle("ix");
  h2_xtalMiscalibCoeffEndcapMinus_->SetYTitle("iy");

  for (int i=0;i<25;i++)
    {
      
      char histoName[50];
      sprintf(histoName,"h_ESCEtrueVsEta_%d",i);
      
      h_ESCEtrueVsEta_[i] = new TH2F(histoName,histoName, 150, 0., 2.7, 300,0.,1.5);
      h_ESCEtrueVsEta_[i]->SetXTitle("|#eta|");
      h_ESCEtrueVsEta_[i]->SetYTitle("E_{SC,raw}/E_{MC}");
      
      sprintf(histoName,"h_ESCEtrue_%d",i);

      h_ESCEtrue_[i] = new TH1F(histoName,histoName, 300,0.,1.5);

      sprintf(histoName,"h2_chi2_%d",i);
      h2_chi2_[i] = new TH2F(histoName,histoName, 1000,-150,150, 1000, -1, 5);

      sprintf(histoName,"h2_iterations_%d",i);
      h2_iterations_[i] = new TH2F(histoName,histoName, 1000,-150,150, 1000, -1, 15);

      sprintf(histoName,"h_ESCcorrEtrueVsEta_%d",i);
      
      h_ESCcorrEtrueVsEta_[i] = new TH2F(histoName,histoName, 150, 0., 2.7, 300,0.,1.5);
      h_ESCcorrEtrueVsEta_[i]->SetXTitle("|#eta|");
      h_ESCcorrEtrueVsEta_[i]->SetYTitle("E_{SC,#eta-corr}/E_{MC}");
      
      sprintf(histoName,"h_ESCcorrEtrue_%d",i);

      h_ESCcorrEtrue_[i] = new TH1F(histoName,histoName, 300,0.,1.5);

      sprintf(histoName,"h2_xtalRecalibCoeffBarrel_%d",i);
      h2_xtalRecalibCoeffBarrel_[i] = new TH2F(histoName,histoName, 171, -85, 85, 360, 0, 360);
      
      h2_xtalRecalibCoeffBarrel_[i]->SetXTitle("ieta");
      h2_xtalRecalibCoeffBarrel_[i]->SetYTitle("iphi");

      sprintf(histoName,"h2_xtalRecalibCoeffEndcapMinus_%d",i);
      h2_xtalRecalibCoeffEndcapMinus_[i] = new TH2F(histoName,histoName, 100, 0,100, 100, 0, 100);
      h2_xtalRecalibCoeffEndcapMinus_[i]->SetXTitle("ix");
      h2_xtalRecalibCoeffEndcapMinus_[i]->SetYTitle("iy");

      sprintf(histoName,"h2_xtalRecalibCoeffEndcapPlus_%d",i);
      h2_xtalRecalibCoeffEndcapPlus_[i] = new TH2F(histoName,histoName, 100, 0,100, 100, 0, 100);
      h2_xtalRecalibCoeffEndcapPlus_[i]->SetXTitle("ix");
      h2_xtalRecalibCoeffEndcapPlus_[i]->SetYTitle("iy");
      
    }                         
                                                                     
  /*
  for (int i=0;i<15;i++)
    {
                                                                                                                             
      char histoName[50];

      sprintf(histoName,"h_DiffZMassDistr_%d",i);
      h_DiffZMassDistr_[i] = new TH1F(histoName,histoName, 400, -20., 20.);
      h_DiffZMassDistr_[i]->SetXTitle("M_{Z, reco} - M_{Z, MC}");
      h_DiffZMassDistr_[i]->SetYTitle("events");

      sprintf(histoName,"h_ZMassDistr_%d",i);
      h_ZMassDistr_[i] = new TH1F(histoName,histoName, 200, 0., 150.);
      h_ZMassDistr_[i]->SetXTitle("RecoZmass (GeV)");
      h_ZMassDistr_[i]->SetYTitle("events");

    }
  */
  
  h1_zMassResol_ = new TH1F("zMassResol", "zMassResol", 200, -50., 50.);
  h1_zMassResol_->SetXTitle("M_{Z, reco} - M_{Z, MC}");
  h1_zMassResol_->SetYTitle("events");

  h1_eleEtaResol_ = new TH1F("eleEtaResol", "eleEtaResol", 100, -0.01, 0.01);
  h1_eleEtaResol_->SetXTitle("#eta_{reco} - #eta_{MC}");
  h1_eleEtaResol_->SetYTitle("events");

  h1_electronCosTheta_TK_ = new TH1F("electronCosTheta_TK", "electronCosTheta_TK", 100, -1, 1);
  h1_electronCosTheta_TK_->SetXTitle("cos #theta_{12}");
  h1_electronCosTheta_TK_->SetYTitle("events");

  h1_electronCosTheta_SC_ = new TH1F("electronCosTheta_SC", "electronCosTheta_SC", 100, -1, 1);
  h1_electronCosTheta_SC_->SetXTitle("cos #theta_{12}");
  h1_electronCosTheta_SC_->SetYTitle("events");

  h1_electronCosTheta_SC_TK_ = new TH1F("electronCosTheta_SC_TK", "electronCosTheta_SC_TK", 200, -0.1, 0.1);
  h1_electronCosTheta_SC_TK_->SetXTitle("cos #theta_{12}^{SC}/ cos #theta_{12}^{TK} - 1");
  h1_electronCosTheta_SC_TK_->SetYTitle("events");

  h1_elePhiResol_ = new TH1F("elePhiResol", "elePhiResol", 100, -0.01, 0.01);
  h1_elePhiResol_->SetXTitle("#phi_{reco} - #phi_{MC}");
  h1_elePhiResol_->SetYTitle("events");


  h1_zEtaResol_ = new TH1F("zEtaResol", "zEtaResol", 200, -1., 1.);
  h1_zEtaResol_->SetXTitle("#eta_{Z, reco} - #eta_{Z, MC}");
  h1_zEtaResol_->SetYTitle("events");


  h1_zPhiResol_ = new TH1F("zPhiResol", "zPhiResol", 200, -1., 1.);
  h1_zPhiResol_->SetXTitle("#phi_{Z, reco} - #phi_{Z, MC}");
  h1_zPhiResol_->SetYTitle("events");

  h1_nEleReco_ = new TH1F("nEleReco","Number of reco electrons",10,-0.5,10.5);
  h1_nEleReco_->SetXTitle("nEleReco");
  h1_nEleReco_->SetYTitle("events");
  
  //  h1_occupancyVsEta_ = new TH1F("occupancyVsEta","occupancyVsEta",EcalRingCalibrationTools::N_RING_TOTAL,0,(float)EcalRingCalibrationTools::N_RING_TOTAL);

  h1_occupancyVsEta_ = new TH1F("occupancyVsEta","occupancyVsEta",249, -124, 124);
  h1_occupancyVsEta_->SetYTitle("Weighted electron statistics");
  h1_occupancyVsEta_->SetXTitle("Eta channel");

  h1_weightSumMeanBarrel_= new TH1F("weightSumMeanBarrel","weightSumMeanBarrel",10000, 0, 10000);
  h1_weightSumMeanEndcap_= new TH1F("weightSumMeanEndcap","weightSumMeanEndcap",10000, 0, 10000);
  
  h1_occupancy_ = new TH1F("occupancy","occupancy",1000,0,10000);
  h1_occupancy_->SetXTitle("Weighted electron statistics");

  h1_occupancyBarrel_ = new TH1F("occupancyBarrel","occupancyBarrel",1000,0,10000);
  h1_occupancyBarrel_->SetXTitle("Weighted electron statistics");

  h1_occupancyEndcap_ = new TH1F("occupancyEndcap","occupancyEndcap",1000,0,10000);
  h1_occupancyEndcap_->SetXTitle("Weighted electron statistics");
 

  h1_eleClasses_= new TH1F("eleClasses","eleClasses",301,-1,300);
  h1_eleClasses_->SetXTitle("classCode");
  h1_eleClasses_->SetYTitle("#");
  


 myZeePlots_ ->bookZMCHistograms();

 myZeePlots_ ->bookZHistograms();
 
 myZeePlots_ ->bookEleMCHistograms();	
 
 myZeePlots_ ->bookEleHistograms();		
 

  h1_ZCandMult_ =new TH1F("ZCandMult","Multiplicity of Z candidates in one event",10,-0.5,10.5);
  h1_ZCandMult_ ->SetXTitle("ZCandMult");
  
  h1_reco_ZMass_ = new TH1F("reco_ZMass","Inv. mass of 2 reco Electrons",200,0.,150.);
  h1_reco_ZMass_->SetXTitle("reco_ZMass (GeV)");
  h1_reco_ZMass_->SetYTitle("events");

  h1_reco_ZMassCorr_ = new TH1F("reco_ZMassCorr","Inv. mass of 2 corrected reco Electrons",200,0.,150.);
  h1_reco_ZMassCorr_->SetXTitle("reco_ZMass (GeV)");
  h1_reco_ZMassCorr_->SetYTitle("events");

  h1_reco_ZMassCorrBB_ = new TH1F("reco_ZMassCorrBB","Inv. mass of 2 corrected reco Electrons",200,0.,150.);
  h1_reco_ZMassCorrBB_->SetXTitle("reco_ZMass (GeV)");
  h1_reco_ZMassCorrBB_->SetYTitle("events");

  h1_reco_ZMassCorrEE_ = new TH1F("reco_ZMassCorrEE","Inv. mass of 2 corrected reco Electrons",200,0.,150.);
  h1_reco_ZMassCorrEE_->SetXTitle("reco_ZMass (GeV)");
  h1_reco_ZMassCorrEE_->SetYTitle("events");

  //  h2_coeffVsEta_= new TH2F("h2_calibCoeffVsEta","h2_calibCoeffVsEta",EcalRingCalibrationTools::N_RING_TOTAL,0, (double)EcalRingCalibrationTools::N_RING_TOTAL, 200, 0., 2.);

  h2_coeffVsEta_= new TH2F("h2_calibCoeffVsEta","h2_calibCoeffVsEta",249,-124,125, 200, 0., 2.);
  h2_coeffVsEta_->SetXTitle("Eta channel");
  h2_coeffVsEta_->SetYTitle("recalibCoeff");

  h2_coeffVsEtaGrouped_= new TH2F("h2_calibCoeffVsEtaGrouped","h2_calibCoeffVsEtaGrouped", 200, 0., 3., 200, 0.6, 1.4);
  h2_coeffVsEtaGrouped_->SetXTitle("|#eta|");
  h2_coeffVsEtaGrouped_->SetYTitle("recalibCoeff");

  h2_zMassVsLoop_= new TH2F("h2_zMassVsLoop","h2_zMassVsLoop",1000,0,40, 90, 80.,95.);

  h2_zMassDiffVsLoop_= new TH2F("h2_zMassDiffVsLoop","h2_zMassDiffVsLoop",1000,0,40, 100, -1., 1.);
  h2_zMassDiffVsLoop_->SetXTitle("Iteration");
  h2_zMassDiffVsLoop_->SetYTitle("M_{Z, reco peak} - M_{Z, true}");

  h2_zWidthVsLoop_= new TH2F("h2_zWidthVsLoop","h2_zWidthVsLoop",1000,0,40, 100, 0.,10.);

  h2_coeffVsLoop_= new TH2F("h2_coeffVsLoop","h2_coeffVsLoop",1000,0,40, 100, 0., 2.);

  h2_residualSigma_= new TH2F("h2_residualSigma","h2_residualSigma",1000, 0, 40, 100, 0., .5);

  h2_miscalRecal_ = new TH2F("h2_miscalRecal","h2_miscalRecal", 500, 0., 2., 500, 0., 2.);
  h2_miscalRecal_->SetXTitle("initCalibCoeff");
  h2_miscalRecal_->SetYTitle("1/RecalibCoeff");
 
  h2_miscalRecalEB_ = new TH2F("h2_miscalRecalEB","h2_miscalRecalEB", 500, 0., 2., 500, 0., 2.);
  h2_miscalRecalEB_->SetXTitle("initCalibCoeff");
  h2_miscalRecalEB_->SetYTitle("1/RecalibCoeff");

  h2_miscalRecalEE_ = new TH2F("h2_miscalRecalEE","h2_miscalRecalEE", 500, 0., 2., 500, 0., 2.);
  h2_miscalRecalEE_->SetXTitle("initCalibCoeff");
  h2_miscalRecalEE_->SetYTitle("1/RecalibCoeff");

  h1_mc_ = new TH1F("h1_residualMiscalib","h1_residualMiscalib", 200, -0.2, 0.2);
  h1_mcEB_ = new TH1F("h1_residualMiscalibEB","h1_residualMiscalibEB", 200, -0.2, 0.2);
  h1_mcEE_ = new TH1F("h1_residualMiscalibEE","h1_residualMiscalibEE", 200, -0.2, 0.2);
 
  for (int i=0;i<25;i++)
    {
      char histoName[50];
      /*     
	     sprintf(histoName,"h2_miscalRecalParz_%d",i);
	     h2_miscalRecalParz_[i] = new TH2F(histoName,histoName,500, 0., 2., 500, 0., 2.);
	     h2_miscalRecalParz_[i]->SetXTitle("initCalibCoeff");
	     h2_miscalRecalParz_[i]->SetYTitle("1/recalibCoeff");
	     
	     sprintf(histoName,"h2_miscalRecalEBParz_%d",i);
	     h2_miscalRecalEBParz_[i] = new TH2F(histoName,histoName,500, 0., 2., 500, 0., 2.);
	     h2_miscalRecalEBParz_[i]->SetXTitle("initCalibCoeff");
	     h2_miscalRecalEBParz_[i]->SetYTitle("1/recalibCoeff");
	     
      sprintf(histoName,"h2_miscalRecalEEParz_%d",i);
      h2_miscalRecalEEParz_[i] = new TH2F(histoName,histoName,500, 0., 2., 500, 0., 2.);
      h2_miscalRecalEEParz_[i]->SetXTitle("initCalibCoeff");
      h2_miscalRecalEEParz_[i]->SetYTitle("1/recalibCoeff");
      */
      
      sprintf(histoName,"h1_residualMiscalibParz_%d",i);
      h1_mcParz_[i] = new TH1F(histoName,histoName, 200, -0.2, 0.2);
      sprintf(histoName,"h1_residualMiscalibEBParz_%d",i);
      h1_mcEBParz_[i] = new TH1F(histoName,histoName, 200, -0.2, 0.2);
      sprintf(histoName,"h1_residualMiscalibEEParz_%d",i);
      h1_mcEEParz_[i] = new TH1F(histoName,histoName, 200, -0.2, 0.2);
      
    }
 

}


double ZeeCalibration::fEtaBarrelBad(double scEta) const{
  
  float p0 = 1.00153e+00;
    float p1 = 3.29331e-02;
    float p2 = 1.21187e-03;

  double x  = (double) fabs(scEta);

  return 1. / ( p0 + p1*x*x + p2*x*x*x*x );  

}
  
double ZeeCalibration::fEtaEndcapGood(double scEta) const{

  // f(eta) for the first 3 classes (100, 110 and 120) 
  // Ivica's new corrections 01/06
  float p0 = 1.06819e+00;
    float p1 = -1.53189e-02;
    float p2 = 4.01707e-04 ;

  double x  = (double) fabs(scEta);

  return 1. / ( p0 + p1*x*x + p2*x*x*x*x );  

}

double ZeeCalibration::fEtaEndcapBad(double scEta) const{
  
  float p0 = 1.17382e+00;
  float p1 = -6.52319e-02; 
  float p2 = 6.26108e-03;

  double x  = (double) fabs(scEta);

 return 1. / ( p0 + p1*x*x + p2*x*x*x*x );  

}
  
double ZeeCalibration::fEtaBarrelGood(double scEta) const{

  float p0 = 9.99782e-01 ;
  float p1 = 1.26983e-02;
  float p2 = 2.16344e-03;

  double x  = (double) fabs(scEta);

 return 1. / ( p0 + p1*x*x + p2*x*x*x*x );  

}


//////////////////////////////////new part

void ZeeCalibration::fillMCmap(const std::vector<const reco::GsfElectron*>* electronCollection,const std::vector<HepMC::GenParticle*>& mcEle,std::map<HepMC::GenParticle*,const reco::GsfElectron*>& myMCmap)
{
  for (unsigned int i=0;i<mcEle.size();i++)
    {
      float minDR=0.1;
      const reco::GsfElectron* myMatchEle=0;
      for (unsigned int j=0;j<electronCollection->size();j++)
        {
          float dr=EvalDR(mcEle[i]->momentum().pseudoRapidity(),(*(*electronCollection)[j]).eta(),mcEle[i]->momentum().phi(),(*(*electronCollection)[j]).phi());
          if (dr < minDR )
            {
              myMatchEle = (*electronCollection)[j];
              minDR = dr;
            }
        }
      myMCmap.insert(std::pair<HepMC::GenParticle*,const reco::GsfElectron*>(mcEle[i],myMatchEle));
      
    }
}
                                                                                                                             
float ZeeCalibration::EvalDR(float Eta,float Eta_ref,float Phi,float Phi_ref)
{
  if (Phi<0) Phi = 2*TMath::Pi() + Phi;
  if (Phi_ref<0) Phi_ref = 2*TMath::Pi() + Phi_ref;
  float DPhi = Phi - Phi_ref ;
  if (fabs(DPhi)>TMath::Pi()) DPhi = 2*TMath::Pi() - fabs(DPhi);
                                                                                                                             
  float DEta = Eta - Eta_ref ;
                                                                                                                             
  float DR = sqrt( DEta*DEta + DPhi*DPhi );
  return DR;
}

float ZeeCalibration::EvalDPhi(float Phi,float Phi_ref)
{
  if (Phi<0) Phi = 2*TMath::Pi() + Phi;
  if (Phi_ref<0) Phi_ref = 2*TMath::Pi() + Phi_ref;
  return (Phi - Phi_ref);
}

void ZeeCalibration::fillEleInfo( std::vector<HepMC::GenParticle*>& mcEle, std::map<HepMC::GenParticle*,const reco::GsfElectron*>& associationMap)
{

  for (unsigned int i=0;i<mcEle.size();i++)
    {

      h_eleEffEta_[0]->Fill(fabs(mcEle[i]->momentum().pseudoRapidity()));
      h_eleEffPhi_[0]->Fill(mcEle[i]->momentum().phi());
      h_eleEffPt_[0]->Fill(mcEle[i]->momentum().perp());

      std::map<HepMC::GenParticle*,const reco::GsfElectron*>::const_iterator mIter = associationMap.find(mcEle[i]);
      if (mIter == associationMap.end() )
        continue;
    
      if((*mIter).second)
        {
          const reco::GsfElectron* myEle=(*mIter).second;
      
	  h_eleEffEta_[1]->Fill(fabs(mcEle[i]->momentum().pseudoRapidity()));
          h_eleEffPhi_[1]->Fill(mcEle[i]->momentum().phi());
          h_eleEffPt_[1]->Fill(mcEle[i]->momentum().perp());
	  h1_eleEtaResol_->Fill( myEle->eta() - mcEle[i]->momentum().eta() );
	  h1_elePhiResol_->Fill( myEle->phi() - mcEle[i]->momentum().phi() );

          const reco::SuperCluster* mySC=&(*(myEle->superCluster()));
	  if (/*fabs(mySC->position().eta()) < 2.4*/1)
	    {
	      //      if(myEle->classification()>=100)std::cout<<"mySC->preshowerEnergy()"<<mySC->preshowerEnergy()<<std::endl;

	      h_ESCEtrue_[loopFlag_]->Fill(mySC->energy()/mcEle[i]->momentum().e());
	      h_ESCEtrueVsEta_[loopFlag_]->Fill(fabs(mySC->position().eta()),mySC->energy()/mcEle[i]->momentum().e());

	      double corrSCenergy = ( mySC->energy() )/getEtaCorrection(myEle);
	      h_ESCcorrEtrue_[loopFlag_]->Fill(corrSCenergy/mcEle[i]->momentum().e());
	      h_ESCcorrEtrueVsEta_[loopFlag_]->Fill(fabs(mySC->position().eta()),corrSCenergy/mcEle[i]->momentum().e());

//	      std::vector<DetId> mySCRecHits = mySC->seed()->getHitsByDetId();

	      h1_seedOverSC_->Fill( mySC->seed()->energy() / mySC->energy() );
	      h1_preshowerOverSC_->Fill( mySC->preshowerEnergy() / mySC->energy() );
	      
	    }

        }
    }
}

int ZeeCalibration::ringNumberCorrector(int k)
{

  int index=-999;

  if( calibMode_ == "RING"){
    if(k>=0 && k<=84)index = k - 85;
    
    if(k>=85 && k<=169)index = k - 84;
    
    if(k>=170 && k<=208)index = - k + 84;
    
    if(k>=209 && k<=247)index = k - 123;
    
  }
  
  else if( calibMode_ == "MODULE"){
    
    if(k>=0 && k<=71)index = k - 72;
    
    if(k>=72 && k<=143)index = k - 71;
    
  }
  return index;
  
}


double ZeeCalibration::getEtaCorrection(const reco::GsfElectron* ele){

  double correction(1.);

  if(ele->classification() ==0 ||
     ele->classification() ==10 ||
     ele->classification() ==20)
    correction = fEtaBarrelGood(ele->superCluster()->eta());
                                                                                                                                               
  if(ele->classification() ==100 ||
     ele->classification() ==110 ||
     ele->classification() ==120)
    correction = fEtaEndcapGood(ele->superCluster()->eta());
                                                                                                                                               
  if(ele->classification() ==30 ||
     ele->classification() ==31 ||
     ele->classification() ==32 ||
     ele->classification() ==33 ||
     ele->classification() ==34)
    correction = fEtaBarrelBad(ele->superCluster()->eta());


  if(ele->classification() ==130 ||
     ele->classification() ==131 ||
     ele->classification() ==132 ||
     ele->classification() ==133 ||
     ele->classification() ==134)
    correction = fEtaEndcapBad(ele->superCluster()->eta());
 
  return correction;                                                                                                                                              
}

std::pair<DetId, double> ZeeCalibration::getHottestDetId(const std::vector<std::pair< DetId,float > >& mySCRecHits, const EBRecHitCollection* ebhits, const EERecHitCollection* eehits){
  

  double maxEnergy = -9999.;
  const EcalRecHit* hottestRecHit=0;
  
  std::pair<DetId, double> myPair (DetId(0), -9999.);


  for(   std::vector<std::pair<DetId,float> >::const_iterator idIt=mySCRecHits.begin(); idIt != mySCRecHits.end(); idIt++){
   
    if (idIt->first.subdetId() == EcalBarrel )
      {
	hottestRecHit  = & (* ( ebhits->find((*idIt).first) ) );

	if( hottestRecHit == & (*( ebhits->end())) )
	  {
	    std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@ NO RECHIT FOUND SHOULD NEVER HAPPEN"<<std::endl;
	    continue;
	  }
      }
    else if (idIt->first.subdetId() == EcalEndcap )
      {
	hottestRecHit  = & (* ( eehits->find((*idIt).first) ) );
	if( hottestRecHit == & (*( eehits->end())) )
	  {
	    std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@ NO RECHIT FOUND SHOULD NEVER HAPPEN"<<std::endl;
	    continue;
	  }
      }    
    
    //std::cout<<"[getHottestDetId] hottestRecHit->energy() "<<hottestRecHit->energy()<<std::endl;
   
    if(hottestRecHit && hottestRecHit->energy() > maxEnergy){

      maxEnergy = hottestRecHit->energy();
      
      myPair.first =   hottestRecHit ->id();
      myPair.second = maxEnergy;
      
    }
    
  }//end loop to find hottest RecHit    
  
  //std::cout<<"[ZeeCalibration::getHottestDetId] going to return..."<<std::endl;

  return myPair;
  
}


bool ZeeCalibration::xtalIsOnModuleBorder( EBDetId myEBDetId ){
  
  bool myBool(false); 

  short ieta = myEBDetId.ieta();
  short iphi = myEBDetId.iphi();

  //  std::cout<<"[xtalIsOnModuleBorder] ieta: "<<ieta<<" iphi "<<iphi<<std::endl;
 
  myBool = ( abs( ieta )  == 1 || abs( ieta ) == 25
	     || abs( ieta )  ==26 || abs( ieta ) == 45
	     || abs( ieta )  ==46 || abs( ieta ) == 65
	     || abs( ieta )  ==66 || abs( ieta ) == 85 );
    
    for(int i = 0; i < 19; i++){
      
      if(iphi == ( 20*i + 1 ) || iphi == 20*i )
      myBool = true;
      
    }
  
  return myBool;
}

float ZeeCalibration::computeCoefficientDistanceAtIteration( float v1[250], float v2[250], int size ){

  float dist(0.);
 
  for(int i =0; i < size; i++){
    
    //    std::cout<< "[ZeeCalibration::computeCoefficientDistanceAtIteration] Adding term "<<pow( v1[i]-v2[i], 2 )<<" from v1 "<<v1[i]<<" and v2 "<<v2[i]<<std::endl;
    
    bool isNearCrack = false;

    if( calibMode_ == "RING"){//exclude non-calibrated rings from computation

      isNearCrack = ( abs( ringNumberCorrector(i) ) == 1 || abs( ringNumberCorrector(i) ) == 25 ||
		      abs( ringNumberCorrector(i) ) == 26 || abs( ringNumberCorrector(i) ) == 45 ||
		      abs( ringNumberCorrector(i) ) == 46 || abs( ringNumberCorrector(i) ) == 65 ||
		      abs( ringNumberCorrector(i) ) == 66 || abs( ringNumberCorrector(i) ) == 85 ||
		      abs( ringNumberCorrector(i) ) == 86 || abs( ringNumberCorrector(i) ) == 124 );
    }
    
    if(!isNearCrack)
      dist += pow( v1[i]-v2[i], 2 );
  }
  
  dist = sqrt(dist) / size;
  
  return dist;
  
}


void ZeeCalibration::resetVariables(){

 BBZN=0;
 EBZN=0;
 EEZN=0;
 BBZN_gg=0;
 EBZN_gg=0;
 EEZN_gg=0;

 BBZN_tt=0;
 EBZN_tt=0;
 EEZN_tt=0;

 BBZN_t0=0;
 EBZN_t0=0;
 EEZN_t0=0;

 TOTAL_ELECTRONS_IN_BARREL=0;
 TOTAL_ELECTRONS_IN_ENDCAP=0;

 GOLDEN_ELECTRONS_IN_BARREL=0;
 GOLDEN_ELECTRONS_IN_ENDCAP=0;
 SILVER_ELECTRONS_IN_BARREL=0;
 SILVER_ELECTRONS_IN_ENDCAP=0;
 SHOWER_ELECTRONS_IN_BARREL=0;
 SHOWER_ELECTRONS_IN_ENDCAP=0;
 CRACK_ELECTRONS_IN_BARREL=0;
 CRACK_ELECTRONS_IN_ENDCAP=0;


 BARREL_ELECTRONS_BEFORE_BORDER_CUT = 0;
 BARREL_ELECTRONS_AFTER_BORDER_CUT = 0;

 return;

}


void ZeeCalibration::resetHistograms(){

 h1_eventsBeforeEWKSelection_ ->Reset();
 h1_eventsAfterEWKSelection_ ->Reset();

 h1_eventsBeforeBorderSelection_ ->Reset();
 h1_eventsAfterBorderSelection_ ->Reset();

 for (int i=0;i<2;i++)
   {
     h_eleEffEta_[i] ->Reset();
     h_eleEffPhi_[i] ->Reset(); 
     h_eleEffPt_[i]  ->Reset();
   }

 h1_seedOverSC_ ->Reset();
 h1_preshowerOverSC_ ->Reset();

 h1_eleEtaResol_->Reset();
 h1_elePhiResol_->Reset();

 h1_zMassResol_->Reset(); 
 
 h1_electronCosTheta_TK_->Reset();
 h1_electronCosTheta_SC_->Reset();
 h1_electronCosTheta_SC_TK_->Reset();

 h2_fEtaBarrelGood_->Reset();
 h2_fEtaBarrelBad_->Reset();
 h2_fEtaEndcapGood_->Reset();
 h2_fEtaEndcapBad_->Reset();
 h1_eleClasses_->Reset();

 h1_ZCandMult_-> Reset();
 h1_reco_ZMass_-> Reset();
  h1_reco_ZMassCorr_-> Reset();
 h1_reco_ZMassCorrBB_-> Reset();
 h1_reco_ZMassCorrEE_-> Reset();
 h1_occupancyVsEta_-> Reset();
 h1_occupancy_-> Reset();
 h1_occupancyBarrel_-> Reset();
 h1_occupancyEndcap_-> Reset();

 return;

}


void ZeeCalibration::printStatistics(){


  std::cout<< "[ CHECK ON BARREL ELECTRON NUMBER ]"<<" first "<<BARREL_ELECTRONS_BEFORE_BORDER_CUT<<" second "<<TOTAL_ELECTRONS_IN_BARREL << std::endl;
  
  
  std::cout<< "[ EFFICIENCY OF THE BORDER SELECTION ]" << (float)BARREL_ELECTRONS_AFTER_BORDER_CUT / (float) BARREL_ELECTRONS_BEFORE_BORDER_CUT << std::endl;
  
  std::cout<< "[ EFFICIENCY OF THE GOLDEN SELECTION ] BARREL: " << (float)GOLDEN_ELECTRONS_IN_BARREL / (float) TOTAL_ELECTRONS_IN_BARREL << " ENDCAP: "<< (float)GOLDEN_ELECTRONS_IN_ENDCAP / (float) TOTAL_ELECTRONS_IN_ENDCAP << std::endl;
  
  std::cout<< "[ EFFICIENCY OF THE SILVER SELECTION ] BARREL: " << (float)SILVER_ELECTRONS_IN_BARREL / (float) TOTAL_ELECTRONS_IN_BARREL << " ENDCAP: "<< (float)SILVER_ELECTRONS_IN_ENDCAP / (float) TOTAL_ELECTRONS_IN_ENDCAP << std::endl;
  
  std::cout<< "[ EFFICIENCY OF THE SHOWER SELECTION ] BARREL: " << (float)SHOWER_ELECTRONS_IN_BARREL / (float) TOTAL_ELECTRONS_IN_BARREL << " ENDCAP: "<< (float)SHOWER_ELECTRONS_IN_ENDCAP / (float) TOTAL_ELECTRONS_IN_ENDCAP << std::endl;
  
  std::cout<< "[ EFFICIENCY OF THE CRACK SELECTION ] BARREL: " << (float)CRACK_ELECTRONS_IN_BARREL / (float) TOTAL_ELECTRONS_IN_BARREL << " ENDCAP: "<< (float)CRACK_ELECTRONS_IN_ENDCAP / (float) TOTAL_ELECTRONS_IN_ENDCAP << std::endl;
  
  
  
  std::ofstream fout("ZeeStatistics.txt");
  
  if(!fout) {
    std::cout << "Cannot open output file.\n";
    }

  fout<<"ZeeStatistics"<<std::endl;

  fout<<"##########################RECO#########################"<<std::endl;
  fout<<"##################Zee with Barrel-Barrel electrons: "<<BBZN<<std::endl;

  fout<<"Golden-Golden fraction: "<<(float)BBZN_gg/BBZN<<" 3-3 fraction is "<<(float)BBZN_tt/BBZN<<" 3-whatever fraction is "<<(float)BBZN_t0/BBZN<<std::endl; 
  fout<<"##################Zee with Barrel-Endcap electrons: "<<EBZN<<std::endl;
  fout<<"Golden-Golden fraction: "<<(float)EBZN_gg/EBZN<<" 3-3 fraction is "<<(float)EBZN_tt/EBZN<<" 3-whatever fraction is "<<(float)EBZN_t0/EBZN<<std::endl; 
  fout<<"##################Zee with Endcap-Endcap electrons: "<<EEZN<<std::endl;
  fout<<"Golden-Golden fraction: "<<(float)EEZN_gg/EEZN<<" 3-3 fraction is "<<(float)EEZN_tt/EEZN<<" 3-whatever fraction is "<<(float)EEZN_t0/EEZN<<std::endl; 

  fout<<"\n"<<std::endl;

  fout<<"##########################GEN#########################"<<std::endl;
  fout<<"##################Zee with Barrel-Barrel electrons: "<<(float)MCZBB/NEVT<<std::endl;
  fout<<"##################Zee with Barrel-Endcap electrons: "<<(float)MCZEB/NEVT<<std::endl;
  fout<<"##################Zee with Endcap-Endcap electrons: "<<(float)MCZEE/NEVT<<std::endl;

  fout.close();



}
