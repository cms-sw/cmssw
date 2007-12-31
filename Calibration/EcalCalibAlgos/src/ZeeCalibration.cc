#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"

#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"

#include "CLHEP/Vector/LorentzVector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPhiSymRecHitsProducer.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeCalibration.h"

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

#include "TTree.h"
#include "TBranch.h"

#include "TFile.h"
#include "TProfile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TGraph.h"
#include "TRandom.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <map>
#include <fstream>

#define MZ 91.1876

//#define DEBUG 1

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

  theParameterSet=iConfig;

  //creating the algorithm
  theAlgorithm_ = new ZIterativeAlgorithmWithFit(iConfig);
  // Tell the framework what data is being produced
  //setWhatProduced(this);
  setWhatProduced (this, &ZeeCalibration::produceEcalIntercalibConstants ) ;
  findingRecord<EcalIntercalibConstantsRcd> () ;

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


void ZeeCalibration::beginOfJob( const edm::EventSetup& iSetup )
{
#ifdef DEBUG
  std::cout<<"[ZeeCalibration] Entering beginOfJob"<<std::endl;
#endif

//inizializzare la geometria di ecal
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<IdealGeometryRecord>().get(pG);     
  EcalRingCalibrationTools::setCaloGeometry(&(*pG));  
     
// go to *OUR* rootfile and book histograms                                                                                                                
  outputFile_->cd();
  bookHistograms();

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

   ////////////////////set miscalibration  
  for(int k = 0; k < theAlgorithm_->getNumberOfChannels(); k++)
    {
      calibCoeff[k]=1.;

      std::vector<DetId> ringIds = EcalRingCalibrationTools::getDetIdsInRing(k);

      if (miscalibMap)
	{
	  initCalibCoeff[k]=0.;	      
	  for (unsigned int iid=0; iid<ringIds.size();++iid)
	    {
	      float miscalib=(*(miscalibMap->get().find(ringIds[iid])));
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
      std::vector<DetId> ringIds = EcalRingCalibrationTools::getDetIdsInRing(k);
      for (unsigned int iid=0; iid<ringIds.size();++iid)
	//	ical->setValue( ringIds[iid], 1. * initCalibCoeff[k] );
	ical->setValue( ringIds[iid], (*(miscalibMap->get().find(ringIds[iid]))) );
    }
  
  read_events = 0;
#ifdef DEBUG
  std::cout<<"[ZeeCalibration] Done with beginOfJob"<<std::endl;
#endif

}


//========================================================================
void
ZeeCalibration::endOfJob() {

  ofstream fout("ZeeStatistics.txt");
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



  //Writing out calibration coefficients
  calibXMLwriter* barrelWriter = new calibXMLwriter(EcalBarrel);
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  barrelWriter->writeLine(ebid,(*(ical->getMap().find(ebid.rawId()))));
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
          endcapWriter->writeLine(eeid,(*(ical->getMap().find(eeid.rawId()))));
	}
      if (EEDetId::validDetId(iX,iY,-1))
	{
	  EEDetId eeid(iX,iY,-1);
          endcapWriter->writeLine(eeid,(*(ical->getMap().find(eeid.rawId()))));
	}
      
    }
  }
  

  std::cout<<"Writing  histos..."<<std::endl;
  outputFile_->cd();
  
  h1_zMassResol_->Write();
  h1_zEtaResol_->Write();
  h1_zPhiResol_->Write();
  h1_eleEtaResol_->Write();
  h1_elePhiResol_->Write();
  h1_seedOverSC_ ->Write();
  h1_preshowerOverSC_ ->Write();

  h1_gen_ZEta_->Write();
  h1_gen_ZRapidity_->Write();
  h1_gen_ZPt_->Write();
  h1_gen_ZMass_->Write();
  h1_gen_ZPhi_->Write();
  
  for(int i =0; i<15; i++){
    if( i < theMaxLoops ){
   
      h_ESCEtrueVsEta_[i]->Write();
      h_ESCEtrue_[i]->Write();

      h_ESCcorrEtrueVsEta_[i]->Write();
      h_ESCcorrEtrue_[i]->Write();
   
    h_DiffZMassDistr_[i]->Write();

    h_ZMassDistr_[i]->Write();
    }
  }

  h2_fEtaBarrelGood_->Write();
  h2_fEtaBarrelBad_->Write();
  h2_fEtaEndcapGood_->Write();
  h2_fEtaEndcapBad_->Write();
  h1_eleClasses_->Write();
  h1_nEleReco_->Write();
  h1_recoEleEnergy_->Write();
  h1_recoElePt_->Write();
  h1_recoEleEta_->Write();
  h1_recoElePhi_->Write();
  h1_ZCandMult_->Write();
  h1_reco_ZMass_->Write();
  h1_reco_ZEta_->Write();
  h1_reco_ZTheta_->Write();
  h1_reco_ZRapidity_->Write();
  h1_reco_ZPhi_->Write();
  h1_reco_ZPt_->Write();
  
  h1_reco_ZMassCorr_->Write();
  h1_reco_ZMassCorrBB_->Write();
  h1_reco_ZMassCorrEE_->Write();

  h1_mcEle_Energy_->Write();
  h1_mcElePt_->Write();
  h1_mcEleEta_->Write();
  h1_mcElePhi_->Write();

  
  h_eleEffEta_[1]->Divide(h_eleEffEta_[0]);
  h_eleEffPhi_[1]->Divide(h_eleEffPhi_[0]);
  h_eleEffPt_[1]->Divide(h_eleEffPt_[0]);

  h_eleEffEta_[0]->Write();
  h_eleEffPhi_[0]->Write();
  h_eleEffPt_[0]->Write();
  
  h_eleEffEta_[1]->Write();
  h_eleEffPhi_[1]->Write();
  h_eleEffPt_[1]->Write();

  
  int j = 0;

  int flag=0;
  
  float mean[25]={0.};
  
  for( int k = 0; k<theAlgorithm_->getNumberOfChannels(); k++ )
    {
      
      //////grouped
      
      if(k<85)
	{
	  if((k+1)%5!=0)
	    {
	      
	      mean[j]+=calibCoeff[k]/10.;
	      mean[j]+=calibCoeff[169 - k]/10.;
	      
	    }
	  
	  else 	{
	    mean[j]+=calibCoeff[k]/10.;
	    mean[j]+=calibCoeff[169 - k]/10.;
	    
	    //#ifdef DEBUG
	    std::cout<<" index(1 to 85) "<<k<<" j is "<<j<<" mean[j] is "<<mean[j]<<std::endl;
	    //#endif
	    j++;
	  }
	
	}
      //EE begin
      
      
      if(k>=170 && k<=204){
	
	if(flag<4){
	  //make groups of 5 Xtals in #eta
	  mean[j]+=calibCoeff[k]/10.;
	  mean[j]+=calibCoeff[k+39]/10.;
	  flag++;
	}

	else if(flag==4){
	  //make groups of 5 Xtals in #eta
	  mean[j]+=calibCoeff[k]/10.;
	  mean[j]+=calibCoeff[k+39]/10.;
	  flag=0;
	  std::cout<<" index(>85) "<<k<<" j is "<<j<<" mean[j] is "<<mean[j]<<std::endl;
	  j++;
	}

      }
      if(k>=205 && k<=208){
	mean[j]+=calibCoeff[k]/8.;
	mean[j]+=calibCoeff[k+39]/8.;
      }
      //EE end

      //////grouped
      
      
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
  
  
  
  Float_t xtalEta[25] = {1.4425, 1.3567,1.2711,1.1855,
			 1.10,1.01,0.92,0.83,
			 0.7468,0.6612,0.5756,0.4897,0.3985,0.3117,0.2250,0.1384,0.0487,
			 1.546, 1.651, 1.771, 1.908, 2.071, 2.267, 2.516, 2.8};
  
  
  for(int j = 0; j <25; j++)
    h2_coeffVsEtaGrouped_->Fill( xtalEta[j],mean[j]);
  
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
  for (int iIteration=0;iIteration<theAlgorithm_->getNumberOfIterations();iIteration++)
    for (int iChannel=0;iChannel<theAlgorithm_->getNumberOfChannels();iChannel++)
      {
	
	if(iChannel%20==0){
	algoHistos->weightedRescaleFactor[iIteration][iChannel]->Write();
	algoHistos->unweightedRescaleFactor[iIteration][iChannel]->Write();
	algoHistos->weight[iIteration][iChannel]->Write();
	}

	if( iIteration==(theAlgorithm_->getNumberOfIterations()-1) ){
	  
	  h1_occupancyVsEta_->Fill((Double_t)ringNumberCorrector(iChannel), algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral() );

	  h1_occupancy_->Fill( algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral() );

#ifdef DEBUG
	  std::cout<<"Writing weighted integral for channel "<<ringNumberCorrector(iChannel)<<" ,value "<<algoHistos->weightedRescaleFactor[iIteration][iChannel]->Integral()<<std::endl;
#endif

	}
	
      }
  
  //  std::cout<<"Done! Closing output file... "<<std::endl;

  h1_occupancyVsEta_->Write();
  h1_occupancyVsEtaGold_->Write();
  h1_occupancyVsEtaSilver_->Write();
  h1_occupancyVsEtaShower_->Write();
  h1_occupancyVsEtaCrack_->Write();
  h1_occupancy_->Write();

  myTree->Write();

  //   outputFile_->Write();//this automatically writes all histos on file
  outputFile_->Close();
  
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
 
  std::vector<HepMC::GenParticle*> mcEle;

  float myGenZMass(-1);
  float myGenZEta(-1);
  float myGenZPhi(-1);
      
  if (!mcProducer_.empty())
    {
      //DUMP GENERATED Z MASS - BEGIN
      Handle< HepMCProduct > hepProd ;
      //   iEvent.getByLabel( "source", hepProd ) ;
      iEvent.getByLabel( mcProducer_.c_str(), hepProd ) ;
                                                                                                                             
      const HepMC::GenEvent * myGenEvent = hepProd->GetEvent();
      HepMC::GenParticle* genZ=0;

      
      for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
            p != myGenEvent->particles_end(); ++p ) {
        //return a pointer to MC Z in the event
        if ( (*p)->pdg_id() == 23 && (*p)->status()==2){

	  myGenZMass = (*p)->momentum().m();
	  myGenZEta = (*p)->momentum().eta();
	  myGenZPhi = (*p)->momentum().phi();
	  
	  h1_gen_ZMass_->Fill((*p)->momentum().m());
	  h1_gen_ZEta_->Fill((*p)->momentum().eta());
	  
	  float genZ_Y = 0.5 * log ( ( (*p)->momentum().e() + (*p)->momentum().pz() ) /  ( (*p)->momentum().e() - (*p)->momentum().pz() ) )   ;

	  h1_gen_ZRapidity_->Fill( genZ_Y );
	  h1_gen_ZPt_->Fill((*p)->momentum().perp());
	  h1_gen_ZPhi_->Fill((*p)->momentum().phi());
          genZ=(*p);
        }
      }
      //DUMP GENERATED Z MASS - END
     
      
      //loop over MC positrons and find the closest MC positron in (eta,phi) phace space - begin
      HepMC::GenParticle MCele;

      for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	    p != myGenEvent->particles_end(); ++p ) {
	
	if (  abs( (*p)->pdg_id() ) == 11 )
	  {
	    mcEle.push_back( (*p) );
	    MCele=*(*p);
	
	    if (loopFlag_ == 0)
	      {
		
		h1_mcEle_Energy_->Fill( MCele.momentum().e() );
		h1_mcElePt_->Fill( MCele.momentum().perp() );
		h1_mcEleEta_->Fill( MCele.momentum().eta() );
		h1_mcElePhi_->Fill( MCele.momentum().phi() );
	      }
	    
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
  iEvent.getByLabel( rechitProducer_, rechitCollection_, phits);
  if (!phits.isValid()) {
    std::cerr << "Error! can't get the product EBRecHitCollection " << std::endl;
  }
  const EBRecHitCollection* hits = phits.product(); // get a ptr to the product

  // Get EERecHits
  Handle<EERecHitCollection> ephits;
  iEvent.getByLabel( erechitProducer_, erechitCollection_, ephits);
  if (!ephits.isValid()) {
    std::cerr << "Error! can't get the product EERecHitCollection " << std::endl;
  }
  const EERecHitCollection* ehits = ephits.product(); // get a ptr to the product

  
  //Get Hybrid SuperClusters
  Handle<reco::SuperClusterCollection> pSuperClusters;
  iEvent.getByLabel(scProducer_, scCollection_, pSuperClusters);
  if (!pSuperClusters.isValid()) {
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
  iEvent.getByLabel(scIslandProducer_, scIslandCollection_, pIslandSuperClusters);
  if (!pIslandSuperClusters.isValid()) {
    std::cerr << "Error! can't get the product IslandSuperClusterCollection "<< std::endl;
  }
  const reco::SuperClusterCollection* scIslandCollection = pIslandSuperClusters.product();

#ifdef DEBUG
  std::cout<<"scCollection->size()"<<scIslandCollection->size()<<std::endl;
#endif

  if(  ( scCollection->size()+scIslandCollection->size() ) < 2) return kContinue;

  // Get Electrons
  Handle<reco::GsfElectronCollection> pElectrons;
  iEvent.getByLabel(electronProducer_, electronCollection_, pElectrons);
  if (!pElectrons.isValid()) {
    std::cerr << "Error! can't get the product ElectronCollection "<< std::endl;
  }
  const reco::GsfElectronCollection* electronCollection = pElectrons.product();

  /*
  //reco-mc association map
  std::map<HepMC::GenParticle*,const reco::GsfElectron*> myMCmap;
  
    fillMCmap(&(*electronCollection),mcEle,myMCmap);
    
    fillEleInfo(mcEle,myMCmap);
  */
    if(electronCollection->size() < 2) return kContinue;
    
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
  
  if(electronCollection->size()<2)
    return kContinue;
  
  h1_nEleReco_->Fill(electronCollection->size());
  
  for(reco::GsfElectronCollection::const_iterator eleIt = electronCollection->begin();   eleIt != electronCollection->end(); eleIt++)
    {
      
      h1_recoElePt_->Fill(eleIt->pt());
      h1_recoEleEta_->Fill(eleIt->eta());
      h1_recoElePhi_->Fill(eleIt->phi());
      
    }   
  
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
      std::cout << calibElectrons.back().getRecoElectron()->superCluster()->energy() << " " << calibElectrons.back().getRecoElectron()->energy() << endl;
#endif
      h1_recoEleEnergy_->Fill(calibElectrons.back().getRecoElectron()->superCluster()->energy());
    }
#ifdef DEBUG
  std::cout << "Filled histos" << std::endl;
#endif  
  
  //COMBINATORY FOR Z MASS - begin                                                                                                                           
  std::vector< pair<calib::CalibElectron*,calib::CalibElectron*> > zeeCandidates;
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
	
	mass =  calculateZMass(std::pair<calib::CalibElectron*,calib::CalibElectron*>(&(calibElectrons[e_it]),&(calibElectrons[p_it])));
	
	if (mass<0)
	  continue;
	
	//#ifdef DEBUG
	std::cout << "#######################mass "<<mass << std::endl;
	//#endif
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


  h1_ZCandMult_->Fill(zeeCandidates.size());
  if(zeeCandidates.size()==0 || myBestZ==-1 )return kContinue;
  
#ifdef DEBUG  
  std::cout << "Found ZCandidates " << myBestZ << std::endl;
#endif  

  //  h1_zMassResol_ ->Fill(mass-myGenZMass);

  /////////////////////////////DUMP ELECTRON CLASS
  

  h1_eleClasses_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->classification());
  h1_eleClasses_->Fill(zeeCandidates[myBestZ].second->getRecoElectron()->classification());
  int class1 = zeeCandidates[myBestZ].first->getRecoElectron()->classification();
  int class2 = zeeCandidates[myBestZ].second->getRecoElectron()->classification();

  ///////////////////////

  if(class1==0 || class1==100)h1_occupancyVsEtaGold_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta());
  if(class2==0 || class2==100)h1_occupancyVsEtaGold_->Fill(zeeCandidates[myBestZ].second->getRecoElectron()->superCluster()->eta());

  if(class1==40 || class1==140)h1_occupancyVsEtaCrack_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta());
  if(class2==40 || class2==140)h1_occupancyVsEtaCrack_->Fill(zeeCandidates[myBestZ].second->getRecoElectron()->superCluster()->eta());

  if( (class1>=30 && class1<=34) || (class1>=130 && class1<=134) )h1_occupancyVsEtaShower_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta());
  if( (class2>=30 && class2<=34) || (class2>=130 && class2<=134) )h1_occupancyVsEtaShower_->Fill(zeeCandidates[myBestZ].second->getRecoElectron()->superCluster()->eta());
 

  if(class1==0 || class1==10 || class1 ==20 || class1==100 || class1==110 || class1 ==120)h1_occupancyVsEtaSilver_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta());
  if(class2==0 || class2==10 || class2 ==20 || class2==100 || class2==110 || class2 ==120)h1_occupancyVsEtaSilver_->Fill(zeeCandidates[myBestZ].second->getRecoElectron()->superCluster()->eta());

  ///////////////////////


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

  //dump f(eta)

  if( zeeCandidates[myBestZ].first->getRecoElectron()->classification()==0 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==10 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==20 ){
   
    h2_fEtaBarrelGood_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta(), 1. / fEtaBarrelGood(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta()) );

  }
  
  if( zeeCandidates[myBestZ].first->getRecoElectron()->classification()==30 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==31 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==32 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==33 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==34 ){
    
    h2_fEtaBarrelBad_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta(), 1. / fEtaBarrelBad(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta()) );
    
  }


  if( zeeCandidates[myBestZ].first->getRecoElectron()->classification()==100 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==110 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==120 ){
   
    h2_fEtaEndcapGood_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta(), 1. / fEtaEndcapGood(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta()) );

  }
  
  if( zeeCandidates[myBestZ].first->getRecoElectron()->classification()==130 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==131 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==132 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==133 || 
      zeeCandidates[myBestZ].first->getRecoElectron()->classification()==134 ){
    
    h2_fEtaEndcapBad_->Fill(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta(), 1. / fEtaEndcapBad(zeeCandidates[myBestZ].first->getRecoElectron()->superCluster()->eta()) );
    
  }
 
  //dump f(eta)


  ///////////////////////////ELECTRON SELECTION///////////////////////////////
  
  bool selectionBool=false;  
  //0 = all electrons (but no crack)
  
  bool invMassBool = ( (mass > minInvMassCut_) && (mass < maxInvMassCut_) );
 
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
                                                                                                                             
  if(electronSelection_==1)selectionBool=( myBestZ != -1 && 
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
	    h_DiffZMassDistr_[loopFlag_]->Fill( (mass-myGenZMass) );
	  }

      //PUT f(eta) IN OUR Zee ALGORITHM
      theAlgorithm_->addEvent(zeeCandidates[myBestZ].first, zeeCandidates[myBestZ].second,MZ*sqrt(ele1EnergyCorrection*ele2EnergyCorrection) );
     
      h1_reco_ZMass_->Fill(calculateZMass(zeeCandidates[myBestZ]));

      h1_reco_ZMassCorr_->Fill(calculateZMassWithCorrectedElectrons(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection));

      if(zeeCandidates[myBestZ].first->getRecoElectron()->classification()<100 && zeeCandidates[myBestZ].second->getRecoElectron()->classification()<100 )
	h1_reco_ZMassCorrBB_->Fill(calculateZMassWithCorrectedElectrons(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection));


      if(zeeCandidates[myBestZ].first->getRecoElectron()->classification()>=100 && zeeCandidates[myBestZ].second->getRecoElectron()->classification()>=100 )
	h1_reco_ZMassCorrEE_->Fill(calculateZMassWithCorrectedElectrons(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection));


      mass4tree = calculateZMassWithCorrectedElectrons(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection);

      massDiff4tree = calculateZMassWithCorrectedElectrons(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection) - myGenZMass;

      h_ZMassDistr_[loopFlag_]->Fill(calculateZMassWithCorrectedElectrons(zeeCandidates[myBestZ],ele1EnergyCorrection,ele2EnergyCorrection));

      h1_reco_ZEta_->Fill(calculateZEta(zeeCandidates[myBestZ]));
      h1_reco_ZTheta_->Fill(calculateZTheta(zeeCandidates[myBestZ]));
      h1_reco_ZRapidity_->Fill(calculateZRapidity(zeeCandidates[myBestZ]));
      h1_reco_ZPhi_->Fill(calculateZPhi(zeeCandidates[myBestZ]));
      h1_reco_ZPt_->Fill(calculateZPt(zeeCandidates[myBestZ]));

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
 h1_gen_ZEta_->Reset();
 h1_gen_ZRapidity_->Reset();
 h1_gen_ZPt_->Reset();
 
 h1_gen_ZMass_->Reset();
 h1_gen_ZPhi_->Reset();
 
 h2_fEtaBarrelGood_->Reset();
 h2_fEtaBarrelBad_->Reset();
 h2_fEtaEndcapGood_->Reset();
 h2_fEtaEndcapBad_->Reset();
 h1_eleClasses_->Reset();
 h1_nEleReco_-> Reset();
 h1_recoEleEnergy_-> Reset();
 h1_recoElePt_-> Reset();
 h1_recoEleEta_-> Reset();
 h1_recoElePhi_-> Reset();
 h1_ZCandMult_-> Reset();
 h1_reco_ZMass_-> Reset();
 h1_reco_ZEta_-> Reset();
 h1_reco_ZTheta_-> Reset();
 h1_reco_ZRapidity_-> Reset();
 h1_reco_ZPhi_-> Reset();
 h1_reco_ZPt_-> Reset();
 h1_reco_ZMassCorr_-> Reset();
 h1_reco_ZMassCorrBB_-> Reset();
 h1_reco_ZMassCorrEE_-> Reset();
 h1_occupancyVsEta_-> Reset();
 h1_occupancyVsEtaGold_-> Reset();
 h1_occupancyVsEtaSilver_-> Reset();
 h1_occupancyVsEtaShower_-> Reset();
 h1_occupancyVsEtaCrack_-> Reset();
 h1_occupancy_-> Reset();
}



// Called at end of loop
edm::EDLooper::Status
ZeeCalibration::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop)
{
  double par[3];
  double errpar[3];
  TF1* zGausFit=ZIterativeAlgorithmWithFit::gausfit(h1_reco_ZMass_,par,errpar,2.,2.);

  h2_zMassVsLoop_ -> Fill(loopFlag_,  par[1] );

  h2_zMassDiffVsLoop_ -> Fill(loopFlag_,  (par[1]-MZ)/MZ );

  h2_zWidthVsLoop_ -> Fill(loopFlag_, par[2] );
 
  //////////////////FIT Z PEAK

  std::cout<< "[ZeeCalibration] Ending loop " << iLoop<<std::endl;
  //RUN the algorithm
  theAlgorithm_->iterate();
  const std::vector<float>& optimizedCoefficients = theAlgorithm_->getOptimizedCoefficients();

#ifdef DEBUG
  std::cout<< "Optimized coefficients " << optimizedCoefficients.size() <<std::endl;
#endif

  h2_coeffVsLoop_->Fill(loopFlag_, optimizedCoefficients[75]); //show the evolution of just 1 ring coefficient (well chosen...)


  for (unsigned int ieta=0;ieta<optimizedCoefficients.size();ieta++)
    {
      calibCoeff[ieta] *= optimizedCoefficients[ieta];
#ifdef DEBUG
      std::cout<< ieta << " " << optimizedCoefficients[ieta] <<std::endl;  
#endif
      std::vector<DetId> ringIds = EcalRingCalibrationTools::getDetIdsInRing(ieta);
      
      for (unsigned int iid=0; iid<ringIds.size();++iid){

	ical->setValue( ringIds[iid], (*(ical->getMap().find(ringIds[iid]))) * optimizedCoefficients[ieta] );
      }    

    }
  
  
  ////////////////////////////////////////set recalib coefficient - end....NEED TO UPDATE FOR ENDCAPS
  

  /////////dump residual miscalibration at each loop

  for( int k = 0; k<theAlgorithm_->getNumberOfChannels(); k++ )
    {
      
      //h2_coeffVsEtaParz_[iLoop]->Fill( k, calibCoeff[k] );
      h2_miscalRecalParz_[iLoop]->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
      h1_mcParz_[iLoop]->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
      
      if(k<170){
	h2_miscalRecalEBParz_[iLoop]->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
	h1_mcEBParz_[iLoop]->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
      }
      
      if(k>=170){
	h2_miscalRecalEEParz_[iLoop]->Fill( initCalibCoeff[k], 1./calibCoeff[k] );
	h1_mcEEParz_[iLoop]->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
      }
    }


  /////////////////////////
  double parResidual[3];
  double errparResidual[3];
  TF1* zGausFitResidual=ZIterativeAlgorithmWithFit::gausfit(h1_mcParz_[iLoop],parResidual,errparResidual,3.,3.);
  //h1_mcParz_[iLoop]->Fit("gaus");
  
  h2_residualSigma_ -> Fill(loopFlag_,  parResidual[2]);
  std::cout<<"Fit on residuals, sigma is "<<parResidual[2]<<std::endl;

  /////////////////////
  outputFile_->cd();

  h2_miscalRecalParz_[iLoop]->Write();
  h1_mcParz_[iLoop]->Write();

  h2_miscalRecalEBParz_[iLoop]->Write();
  h1_mcEBParz_[iLoop]->Write();

  h2_miscalRecalEEParz_[iLoop]->Write();
  h1_mcEEParz_[iLoop]->Write();

  /////////dump residual miscalibration at each loop
  
  loopFlag_++;
  
#ifdef DEBUG  
  std::cout<<" loopFlag_ is "<<loopFlag_<<std::endl;
#endif  
  
  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;
}

float ZeeCalibration::calculateZMassWithCorrectedElectrons(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection)
{
  return ZIterativeAlgorithmWithFit::invMassCalc(aZCandidate.first->getRecoElectron()->superCluster()->energy() / ele1EnergyCorrection, aZCandidate.first->getRecoElectron()->eta(), aZCandidate.first->getRecoElectron()->phi(), aZCandidate.second->getRecoElectron()->superCluster()->energy() / ele2EnergyCorrection, aZCandidate.second->getRecoElectron()->eta(), aZCandidate.second->getRecoElectron()->phi());

}

float ZeeCalibration::calculateZMass(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  return  ZIterativeAlgorithmWithFit::invMassCalc(aZCandidate.first->getRecoElectron()->superCluster()->energy(), aZCandidate.first->getRecoElectron()->eta(), aZCandidate.first->getRecoElectron()->phi(), aZCandidate.second->getRecoElectron()->superCluster()->energy(), aZCandidate.second->getRecoElectron()->eta(), aZCandidate.second->getRecoElectron()->phi());  

}


float ZeeCalibration::calculateZRapidity(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  

  return  (ele1LV + ele2LV).Rapidity();

}

float ZeeCalibration::calculateZEta(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  
  return  (ele1LV + ele2LV).Eta();
  
}

float ZeeCalibration::calculateZTheta(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  
  return  (ele1LV + ele2LV).Theta();
  
}

float ZeeCalibration::calculateZPhi(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  
  return  (ele1LV + ele2LV).Phi();

}


float ZeeCalibration::calculateZPt(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{

  TLorentzVector ele1LV( aZCandidate.first->getRecoElectron()->px(), aZCandidate.first->getRecoElectron()->py(), aZCandidate.first->getRecoElectron()->pz(), aZCandidate.first->getRecoElectron()->superCluster()->energy());

  TLorentzVector ele2LV( aZCandidate.second->getRecoElectron()->px(), aZCandidate.second->getRecoElectron()->py(), aZCandidate.second->getRecoElectron()->pz(), aZCandidate.second->getRecoElectron()->superCluster()->energy());
  
  return  (ele1LV + ele2LV).Pt();

}



void ZeeCalibration::bookHistograms()
{
  h1_seedOverSC_= new TH1F("h1_seedOverSC", "h1_seedOverSC", 400, 0., 2.);

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
  
  for (int i=0;i<15;i++)
    {
      
      char histoName[50];
      sprintf(histoName,"h_ESCEtrueVsEta_%d",i);
      
      h_ESCEtrueVsEta_[i] = new TH2F(histoName,histoName, 150, 0., 2.7, 300,0.,1.5);
      h_ESCEtrueVsEta_[i]->SetXTitle("|#eta|");
      h_ESCEtrueVsEta_[i]->SetYTitle("E_{SC,raw}/E_{MC}");
      
      sprintf(histoName,"h_ESCEtrue_%d",i);

      h_ESCEtrue_[i] = new TH1F(histoName,histoName, 300,0.,1.5);

      sprintf(histoName,"h_ESCcorrEtrueVsEta_%d",i);
      
      h_ESCcorrEtrueVsEta_[i] = new TH2F(histoName,histoName, 150, 0., 2.7, 300,0.,1.5);
      h_ESCcorrEtrueVsEta_[i]->SetXTitle("|#eta|");
      h_ESCcorrEtrueVsEta_[i]->SetYTitle("E_{SC,#eta-corr}/E_{MC}");
      
      sprintf(histoName,"h_ESCcorrEtrue_%d",i);

      h_ESCcorrEtrue_[i] = new TH1F(histoName,histoName, 300,0.,1.5);
      
    }                                                                                              

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

  
  h1_zMassResol_ = new TH1F("zMassResol", "zMassResol", 200, -50., 50.);
  h1_zMassResol_->SetXTitle("M_{Z, reco} - M_{Z, MC}");
  h1_zMassResol_->SetYTitle("events");

  h1_eleEtaResol_ = new TH1F("eleEtaResol", "eleEtaResol", 100, -0.01, 0.01);
  h1_eleEtaResol_->SetXTitle("#eta_{reco} - #eta_{MC}");
  h1_eleEtaResol_->SetYTitle("events");

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

  /////single class occupancy
  
  h1_occupancyVsEtaGold_ = new TH1F("occupancyVsEtaGold","occupancyVsEtaGold", 200, -4.,4.);
  h1_occupancyVsEtaGold_->SetYTitle("Electron statistics");
  h1_occupancyVsEtaGold_->SetXTitle("Eta channel");

  h1_occupancyVsEtaSilver_ = new TH1F("occupancyVsEtaSilver","occupancyVsEtaSilver", 200, -4.,4.);
  h1_occupancyVsEtaSilver_->SetYTitle("Electron statistics");
  h1_occupancyVsEtaSilver_->SetXTitle("Eta channel");

  h1_occupancyVsEtaShower_ = new TH1F("occupancyVsEtaShower","occupancyVsEtaShower", 200, -4.,4.);
  h1_occupancyVsEtaShower_->SetYTitle("Electron statistics");
  h1_occupancyVsEtaShower_->SetXTitle("Eta channel");

  h1_occupancyVsEtaCrack_ = new TH1F("occupancyVsEtaCrack","occupancyVsEtaCrack", 200, -4.,4.);
  h1_occupancyVsEtaCrack_->SetYTitle("Electron statistics");
  h1_occupancyVsEtaCrack_->SetXTitle("Eta channel");

  /////single class occupancy
  
  h1_occupancy_ = new TH1F("occupancy","occupancy",100,0,400);
  h1_occupancy_->SetXTitle("Weighted electron statistics");
 

  h1_eleClasses_= new TH1F("eleClasses","eleClasses",301,-1,300);
  h1_eleClasses_->SetXTitle("classCode");
  h1_eleClasses_->SetYTitle("#");
  
  h1_recoEleEnergy_ = new TH1F("recoEleEnergy","EleEnergy from SC",300,0.,300.);
  h1_recoEleEnergy_->SetXTitle("eleSCEnergy(GeV)");
  h1_recoEleEnergy_->SetYTitle("events");
  
  h1_mcEle_Energy_ = new TH1F("mcEleEnergy","mc EleEnergy",300,0.,300.);
  h1_mcEle_Energy_->SetXTitle("E (GeV)");
  h1_mcEle_Energy_->SetYTitle("events");

  h1_recoElePt_ = new TH1F("recoElePt","p_{T} of reco electrons",300,0.,300.);
  h1_recoElePt_->SetXTitle("p_{T}(GeV/c)");
  h1_recoElePt_->SetYTitle("events");

  h1_mcElePt_ = new TH1F("mcElePt","p_{T} of MC electrons",300,0.,300.);
  h1_mcElePt_->SetXTitle("p_{T}(GeV/c)");
  h1_mcElePt_->SetYTitle("events");
  
  h1_recoEleEta_ = new TH1F("recoEleEta","Eta of reco electrons",100,-4.,4.);
  h1_recoEleEta_->SetXTitle("#eta");
  h1_recoEleEta_->SetYTitle("events");

  h1_mcEleEta_ = new TH1F("mcEleEta","Eta of MC electrons",100,-4.,4.);
  h1_mcEleEta_->SetXTitle("#eta");
  h1_mcEleEta_->SetYTitle("events");
  
  h1_recoElePhi_ = new TH1F("recoElePhi","Phi of reco electrons",100,-4.,4.);
  h1_recoElePhi_->SetXTitle("#phi");
  h1_recoElePhi_->SetYTitle("events");

  h1_mcElePhi_ = new TH1F("mcElePhi","Phi of MC electrons",100,-4.,4.);
  h1_mcElePhi_->SetXTitle("#phi");
  h1_mcElePhi_->SetYTitle("events");

  h1_gen_ZMass_ = new TH1F("gen_ZMass","Generated Z mass",200,0.,150.);
  h1_gen_ZMass_->SetXTitle("gen_ZMass (GeV)");
  h1_gen_ZMass_->SetYTitle("events");

  h1_gen_ZEta_ = new TH1F("gen_ZEta","Eta of gen Z",200,-6.,6.);
  h1_gen_ZEta_->SetXTitle("#eta");
  h1_gen_ZEta_->SetYTitle("events");

  h1_gen_ZRapidity_ = new TH1F("gen_ZRapidity","Rapidity of gen Z",200,-6.,6.);
  h1_gen_ZRapidity_->SetXTitle("Y");
  h1_gen_ZRapidity_->SetYTitle("events");

  h1_gen_ZPt_ = new TH1F("gen_ZPt","Pt of gen Z",200, 0.,100.);
  h1_gen_ZPt_->SetXTitle("p_{T} (GeV/c)");
  h1_gen_ZPt_->SetYTitle("events");

  h1_reco_ZEta_ = new TH1F("reco_ZEta","Eta of reco Z",200,-6.,6.);
  h1_reco_ZEta_->SetXTitle("#eta");
  h1_reco_ZEta_->SetYTitle("events");

  h1_reco_ZTheta_ = new TH1F("reco_ZTheta","Theta of reco Z",200, 0., 4.);
  h1_reco_ZTheta_->SetXTitle("#theta");
  h1_reco_ZTheta_->SetYTitle("events");

  h1_reco_ZRapidity_ = new TH1F("reco_ZRapidity","Rapidity of reco Z",200,-6.,6.);
  h1_reco_ZRapidity_->SetXTitle("Y");
  h1_reco_ZRapidity_->SetYTitle("events");

  h1_reco_ZPhi_ = new TH1F("reco_ZPhi","Phi of reco Z",100,-4.,4.);
  h1_reco_ZPhi_->SetXTitle("#phi");
  h1_reco_ZPhi_->SetYTitle("events");

  h1_reco_ZPt_ = new TH1F("reco_ZPt","Pt of reco Z",200,0.,100.);
  h1_reco_ZPt_->SetXTitle("p_{T} (GeV/c)");
  h1_reco_ZPt_->SetYTitle("events");

  h1_gen_ZPhi_ = new TH1F("gen_ZPhi","Phi of gen Z",100,-4.,4.);
  h1_gen_ZPhi_->SetXTitle("#phi");
  h1_gen_ZPhi_->SetYTitle("events");

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

  h2_zMassVsLoop_= new TH2F("h2_zMassVsLoop","h2_zMassVsLoop",20,0,20, 90, 80.,95.);

  h2_zMassDiffVsLoop_= new TH2F("h2_zMassDiffVsLoop","h2_zMassDiffVsLoop",20,0,20, 100, -1., 1.);
  h2_zMassDiffVsLoop_->SetXTitle("Iteration");
  h2_zMassDiffVsLoop_->SetYTitle("M_{Z, reco peak} - M_{Z, true}");

  h2_zWidthVsLoop_= new TH2F("h2_zWidthVsLoop","h2_zWidthVsLoop",20,0,20, 100, 0.,10.);

  h2_coeffVsLoop_= new TH2F("h2_coeffVsLoop","h2_coeffVsLoop",20,0,20, 100, 0., 2.);

  h2_residualSigma_= new TH2F("h2_residualSigma","h2_residualSigma",20, 0, 20, 100, 0., .5);

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

  for (int i=0;i<15;i++)
    {
      char histoName[50];
      
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

      sprintf(histoName,"h1_residualMiscalibParz_%d",i);
      h1_mcParz_[i] = new TH1F(histoName,histoName, 200, -0.2, 0.2);
      sprintf(histoName,"h1_residualMiscalibEBParz_%d",i);
      h1_mcEBParz_[i] = new TH1F(histoName,histoName, 200, -0.2, 0.2);
      sprintf(histoName,"h1_residualMiscalibEEParz_%d",i);
      h1_mcEEParz_[i] = new TH1F(histoName,histoName, 200, -0.2, 0.2);
      
    }


}


double ZeeCalibration::fEtaBarrelBad(double scEta) const{
  
  // f(eta) for the class = 30 (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  9.99063e-01; 
  float p1 = -2.63341e-02; 
  float p2 =  5.16054e-02; 
  float p3 = -4.95976e-02; 
  float p4 =  3.62304e-03; 

  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;  

}
  
double ZeeCalibration::fEtaEndcapGood(double scEta) const{

  // f(eta) for the first 3 classes (100, 110 and 120) 
  // Ivica's new corrections 01/06
  float p0 =        -8.51093e-01; 
  float p1 =         3.54266e+00;
  float p2 =        -2.59288e+00;
  float p3 =         8.58945e-01;
  float p4 =        -1.07844e-01; 

  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x; 

}

double ZeeCalibration::fEtaEndcapBad(double scEta) const{
  
  // f(eta) for the class = 130-134 
  // Ivica's new corrections 01/06
  float p0 =        -4.25221e+00; 
  float p1 =         1.01936e+01;
  float p2 =        -7.48247e+00;
  float p3 =         2.45520e+00;
  float p4 =        -3.02872e-01;

  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;  

}
  
double ZeeCalibration::fEtaBarrelGood(double scEta) const{

  // f(eta) for the first 3 classes (0, 10 and 20) (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  1.00149e+00; 
  float p1 = -2.06622e-03; 
  float p2 = -1.08793e-02; 
  float p3 =  1.54392e-02; 
  float p4 = -1.02056e-02; 

  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x; 

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
	  h1_eleEtaResol_->Fill(myEle->eta() - mcEle[i]->momentum().eta() );
	  h1_elePhiResol_->Fill(myEle->phi() - mcEle[i]->momentum().phi() );

          const reco::SuperCluster* mySC=&(*(myEle->superCluster()));
	  if (/*fabs(mySC->position().eta()) < 2.4*/1)
	    {
	      if(myEle->classification()>=100)std::cout<<"mySC->preshowerEnergy()"<<mySC->preshowerEnergy()<<std::endl;

	      h_ESCEtrue_[loopFlag_]->Fill(mySC->energy()/mcEle[i]->momentum().e());
	      h_ESCEtrueVsEta_[loopFlag_]->Fill(fabs(mySC->position().eta()),mySC->energy()/mcEle[i]->momentum().e());

	      double corrSCenergy = ( mySC->energy() )/getEtaCorrection(myEle);
	      h_ESCcorrEtrue_[loopFlag_]->Fill(corrSCenergy/mcEle[i]->momentum().e());
	      h_ESCcorrEtrueVsEta_[loopFlag_]->Fill(fabs(mySC->position().eta()),corrSCenergy/mcEle[i]->momentum().e());

	      std::vector<DetId> mySCRecHits = mySC->seed()->getHitsByDetId();
	      
	      
	      
	      

	      h1_seedOverSC_->Fill( mySC->seed()->energy() / mySC->energy() );
	      h1_preshowerOverSC_->Fill( mySC->preshowerEnergy() / mySC->energy() );
	      
	    }

        }
    }
}

int ZeeCalibration::ringNumberCorrector(int k)
{

  int index=-999;

  if(k>=0 && k<=84)index = k - 85;

  if(k>=85 && k<=169)index = k - 84;

  if(k>=170 && k<=208)index = - k + 84;

  if(k>=209 && k<=247)index = k - 123;
  
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

/*
 DetId* getHottestDetId(std::vector<DetId> mySCRecHits){

   reco::EBRecHitCollection myEBHitCollection;

   reco::EERecHitCollection myEEHitCollection;

   for(   std::vector<DetId>::const_iterator idIt=mySCRecHits.begin(); idIt != mySCRecHits.end(); idIt++){


  for(reco::EBRecHitCollection::const_iterator ebIt = hits->begin(); ebIt != hits->end(); ebIt++){
    if( ebIt->id() ==*(&idIt)  )myRhcollection.push_back(ebIt);
  }

  for(reco::EERecHitCollection::const_iterator eeIt = ehits->begin(); eeIt != ehits->end(); eeIt++){

  }



   }

}

*/


