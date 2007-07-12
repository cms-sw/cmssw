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

#include "CLHEP/Vector/LorentzVector.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPhiSymRecHitsProducer.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeCalibration.h"

#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <map>
#include <fstream>

#define MZ 91.1876


ZeeCalibration::ZeeCalibration(const edm::ParameterSet& iConfig)
{
  theMaxLoops =  iConfig.getUntrackedParameter<unsigned int>("maxLoops",0);
  
  outputFileName_  = iConfig.getParameter<std::string>("outputFile");

  rechitProducer_ = iConfig.getParameter<std::string>("rechitProducer");
  rechitCollection_ = iConfig.getParameter<std::string>("rechitCollection");
  scProducer_ = iConfig.getParameter<std::string>("scProducer");
  mcProducer_ = iConfig.getUntrackedParameter<std::string>("mcProducer","");
  scCollection_ = iConfig.getParameter<std::string>("scCollection");

  electronProducer_ = iConfig.getParameter<std::string > ("electronProducer");
  electronCollection_ = iConfig.getParameter<std::string > ("electronCollection");

  outputFile_ = TFile::Open(outputFileName_.c_str(),"RECREATE"); // open output file to store histograms  
  
  barrelfile_=iConfig.getUntrackedParameter<std::string> ("initialMiscalibrationBarrel","");

  theParameterSet=iConfig;

  //creating the algorithm
  theAlgorithm_ = new ZIterativeAlgorithmWithFit(iConfig);
  // Tell the framework what data is being produced
  //setWhatProduced(this);
  setWhatProduced (this, &ZeeCalibration::produceEcalIntercalibConstants ) ;
  findingRecord<EcalIntercalibConstantsRcd> () ;

}


ZeeCalibration::~ZeeCalibration()
{
  if (theAlgorithm_)
    delete theAlgorithm_;
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
  //  std::cout << "Zee initializing" << std::endl;
//========================================================================
//void
//ZeeCalibration::beginJob(edm::EventSetup const& iSetup) {
//========================================================================
  
  
// go to *OUR* rootfile and book histograms                                                                                                                
  outputFile_->cd();

// Book histograms 
  bookHistograms();


  loopFlag_ = 0;

  //Read miscalibration map if requested
  CaloMiscalibMapEcal* miscalibMap=0;
  if(!barrelfile_.empty())
    {
      miscalibMap=new CaloMiscalibMapEcal();
      miscalibMap->prefillMap();
      MiscalibReaderFromXMLEcalBarrel barrelreader_(*miscalibMap);
      barrelreader_.parseXMLMiscalibFile(barrelfile_);
    }
  
  for(int k = 0; k < theAlgorithm_->getNumberOfChannels(); k++)
    {
      calibCoeff[k]=1.;
      if (miscalibMap)
	{
	  int etaIndex =0;

	  if (k<85)
	    etaIndex=-85 + k;
	  else
	    etaIndex= k - 84;
	  int xtalsInPhi=0;
	  initCalibCoeff[k]=0.;
	  for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) 
	    {
	      try
		{
		  EBDetId ebid(etaIndex,iphi);
		  float miscalib=miscalibMap->get().getMap().find(ebid.rawId())->second;
		  initCalibCoeff[k]+=miscalib;
		  xtalsInPhi++;
		}
	      catch(...)
		{
		}
	    }
	  initCalibCoeff[k]/=(float)xtalsInPhi;
	  std::cout << k << " " << initCalibCoeff[k] << " " << xtalsInPhi << std::endl;
	}
      else
	{
	  initCalibCoeff[k]=1.;
	}
    }

  ical = boost::shared_ptr<EcalIntercalibConstants>( new EcalIntercalibConstants() );
  
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) 
    {

      if(ieta==0) continue;
      
      int etaIndex;
      if (ieta < 0)
	etaIndex =  ieta + 85;
      else
	etaIndex =  ieta + 84;
      
      for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
	// make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
	try
	  {
	    EBDetId ebid(ieta,iphi);
	    ical->setValue( ebid.rawId(), 1. * initCalibCoeff[etaIndex] );
	  }
	catch (...)
	  {
	  }
      }
    }
  
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      try
	{
	  EEDetId eedetidpos(iX,iY,1);
	  ical->setValue( eedetidpos.rawId(), 1.);
	}
      catch(...)
	{
	}
      try
        {
	  EEDetId eedetidneg(iX,iY,-1);
	  ical->setValue( eedetidneg.rawId(), 1.);
	}
      catch(...)
	{
	}
    }
  }

  read_events = 0;
  //  std::cout << "Zee initialized" << std::endl;
}


//========================================================================
void
ZeeCalibration::endOfJob() {

  //  std::cout<<"Writing  histos..."<<std::endl;
  outputFile_->cd();

  h1_nEleReco_->Write();
  h1_recoEleEnergy_->Write();
  h1_recoElePt_->Write();
  h1_recoEleEta_->Write();
  h1_recoElePhi_->Write();
  h1_ZCandMult_->Write();
  h1_reco_ZMass_->Write();

  h1_mcEle_Energy_->Write();
  h1_mcElePt_->Write();
  h1_mcEleEta_->Write();
  h1_mcElePhi_->Write();

  for( int k = 0; k<170; k++ ){
    
    h2_coeffVsEta_->Fill( k, calibCoeff[k] );
    h2_miscalRecal_->Fill( initCalibCoeff[k], calibCoeff[k] );
    h1_mc_->Fill( initCalibCoeff[k]*calibCoeff[k] -1. );
  }
  
  h2_coeffVsEta_->Write();
  h2_zMassVsLoop_->Write();
  h2_zWidthVsLoop_->Write();
  h2_coeffVsLoop_->Write();
  h2_miscalRecal_->Write();
  h1_mc_->Write();
  
  const ZIterativeAlgorithmWithFit::ZIterativeAlgorithmWithFitPlots* algoHistos=theAlgorithm_->getHistos();
  for (int iIteration=0;iIteration<theAlgorithm_->getNumberOfIterations();iIteration++)
    for (int iChannel=0;iChannel<theAlgorithm_->getNumberOfChannels();iChannel++)
      {
	algoHistos->weightedRescaleFactor[iIteration][iChannel]->Write();
	algoHistos->unweightedRescaleFactor[iIteration][iChannel]->Write();
	algoHistos->weight[iIteration][iChannel]->Write();
      }

  //  std::cout<<"Done! Closing output file... "<<std::endl;
  
  outputFile_->Close();

  //Writing out calibration coefficients
  calibXMLwriter barrelWriter(EcalBarrel);
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      try
	{
	  EBDetId ebid(ieta,iphi);
	  barrelWriter.writeLine(ebid,ical->getMap().find(ebid.rawId())->second);
	}
      catch (...)
	{
	}
    }
  }
  
  //  std::cout<<"Done!"<<std::endl;
  
}

//_____________________________________________________________________________
// Called at each event
//________________________________________

edm::EDLooper::Status
ZeeCalibration::duringLoop( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;

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
	  //      h1_gen_ZMass_->Fill((*p)->momentum().m());
	  genZ=(*p);
	}
      }
      //DUMP GENERATED Z MASS - END
      
      //loop over MC positrons and find the closest MC positron in (eta,phi) phace space - begin
      HepMC::GenParticle MCele;
      if (loopFlag_ == 0)
	{
	  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
		p != myGenEvent->particles_end(); ++p ) {
	    
	    if (  abs( (*p)->pdg_id() ) == 11 )
	      //&& (*p)->Mother() == 7 )
	      {
		MCele=*(*p);
		h1_mcEle_Energy_->Fill( MCele.momentum().e() );
		h1_mcElePt_->Fill( MCele.momentum().perp() );
		h1_mcEleEta_->Fill( MCele.momentum().eta() );
		h1_mcElePhi_->Fill( MCele.momentum().phi() );
	      }
	  }
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
  
  //Get SuperClusters
  Handle<reco::SuperClusterCollection> pSuperClusters;
  try {
    iEvent.getByLabel(scProducer_, scCollection_, pSuperClusters);
  } catch (std::exception& ex ) {
    std::cerr << "Error! can't get the product SuperClusterCollection "<< std::endl;
  }
  const reco::SuperClusterCollection* scCollection = pSuperClusters.product();

#ifdef DEBUG
  std::cout<<"scCollection->size()"<<scCollection->size()<<std::endl;
#endif

  if(scCollection->size() < 2) return kContinue;

  // Get Electrons
  Handle<reco::PixelMatchGsfElectronCollection> pElectrons;
  try {
    iEvent.getByLabel(electronProducer_, electronCollection_, pElectrons);
  } catch (std::exception& ex ) {
    std::cerr << "Error! can't get the product ElectronCollection "<< std::endl;
  }
  const reco::PixelMatchGsfElectronCollection* electronCollection = pElectrons.product();
  
  if(electronCollection->size() < 2) return kContinue;
  
  if (!hits){
    std::cout << "!hits" << std::endl;   
    return kContinue;
  }
  
  if (hits->size() == 0){
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
  
  ////////////////////////////////////////////////////////////////////////////////////////
  ///                          START HERE....
  ///////////////////////////////////////////////////////////////////////////////////////
  read_events++;
  //  std::cout << "read_events = " << read_events << std::endl;
  
  ////////////////////////CHOOSE EVENTS WITH 2 OR MORE ELECTRONS IN THE BARREL/////////////////////
  
  if(electronCollection->size()<2)
    return kContinue;
  
  h1_nEleReco_->Fill(electronCollection->size());
  
  for(reco::PixelMatchGsfElectronCollection::const_iterator eleIt = electronCollection->begin();   eleIt != electronCollection->end(); eleIt++)
    {
      
      h1_recoElePt_->Fill(eleIt->pt());
      h1_recoEleEta_->Fill(eleIt->eta());
      h1_recoElePhi_->Fill(eleIt->phi());
      
    }   
  
  //FILL an electron vector - end
  //###################################Electron-SC association map: begin#####################################################
  //Filling new ElectronCollection with new SC ref and calibElectron container
#ifdef DEBUG
  std::cout << "Filling new electrons  " << std::endl;
#endif
  reco::PixelMatchGsfElectronCollection newElectrons;
  std::vector<calib::CalibElectron> calibElectrons;
  //  std::map< const reco::PixelMatchGsfElectron* , const reco::SuperCluster* > eleScMap;
  
  for(reco::PixelMatchGsfElectronCollection::const_iterator eleIt = electronCollection->begin(); eleIt != electronCollection->end(); eleIt++)
    {
      float DeltaRMineleSC(0.15); 
      const reco::SuperCluster* nearestSC=0;
      int iscRef=-1;
      int iSC=0;
      for(reco::SuperClusterCollection::const_iterator scIt = scCollection->begin();
	  scIt != scCollection->end(); scIt++){
#ifdef DEBUG	
	std::cout << scIt->energy() << " " << scIt->eta() << " " << scIt->phi() << " " << eleIt->eta() << " " << eleIt->phi() << std::endl;
#endif
	double DeltaReleSC = sqrt ( pow(  eleIt->eta() - scIt->eta(),2) + pow(eleIt->phi() - scIt->phi(),2));
	
	if(DeltaReleSC<DeltaRMineleSC)
	  {
	    DeltaRMineleSC = DeltaReleSC;
	    nearestSC = &*scIt;
	    iscRef = iSC;
	  }
	iSC++;
      }
      
      if(nearestSC){
	//NB an iterator is NOT a pointer! You MUST first dereference it, and then take the address!
	reco::PixelMatchGsfElectron newEle(*eleIt);
	newEle.setGsfTrack(eleIt->gsfTrack());
	reco::SuperClusterRef scRef(reco::SuperClusterRef(pSuperClusters, iscRef));
	newEle.setSuperCluster(scRef);
	newElectrons.push_back(newEle);
      }  
    }
  
#ifdef DEBUG
  std::cout << "Filled new electrons  " << newElectrons.size() << std::endl;
#endif
  //#####################################Electron-SC association map: end#####################################################  
  for(unsigned int e_it = 0 ; e_it != newElectrons.size() ; e_it++)
    {
      calibElectrons.push_back(calib::CalibElectron(&(newElectrons[e_it]),hits));
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
  
  double mass(-1.);
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
	
#ifdef DEBUG
	std::cout << mass << std::endl;
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
  
  h1_ZCandMult_->Fill(zeeCandidates.size());
#ifdef DEBUG  
  std::cout << "Found ZCandidates " << myBestZ << std::endl;
#endif  
  if (myBestZ != -1)
    {
      theAlgorithm_->addEvent(zeeCandidates[myBestZ].first, zeeCandidates[myBestZ].second, MZ);
      h1_reco_ZMass_->Fill(calculateZMass(zeeCandidates[myBestZ]));
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
 h1_nEleReco_-> Reset();
 h1_recoEleEnergy_-> Reset();
 h1_recoElePt_-> Reset();
 h1_recoEleEta_-> Reset();
 h1_recoElePhi_-> Reset();
 h1_ZCandMult_-> Reset();
 h1_reco_ZMass_-> Reset();
}



// Called at end of loop
edm::EDLooper::Status
ZeeCalibration::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop)
{
  double par[3];
  double errpar[3];
  TF1* zGausFit=ZIterativeAlgorithmWithFit::gausfit(h1_reco_ZMass_,par,errpar,2.,2.);

  h2_zMassVsLoop_ -> Fill(loopFlag_,  par[1] );
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
      int etaIndex =0;
      if (ieta<85)
	etaIndex=-85 + ieta;
      else
	etaIndex= ieta - 84;
      
      for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) 
	{
	  // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
	  try
	    {
	      EBDetId ebid(etaIndex,iphi);
#ifdef DEBUG
	      std::cout << "Before " << ical->getMap().find(ebid.rawId())->second << " " ;
#endif
	      ical->setValue( ebid.rawId(), ical->getMap().find(ebid.rawId())->second * optimizedCoefficients[ieta] );
#ifdef DEBUG
	      std::cout << "After " << ical->getMap().find(ebid.rawId())->second << std::endl;
#endif
	    }
	  catch (...)
	    {
	    }  
	}    
      }

  loopFlag_++;

#ifdef DEBUG  
  std::cout<<" loopFlag_ is "<<loopFlag_<<std::endl;
#endif  

  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;
}

float ZeeCalibration::calculateZMass(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate)
{
	return  ZIterativeAlgorithmWithFit::invMassCalc(aZCandidate.first->getRecoElectron()->superCluster()->energy(), aZCandidate.first->getRecoElectron()->eta(), aZCandidate.first->getRecoElectron()->phi(), aZCandidate.second->getRecoElectron()->superCluster()->energy(), aZCandidate.second->getRecoElectron()->eta(), aZCandidate.second->getRecoElectron()->phi());  
}


void ZeeCalibration::bookHistograms()
{
  h1_nEleReco_ = new TH1F("nEleReco","Number of reco electrons",10,-0.5,10.5);
  h1_nEleReco_->SetXTitle("nEleReco");
  h1_nEleReco_->SetYTitle("events");
  
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
  
  h1_ZCandMult_ =new TH1F("ZCandMult","Multiplicity of Z candidates in one event",10,-0.5,10.5);
  h1_ZCandMult_ ->SetXTitle("ZCandMult");
  
  h1_reco_ZMass_ = new TH1F("reco_ZMass","Inv. mass of 2 reco Electrons",200,0.,150.);
  h1_reco_ZMass_->SetXTitle("reco_ZMass");
  h1_reco_ZMass_->SetYTitle("events");

  h2_coeffVsEta_= new TH2F("h2_calibCoeffVsEta","h2_calibCoeffVsEta",171,0,171, 100, 0., 2.);

  h2_zMassVsLoop_= new TH2F("h2_zMassVsLoop","h2_zMassVsLoop",20,0,20, 90, 80.,95.);

  h2_zWidthVsLoop_= new TH2F("h2_zWidthVsLoop","h2_zWidthVsLoop",20,0,20, 100, 0.,10.);

  h2_coeffVsLoop_= new TH2F("h2_coeffVsLoop","h2_coeffVsLoop",20,0,20, 100, 0., 2.);

  h2_miscalRecal_ = new TH2F("h2_miscalRecal","h2_miscalRecal", 100, 0., 2., 100, 0., 2.);

  h1_mc_ = new TH1F("h1_mc","h1_mc", 200, -1., 1.);
}
