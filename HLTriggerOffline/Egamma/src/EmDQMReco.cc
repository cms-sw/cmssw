////////////////////////////////////////////////////////////////////////////
//                    Header file for this                                    //
////////////////////////////////////////////////////////////////////////////////
#include "HLTriggerOffline/Egamma/interface/EmDQMReco.h"

////////////////////////////////////////////////////////////////////////////////
//                    Collaborating Class Header                              //
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
//#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
//#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/TriggerResults.h" 
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
////////////////////////////////////////////////////////////////////////////////
//                           Root include files                               //
////////////////////////////////////////////////////////////////////////////////
#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"
#include <iostream>
#include <string>
#include <Math/VectorUtil.h>
using namespace ROOT::Math::VectorUtil ;


////////////////////////////////////////////////////////////////////////////////
//                             Constructor                                    //
////////////////////////////////////////////////////////////////////////////////
EmDQMReco::EmDQMReco(const edm::ParameterSet& pset)  
{

  dbe = edm::Service < DQMStore > ().operator->();
  dbe->setVerbose(0);


  ////////////////////////////////////////////////////////////
  //          Read from configuration file                  //
  ////////////////////////////////////////////////////////////
  dirname_="HLT/HLTEgammaValidation/"+pset.getParameter<std::string>("@module_label");
  dbe->setCurrentFolder(dirname_);

  // paramters for generator study 
  reqNum    = pset.getParameter<unsigned int>("reqNum");
  pdgGen    = pset.getParameter<int>("pdgGen");
  recoEtaAcc = pset.getParameter<double>("genEtaAcc");
  recoEtAcc  = pset.getParameter<double>("genEtAcc");
  // plotting paramters (untracked because they don't affect the physics)
  plotPtMin  = pset.getUntrackedParameter<double>("PtMin",0.);
  plotPtMax  = pset.getUntrackedParameter<double>("PtMax",1000.);
  plotEtaMax = pset.getUntrackedParameter<double>("EtaMax", 2.7);
  plotBins   = pset.getUntrackedParameter<unsigned int>("Nbins",50);
  useHumanReadableHistTitles = pset.getUntrackedParameter<bool>("useHumanReadableHistTitles", false);

  //preselction cuts 
  // recocutCollection_= pset.getParameter<edm::InputTag>("cutcollection");
  recocut_          = pset.getParameter<int>("cutnum");

  // prescale = 10;
  eventnum = 0;

  // just init
  isHltConfigInitialized_ = false;

  ////////////////////////////////////////////////////////////
  //         Read in the Vector of Parameter Sets.          //
  //           Information for each filter-step             //
  ////////////////////////////////////////////////////////////
  std::vector<edm::ParameterSet> filters = 
       pset.getParameter<std::vector<edm::ParameterSet> >("filters");

  int i = 0;
  for(std::vector<edm::ParameterSet>::iterator filterconf = filters.begin() ; filterconf != filters.end() ; filterconf++)
  {

    theHLTCollectionLabels.push_back(filterconf->getParameter<edm::InputTag>("HLTCollectionLabels"));
    theHLTOutputTypes.push_back(filterconf->getParameter<int>("theHLTOutputTypes"));
    // Grab the human-readable name, if it is not specified, use the Collection Label
    theHLTCollectionHumanNames.push_back(filterconf->getUntrackedParameter<std::string>("HLTCollectionHumanName",theHLTCollectionLabels[i].label()));

    std::vector<double> bounds = filterconf->getParameter<std::vector<double> >("PlotBounds");
    // If the size of plot "bounds" vector != 2, abort
    assert(bounds.size() == 2);
    plotBounds.push_back(std::pair<double,double>(bounds[0],bounds[1]));
    isoNames.push_back(filterconf->getParameter<std::vector<edm::InputTag> >("IsoCollections"));
    // If the size of the isoNames vector is not greater than zero, abort
    assert(isoNames.back().size()>0);
    if (isoNames.back().at(0).label()=="none") {
      plotiso.push_back(false);
    } else {
      plotiso.push_back(true);
    }
    i++;
  } // END of loop over parameter sets

  // Record number of HLTCollectionLabels
  numOfHLTCollectionLabels = theHLTCollectionLabels.size();
  
}



///
///
///
void EmDQMReco::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup ) {

  bool isHltConfigChanged = false; // change of cfg at run boundaries?
  isHltConfigInitialized_ = hltConfig_.init( iRun, iSetup, "HLT", isHltConfigChanged );

}





////////////////////////////////////////////////////////////////////////////////
//       method called once each job just before starting event loop          //
////////////////////////////////////////////////////////////////////////////////
void 
EmDQMReco::beginJob()
{
  //edm::Service<TFileService> fs;
  dbe->setCurrentFolder(dirname_);

  ////////////////////////////////////////////////////////////
  //  Set up Histogram of Effiency vs Step.                 //
  //   theHLTCollectionLabels is a vector of InputTags      //
  //    from the configuration file.                        //
  ////////////////////////////////////////////////////////////

  std::string histName="total_eff";
  std::string histTitle = "total events passing";
  // This plot will have bins equal to 2+(number of
  //        HLTCollectionLabels in the config file)
  totalreco = dbe->book1D(histName.c_str(),histTitle.c_str(),numOfHLTCollectionLabels+2,0,numOfHLTCollectionLabels+2);
  totalreco->setBinLabel(numOfHLTCollectionLabels+1,"Total");
  totalreco->setBinLabel(numOfHLTCollectionLabels+2,"Reco");
  for (unsigned int u=0; u<numOfHLTCollectionLabels; u++){totalreco->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());}

  histName="total_eff_RECO_matched";
  histTitle="total events passing (Reco matched)";
  totalmatchreco = dbe->book1D(histName.c_str(),histTitle.c_str(),numOfHLTCollectionLabels+2,0,numOfHLTCollectionLabels+2);
  totalmatchreco->setBinLabel(numOfHLTCollectionLabels+1,"Total");
  totalmatchreco->setBinLabel(numOfHLTCollectionLabels+2,"Reco");
  for (unsigned int u=0; u<numOfHLTCollectionLabels; u++){totalmatchreco->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());}

  MonitorElement* tmphisto;
  MonitorElement* tmpiso;

  ////////////////////////////////////////////////////////////
  // Set up generator-level histograms                      //
  ////////////////////////////////////////////////////////////
  std::string pdgIdString;
  switch(pdgGen) {
  case 11:
    pdgIdString="Electron";break;
  case 22:
    pdgIdString="Photon";break;
  default:
    pdgIdString="Particle";
  }

  histName = "reco_et";
  histTitle= "E_{T} of " + pdgIdString + "s" ;
  etreco =  dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
  histName = "reco_eta";
  histTitle= "#eta of "+ pdgIdString +"s " ;
  etareco = dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
 
  histName = "reco_et_monpath";
  histTitle= "E_{T} of " + pdgIdString + "s monpath" ;
  etrecomonpath =  dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
  histName = "reco_eta_monpath";
  histTitle= "#eta of "+ pdgIdString +"s monpath" ;
  etarecomonpath = dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
 
  histName = "final_et_monpath";
  histTitle = "Final Et Monpath";
  ethistmonpath = dbe->book1D(histName.c_str(), histTitle.c_str(), plotBins, plotPtMin, plotPtMax); 

  histName = "final_eta_monpath";
  histTitle = "Final Eta Monpath";
  etahistmonpath = dbe->book1D(histName.c_str(), histTitle.c_str(), plotBins, -plotEtaMax, plotEtaMax); 

  ////////////////////////////////////////////////////////////
  //  Set up histograms of HLT objects                      //
  ////////////////////////////////////////////////////////////

  // Determine what strings to use for histogram titles
  std::vector<std::string> HltHistTitle;
  if ( theHLTCollectionHumanNames.size() == numOfHLTCollectionLabels && useHumanReadableHistTitles ) {
    HltHistTitle = theHLTCollectionHumanNames;
  } else {
    for (unsigned int i =0; i < numOfHLTCollectionLabels; i++) {
      HltHistTitle.push_back(theHLTCollectionLabels[i].label());
    }
  }
 
  for(unsigned int i = 0; i< numOfHLTCollectionLabels ; i++){
    // Et distribution of HLT objects passing filter i
    histName = theHLTCollectionLabels[i].label()+"et_all";
    histTitle = HltHistTitle[i]+" Et (ALL)";
    tmphisto =  dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
    ethist.push_back(tmphisto);
    
    // Eta distribution of HLT objects passing filter i
    histName = theHLTCollectionLabels[i].label()+"eta_all";
    histTitle = HltHistTitle[i]+" #eta (ALL)";
    tmphisto =  dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
    etahist.push_back(tmphisto);          

    // Et distribution of reco object matching HLT object passing filter i
    histName = theHLTCollectionLabels[i].label()+"et_RECO_matched";
    histTitle = HltHistTitle[i]+" Et (RECO matched)";
    tmphisto =  dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
    ethistmatchreco.push_back(tmphisto);
    
    // Eta distribution of Reco object matching HLT object passing filter i
    histName = theHLTCollectionLabels[i].label()+"eta_RECO_matched";
    histTitle = HltHistTitle[i]+" #eta (RECO matched)";
    tmphisto =  dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
    etahistmatchreco.push_back(tmphisto);


    // Et distribution of reco object matching HLT object passing filter i
    histName = theHLTCollectionLabels[i].label()+"et_RECO_matched_monpath";
    histTitle = HltHistTitle[i]+" Et (RECO matched, monpath)";
    tmphisto =  dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
    ethistmatchrecomonpath.push_back(tmphisto);
    
    // Eta distribution of Reco object matching HLT object passing filter i
    histName = theHLTCollectionLabels[i].label()+"eta_RECO_matched_monpath";
    histTitle = HltHistTitle[i]+" #eta (RECO matched, monpath)";
    tmphisto =  dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
    etahistmatchrecomonpath.push_back(tmphisto);

    // Et distribution of HLT object that is closest delta-R match to sorted reco particle(s)
    histName  = theHLTCollectionLabels[i].label()+"et_reco";
    histTitle = HltHistTitle[i]+" Et (reco)";
    tmphisto  = dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
    histEtOfHltObjMatchToReco.push_back(tmphisto);

    // eta distribution of HLT object that is closest delta-R match to sorted reco particle(s)
    histName  = theHLTCollectionLabels[i].label()+"eta_reco";
    histTitle = HltHistTitle[i]+" eta (reco)";
    tmphisto  = dbe->book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
    histEtaOfHltObjMatchToReco.push_back(tmphisto);
    
    if (!plotiso[i]) {
      tmpiso = NULL;
      etahistiso.push_back(tmpiso);
      ethistiso.push_back(tmpiso);
      etahistisomatchreco.push_back(tmpiso);
      ethistisomatchreco.push_back(tmpiso);
      histEtaIsoOfHltObjMatchToReco.push_back(tmpiso);
      histEtIsoOfHltObjMatchToReco.push_back(tmpiso);
    } else {
      // 2D plot: Isolation values vs eta for all objects
      histName  = theHLTCollectionLabels[i].label()+"eta_isolation_all";
      histTitle = HltHistTitle[i]+" isolation vs #eta (all)";
      tmpiso    = dbe->book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      etahistiso.push_back(tmpiso);

      // 2D plot: Isolation values vs et for all objects
      histName  = theHLTCollectionLabels[i].label()+"et_isolation_all";
      histTitle = HltHistTitle[i]+" isolation vs Et (all)";
      tmpiso    = dbe->book2D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      ethistiso.push_back(tmpiso);

      // 2D plot: Isolation values vs eta for reco matched objects
      histName  = theHLTCollectionLabels[i].label()+"eta_isolation_RECO_matched";
      histTitle = HltHistTitle[i]+" isolation vs #eta (reco matched)";
      tmpiso    = dbe->book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      etahistisomatchreco.push_back(tmpiso);

      // 2D plot: Isolation values vs et for matched objects
      histName  = theHLTCollectionLabels[i].label()+"et_isolation_RECO_matched";
      histTitle = HltHistTitle[i]+" isolation vs Et (reco matched)";
      tmpiso    = dbe->book2D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      ethistisomatchreco.push_back(tmpiso);

      // 2D plot: Isolation values vs eta for HLT object that 
      // is closest delta-R match to sorted reco particle(s)
      histName  = theHLTCollectionLabels[i].label()+"eta_isolation_reco";
      histTitle = HltHistTitle[i]+" isolation vs #eta (reco)";
      tmpiso    = dbe->book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      histEtaIsoOfHltObjMatchToReco.push_back(tmpiso);

      // 2D plot: Isolation values vs et for HLT object that 
      // is closest delta-R match to sorted reco particle(s)
      histName  = theHLTCollectionLabels[i].label()+"et_isolation_reco";
      histTitle = HltHistTitle[i]+" isolation vs Et (reco)";
      tmpiso    = dbe->book2D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      histEtIsoOfHltObjMatchToReco.push_back(tmpiso);
      
    } // END of HLT histograms

  }
}


////////////////////////////////////////////////////////////////////////////////
//                                Destructor                                  //
////////////////////////////////////////////////////////////////////////////////
EmDQMReco::~EmDQMReco(){
}


////////////////////////////////////////////////////////////////////////////////
//                     method called to for each event                        //
////////////////////////////////////////////////////////////////////////////////
void 
EmDQMReco::analyze(const edm::Event & event , const edm::EventSetup& setup)
{

  // protect from hlt config failure
  if( !isHltConfigInitialized_ ) return;

  eventnum++;
  bool plotMonpath = false;
  bool plotReco = true; 

  edm::Handle<edm::View<reco::Candidate> > recoObjects;
  edm::Handle<std::vector<reco::SuperCluster> > recoObjectsEB;
  edm::Handle<std::vector<reco::SuperCluster> > recoObjectsEE;

  if (pdgGen == 11) {

    event.getByLabel("gsfElectrons", recoObjects);
    
    if (recoObjects->size() < (unsigned int)recocut_) {
      // edm::LogWarning("EmDQMReco") << "Less than "<< recocut_ <<" Reco particles with pdgId=" << pdgGen << ".  Only " << cutRecoCounter->size() << " particles.";
      return;
    }
  } else if (pdgGen == 22) {

    event.getByLabel("correctedHybridSuperClusters", recoObjectsEB);
    event.getByLabel("correctedMulti5x5SuperClustersWithPreshower", recoObjectsEE);
    
    if (recoObjectsEB->size() + recoObjectsEE->size() < (unsigned int)recocut_) {
      // edm::LogWarning("EmDQMReco") << "Less than "<< recocut_ <<" Reco particles with pdgId=" << pdgGen << ".  Only " << cutRecoCounter.size() << " particles.";
      return;
    }
  }
  
  edm::Handle<edm::TriggerResults> HLTR;
  event.getByLabel(edm::InputTag("TriggerResults","","HLT"), HLTR);
  
  ///
  /// NOTE:
  /// hltConfigProvider initialization has been moved to beginRun()
  ///

  /* if (theHLTCollectionHumanNames[0] == "hltL1sRelaxedSingleEgammaEt8"){
    triggerIndex = hltConfig.triggerIndex("HLT_L1SingleEG8");
  } else if (theHLTCollectionHumanNames[0] == "hltL1sRelaxedSingleEgammaEt5") {
    triggerIndex = hltConfig.triggerIndex("HLT_L1SingleEG5");
  } else if (theHLTCollectionHumanNames[0] == "hltL1sRelaxedDoubleEgammaEt5") {
    triggerIndex = hltConfig.triggerIndex("HLT_L1DoubleEG5"); 
  } else { 
    triggerIndex = hltConfig.triggerIndex("");
    } */

  unsigned int triggerIndex; 
  triggerIndex = hltConfig_.triggerIndex("HLT_MinBias");
  
  //triggerIndex must be less than the size of HLTR or you get a CMSException
  bool isFired = false;
  if (triggerIndex < HLTR->size()){
    isFired = HLTR->accept(triggerIndex); 
  }

  // fill L1 and HLT info
  // get objects possed by each filter
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("hltTriggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("EmDQMReco") << "RAW-type HLT results not found, skipping event";
    return;
  }

  ////////////////////////////////////////////////////////////
  //  Fill the bin labeled "Total"                          //
  //   This will be the number of events looked at.         //
  ////////////////////////////////////////////////////////////
  totalreco->Fill(numOfHLTCollectionLabels+0.5);
  totalmatchreco->Fill(numOfHLTCollectionLabels+.5);

  /* edm::Handle< edm::View<reco::GsfElectron> > recoParticles;
  event.getByLabel("gsfElectrons", recoParticles);

  std::vector<reco::GsfElectron> allSortedRecoParticles;

  for(edm::View<reco::GsfElectron>::const_iterator currentRecoParticle = recoParticles->begin(); currentRecoParticle != recoParticles->end(); currentRecoParticle++){
  if (  !((*currentRecoParticle).et() > 2.0)  )  continue;
    reco::GsfElectron tmpcand( *(currentRecoParticle) );
    allSortedRecoParticles.push_back(tmpcand);
  }

  std::sort(allSortedRecoParticles.begin(), allSortedRecoParticles.end(),pTRecoComparator_);*/

  // Were enough high energy gen particles found?
  // It was an event worth keeping. Continue.

  ////////////////////////////////////////////////////////////
  //  Fill the bin labeled "Total"                          //
  //   This will be the number of events looked at.         //
  ////////////////////////////////////////////////////////////
  //total->Fill(numOfHLTCollectionLabels+0.5);
  //totalmatch->Fill(numOfHLTCollectionLabels+0.5);


  ////////////////////////////////////////////////////////////
  //               Fill reconstruction info                      //
  ////////////////////////////////////////////////////////////
  // the recocut_ highest Et generator objects of the preselected type are our matches

  std::vector<reco::Particle> sortedReco;
  if (plotReco == true) {
    if (pdgGen == 11) {
      for(edm::View<reco::Candidate>::const_iterator recopart = recoObjects->begin(); recopart != recoObjects->end();recopart++){
	reco::Particle tmpcand(  recopart->charge(), recopart->p4(), recopart->vertex(),recopart->pdgId(),recopart->status() );
	sortedReco.push_back(tmpcand);
      }
    }
    else if (pdgGen == 22) {
      for(std::vector<reco::SuperCluster>::const_iterator recopart2 = recoObjectsEB->begin(); recopart2 != recoObjectsEB->end();recopart2++){
	float en = recopart2->energy();
	float er = sqrt(pow(recopart2->x(),2) + pow(recopart2->y(),2) + pow(recopart2->z(),2) );
	float px = recopart2->energy()*recopart2->x()/er;
	float py = recopart2->energy()*recopart2->y()/er;
	float pz = recopart2->energy()*recopart2->z()/er;
	reco::Candidate::LorentzVector thisLV(px,py,pz,en);
	reco::Particle tmpcand(  0, thisLV, math::XYZPoint(0.,0.,0.), 22, 1 );
	sortedReco.push_back(tmpcand);
      }
      for(std::vector<reco::SuperCluster>::const_iterator recopart2 = recoObjectsEE->begin(); recopart2 != recoObjectsEE->end();recopart2++){
	float en = recopart2->energy();
	float er = sqrt(pow(recopart2->x(),2) + pow(recopart2->y(),2) + pow(recopart2->z(),2) );
	float px = recopart2->energy()*recopart2->x()/er;
	float py = recopart2->energy()*recopart2->y()/er;
	float pz = recopart2->energy()*recopart2->z()/er;
        reco::Candidate::LorentzVector thisLV(px,py,pz,en);
	reco::Particle tmpcand(  0, thisLV, math::XYZPoint(0.,0.,0.), 22, 1 );
	sortedReco.push_back(tmpcand);
      }
    }

    std::sort(sortedReco.begin(),sortedReco.end(),pTComparator_ );
      
    // Now the collection of gen particles is sorted by pt.
    // So, remove all particles from the collection so that we 
    // only have the top "1 thru recocut_" particles in it
    
    sortedReco.erase(sortedReco.begin()+recocut_,sortedReco.end());
    
    for (unsigned int i = 0 ; i < recocut_ ; i++ ) {
      etreco ->Fill( sortedReco[i].et()  ); //validity has been implicitily checked by the cut on recocut_ above
      etareco->Fill( sortedReco[i].eta() );
      
      if (isFired) {
	etrecomonpath->Fill( sortedReco[i].et() ); 
	etarecomonpath->Fill( sortedReco[i].eta() );
	plotMonpath = true;
      }
      
    } // END of loop over Reconstructed particles
    
    if (recocut_ >= reqNum) totalreco->Fill(numOfHLTCollectionLabels+1.5); // this isn't really needed anymore keep for backward comp.
    if (recocut_ >= reqNum) totalmatchreco->Fill(numOfHLTCollectionLabels+1.5); // this isn't really needed anymore keep for backward comp.

  } 

  


   ////////////////////////////////////////////////////////////
  //            Loop over filter modules                    //
  ////////////////////////////////////////////////////////////
  for(unsigned int n=0; n < numOfHLTCollectionLabels ; n++) {
    // These numbers are from the Parameter Set, such as:
    //   theHLTOutputTypes = cms.uint32(100)
    switch(theHLTOutputTypes[n]) 
    {
      case trigger::TriggerL1NoIsoEG: // Non-isolated Level 1
        fillHistos<l1extra::L1EmParticleCollection>(triggerObj,event,n, sortedReco, plotReco, plotMonpath);break;
      case trigger::TriggerL1IsoEG: // Isolated Level 1
        fillHistos<l1extra::L1EmParticleCollection>(triggerObj,event,n, sortedReco, plotReco, plotMonpath);break;
      case trigger::TriggerPhoton: // Photon 
        fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,event,n, sortedReco, plotReco, plotMonpath);break;
      case trigger::TriggerElectron: // Electron 
        fillHistos<reco::ElectronCollection>(triggerObj,event,n, sortedReco, plotReco, plotMonpath);break;
      case trigger::TriggerCluster: // TriggerCluster
        fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,event,n, sortedReco, plotReco, plotMonpath);break;
      default: 
        throw(cms::Exception("Release Validation Error") << "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
    } // END of loop over filter modules
}


////////////////////////////////////////////////////////////////////////////////
// fillHistos                                                                 //
//   Called by analyze method.                                                //
////////////////////////////////////////////////////////////////////////////////
template <class T> void EmDQMReco::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent ,unsigned int n, std::vector<reco::Particle>& sortedReco, bool plotReco, bool plotMonpath)
{
  std::vector<edm::Ref<T> > recoecalcands;
  if ( ( triggerObj->filterIndex(theHLTCollectionLabels[n])>=triggerObj->size() )){ // only process if available
    return;
  }

  ////////////////////////////////////////////////////////////
  //      Retrieve saved filter objects                     //
  ////////////////////////////////////////////////////////////
  triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),theHLTOutputTypes[n],recoecalcands);
  //Danger: special case, L1 non-isolated
  // needs to be merged with L1 iso
  if (theHLTOutputTypes[n] == trigger::TriggerL1NoIsoEG){
    std::vector<edm::Ref<T> > isocands;
    triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),trigger::TriggerL1IsoEG,isocands);
    if (isocands.size()>0) 
      {
	for (unsigned int i=0; i < isocands.size(); i++)
	  recoecalcands.push_back(isocands[i]);
      }
  } // END of if theHLTOutputTypes == 82
  

  if (recoecalcands.size() < 1){ // stop if no object passed the previous filter
    return;
  }


  if (recoecalcands.size() >= reqNum ) 
    totalreco->Fill(n+0.5);


  ///////////////////////////////////////////////////
  // check for validity                            //
  // prevents crash in CMSSW_3_1_0_pre6            //
  ///////////////////////////////////////////////////
  for (unsigned int j=0; j<recoecalcands.size(); j++){
    if(!( recoecalcands.at(j).isAvailable())){
      edm::LogError("EmDQMReco") << "Event content inconsistent: TriggerEventWithRefs contains invalid Refs" << std::endl << "invalid refs for: " << theHLTCollectionLabels[n].label();
      return;
    }
  } 

  ////////////////////////////////////////////////////////////
  //  Loop over all HLT objects in this filter step, and    //
  //  fill histograms.                                      //
  ////////////////////////////////////////////////////////////
  //  bool foundAllMatches = false;
  //  unsigned int numOfHLTobjectsMatched = 0;
  for (unsigned int i=0; i<recoecalcands.size(); i++) {
   
    ethist[n] ->Fill(recoecalcands[i]->et() );
    etahist[n]->Fill(recoecalcands[i]->eta() );

    ////////////////////////////////////////////////////////////
    //  Plot isolation variables (show the not-yet-cut        //
    //  isolation, i.e. associated to next filter)            //
    ////////////////////////////////////////////////////////////
    if ( n+1 < numOfHLTCollectionLabels ) { // can't plot beyond last
      if (plotiso[n+1]) {
	for (unsigned int j =  0 ; j < isoNames[n+1].size() ;j++  ){
	  edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMap; 
	  iEvent.getByLabel(isoNames[n+1].at(j),depMap);
	  if (depMap.isValid()){ //Map may not exist if only one candidate passes a double filter
	    typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMap->find(recoecalcands[i]);
	    if (mapi!=depMap->end()){  // found candidate in isolation map! 
	      etahistiso[n+1]->Fill(recoecalcands[i]->eta(),mapi->val);
	      ethistiso[n+1]->Fill(recoecalcands[i]->et(),mapi->val);
	    }
	  }
	}
      }
    } // END of if n+1 < then the number of hlt collections
  }

  ////////////////////////////////////////////////////////////
  // Loop over the Reconstructed Particles, and find the        //
  // closest HLT object match.                              //
  ////////////////////////////////////////////////////////////
  if (plotReco == true) {
    for (unsigned int i=0; i < recocut_; i++) {
      math::XYZVector currentRecoParticleMomentum = sortedReco[i].momentum();

      // float closestRecoDeltaR = 0.5;
      float closestRecoDeltaR = 1000.;
      int closestRecoEcalCandIndex = -1;
      for (unsigned int j=0; j<recoecalcands.size(); j++) {
	float deltaR = DeltaR(recoecalcands[j]->momentum(),currentRecoParticleMomentum);

	if (deltaR < closestRecoDeltaR) {
	  closestRecoDeltaR = deltaR;
	  closestRecoEcalCandIndex = j;
	}
    }
      
      // If an HLT object was found within some delta-R
      // of this reco particle, store it in a histogram
      if ( closestRecoEcalCandIndex >= 0 ) {
	histEtOfHltObjMatchToReco[n] ->Fill( recoecalcands[closestRecoEcalCandIndex]->et()  );
	histEtaOfHltObjMatchToReco[n]->Fill( recoecalcands[closestRecoEcalCandIndex]->eta() );
	
	// Also store isolation info
	if (n+1 < numOfHLTCollectionLabels){ // can't plot beyond last
	  if (plotiso[n+1] ){  // only plot if requested in config
	    for (unsigned int j =  0 ; j < isoNames[n+1].size() ;j++  ){
	      edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMap; 
	      iEvent.getByLabel(isoNames[n+1].at(j),depMap);
	      if (depMap.isValid()){ //Map may not exist if only one candidate passes a double filter
		typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMap->find(recoecalcands[closestRecoEcalCandIndex]);
		if (mapi!=depMap->end()) {  // found candidate in isolation map! 
		  histEtaIsoOfHltObjMatchToReco[n+1]->Fill( recoecalcands[closestRecoEcalCandIndex]->eta(),mapi->val);
		  histEtIsoOfHltObjMatchToReco[n+1] ->Fill( recoecalcands[closestRecoEcalCandIndex]->et(), mapi->val);
		}
	      }
	    }
	  }
	}
      } // END of if closestEcalCandIndex >= 0
    }
   
    ////////////////////////////////////////////////////////////
    //        Fill reco matched objects into histograms         //
    ////////////////////////////////////////////////////////////
    unsigned int mtachedRecoParts = 0;
    float minrecodist=0.3;
    if(n==0) minrecodist=0.5; //low L1-resolution => allow wider matching 
    for(unsigned int i =0; i < recocut_; i++){
      //match generator candidate    
      bool matchThis= false;
      math::XYZVector candDir=sortedReco[i].momentum();
      unsigned int closest = 0;
      double closestDr = 1000.;
      for(unsigned int trigOb = 0 ; trigOb < recoecalcands.size(); trigOb++){
	double dr = DeltaR(recoecalcands[trigOb]->momentum(),candDir);
	if (dr < closestDr) {
	  closestDr = dr;
	  closest = trigOb;
	}
	if (closestDr > minrecodist) { // it's not really a "match" if it's that far away
	  closest = -1;
	} else {
	  mtachedRecoParts++;
	  matchThis = true;
	}
      }
      if ( !matchThis ) continue; // only plot matched candidates
      // fill coordinates of mc particle matching trigger object
      ethistmatchreco[n] ->Fill( sortedReco[i].et()  );
      etahistmatchreco[n]->Fill( sortedReco[i].eta() );
      if (plotMonpath) {
	ethistmatchrecomonpath[n]->Fill( sortedReco[i].et() );
	etahistmatchrecomonpath[n]->Fill( sortedReco[i].eta() );
      }
      ////////////////////////////////////////////////////////////
      //  Plot isolation variables (show the not-yet-cut        //
      //  isolation, i.e. associated to next filter)            //
      ////////////////////////////////////////////////////////////
      if (n+1 < numOfHLTCollectionLabels){ // can't plot beyond last
	if (plotiso[n+1] ){  // only plot if requested in config
	  for (unsigned int j =  0 ; j < isoNames[n+1].size() ;j++  ){
	    edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMapReco; 
	    iEvent.getByLabel(isoNames[n+1].at(j),depMapReco);
	    if (depMapReco.isValid()){ //Map may not exist if only one candidate passes a double filter
	      typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMapReco->find(recoecalcands[closest]);
	      if (mapi!=depMapReco->end()){  // found candidate in isolation map!
		etahistisomatchreco[n+1]->Fill(sortedReco[i].eta(),mapi->val);
		ethistisomatchreco[n+1]->Fill(sortedReco[i].et(),mapi->val);
	      }
	    }
	  }
	}
      } // END of if n+1 < then the number of hlt collections
    }
    // fill total reco matched efficiency
    if (mtachedRecoParts >= reqNum ) 
      totalmatchreco->Fill(n+0.5);
  }
  
}


//////////////////////////////////////////////////////////////////////////////// 
//      method called once each job just after ending the event loop          //
//////////////////////////////////////////////////////////////////////////////// 
void EmDQMReco::endJob(){

}

DEFINE_FWK_MODULE(EmDQMReco);
