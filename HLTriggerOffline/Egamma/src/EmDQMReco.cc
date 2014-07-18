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
#include <boost/format.hpp>
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

//----------------------------------------------------------------------
// class EmDQMReco::FourVectorMonitorElements
//----------------------------------------------------------------------
EmDQMReco::FourVectorMonitorElements::FourVectorMonitorElements(EmDQMReco *_parent,
    DQMStore::IBooker &iBooker,
    const std::string &histogramNameTemplate,
    const std::string &histogramTitleTemplate
  ) :
  parent(_parent)
{
  // introducing variables for better code readability later on
  std::string histName;
  std::string histTitle;

  // et
  histName = boost::str(boost::format(histogramNameTemplate) % "et");
  histTitle = boost::str(boost::format(histogramTitleTemplate) % "E_{T}");
  etMonitorElement =  iBooker.book1D(histName.c_str(),
                                   histTitle.c_str(),
                                   parent->plotBins,
                                   parent->plotPtMin,
                                   parent->plotPtMax);

  // eta
  histName = boost::str(boost::format(histogramNameTemplate) % "eta");
  histTitle= boost::str(boost::format(histogramTitleTemplate) % "#eta");
  etaMonitorElement = iBooker.book1D(histName.c_str(),
                                  histTitle.c_str(),
                                  parent->plotBins,
                                  - parent->plotEtaMax,
                                    parent->plotEtaMax);

  // phi
  histName = boost::str(boost::format(histogramNameTemplate) % "phi");
  histTitle= boost::str(boost::format(histogramTitleTemplate) % "#phi");
  phiMonitorElement = iBooker.book1D(histName.c_str(),
                                  histTitle.c_str(),
                                  parent->plotBins,
                                  - parent->plotPhiMax,
                                    parent->plotPhiMax);
}

//----------------------------------------------------------------------

void
EmDQMReco::FourVectorMonitorElements::fill(const math::XYZTLorentzVector &momentum)
{
  etMonitorElement->Fill(momentum.Et());
  etaMonitorElement->Fill(momentum.eta() );
  phiMonitorElement->Fill(momentum.phi() );
}

//----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
//                             Constructor                                    //
////////////////////////////////////////////////////////////////////////////////
EmDQMReco::EmDQMReco(const edm::ParameterSet& pset)
{
  ////////////////////////////////////////////////////////////
  //          Read from configuration file                  //
  ////////////////////////////////////////////////////////////
  dirname_="HLT/HLTEgammaValidation/"+pset.getParameter<std::string>("@module_label");

  // parameters for generator study
  reqNum    = pset.getParameter<unsigned int>("reqNum");
  pdgGen    = pset.getParameter<int>("pdgGen");
  recoEtaAcc = pset.getParameter<double>("genEtaAcc");
  recoEtAcc  = pset.getParameter<double>("genEtAcc");
  // plotting parameters (untracked because they don't affect the physics)
  plotPtMin  = pset.getUntrackedParameter<double>("PtMin",0.);
  plotPtMax  = pset.getUntrackedParameter<double>("PtMax",1000.);
  plotEtaMax = pset.getUntrackedParameter<double>("EtaMax", 2.7);
  plotPhiMax = pset.getUntrackedParameter<double>("PhiMax", 3.15);
  plotBins   = pset.getUntrackedParameter<unsigned int>("Nbins",50);
  useHumanReadableHistTitles = pset.getUntrackedParameter<bool>("useHumanReadableHistTitles", false);

  triggerNameRecoMonPath = pset.getUntrackedParameter<std::string>("triggerNameRecoMonPath","HLT_MinBias");
  processNameRecoMonPath = pset.getUntrackedParameter<std::string>("processNameRecoMonPath","HLT");

  recoElectronsInput = consumes<reco::GsfElectronCollection>(pset.getUntrackedParameter<edm::InputTag>("recoElectrons",edm::InputTag("gsfElectrons")));
  recoObjectsEBT = consumes<std::vector<reco::SuperCluster>>(edm::InputTag("correctedHybridSuperClusters"));
  recoObjectsEET = consumes<std::vector<reco::SuperCluster>>(edm::InputTag("correctedMulti5x5SuperClustersWithPreshower"));
  hltResultsT    = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults","",processNameRecoMonPath));
  triggerObjT    = consumes<trigger::TriggerEventWithRefs>(edm::InputTag("hltTriggerSummaryRAW"));

  // preselction cuts
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
    
    for (unsigned int i=0; i<isoNames.back().size(); i++) {
      switch(theHLTOutputTypes.back())  {
      case trigger::TriggerL1NoIsoEG: 
	histoFillerL1NonIso->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<l1extra::L1EmParticleCollection , float>>>(isoNames.back()[i]));
	break;
      case trigger::TriggerL1IsoEG: // Isolated Level 1
	histoFillerL1Iso->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<l1extra::L1EmParticleCollection , float>>>(isoNames.back()[i]));
	break;
      case trigger::TriggerPhoton: // Photon 
	histoFillerPho->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<reco::RecoEcalCandidateCollection , float>>>(isoNames.back()[i]));
	break;
      case trigger::TriggerElectron: // Electron 
	histoFillerEle->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<reco::ElectronCollection , float>>>(isoNames.back()[i]));
	break;
      case trigger::TriggerCluster: // TriggerCluster
	histoFillerClu->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<reco::RecoEcalCandidateCollection , float>>>(isoNames.back()[i]));
	break;
      default: 
	throw(cms::Exception("Release Validation Error") << "HLT output type not implemented: theHLTOutputTypes[n]" );
      }
    }
    
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
void EmDQMReco::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup ) {

  bool isHltConfigChanged = false; // change of cfg at run boundaries?
  isHltConfigInitialized_ = hltConfig_.init( iRun, iSetup, "HLT", isHltConfigChanged );

}

////////////////////////////////////////////////////////////////////////////////
//       book DQM histograms                                                  //
////////////////////////////////////////////////////////////////////////////////
void
EmDQMReco::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &iRun, edm::EventSetup const &iSetup)
{
  //edm::Service<TFileService> fs;
  iBooker.setCurrentFolder(dirname_);

  ////////////////////////////////////////////////////////////
  //  Set up Histogram of Effiency vs Step.                 //
  //   theHLTCollectionLabels is a vector of InputTags      //
  //    from the configuration file.                        //
  ////////////////////////////////////////////////////////////

  std::string histName="total_eff";
  std::string histTitle = "total events passing";
  // This plot will have bins equal to 2+(number of
  //        HLTCollectionLabels in the config file)
  totalreco = iBooker.book1D(histName.c_str(),histTitle.c_str(),numOfHLTCollectionLabels+2,0,numOfHLTCollectionLabels+2);
  totalreco->setBinLabel(numOfHLTCollectionLabels+1,"Total");
  totalreco->setBinLabel(numOfHLTCollectionLabels+2,"Reco");
  for (unsigned int u=0; u<numOfHLTCollectionLabels; u++){totalreco->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());}

  histName="total_eff_RECO_matched";
  histTitle="total events passing (Reco matched)";
  totalmatchreco = iBooker.book1D(histName.c_str(),histTitle.c_str(),numOfHLTCollectionLabels+2,0,numOfHLTCollectionLabels+2);
  totalmatchreco->setBinLabel(numOfHLTCollectionLabels+1,"Total");
  totalmatchreco->setBinLabel(numOfHLTCollectionLabels+2,"Reco");
  for (unsigned int u=0; u<numOfHLTCollectionLabels; u++){totalmatchreco->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());}

  // MonitorElement* tmphisto;
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

  //--------------------

  // reco
  // (note that reset(..) must be used to set the value of the scoped_ptr...)
  histReco.reset(
      new FourVectorMonitorElements(this, iBooker,
          "reco_%s",             // pattern for histogram name
          "%s of " + pdgIdString + "s"
      ));

  //--------------------

  // monpath
  histRecoMonpath.reset(
       new FourVectorMonitorElements(this, iBooker,
           "reco_%s_monpath",   // pattern for histogram name
           "%s of " + pdgIdString + "s monpath"
       )
  );

  //--------------------

  // TODO: WHAT ARE THESE HISTOGRAMS FOR ? THEY SEEM NEVER REFERENCED ANYWHERE IN THIS FILE...
  // final X monpath
  histMonpath.reset(
       new FourVectorMonitorElements(this, iBooker,
           "final_%s_monpath",   // pattern for histogram name
           "Final %s Monpath"
       )
  );

  //--------------------

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
    //--------------------
    // distributions of HLT objects passing filter i
    //--------------------

//    // Et
//    histName = theHLTCollectionLabels[i].label()+"et_all";
//    histTitle = HltHistTitle[i]+" Et (ALL)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
//    ethist.push_back(tmphisto);
//
//    // Eta
//    histName = theHLTCollectionLabels[i].label()+"eta_all";
//    histTitle = HltHistTitle[i]+" #eta (ALL)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
//    etahist.push_back(tmphisto);
//
//    // phi
//    histName = theHLTCollectionLabels[i].label()+"phi_all";
//    histTitle = HltHistTitle[i]+" #phi (ALL)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotPhiMax,plotPhiMax);
//    phiHist.push_back(tmphisto);

    standardHist.push_back(new FourVectorMonitorElements(this, iBooker,
							 theHLTCollectionLabels[i].label()+"%s_all", // histogram name
							 HltHistTitle[i]+" %s (ALL)"                 // histogram title
							 ));

    //--------------------
    // distributions of reco object matching HLT object passing filter i
    //--------------------

    // Et
//    histName = theHLTCollectionLabels[i].label()+"et_RECO_matched";
//    histTitle = HltHistTitle[i]+" Et (RECO matched)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
//    ethistmatchreco.push_back(tmphisto);

//    // Eta
//    histName = theHLTCollectionLabels[i].label()+"eta_RECO_matched";
//    histTitle = HltHistTitle[i]+" #eta (RECO matched)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
//    etahistmatchreco.push_back(tmphisto);
//
//    // phi
//    histName = theHLTCollectionLabels[i].label()+"phi_RECO_matched";
//    histTitle = HltHistTitle[i]+" #phi (RECO matched)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotPhiMax,plotPhiMax);
//    phiHistMatchReco.push_back(tmphisto);
    histMatchReco.push_back(new FourVectorMonitorElements(this, iBooker,
        theHLTCollectionLabels[i].label()+"%s_RECO_matched", // histogram name
        HltHistTitle[i]+" %s (RECO matched)"                 // histogram title
        ));

    //--------------------
    // distributions of reco object matching HLT object passing filter i
    //--------------------

//    // Et
//    histName = theHLTCollectionLabels[i].label()+"et_RECO_matched_monpath";
//    histTitle = HltHistTitle[i]+" Et (RECO matched, monpath)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
//    ethistmatchrecomonpath.push_back(tmphisto);
//
//    // Eta
//    histName = theHLTCollectionLabels[i].label()+"eta_RECO_matched_monpath";
//    histTitle = HltHistTitle[i]+" #eta (RECO matched, monpath)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
//    etahistmatchrecomonpath.push_back(tmphisto);
//
//    // phi
//    histName = theHLTCollectionLabels[i].label()+"phi_RECO_matched_monpath";
//    histTitle = HltHistTitle[i]+" #phi (RECO matched, monpath)";
//    tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotPhiMax,plotPhiMax);
//    phiHistMatchRecoMonPath.push_back(tmphisto);

    histMatchRecoMonPath.push_back(new FourVectorMonitorElements(this, iBooker,
        theHLTCollectionLabels[i].label()+"%s_RECO_matched_monpath", // histogram name
        HltHistTitle[i]+" %s (RECO matched, monpath)"                // histogram title
        ));
    //--------------------
    // distributions of HLT object that is closest delta-R match to sorted reco particle(s)
    //--------------------

    // Et
//    histName  = theHLTCollectionLabels[i].label()+"et_reco";
//    histTitle = HltHistTitle[i]+" Et (reco)";
//    tmphisto  = iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax);
//    histEtOfHltObjMatchToReco.push_back(tmphisto);
//
//    // eta
//    histName  = theHLTCollectionLabels[i].label()+"eta_reco";
//    histTitle = HltHistTitle[i]+" eta (reco)";
//    tmphisto  = iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax);
//    histEtaOfHltObjMatchToReco.push_back(tmphisto);
//
//    // phi
//    histName  = theHLTCollectionLabels[i].label()+"phi_reco";
//    histTitle = HltHistTitle[i]+" phi (reco)";
//    tmphisto  = iBooker.book1D(histName.c_str(),histTitle.c_str(),plotBins,-plotPhiMax,plotPhiMax);
//    histPhiOfHltObjMatchToReco.push_back(tmphisto);

    histHltObjMatchToReco.push_back(new FourVectorMonitorElements(this, iBooker,
        theHLTCollectionLabels[i].label()+"%s_reco",   // histogram name
        HltHistTitle[i]+" %s (reco)"                  // histogram title
        ));

    //--------------------

    if (!plotiso[i]) {
      tmpiso = NULL;
      etahistiso.push_back(tmpiso);
      ethistiso.push_back(tmpiso);
      phiHistIso.push_back(tmpiso);

      etahistisomatchreco.push_back(tmpiso);
      ethistisomatchreco.push_back(tmpiso);
      phiHistIsoMatchReco.push_back(tmpiso);

      histEtaIsoOfHltObjMatchToReco.push_back(tmpiso);
      histEtIsoOfHltObjMatchToReco.push_back(tmpiso);
      histPhiIsoOfHltObjMatchToReco.push_back(tmpiso);

    } else {

      //--------------------
      // 2D plot: Isolation values vs X for all objects
      //--------------------

      // X = eta
      histName  = theHLTCollectionLabels[i].label()+"eta_isolation_all";
      histTitle = HltHistTitle[i]+" isolation vs #eta (all)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      etahistiso.push_back(tmpiso);

      // X = et
      histName  = theHLTCollectionLabels[i].label()+"et_isolation_all";
      histTitle = HltHistTitle[i]+" isolation vs Et (all)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      ethistiso.push_back(tmpiso);

      // X = phi
      histName  = theHLTCollectionLabels[i].label()+"phi_isolation_all";
      histTitle = HltHistTitle[i]+" isolation vs #phi (all)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotPhiMax,plotPhiMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      phiHistIso.push_back(tmpiso);

      //--------------------
      // 2D plot: Isolation values vs X for reco matched objects
      //--------------------

      // X = eta
      histName  = theHLTCollectionLabels[i].label()+"eta_isolation_RECO_matched";
      histTitle = HltHistTitle[i]+" isolation vs #eta (reco matched)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      etahistisomatchreco.push_back(tmpiso);

      // X = et
      histName  = theHLTCollectionLabels[i].label()+"et_isolation_RECO_matched";
      histTitle = HltHistTitle[i]+" isolation vs Et (reco matched)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      ethistisomatchreco.push_back(tmpiso);

      // X = eta
      histName  = theHLTCollectionLabels[i].label()+"phi_isolation_RECO_matched";
      histTitle = HltHistTitle[i]+" isolation vs #phi (reco matched)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotPhiMax,plotPhiMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      phiHistIsoMatchReco.push_back(tmpiso);

      //--------------------
      // 2D plot: Isolation values vs X for HLT object that
      // is closest delta-R match to sorted reco particle(s)
      //--------------------

      // X = eta
      histName  = theHLTCollectionLabels[i].label()+"eta_isolation_reco";
      histTitle = HltHistTitle[i]+" isolation vs #eta (reco)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotEtaMax,plotEtaMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      histEtaIsoOfHltObjMatchToReco.push_back(tmpiso);

      // X = et
      histName  = theHLTCollectionLabels[i].label()+"et_isolation_reco";
      histTitle = HltHistTitle[i]+" isolation vs Et (reco)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,plotPtMin,plotPtMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      histEtIsoOfHltObjMatchToReco.push_back(tmpiso);

      // X = phi
      histName  = theHLTCollectionLabels[i].label()+"phi_isolation_reco";
      histTitle = HltHistTitle[i]+" isolation vs #phi (reco)";
      tmpiso    = iBooker.book2D(histName.c_str(),histTitle.c_str(),plotBins,-plotPhiMax,plotPhiMax,plotBins,plotBounds[i].first,plotBounds[i].second);
      histPhiIsoOfHltObjMatchToReco.push_back(tmpiso);
      //--------------------

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

    event.getByToken(recoElectronsInput, recoObjects);

    if (recoObjects->size() < (unsigned int)recocut_) {
      // edm::LogWarning("EmDQMReco") << "Less than "<< recocut_ <<" Reco particles with pdgId=" << pdgGen << ".  Only " << cutRecoCounter->size() << " particles.";
      return;
    }
  } else if (pdgGen == 22) {

    event.getByToken(recoObjectsEBT, recoObjectsEB);
    event.getByToken(recoObjectsEET, recoObjectsEE);

    if (recoObjectsEB->size() + recoObjectsEE->size() < (unsigned int)recocut_) {
      // edm::LogWarning("EmDQMReco") << "Less than "<< recocut_ <<" Reco particles with pdgId=" << pdgGen << ".  Only " << cutRecoCounter.size() << " particles.";
      return;
    }
  }

  edm::Handle<edm::TriggerResults> HLTR;
  event.getByToken(hltResultsT, HLTR);

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
  triggerIndex = hltConfig_.triggerIndex(triggerNameRecoMonPath);

  //triggerIndex must be less than the size of HLTR or you get a CMSException
  bool isFired = false;
  if (triggerIndex < HLTR->size()){
    isFired = HLTR->accept(triggerIndex);
  }

  // fill L1 and HLT info
  // get objects possed by each filter
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByToken(triggerObjT, triggerObj);

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
        //validity has been implicitily checked by the cut on recocut_ above
        histReco->fill(sortedReco[i].p4());

//      etreco ->Fill( sortedReco[i].et()  );
//      etareco->Fill( sortedReco[i].eta() );
//      phiReco->Fill( sortedReco[i].phi() );

      if (isFired) {
        histRecoMonpath->fill(sortedReco[i].p4());
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
        histoFillerL1NonIso->fillHistos(triggerObj,event,n, sortedReco, plotReco, plotMonpath);
	break;
      case trigger::TriggerL1IsoEG: // Isolated Level 1
        histoFillerL1Iso->fillHistos(triggerObj,event,n, sortedReco, plotReco, plotMonpath);
	break;
      case trigger::TriggerPhoton: // Photon
        histoFillerPho->fillHistos(triggerObj,event,n, sortedReco, plotReco, plotMonpath);
	break;
      case trigger::TriggerElectron: // Electron
        histoFillerEle->fillHistos(triggerObj,event,n, sortedReco, plotReco, plotMonpath);
	break;
      case trigger::TriggerCluster: // TriggerCluster
        histoFillerClu->fillHistos(triggerObj,event,n, sortedReco, plotReco, plotMonpath);
	break;
      default:
        throw(cms::Exception("Release Validation Error") << "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
    } // END of loop over filter modules
}


////////////////////////////////////////////////////////////////////////////////
// fillHistos                                                                 //
//   Called by analyze method.                                                //
////////////////////////////////////////////////////////////////////////////////
template <class T> void HistoFillerReco<T>::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent ,unsigned int n, std::vector<reco::Particle>& sortedReco, bool plotReco, bool plotMonpath)
{
  std::vector<edm::Ref<T> > recoecalcands;
  if ( ( triggerObj->filterIndex(dqm->theHLTCollectionLabels[n])>=triggerObj->size() )){ // only process if available
    return;
  }

  ////////////////////////////////////////////////////////////
  //      Retrieve saved filter objects                     //
  ////////////////////////////////////////////////////////////
  triggerObj->getObjects(triggerObj->filterIndex(dqm->theHLTCollectionLabels[n]),dqm->theHLTOutputTypes[n],recoecalcands);
  //Danger: special case, L1 non-isolated
  // needs to be merged with L1 iso
  if (dqm->theHLTOutputTypes[n] == trigger::TriggerL1NoIsoEG){
    std::vector<edm::Ref<T> > isocands;
    triggerObj->getObjects(triggerObj->filterIndex(dqm->theHLTCollectionLabels[n]),trigger::TriggerL1IsoEG,isocands);
    if (isocands.size()>0)
      {
        for (unsigned int i=0; i < isocands.size(); i++)
          recoecalcands.push_back(isocands[i]);
      }
  } // END of if theHLTOutputTypes == 82


  if (recoecalcands.size() < 1){ // stop if no object passed the previous filter
    return;
  }


  if (recoecalcands.size() >= dqm->reqNum )
    dqm->totalreco->Fill(n+0.5);


  ///////////////////////////////////////////////////
  // check for validity                            //
  // prevents crash in CMSSW_3_1_0_pre6            //
  ///////////////////////////////////////////////////
  for (unsigned int j=0; j<recoecalcands.size(); j++){
    if(!( recoecalcands.at(j).isAvailable())){
      edm::LogError("EmDQMReco") << "Event content inconsistent: TriggerEventWithRefs contains invalid Refs" << std::endl << "invalid refs for: " << dqm->theHLTCollectionLabels[n].label();
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

    dqm->standardHist[n].fill(recoecalcands[i]->p4());

    ////////////////////////////////////////////////////////////
    //  Plot isolation variables (show the not-yet-cut        //
    //  isolation, i.e. associated to next filter)            //
    ////////////////////////////////////////////////////////////
    if ( n+1 < dqm->numOfHLTCollectionLabels ) { // can't plot beyond last
      if (dqm->plotiso[n+1]) {
        for (unsigned int j =  0 ; j < isoNameTokens_.size() ;j++  ){
          edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMap;
          iEvent.getByToken(isoNameTokens_.at(j),depMap);
          if (depMap.isValid()){ //Map may not exist if only one candidate passes a double filter
            typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMap->find(recoecalcands[i]);
            if (mapi!=depMap->end()){  // found candidate in isolation map!
              dqm->etahistiso[n+1]->Fill(recoecalcands[i]->eta(),mapi->val);
              dqm->ethistiso[n+1]->Fill(recoecalcands[i]->et()  ,mapi->val);
              dqm->phiHistIso[n+1]->Fill(recoecalcands[i]->phi(),mapi->val);
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
    for (unsigned int i=0; i < dqm->recocut_; i++) {
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
//        histEtOfHltObjMatchToReco[n] ->Fill( recoecalcands[closestRecoEcalCandIndex]->et()  );
//        histEtaOfHltObjMatchToReco[n]->Fill( recoecalcands[closestRecoEcalCandIndex]->eta() );
//        histPhiOfHltObjMatchToReco[n]->Fill( recoecalcands[closestRecoEcalCandIndex]->phi() );

          dqm->histHltObjMatchToReco[n].fill(recoecalcands[closestRecoEcalCandIndex]->p4());

        // Also store isolation info
        if (n+1 < dqm->numOfHLTCollectionLabels){ // can't plot beyond last
          if (dqm->plotiso[n+1] ){  // only plot if requested in config
            for (unsigned int j =  0 ; j < isoNameTokens_.size() ;j++  ){
              edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMap;
              iEvent.getByToken(isoNameTokens_.at(j),depMap);
              if (depMap.isValid()){ //Map may not exist if only one candidate passes a double filter
                typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMap->find(recoecalcands[closestRecoEcalCandIndex]);
                if (mapi!=depMap->end()) {  // found candidate in isolation map!
                  dqm->histEtaIsoOfHltObjMatchToReco[n+1]->Fill( recoecalcands[closestRecoEcalCandIndex]->eta(),mapi->val);
                  dqm->histEtIsoOfHltObjMatchToReco[n+1] ->Fill( recoecalcands[closestRecoEcalCandIndex]->et(), mapi->val);
                  dqm->histPhiIsoOfHltObjMatchToReco[n+1] ->Fill( recoecalcands[closestRecoEcalCandIndex]->phi(), mapi->val);
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
    for(unsigned int i =0; i < dqm->recocut_; i++){
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
//      ethistmatchreco[n] ->Fill( sortedReco[i].et()  );
//      etahistmatchreco[n]->Fill( sortedReco[i].eta() );
//      phiHistMatchReco[n]->Fill( sortedReco[i].phi() );
      dqm->histMatchReco[n].fill(sortedReco[i].p4());

      if (plotMonpath) {
//        ethistmatchrecomonpath[n]->Fill( sortedReco[i].et() );
//        etahistmatchrecomonpath[n]->Fill( sortedReco[i].eta() );
//        phiHistMatchRecoMonPath[n]->Fill( sortedReco[i].phi() );
          dqm->histMatchRecoMonPath[n].fill(sortedReco[i].p4());

      }
      ////////////////////////////////////////////////////////////
      //  Plot isolation variables (show the not-yet-cut        //
      //  isolation, i.e. associated to next filter)            //
      ////////////////////////////////////////////////////////////
      if (n+1 < dqm->numOfHLTCollectionLabels){ // can't plot beyond last
        if (dqm->plotiso[n+1] ){  // only plot if requested in config
          for (unsigned int j =  0 ; j < isoNameTokens_.size() ;j++  ){
            edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMapReco;
            iEvent.getByToken(isoNameTokens_.at(j),depMapReco);
            if (depMapReco.isValid()){ //Map may not exist if only one candidate passes a double filter
              typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMapReco->find(recoecalcands[closest]);
              if (mapi!=depMapReco->end()){  // found candidate in isolation map!
                dqm->etahistisomatchreco[n+1]->Fill(sortedReco[i].eta(),mapi->val);
                dqm->ethistisomatchreco[n+1]->Fill(sortedReco[i].et(),mapi->val);
                dqm->phiHistIsoMatchReco[n+1]->Fill(sortedReco[i].eta(),mapi->val);
              }
            }
          }
        }
      } // END of if n+1 < then the number of hlt collections
    }
    // fill total reco matched efficiency
    if (mtachedRecoParts >= dqm->reqNum )
     dqm-> totalmatchreco->Fill(n+0.5);
  }

}


////////////////////////////////////////////////////////////////////////////////
//      method called once each job just after ending the event loop          //
////////////////////////////////////////////////////////////////////////////////
void EmDQMReco::endJob(){

}

DEFINE_FWK_MODULE(EmDQMReco);
