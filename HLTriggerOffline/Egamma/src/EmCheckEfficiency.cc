#include "HLTriggerOffline/Egamma/interface/EmCheckEfficiency.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"
#include <iostream>
#include <string>
#include <Math/VectorUtil.h>
using namespace ROOT::Math::VectorUtil ;

/// Constructor
EmCheckEfficiency::EmCheckEfficiency(const edm::ParameterSet& pset)  
{

  //parameters for generator study
  _doMC = pset.getParameter<bool>("doMC");
  _doOffline = pset.getParameter<bool>("doOffline");
  reqNum = pset.getParameter<unsigned int>("reqNum");
  pdgGen =  pset.getParameter<int>("pdgGen");
  genEtaAcc = pset.getParameter<double>("genEtaAcc");
  genEtAcc = pset.getParameter<double>("genEtAcc");
  //plotting parameters
  thePtMin = pset.getUntrackedParameter<double>("PtMin",0.);
  thePtMax = pset.getUntrackedParameter<double>("PtMax",1000.);
  theNbins = pset.getUntrackedParameter<unsigned int>("Nbins",40);
  
  //info for each filter-step
  std::vector<edm::ParameterSet> filters = pset.getParameter<std::vector<edm::ParameterSet> >("filters");

  for(std::vector<edm::ParameterSet>::iterator filterconf = filters.begin() ; filterconf != filters.end() ; filterconf++){
    theHLTCollectionLabels.push_back(filterconf->getParameter<edm::InputTag>("HLTCollectionLabels"));
    theHLTOutputTypes.push_back(filterconf->getParameter<unsigned int>("theHLTOutputTypes"));
    std::vector<double> bounds = filterconf->getParameter<std::vector<double> >("PlotBounds");
    assert(bounds.size() == 2);
    plotBounds.push_back(std::pair<double,double>(bounds[0],bounds[1]));
    isoNames.push_back(filterconf->getParameter<std::vector<edm::InputTag> >("IsoCollections"));
    assert(isoNames.back().size()>0);
    if (isoNames.back().at(0).label()=="none")
      plotiso.push_back(false);
    else{
      plotiso.push_back(true);
       //std::cout << "plotting isolation for: " <<  isoNames.back().at(0).label() << std::endl;
    }   
  }
}


void EmCheckEfficiency::beginJob(const edm::EventSetup&){
  edm::Service<TFileService> fs;

  std::string histoname="";

  if (_doMC) {
 
    histoname="total eff";
    
    total = fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theHLTCollectionLabels.size()+2,0,theHLTCollectionLabels.size()+2);
    total->GetXaxis()->SetBinLabel(1,"Total");
    total->GetXaxis()->SetBinLabel(2,"Gen");
    for (unsigned int u=0; u<theHLTCollectionLabels.size(); u++){total->GetXaxis()->SetBinLabel(u+3,theHLTCollectionLabels[u].label().c_str());}
  }

  if (_doOffline) {
    histoname = "total eff offline";
    totaloff = fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theHLTCollectionLabels.size()+2,0,theHLTCollectionLabels.size()+2);
    totaloff->GetXaxis()->SetBinLabel(1,"Total");
    totaloff->GetXaxis()->SetBinLabel(2,"All Offline");
    for (unsigned int u=0; u<theHLTCollectionLabels.size(); u++){totaloff->GetXaxis()->SetBinLabel(u+3,theHLTCollectionLabels[u].label().c_str());}
  }

  TH1F* tmphisto;
  TH2F* tmpiso;

  if (_doMC) {
    histoname = "gen et";
    etgen =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
    histoname = "gen eta";
    etagen = fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
  }
  
  if (_doOffline) {
    histoname = "off et";
    etoff =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
    histoname = "off eta";
    etaoff = fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
  }

  for(unsigned int i = 0; i< theHLTCollectionLabels.size() ; i++){
    histoname = theHLTCollectionLabels[i].label()+"et";
    tmphisto =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
    ethist.push_back(tmphisto);
    
    histoname = theHLTCollectionLabels[i].label()+"eta";
    tmphisto =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
    etahist.push_back(tmphisto);          

    if (_doMC) {
      histoname = theHLTCollectionLabels[i].label()+"et_MC_matched";
      tmphisto =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
      ethistmatch.push_back(tmphisto);
      
      histoname = theHLTCollectionLabels[i].label()+"eta_MC_matched";
      tmphisto =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
      etahistmatch.push_back(tmphisto);   
    }

    if (_doOffline) {
      histoname = theHLTCollectionLabels[i].label()+"et_offline_matched";
      tmphisto =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
      ethistoff.push_back(tmphisto);
      
      histoname = theHLTCollectionLabels[i].label()+"eta_offline_matched";
      tmphisto =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
      etahistoff.push_back(tmphisto);  
    }
    
    if(plotiso[i]){
      histoname = theHLTCollectionLabels[i].label()+"eta_isolation";
      tmpiso = fs->make<TH2F>(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7,theNbins,plotBounds[i].first,plotBounds[i].second);
    }
    else{
      tmpiso = NULL;
    }
    etahistiso.push_back(tmpiso);

    if(plotiso[i]){
      histoname = theHLTCollectionLabels[i].label()+"et_isolation";
      tmpiso = fs->make<TH2F>(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax,theNbins,plotBounds[i].first,plotBounds[i].second);
    }
    else{
      tmpiso = NULL;
    }
    ethistiso.push_back(tmpiso);
 
  }
  
  // histoname = "Delta phi";
  // dphi =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),50,0.,0.3);
  // histoname = "Delta eta";
  // deta =  fs->make<TH1F>(histoname.c_str(),histoname.c_str(),50,0.,0.05);
}


/// Destructor
EmCheckEfficiency::~EmCheckEfficiency(){
}

void EmCheckEfficiency::analyze(const edm::Event & event , const edm::EventSetup& setup){

  // fill L1 and HLT info
  // get objects passed by each filter
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("hltTriggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    std::cout << "RAW-type HLT results not found, skipping event" << std::endl;
    // return;
  }

  unsigned int ncand = 0;
  std::vector<HepMC::GenParticle> mcparts;
  reco::GsfElectronCollection offele;

  // fill generator info
  if (_doMC) {
    // total event number
    total->Fill(0.5); 
    
    edm::Handle<edm::HepMCProduct> genEvt;
    event.getByLabel("source", genEvt);
    
    const HepMC::GenEvent * myGenEvent = genEvt->GetEvent();
    
    for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) {
      if (  !( abs((*p)->pdg_id())==pdgGen  && (*p)->status()==1 )   )  continue;
      float eta   =(*p)->momentum().eta();
      float e     =(*p)->momentum().e();
      float theta =2*atan(exp(-eta));
      float Et    =e*sin(theta);
      if(fabs(eta) < genEtaAcc &&  Et > genEtAcc) {
	ncand++;
	etgen->Fill(Et);
	etagen->Fill(eta);
	mcparts.push_back(*(*p));
      }
    } //end of loop over MC particles
    if (ncand >= reqNum) total->Fill(1.5);
  }

  ncand = 0; 

  // fill offline electrons info
  if (_doOffline) {
    // total event number
    totaloff->Fill(0.5);

    edm::Handle<reco::GsfElectronCollection> electrons;
    event.getByLabel("pixelMatchGsfElectrons",electrons);
    
    // unsigned int ncand = 0;
    for ( reco::GsfElectronCollection::const_iterator it=electrons->begin(); it != electrons->end(); ++it ) {
      float eta   =(*it).eta();
      float Et    =(*it).et();
      ncand++;
      etoff->Fill(Et);
      etaoff->Fill(eta);
      offele.push_back(*it);
    }
    if (ncand >= reqNum) totaloff->Fill(1.5);
  }
  
  for(unsigned int n=0; n < theHLTCollectionLabels.size() ; n++) { //loop over filter modules
    // std::cout << "ThisIndex = " << " " << theHLTCollectionLabels[n].label() << " " << triggerObj->filterIndex(theHLTCollectionLabels[n]) << std::endl;
    //  std::cout << "TheSize = " << triggerObj->size() << std::endl;
    
    switch(theHLTOutputTypes[n]){
    case 82: // non-iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,event,n,mcparts,offele);break;
    case 83: // iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,event,n,mcparts,offele);break;
    case 91: //photon 
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,event,n,mcparts,offele);break;
    case 92: //electron 
      fillHistos<reco::ElectronCollection>(triggerObj,event,n,mcparts,offele);break;
    case 100: // TriggerCluster
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,event,n,mcparts,offele);break;
    default: throw(cms::Exception("Release Validation Error")<< "HLT output type not implemented: " << theHLTOutputTypes[n]);
    }
  }
}

template <class T> void EmCheckEfficiency::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent ,unsigned int n,std::vector<HepMC::GenParticle>& mcparts, reco::GsfElectronCollection& offele) {
  
  double dRmax = 0.3;

  std::vector< edm::Ref<T> > recoecalcands;
  if (!( triggerObj->filterIndex(theHLTCollectionLabels[n])>=triggerObj->size() )){ // only process if available
  
    // retrieve saved filter objects
    triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),theHLTOutputTypes[n],recoecalcands);
    // std::cout << "EcalCandsSize = " << recoecalcands.size() << std::endl;
    // WARNING: special case, L1 non-isolated
    // needs to be merged with L1 iso
    if(theHLTOutputTypes[n]==82) {
      dRmax = 0.4;
      std::vector<edm::Ref<T> > isocands;
      triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),83,isocands);
      if(isocands.size()>0)
	for(unsigned int i=0; i < isocands.size(); i++)
	  recoecalcands.push_back(isocands[i]);
    }
    if(theHLTOutputTypes[n]==83) {dRmax = 0.4;}
  }  

  // NEW WAY
  if (recoecalcands.size()!=0) {
    
    std::vector<HepMC::GenParticle>::iterator closest = mcparts.end();
    for(std::vector<HepMC::GenParticle>::iterator mc = mcparts.begin(); mc !=  mcparts.end() ; mc++){
      math::XYZVector mcDir( mc->momentum().px(),
			     mc->momentum().py(),
			     mc->momentum().pz());
      for (unsigned int i=0; i<recoecalcands.size(); i++) { 
	math::XYZVector candDir=recoecalcands[i]->momentum();
	if (DeltaR(mcDir,candDir) < dRmax) {
          closest = mc;
          float eta   =closest->momentum().eta();
	  float e     =closest->momentum().e();
	  float theta =2*atan(exp(-eta));
	  float Et    =e*sin(theta);
	  ethistmatch[n]->Fill( Et );
	  etahistmatch[n]->Fill( eta );
          break;
	}
      }
    }
    if( recoecalcands.size() >= reqNum && closest != mcparts.end()) total->Fill(n+2.5);
    
    reco::GsfElectronCollection::iterator closestoff = offele.end();
    for (reco::GsfElectronCollection::iterator it = offele.begin(); it !=  offele.end() ; it++){
      math::XYZVector offlineDir( (*it).px(),
				  (*it).py(),
				  (*it).pz());
      for (unsigned int i=0; i<recoecalcands.size(); i++) { 
	math::XYZVector candDir=recoecalcands[i]->momentum();
	if (DeltaR(offlineDir,candDir) < dRmax) {
	  closestoff = it;
	  float eta   =closestoff->eta();
	  float Et    =closestoff->et();
	  ethistoff[n]->Fill( Et );
	  etahistoff[n]->Fill( eta );
	  break;
	}
      }
    }
    if( recoecalcands.size() >= reqNum && closestoff != offele.end()) totaloff->Fill(n+2.5);
    
    // OLD WAY : EFFICIENCY CAN BE > 1.0
    /*
    //fill filter objects into histos
    if (recoecalcands.size()!=0){
      // if(recoecalcands.size() >= reqNum ) 
      //	total->Fill(n+2.5);
      for (unsigned int i=0; i<recoecalcands.size(); i++) { 
	//unmatched
	ethist[n]->Fill(recoecalcands[i]->et() );
	etahist[n]->Fill(recoecalcands[i]->eta() );
	// MC matched
	math::XYZVector candDir=recoecalcands[i]->momentum();
	std::vector<HepMC::GenParticle>::iterator closest = mcparts.end();
	double closestDr=1000. ;
	for(std::vector<HepMC::GenParticle>::iterator mc = mcparts.begin(); mc !=  mcparts.end() ; mc++){
	  math::XYZVector mcDir( mc->momentum().px(),
				 mc->momentum().py(),
				 mc->momentum().pz());	  
	  double dr = DeltaR(mcDir,candDir);
	  if(dr < closestDr){
	    closestDr = dr;
	    closest = mc;
	  }
	}
	if (closest == mcparts.end())  edm::LogWarning("EmCheckEfficiency") << "no MC match, may skew efficiency";
	else{
          if(recoecalcands.size() >= reqNum ) total->Fill(n+2.5);
	  float eta   =closest->momentum().eta();
	  float e     =closest->momentum().e();
	  float theta =2*atan(exp(-eta));
	  float Et    =e*sin(theta);
	  ethistmatch[n]->Fill( Et );
	  etahistmatch[n]->Fill( eta );
	}

        // offline matched 
        reco::GsfElectronCollection::iterator closestoff = offele.end();
	double closestoffDr=1000. ;
	for(reco::GsfElectronCollection::iterator it = offele.begin(); it !=  offele.end() ; it++){
	  math::XYZVector offlineDir( (*it).px(),
				      (*it).py(),
				      (*it).pz());	  
	  double dr = DeltaR(offlineDir,candDir);
	  if(dr < closestoffDr){
	    closestoffDr = dr;
	    closestoff = it;
	  }
	}
	if (closestoff == offele.end())  edm::LogWarning("EmCheckEfficiency") << "no offline match, may skew efficiency";
	else {
          if(recoecalcands.size() >= reqNum ) totaloff->Fill(n+2.5);
	  float eta   =closestoff->eta();
	  float Et    =closestoff->et();
	  // float theta =2*atan(exp(-eta));
	  // float Et    =e*sin(theta);
	  ethistoff[n]->Fill( Et );
	  etahistoff[n]->Fill( eta );
	}
    */

    //plot isolation variables (show not yet cut iso, i.e. associated to next filter)

    for (unsigned int i=0; i<recoecalcands.size(); i++) { 
      //unmatched
      ethist[n]->Fill( recoecalcands[i]->et() );
      etahist[n]->Fill( recoecalcands[i]->eta() );
      if(n+1 < theHLTCollectionLabels.size()){ // can't plot beyond last
	if(plotiso[n+1] ){
	  for(unsigned int j =  0 ; j < isoNames[n+1].size() ;j++  ){
	    edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMap; 
	    if(depMap.isValid()){ // Map may not exist if only one candidate passes a double filter
	      iEvent.getByLabel(isoNames[n+1].at(j).label(),depMap);
	      typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMap->find(recoecalcands[i]);
	      if(mapi!=depMap->end()){  // found candidate in isolation map! 
		etahistiso[n+1]->Fill(recoecalcands[i]->eta(),mapi->val);
		ethistiso[n+1]->Fill(recoecalcands[i]->et(),mapi->val);
		break; // to avoid multiple filling we only look until we found the candidate once.
	      }
	    }
	  }
	}
      }
    } 
  }
}

void EmCheckEfficiency::endJob(){
  //  total->Scale(1./total->GetBinContent(1));
  //for(unsigned int n= theHLTCollectionLabels.size()-1 ; n>0;n--){
  //  ethist[n]->Divide(ethist[n-1]);
  //  etahist[n]->Divide(etahist[n-1]);
  //}
}

DEFINE_FWK_MODULE(EmCheckEfficiency);
