
//===========Original Author: Matthias Mozer
//===========Modified for purpose of Tau Group: Konstantinos A. Petridis


#include "DQMOffline/Trigger/interface/HLTTauElDQMOfflineSource.h"

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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/AssociationMap.h"

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
HLTTauElDQMOfflineSource::HLTTauElDQMOfflineSource(const edm::ParameterSet& pset)  
{

  //paramters for generator study

  refCollection_=pset.getUntrackedParameter<edm::InputTag>("refCollection");
  reqNum_ = pset.getParameter<unsigned int>("reqNum");
  pdgGen_ =  pset.getParameter<int>("pdgGen");
  genEtaAcc_ = pset.getParameter<double>("genEtaAcc");
  genEtAcc_ = pset.getParameter<double>("genEtAcc");
  outputFile_= pset.getParameter<std::string>("outputFile");
  triggerName_=pset.getParameter<std::string>("triggerName");
  //plotting paramters
  thePtMin_ = pset.getUntrackedParameter<double>("PtMin",0.);
  thePtMax_ = pset.getUntrackedParameter<double>("PtMax",1000.);
  theNbins_ = pset.getUntrackedParameter<unsigned int>("Nbins",40);  
  //info for each filter-step
  std::vector<edm::ParameterSet> filters_ = pset.getParameter<std::vector<edm::ParameterSet> >("filters");

  for(std::vector<edm::ParameterSet>::iterator filterconf = filters_.begin() ; filterconf != filters_.end() ; filterconf++){
    m_theHLTCollectionLabels.push_back(filterconf->getParameter<edm::InputTag>("HLTCollectionLabels"));
    m_theHLTOutputTypes.push_back(filterconf->getParameter<unsigned int>("theHLTOutputTypes"));
    std::vector<double> bounds = filterconf->getParameter<std::vector<double> >("PlotBounds");
    assert(bounds.size() == 2);
    m_plotBounds.push_back(std::pair<double,double>(bounds[0],bounds[1]));
    m_isoNames.push_back(filterconf->getParameter<std::vector<edm::InputTag> >("IsoCollections"));
    assert(m_isoNames.back().size()>0);
    if (m_isoNames.back().at(0).label()=="none")
      m_plotiso.push_back(false);
    else{
      m_plotiso.push_back(true);
       //std::cout << "plotting isolation for: " <<  m_isoNames.back().at(0).label() << std::endl;
    }

  }
}


void HLTTauElDQMOfflineSource::beginJob(const edm::EventSetup&){
  std::string histoname="total eff";
  DQMStore* store = &*edm::Service<DQMStore>();

  if(store){


    store->setCurrentFolder(triggerName_);
    m_total = store->book1D(histoname.c_str(),histoname.c_str(),m_theHLTCollectionLabels.size()+2,0,m_theHLTCollectionLabels.size()+2);
    m_total->setBinLabel(m_theHLTCollectionLabels.size()+1,"Total");
    m_total->setBinLabel(m_theHLTCollectionLabels.size()+2,"Gen");
    for (unsigned int u=0; u<m_theHLTCollectionLabels.size(); u++){m_total->setBinLabel(u+1,m_theHLTCollectionLabels[u].label().c_str());}
    
    MonitorElement* tmphisto;
    
    histoname = "gen et";
    m_etgen =  store->book1D(histoname.c_str(),histoname.c_str(),theNbins_,thePtMin_,thePtMax_);
    histoname = "gen eta";
    m_etagen = store->book1D(histoname.c_str(),histoname.c_str(),theNbins_,-2.7,2.7);
    
    for(unsigned int i = 0; i< m_theHLTCollectionLabels.size() ; i++){
      histoname = m_theHLTCollectionLabels[i].label()+"et";
      tmphisto = store->book1D(histoname.c_str(),histoname.c_str(),theNbins_,thePtMin_,thePtMax_);
      m_ethist.push_back(tmphisto);
      
      histoname = m_theHLTCollectionLabels[i].label()+"eta";
      tmphisto =  store->book1D(histoname.c_str(),histoname.c_str(),theNbins_,-2.7,2.7);
      m_etahist.push_back(tmphisto);          
      
      histoname = m_theHLTCollectionLabels[i].label()+"etMatched";
      tmphisto = store->book1D(histoname.c_str(),histoname.c_str(),theNbins_,thePtMin_,thePtMax_);
      m_ethistmatch.push_back(tmphisto);
      
      histoname = m_theHLTCollectionLabels[i].label()+"etaMatched";
      tmphisto =  store->book1D(histoname.c_str(),histoname.c_str(),theNbins_,-2.7,2.7);
      m_etahistmatch.push_back(tmphisto);          
    }
  }
}

/// Destructor
HLTTauElDQMOfflineSource::~HLTTauElDQMOfflineSource(){
}

void HLTTauElDQMOfflineSource::analyze(const edm::Event & event , const edm::EventSetup& setup){



  // fill L1 and HLT info
  // get objects possed by each filter
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("hltTriggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("HLTTauElDQMOfflineSource") << "RAW-type HLT results not found, skipping event";
    return;
  }
// throw(cms::Exception("Release Validation Error")<< "RAW-type HLT results not found" );



//  for(int i = 0; i<triggerObj->size() ;i++ ){
//    std::cout << triggerObj->filterTag(i) << std::endl;
//  }  

  // total event number
  m_total->Fill(m_theHLTCollectionLabels.size()+0.5);
  // get matching objects
  edm::Handle<LVColl> refC;
  bool ref_ok=false;
  LVColl mcparts;
  if(event.getByLabel(refCollection_,refC)){
    ref_ok=true;
    if(refC->size())m_total->Fill(m_theHLTCollectionLabels.size()+1.5);
    for (size_t i=0;i<refC->size();++i)
      {
	double eta=refC->at(i).Eta();double et=refC->at(i).Et();
	m_etgen->Fill(et);
	m_etagen->Fill(eta);
	mcparts.push_back(refC->at(i));
      }
  }
  
  for(unsigned int n=0; n < m_theHLTCollectionLabels.size() ; n++) { //loop over filter modules

    switch(m_theHLTOutputTypes[n]){
    case 82: // non-iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,event,n,mcparts);break;
    case 83: // iso L1
      fillHistos<l1extra::L1EmParticleCollection>(triggerObj,event,n,mcparts);break;
    case 91: //photon 
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,event,n,mcparts);break;
    case 92: //electron 
      fillHistos<reco::ElectronCollection>(triggerObj,event,n,mcparts);break;
    case 100: // TriggerCluster
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,event,n,mcparts);break;
    default: throw(cms::Exception("Release Validation Error")<< "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
  }

}

template <class T> void HLTTauElDQMOfflineSource::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent ,unsigned int n,LVColl& mcparts){
  
  std::vector<edm::Ref<T> > recoecalcands;
  if (!( triggerObj->filterIndex(m_theHLTCollectionLabels[n])>=triggerObj->size() )){ // only process if availabel
    
    // retrieve saved filter objects
    triggerObj->getObjects(triggerObj->filterIndex(m_theHLTCollectionLabels[n]),m_theHLTOutputTypes[n],recoecalcands);
    //Danger: special case, L1 non-isolated
    // needs to be merged with L1 iso
    if(m_theHLTOutputTypes[n]==82){
      std::vector<edm::Ref<T> > isocands;
      triggerObj->getObjects(triggerObj->filterIndex(m_theHLTCollectionLabels[n]),83,isocands);
      if(isocands.size()>0)
	for(unsigned int i=0; i < isocands.size(); i++)
	  recoecalcands.push_back(isocands[i]);
    }
    
    //fill filter objects into histos
    if (recoecalcands.size()&&mcparts.size()){

      if(recoecalcands.size() >= reqNum_) 
	m_total->Fill(n+0.5);
      for (unsigned int i=0; i<recoecalcands.size(); i++) {
	//unmatched
	m_ethist[n]->Fill(recoecalcands[i]->et() );
	m_etahist[n]->Fill(recoecalcands[i]->eta() );
      }
    
      
      for(size_t i=0;i<mcparts.size();++i)
	{
	  double closestDr=1000.;
	  LV closest;
	  for (size_t ii=0; ii<recoecalcands.size(); ++ii) {
	    //matched
	    math::XYZVector candDir=recoecalcands[ii]->momentum();
	    double dr = DeltaR(mcparts[i],candDir);
	    if(dr < closestDr){
		closestDr = dr;
		//closest = mcparts[i];
	    }
	  }
	  if( (closestDr<0.5&&(m_theHLTOutputTypes[n]==82 || m_theHLTOutputTypes[n]==83)) ||
	      (closestDr<0.1&&m_theHLTOutputTypes[n]>90) ){
	    float eta   =mcparts[i].Eta();//closest.Eta();
	    float et    =mcparts[i].Et();//closest.Et();
	    m_ethistmatch[n]->Fill( et );
	    m_etahistmatch[n]->Fill( eta );
	  }
	}
    }
  }
}
  


void HLTTauElDQMOfflineSource::endJob(){

  if(outputFile_.size()>0)
  if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outputFile_);
  //  total->Scale(1./total->GetBinContent(1));
  //for(unsigned int n= m_theHLTCollectionLabels.size()-1 ; n>0;n--){
  //  ethist[n]->Divide(ethist[n-1]);
  //  etahist[n]->Divide(etahist[n-1]);
  //}
}

