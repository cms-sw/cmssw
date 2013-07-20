/** \class EgammaHLTCaloIsolFilterPairs
 *
 * $Id: HLTEgammaCaloIsolFilterPairs.cc,v 1.2 2012/01/21 14:56:56 fwyzard Exp $
 * 
 *  \author Alessio Ghezzi
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaCaloIsolFilterPairs.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTEgammaCaloIsolFilterPairs::HLTEgammaCaloIsolFilterPairs(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");

  isolcut_EB1  = iConfig.getParameter<double> ("isolcutEB1");
  FracCut_EB1 = iConfig.getParameter<double> ("IsoOverEtCutEB1");
  IsoloEt2_EB1 = iConfig.getParameter<double> ("IsoOverEt2CutEB1");
  isolcut_EE1  = iConfig.getParameter<double> ("isolcutEE1");
  FracCut_EE1 = iConfig.getParameter<double> ("IsoOverEtCutEE1");
  IsoloEt2_EE1 = iConfig.getParameter<double> ("IsoOverEt2CutEE1");

  isolcut_EB2  = iConfig.getParameter<double> ("isolcutEB2");
  FracCut_EB2 = iConfig.getParameter<double> ("IsoOverEtCutEB2");
  IsoloEt2_EB2 = iConfig.getParameter<double> ("IsoOverEt2CutEB2");
  isolcut_EE2  = iConfig.getParameter<double> ("isolcutEE2");
  FracCut_EE2 = iConfig.getParameter<double> ("IsoOverEtCutEE2");
  IsoloEt2_EE2 = iConfig.getParameter<double> ("IsoOverEt2CutEE2");


  AlsoNonIso_1 = iConfig.getParameter<bool> ("AlsoNonIso1");
  AlsoNonIso_2 = iConfig.getParameter<bool> ("AlsoNonIso2");
}

HLTEgammaCaloIsolFilterPairs::~HLTEgammaCaloIsolFilterPairs(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaCaloIsolFilterPairs::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace trigger;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);

  //get hold of ecal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of ecal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(AlsoNonIso_1 || AlsoNonIso_2) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
  

  int n = 0;
   // the list should be interpreted as pairs:
  // <recoecalcands[0],recoecalcands[1]>
  // <recoecalcands[2],recoecalcands[3]>
  // <recoecalcands[4],recoecalcands[5]>
  // .......

  // Should I check that the size of recoecalcands is even ?
  for (unsigned int i=0; i<recoecalcands.size(); i=i+2) {
    
    edm::Ref<reco::RecoEcalCandidateCollection> r1 = recoecalcands[i];
    edm::Ref<reco::RecoEcalCandidateCollection> r2 = recoecalcands[i+1];
    // std::cout<<"CaloIsol 1) Et Eta phi: "<<r1->et()<<" "<<r1->eta()<<" "<<r1->phi()<<" 2) Et eta phi: "<<r2->et()<<" "<<r2->eta()<<" "<<r2->phi()<<std::endl;

    if( PassCaloIsolation(r1,*depMap,*depNonIsoMap,1,AlsoNonIso_1) && PassCaloIsolation(r2,*depMap,*depNonIsoMap,2,AlsoNonIso_2)    )
      {
	n++;
	filterproduct.addObject(TriggerCluster, r1);
	filterproduct.addObject(TriggerCluster, r2);
      }
  }
  
  // filter decision
  bool accept(n>=1);
  
  return accept;
}

bool HLTEgammaCaloIsolFilterPairs::PassCaloIsolation(edm::Ref<reco::RecoEcalCandidateCollection> ref,reco::RecoEcalCandidateIsolationMap IsoMap,reco::RecoEcalCandidateIsolationMap NonIsoMap, int which, bool ChekAlsoNonIso){

  
  reco::RecoEcalCandidateIsolationMap::const_iterator mapi = IsoMap.find( ref );

  if(mapi==IsoMap.end()) {
    if(ChekAlsoNonIso) mapi = NonIsoMap.find( ref ); 
  }

  float vali = mapi->val;
  float IsoOE= vali/ref->et();
  float IsoOE2= IsoOE/ref->et();
  double isolcut=0,FracCut=0,IsoloEt2=0;
  if( fabs(ref->eta()) < 1.479){
    if(which==1){
      isolcut = isolcut_EB1;
      FracCut =  FracCut_EB1;
      IsoloEt2  =  IsoloEt2_EB1;
    }
    else if(which==2){
      isolcut = isolcut_EB2;
      FracCut =  FracCut_EB2;
      IsoloEt2  =  IsoloEt2_EB2;    
    }
    else {return false;}
  }
  else {
    if(which==1){
      isolcut = isolcut_EE1;
      FracCut =  FracCut_EE1;
       IsoloEt2 =  IsoloEt2_EE1;
    }
    else if(which==2){
      isolcut = isolcut_EE2;
      FracCut =  FracCut_EE2;
      IsoloEt2 = IsoloEt2_EE2;    
    }
    else {return false;}   
  }
  
  if ( vali < isolcut || IsoOE < FracCut || IsoOE2 < IsoloEt2 ) { return true;}
  return false;
}
