/** \class EgammaHLTCaloIsolFilterPairs
 *
 *
 *  \author Alessio Ghezzi
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaCaloIsolFilterPairs.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
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

  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  isoToken_ = consumes<reco::RecoEcalCandidateIsolationMap>(isoTag_);
  if(AlsoNonIso_1 || AlsoNonIso_2) nonIsoToken_ = consumes<reco::RecoEcalCandidateIsolationMap>(nonIsoTag_);
}

HLTEgammaCaloIsolFilterPairs::~HLTEgammaCaloIsolFilterPairs(){}

void
HLTEgammaCaloIsolFilterPairs::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
   edm::ParameterSetDescription desc;
   makeHLTFilterDescription(desc);
   desc.add<edm::InputTag>("candTag",edm::InputTag(""));
   desc.add<edm::InputTag>("isoTag",edm::InputTag(""));
   desc.add<edm::InputTag>("nonIsoTag",edm::InputTag(""));
   desc.add<double>("isolcutEB1",0.0);
   desc.add<double>("IsoOverEtCutEB1",0.0);
   desc.add<double>("IsoOverEt2CutEB1",0.0);
   desc.add<double>("isolcutEE1",0.0);
   desc.add<double>("IsoOverEtCutEE1",0.0);
   desc.add<double>("IsoOverEt2CutEE1",0.0);
   desc.add<double>("isolcutEB2",0.0);
   desc.add<double>("IsoOverEtCutEB2",0.0);
   desc.add<double>("IsoOverEt2CutEB2",0.0);
   desc.add<double>("isolcutEE2",0.0);
   desc.add<double>("IsoOverEtCutEE2",0.0);
   desc.add<double>("IsoOverEt2CutEE2",0.0);
   desc.add<bool>("AlsoNonIso1",false);
   desc.add<bool>("AlsoNonIso2",false);
   descriptions.add("hltEgammaCaloIsolFilterPairs",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTEgammaCaloIsolFilterPairs::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace trigger;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (candToken_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton, recoecalcands);

  //get hold of ecal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByToken (isoToken_,depMap);

  //get hold of ecal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(AlsoNonIso_1 || AlsoNonIso_2) iEvent.getByToken (nonIsoToken_,depNonIsoMap);


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

bool HLTEgammaCaloIsolFilterPairs::PassCaloIsolation(edm::Ref<reco::RecoEcalCandidateCollection> ref,const reco::RecoEcalCandidateIsolationMap& IsoMap,const reco::RecoEcalCandidateIsolationMap& NonIsoMap, int which, bool ChekAlsoNonIso) const {


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
