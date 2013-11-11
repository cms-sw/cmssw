/** \class HLTElectronPFMTFilter
 *
 *
 *  \author Gheorghe Lungu
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronPFMTFilter.h"

//
// constructors and destructor
//
HLTElectronPFMTFilter::HLTElectronPFMTFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  // MHT parameters
  inputMetTag_ = iConfig.getParameter< edm::InputTag > ("inputMetTag");
  minMht_      = iConfig.getParameter<double> ("minMht");
  // Electron parameters
  inputEleTag_ = iConfig.getParameter< edm::InputTag > ("inputEleTag");
  lowerMTCut_  = iConfig.getParameter<double> ("lowerMTCut");
  upperMTCut_  = iConfig.getParameter<double> ("upperMTCut");
  relaxed_     = iConfig.getParameter<bool> ("relaxed");
  minN_        = iConfig.getParameter<int>("minN");
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

  inputMetToken_ = consumes<reco::METCollection>(inputMetTag_);
  inputEleToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(inputEleTag_);
}

HLTElectronPFMTFilter::~HLTElectronPFMTFilter(){}

void HLTElectronPFMTFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputMetTag",edm::InputTag("hltPFMHT"));
  desc.add<edm::InputTag>("inputEleTag",edm::InputTag("hltEle25CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter"));
  desc.add<edm::InputTag>("L1IsoCand",edm::InputTag("hltL1IsoRecoEcalCandidate"));
  desc.add<edm::InputTag>("L1NonIsoCand",edm::InputTag("hltL1NonIsoRecoEcalCandidate"));
  desc.add<bool>("relaxed",true);
  desc.add<int>("minN",0);
  desc.add<double>("minMht",0.0);
  desc.add<double>("lowerMTCut",0.0);
  desc.add<double>("upperMTCut",9999.0);
  descriptions.add("hltElectronPFMTFilter",desc);
}



// ------------ method called to produce the data  ------------
bool
    HLTElectronPFMTFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(inputMetTag_);
    filterproduct.addCollectionTag(L1IsoCollTag_);
    if (relaxed_) filterproduct.addCollectionTag(L1NonIsoCollTag_);
  }
  
  // Get the Met collection
  edm::Handle<reco::METCollection> pfMHT;
  iEvent.getByToken(inputMetToken_,pfMHT);

  // Sanity check:
  if(!pfMHT.isValid()) {
    
    edm::LogError("HLTElectronPFMTFilter") << "missing input Met collection!";
    
  }
  
  const METCollection *metcol = pfMHT.product();
  const MET *met;
  met = &(metcol->front());
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (inputEleToken_,PrevFilterOutput); 
   
  int nW = 0;
    
  Ref< ElectronCollection > refele;
    
  vector< Ref< ElectronCollection > > electrons;
  PrevFilterOutput->getObjects(TriggerElectron, electrons);

  TLorentzVector pMET(met->px(), met->py(),0.0,sqrt(met->px()*met->px() + met->py()*met->py()));
    
  for (unsigned int i=0; i<electrons.size(); i++) {
    
    refele = electrons[i];
    TLorentzVector pThisEle(refele->px(), refele->py(), 
                            0.0, refele->et() );
    TLorentzVector pTot = pMET + pThisEle;
    double mass = pTot.M();
       
    if(mass>=lowerMTCut_ && mass<=upperMTCut_ && pMET.E()>= minMht_)
    {
      nW++;
      refele = electrons[i];
      filterproduct.addObject(TriggerElectron, refele);
    }
  }

  // filter decision
  const bool accept(nW>=minN_);

  return accept;
}
