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
HLTElectronPFMTFilter::HLTElectronPFMTFilter(const edm::ParameterSet& iConfig)
{
  // MHT parameters
  inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
  saveTags_     = iConfig.getParameter<bool>("saveTags");
  minMht_= iConfig.getParameter<double> ("minMht");
  minPtJet_= iConfig.getParameter<std::vector<double> > ("minPtJet");
  minNJet_= iConfig.getParameter<int> ("minNJet");
  mode_= iConfig.getParameter<int>("mode");
  //----mode=1 for MHT only
  //----mode=2 for Meff
  //----mode=3 for PT12
  //----mode=4 for HT only
  //----mode=5 for HT and AlphaT cross trigger (ALWAYS uses jet ET, not pT)
  etaJet_= iConfig.getParameter<std::vector<double> > ("etaJet");
  usePt_= iConfig.getParameter<bool>("usePt");
  minPT12_= iConfig.getParameter<double> ("minPT12");
  minMeff_= iConfig.getParameter<double> ("minMeff");
  minHt_= iConfig.getParameter<double> ("minHt");
  minAlphaT_= iConfig.getParameter<double> ("minAlphaT");
  // Electron parameters
  inputEleTag_            = iConfig.getParameter< edm::InputTag > ("inputEleTag");
  lowerMTCut_       = iConfig.getParameter<double> ("lowerMTCut");
  upperMTCut_       = iConfig.getParameter<double> ("upperMTCut");
  relaxed_ = iConfig.getUntrackedParameter<bool> ("relaxed",true) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

  // sanity checks
  if (       (minPtJet_.size()    !=  etaJet_.size())
       || (  (minPtJet_.size()<1) || (etaJet_.size()<1) )
       || ( ((minPtJet_.size()<2) || (etaJet_.size()<2))
	    && ( (mode_==1) || (mode_==2) || (mode_ == 5))) ) {
    edm::LogError("HLTElectronPFMTFilter") << "inconsistent module configuration!";
  }

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTElectronPFMTFilter::~HLTElectronPFMTFilter(){}

void HLTElectronPFMTFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("minMht",0.0);
  {
    std::vector<double> temp1;
    temp1.reserve(2);
    temp1.push_back(20.0);
    temp1.push_back(20.0);
    desc.add<std::vector<double> >("minPtJet",temp1);
  }
  desc.add<int>("minNJet",0);
  desc.add<int>("mode",2);
  {
    std::vector<double> temp1;
    temp1.reserve(2);
    temp1.push_back(9999.0);
    temp1.push_back(9999.0);
    desc.add<std::vector<double> >("etaJet",temp1);
  }
  desc.add<bool>("usePt",true);
  desc.add<double>("minPT12",0.0);
  desc.add<double>("minMeff",180.0);
  desc.add<double>("minHt",0.0);
  desc.add<double>("minAlphaT",0.0);
  descriptions.add("hltElectronPFMTFilter",desc);
}



// ------------ method called to produce the data  ------------
bool
  HLTElectronPFMTFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTags_) filterobject->addCollectionTag(inputJetTag_);

  CaloJetRef ref;
  // Get the Candidates

  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0), nj(0), flag(0);
  double ht=0.;
  double mhtx=0., mhty=0.;
  double jetVar;
  double dht = 0.;
  double aT = 0.;
  if(recocalojets->size() > 0){
    // events with at least one jet
    for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin();
    recocalojet != recocalojets->end(); recocalojet++) {
      if (flag == 1){break;}
      jetVar = recocalojet->pt();
      if (!usePt_ || mode_==3 ) jetVar = recocalojet->et();

      if (mode_==1 || mode_==2 || mode_ == 5) {//---get MHT
        if (jetVar > minPtJet_.at(1) && fabs(recocalojet->eta()) < etaJet_.at(1)) {
          mhtx -= jetVar*cos(recocalojet->phi());
          mhty -= jetVar*sin(recocalojet->phi());
        }
      }
      if (mode_==2 || mode_==4 || mode_==5) {//---get HT
        if (jetVar > minPtJet_.at(0) && fabs(recocalojet->eta()) < etaJet_.at(0)) {
          ht += jetVar;
          nj++;
        }
      }
      if (mode_==3) {//---get PT12
        if (jetVar > minPtJet_.at(0) && fabs(recocalojet->eta()) < etaJet_.at(0)) {
          nj++;
          mhtx -= jetVar*cos(recocalojet->phi());
          mhty -= jetVar*sin(recocalojet->phi());
          if (nj==2) break;
        }
      }
      if(mode_ == 5){
        double mHT = sqrt( (mhtx*mhtx) + (mhty*mhty) );
        dht += ( nj < 2 ? jetVar : -1.* jetVar ); //@@ only use for njets < 4
        if ( nj == 2 || nj == 3 ) {
          aT = ( ht - fabs(dht) ) / ( 2. * sqrt( ( ht*ht ) - ( mHT*mHT  ) ) );
        } else if ( nj > 3 ) {
          aT = ht / ( 2.*sqrt( ( ht*ht ) - ( mHT*mHT  ) ) );
        }
        if(ht > minHt_ && aT > minAlphaT_){
  // put filter object into the Event
          flag = 1;
        }
      }
    }

  if( mode_==1 && sqrt(mhtx*mhtx + mhty*mhty) > minMht_) flag=1;
  if( mode_==2 && sqrt(mhtx*mhtx + mhty*mhty)+ht > minMeff_) flag=1;
  if( mode_==3 && sqrt(mhtx*mhtx + mhty*mhty) > minPT12_ && nj>1) flag=1;
  if( mode_==4 && ht > minHt_) flag=1;

  if (flag==1) {
    for (reco::CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); recocalojet!=recocalojets->end(); recocalojet++) {
      jetVar = recocalojet->pt();
      if (!usePt_ || mode_==3) jetVar = recocalojet->et();

      if (jetVar > minPtJet_.at(0)) {
        ref = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
        filterobject->addObject(TriggerJet,ref);
        n++;
      }
    }
  }
} // events with at least one jet


  if( saveTags_ ){filterobject->addCollectionTag(L1IsoCollTag_);}
  if( saveTags_ && relaxed_){filterobject->addCollectionTag(L1NonIsoCollTag_);}
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel (inputEleTag_,PrevFilterOutput); 

   
  int nW = 0;
    
  Ref< ElectronCollection > refele;
    
  vector< Ref< ElectronCollection > > electrons;
  PrevFilterOutput->getObjects(TriggerElectron, electrons);
  
  TLorentzVector pMET(mhtx, mhty,0.0,sqrt(mhtx*mhtx + mhty*mhty));

    
  for (unsigned int i=0; i<electrons.size(); i++) {
    
      refele = electrons[i];
      TLorentzVector pThisEle(refele->px(), refele->py(), 
			      0.0, refele->et() );
      TLorentzVector pTot = pMET + pThisEle;
      double mass = pTot.M();
       
      if(mass>=lowerMTCut_ && mass<=upperMTCut_)
        {
	 if (flag==1) nW++;
	 refele = electrons[i];
	 filterobject->addObject(TriggerElectron, refele);
        }
  }

  // filter decision
  bool accept(nW>0);
  iEvent.put(filterobject);

return accept;
}
