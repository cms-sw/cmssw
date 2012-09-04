#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DQM/DataScouting/interface/ThreeJetVarProducer.h"

#include "TVector3.h"
#include <memory>
#include <algorithm>
#include <vector>

namespace {
  bool bigger(double a, double b)
  { return (a > b); }
  bool sortLorentzPt(TLorentzVector a,TLorentzVector b)
  {return (a.Pt() > b.Pt()); }
}
//
// constructors and destructor
//

ThreeJetVarProducer::ThreeJetVarProducer(const edm::ParameterSet& iConfig) :
  inputJetTag_    (iConfig.getParameter<edm::InputTag>("inputJetTag")){
  
  produces<std::vector<double> >();

  LogDebug("") << "Inputs: "
	       << inputJetTag_.encode() << " ";
}

ThreeJetVarProducer::~ThreeJetVarProducer()
{
}

// ------------ method called to produce the data  ------------ 
void
ThreeJetVarProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;


   // get hold of collection of objects
   edm::Handle<reco::CaloJetCollection> calojet_handle; 
   iEvent.getByLabel(inputJetTag_,calojet_handle);

   std::auto_ptr<std::vector<double> > jetPt(new std::vector<double>);
   std::auto_ptr<std::vector<double> > tripMass(new std::vector<double>);
   std::auto_ptr<std::vector<double> > tripSumPt(new std::vector<double>);
   jetPt->reserve(6);
   tripMass->reserve(20);
   tripSumPt->reserve(20);
   // check the the input collections are available
   if (calojet_handle.isValid()){
     std::vector<TLorentzVector> myJets;
     reco::CaloJetCollection::const_iterator jetIt;
     //Select jets
     for(jetIt = calojet_handle->begin(); jetIt != calojet_handle->end(); jetIt++){
       if (jetIt->pt()        < 30.0) continue;
       if (fabs(jetIt->eta()) >  3.0) continue; 
       TLorentzVector j; j.SetPtEtaPhiE(jetIt->pt(),jetIt->eta(), jetIt->phi(), jetIt->energy());
       myJets.push_back(j);
       jetPt->push_back(jetIt->pt());
     }
     //Let's make sure these jets are really pT sorted
     std::sort(myJets.begin(), myJets.end(),sortLorentzPt);
     std::sort(jetPt->begin(), jetPt->end(),bigger);
     //Protection against too many jets...
     int nGoodJets = myJets.size();
     if (nGoodJets > 15) nGoodJets = 15;

     //Making triplets (format allows ability to look at *all* jet triplets in code)
     if (nGoodJets > 5){
       for (int k = 2; k < nGoodJets; ++k){
	 for (int i = 0; i < k-1; ++i){
	   for (int j = i+1; j < k; ++j){
	     double sumPt = myJets[i].Pt() + myJets[j].Pt() + myJets[k].Pt();
	     double mass  = (myJets[i] + myJets[j] + myJets[k]).M();
	     tripMass->push_back(mass);
	     tripSumPt->push_back(sumPt);
	   }}}//end of 3jet loop
     }//end 6 jet requirement
   }
   iEvent.put(jetPt, "jetPt");
   iEvent.put(tripMass, "tripMass");
   iEvent.put(tripSumPt, "tripSumPt");
}

DEFINE_FWK_MODULE(ThreeJetVarProducer);
