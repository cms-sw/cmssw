#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DQM/DataScouting/interface/DiJetPairsVarProducer.h"

#include "TVector3.h"
#include <memory>
#include <algorithm>
#include <vector>

namespace{
  bool larger(double a, double b)
  { return (a > b); }
  bool sortmyLorentzPt(TLorentzVector a,TLorentzVector b)
  {return (a.Pt() > b.Pt()); }
}
//
// constructors and destructor
//

DiJetPairsVarProducer::DiJetPairsVarProducer(const edm::ParameterSet& iConfig) :
  inputJetTag_    (iConfig.getParameter<edm::InputTag>("inputJetTag")){
  
  produces<std::vector<double> >();

  LogDebug("") << "Inputs: "
	       << inputJetTag_.encode() << " ";
}

DiJetPairsVarProducer::~DiJetPairsVarProducer()
{
}

// ------------ method called to produce the data  ------------ 
void
DiJetPairsVarProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;


   // get hold of collection of objects
   edm::Handle<reco::CaloJetCollection> calojet_handle; 
   iEvent.getByLabel(inputJetTag_,calojet_handle);

   std::auto_ptr<std::vector<double> > jetPt(new std::vector<double>);
   std::auto_ptr<std::vector<double> > dijetMass(new std::vector<double>);
   std::auto_ptr<std::vector<double> > dijetSumPt(new std::vector<double>);
   std::auto_ptr<std::vector<double> > dijetdRjj(new std::vector<double>);
   jetPt->reserve(4);
   dijetMass->reserve(6);
   dijetSumPt->reserve(6);
   dijetdRjj->reserve(6);
   // check the the input collections are available
   if (calojet_handle.isValid()){
     std::vector<TLorentzVector> myJets;
     reco::CaloJetCollection::const_iterator jetIt;
     //Select jets
     for(jetIt = calojet_handle->begin(); jetIt != calojet_handle->end(); jetIt++){
       if (jetIt->pt()        < 30.0) continue;
       if (fabs(jetIt->eta()) >  2.5) continue;
       TLorentzVector j; j.SetPtEtaPhiE(jetIt->pt(),jetIt->eta(), jetIt->phi(), jetIt->energy());
       myJets.push_back(j);
       jetPt->push_back(jetIt->pt());
     }
     //Let's make sure these jets are really pT sorted
     std::sort(myJets.begin(), myJets.end(),sortmyLorentzPt);
     std::sort(jetPt->begin(), jetPt->end(),larger);
     //Protection against too many jets...
     int nGoodJets = myJets.size();
     //if (nGoodJets > 15) nGoodJets = 15;

     //Making triplets (format allows ability to look at *all* jet triplets in code)
     if (nGoodJets > 3){
       for (int i = 0; i < 3; ++i){
	 for (int j = i+1; j < 4; ++j){
	   std::cout << i << j << std::endl;
	   double sumPt = myJets[i].Pt() + myJets[j].Pt();
	   double mass  = (myJets[i] + myJets[j]).M();
	   double dR    = sqrt(pow(myJets[i].Eta()-myJets[j].Eta(),2) +
			       pow(myJets[i].Phi()-myJets[j].Phi(),2));
	   dijetMass->push_back(mass);
	   dijetSumPt->push_back(sumPt);
	   dijetdRjj->push_back(dR);
	 }}
     }//end 4 jet requirement
   }
   iEvent.put(jetPt, "jetPt");
   iEvent.put(dijetMass, "dijetMass");
   iEvent.put(dijetSumPt, "dijetSumPt");
   iEvent.put(dijetdRjj, "dijetdRjj");
}

DEFINE_FWK_MODULE(DiJetPairsVarProducer);
