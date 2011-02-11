#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/JetMET/interface/HLTRFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TVector3.h"
#include "TLorentzVector.h"
//
// constructors and destructor
//
HLTRFilter::HLTRFilter(const edm::ParameterSet& iConfig) :
  inputTag_    (iConfig.getParameter<edm::InputTag>("inputTag")),
  inputMetTag_ (iConfig.getParameter<edm::InputTag>("inputMetTag")),
  min_R_       (iConfig.getParameter<double>       ("minR"   )),
  min_MR_      (iConfig.getParameter<double>       ("minMR"   )),
  DoRPrime_    (iConfig.getParameter<bool>       ("doRPrime"   )),
  accept_NJ_    (iConfig.getParameter<bool>       ("acceptNJ"   ))

{
   LogDebug("") << "Inputs/minR/minMR/doRPrime/acceptNJ : "
		<< inputTag_.encode() << " "
		<< inputMetTag_.encode() << " "
		<< min_R_ << " "
		<< min_MR_ << " "
		<< DoRPrime_ << " "
		<< accept_NJ_ << ".";

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTRFilter::~HLTRFilter()
{
}

void
HLTRFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltRHemisphere"));
  desc.add<edm::InputTag>("inputMetTag",edm::InputTag("hltMet"));
  desc.add<double>("minR",0.3);
  desc.add<double>("minMR",100.0);
  desc.add<bool>("doRPrime",false);
  desc.add<bool>("acceptNJ",true);
  descriptions.add("hltRFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool 
HLTRFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterobject (new TriggerFilterObjectWithRefs(path(),module()));

   // get hold of collection of objects
   Handle< vector<math::XYZTLorentzVector> > hemispheres;
   iEvent.getByLabel (inputTag_,hemispheres);

   // get hold of the MET Collection
   Handle<CaloMETCollection> inputMet;
   iEvent.getByLabel(inputMetTag_,inputMet);  

   iEvent.put(filterobject);

   // check the the input collections are available
   if (not hemispheres.isValid() or not inputMet.isValid())
     return false;

   if(hemispheres->size() ==0){  // the Hemisphere Maker will produce an empty collection of hemispheres iff the number of jets in the
     return accept_NJ_;   // event is greater than the maximum number of jets
   }


   //***********************************
   //Calculate R or R prime

   TLorentzVector j1(hemispheres->at(0).x(),hemispheres->at(0).y(),hemispheres->at(0).z(),hemispheres->at(0).t());
   TLorentzVector j2(hemispheres->at(1).x(),hemispheres->at(1).y(),hemispheres->at(1).z(),hemispheres->at(1).t());

  j1.SetPtEtaPhiM(j1.Pt(),j1.Eta(),j1.Phi(),0.0);
  j2.SetPtEtaPhiM(j2.Pt(),j2.Eta(),j2.Phi(),0.0);
  
  if(j2.Pt() > j1.Pt()){
    TLorentzVector temp = j1;
    j1 = j2;
    j2 = temp;
  }
  
  double num = j1.P()-j2.P();
  double den = j1.Pz()-j2.Pz();
  if(fabs(num)==fabs(den)) return false; //ignore if beta=1
  if(fabs(num)<fabs(den) && DoRPrime_) return false; //num<den ==> R event
  if(fabs(num)>fabs(den) && !DoRPrime_) return false; // num>den ==> R' event

 //now we can calculate MTR
  TVector3 met;
  met.SetPtEtaPhi((inputMet->front()).pt(),0.0,(inputMet->front()).phi());
  double MTR = sqrt(0.5*(met.Mag()*(j1.Pt()+j2.Pt()) - met.Dot(j1.Vect()+j2.Vect())));

 //calculate MR or MRP
  double MR=0;
  if(!DoRPrime_){    //CALCULATE MR
    double temp = (j1.P()*j2.Pz()-j2.P()*j1.Pz())*(j1.P()*j2.Pz()-j2.P()*j1.Pz());
    temp /= (j1.Pz()-j2.Pz())*(j1.Pz()-j2.Pz())-(j1.P()-j2.P())*(j1.P()-j2.P());    
    MR = 2.*sqrt(temp);
  }else{      //CALCULATE MRP   
    double jaP = j1.Pt()*j1.Pt() +j1.Pz()*j2.Pz()-j1.P()*j2.P();
    double jbP = j2.Pt()*j2.Pt() +j1.Pz()*j2.Pz()-j1.P()*j2.P();
    jbP *= -1.;
    double den = sqrt((j1.P()-j2.P())*(j1.P()-j2.P())-(j1.Pz()-j2.Pz())*(j1.Pz()-j2.Pz()));
    
    jaP /= den;
    jbP /= den;
    
    double temp = jaP*met.Dot(j2.Vect())/met.Mag() + jbP*met.Dot(j1.Vect())/met.Mag();
    temp = temp*temp;
    
    den = (met.Dot(j1.Vect()+j2.Vect())/met.Mag())*(met.Dot(j1.Vect()+j2.Vect())/met.Mag())-(jaP-jbP)*(jaP-jbP);
    
    if(den <= 0.0) return false;

    temp /= den;
    temp = 2.*sqrt(temp);
    
    double bR = (jaP-jbP)/(met.Dot(j1.Vect()+j2.Vect())/met.Mag());
    double gR = 1./sqrt(1.-bR*bR);
    
    temp *= gR;

    MR = temp;
  }

  // filter decision
  bool accept(MR>=min_MR_ && float(MTR)/float(MR)>=min_R_);


  return accept;
}

DEFINE_FWK_MODULE(HLTRFilter);

