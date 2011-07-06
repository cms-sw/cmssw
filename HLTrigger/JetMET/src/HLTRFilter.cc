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

   if(hemispheres->size() ==0){  // the Hemisphere Maker will produce an empty collection of hemispheres if the number of jets in the
     return accept_NJ_;   // event is greater than the maximum number of jets
   }

   //***********************************
   //Calculate R or R prime

   TLorentzVector j1R(hemispheres->at(0).x(),hemispheres->at(0).y(),hemispheres->at(0).z(),hemispheres->at(0).t());
   TLorentzVector j2R(hemispheres->at(1).x(),hemispheres->at(1).y(),hemispheres->at(1).z(),hemispheres->at(1).t());
   TLorentzVector j1Rp(hemispheres->at(2).x(),hemispheres->at(2).y(),hemispheres->at(2).z(),hemispheres->at(2).t());
   TLorentzVector j2Rp(hemispheres->at(3).x(),hemispheres->at(3).y(),hemispheres->at(3).z(),hemispheres->at(3).t());

   if(j1R.Pt() > 0.1) { // some good R combination was found

     j1R.SetPtEtaPhiM(j1R.Pt(),j1R.Eta(),j1R.Phi(),0.0);
     j2R.SetPtEtaPhiM(j2R.Pt(),j2R.Eta(),j2R.Phi(),0.0);
  
     if(j2R.Pt() > j1R.Pt()){
       TLorentzVector temp = j1R;
       j1R = j2R;
       j2R = temp;
     }
  
     double num = j1R.P()-j2R.P();
     double den = j1R.Pz()-j2R.Pz();
     if(fabs(num)<fabs(den)) {

       //now we can calculate MTR
       TVector3 met;
       met.SetPtEtaPhi((inputMet->front()).pt(),0.0,(inputMet->front()).phi());
       double MTR = sqrt(0.5*(met.Mag()*(j1R.Pt()+j2R.Pt()) - met.Dot(j1R.Vect()+j2R.Vect())));

       //calculate MR
       double MR=0;
       double temp = (j1R.P()*j2R.Pz()-j2R.P()*j1R.Pz())*(j1R.P()*j2R.Pz()-j2R.P()*j1R.Pz());
       temp /= (j1R.Pz()-j2R.Pz())*(j1R.Pz()-j2R.Pz())-(j1R.P()-j2R.P())*(j1R.P()-j2R.P());    
       MR = 2.*sqrt(temp);
       if(MR>=min_MR_ && float(MTR)/float(MR)>=min_R_) return true;
     }
   }

   if(j1Rp.Pt() > 0.1) { // some good R' combination was found  
     
     j1Rp.SetPtEtaPhiM(j1Rp.Pt(),j1Rp.Eta(),j1Rp.Phi(),0.0);
     j2Rp.SetPtEtaPhiM(j2Rp.Pt(),j2Rp.Eta(),j2Rp.Phi(),0.0);

     if(j2Rp.Pt() > j1Rp.Pt()){
       TLorentzVector temp = j1Rp;
       j1Rp = j2Rp;
       j2Rp = temp;
     }

     double num = j1Rp.P()-j2Rp.P();
     double den = j1Rp.Pz()-j2Rp.Pz();
     if(fabs(num)>fabs(den)) {
       //now we can calculate MTR

       TVector3 met;
       met.SetPtEtaPhi((inputMet->front()).pt(),0.0,(inputMet->front()).phi());
       double MTR = sqrt(0.5*(met.Mag()*(j1Rp.Pt()+j2Rp.Pt()) - met.Dot(j1Rp.Vect()+j2Rp.Vect())));

       double jaP = j1Rp.Pt()*j1Rp.Pt() +j1Rp.Pz()*j2Rp.Pz()-j1Rp.P()*j2Rp.P();
       double jbP = j2Rp.Pt()*j2Rp.Pt() +j1Rp.Pz()*j2Rp.Pz()-j1Rp.P()*j2Rp.P();
       jbP *= -1.;
       double den = sqrt((j1Rp.P()-j2Rp.P())*(j1Rp.P()-j2Rp.P())-(j1Rp.Pz()-j2Rp.Pz())*(j1Rp.Pz()-j2Rp.Pz()));
       
       jaP /= den;
       jbP /= den;
    
       double temp = jaP*met.Dot(j2Rp.Vect())/met.Mag() + jbP*met.Dot(j1Rp.Vect())/met.Mag();
       temp = temp*temp;
    
       den = (met.Dot(j1Rp.Vect()+j2Rp.Vect())/met.Mag())*(met.Dot(j1Rp.Vect()+j2Rp.Vect())/met.Mag())-(jaP-jbP)*(jaP-jbP);
    
       if(den <= 0.0) return false;
       
       temp /= den;
       temp = 2.*sqrt(temp);
       
       double bR = (jaP-jbP)/(met.Dot(j1Rp.Vect()+j2Rp.Vect())/met.Mag());
       double gR = 1./sqrt(1.-bR*bR);
    
       temp *= gR;

       double MRp = temp;

       if(MRp>=min_MR_ && float(MTR)/float(MRp)>=min_R_) return true;
     }
   }

   // filter decision
   return false;
}

DEFINE_FWK_MODULE(HLTRFilter);

