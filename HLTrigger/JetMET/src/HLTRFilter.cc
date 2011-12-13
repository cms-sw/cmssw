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
  accept_NJ_   (iConfig.getParameter<bool>       ("acceptNJ"   )),
  R_offset_    ((iConfig.existsAs<double>("R2Offset") ? iConfig.getParameter<double>("R2Offset"):0)),
  MR_offset_   ((iConfig.existsAs<double>("MROffset") ? iConfig.getParameter<double>("MROffset"):0)),
  R_MR_cut_    ((iConfig.existsAs<double>("RMRCut") ? iConfig.getParameter<double>("RMRCut"):-999999.))
  

{
   LogDebug("") << "Inputs/minR/minMR/doRPrime/acceptNJ/R2Offset/MROffset/RMRCut : "
		<< inputTag_.encode() << " "
		<< inputMetTag_.encode() << " "
		<< min_R_ << " "
		<< min_MR_ << " "
		<< accept_NJ_ 
		<< R_offset_ << " "
		<< MR_offset_ << " "
		<< R_MR_cut_
		<< ".";

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
  desc.add<double>("R2Offset",0.0);
  desc.add<double>("MROffset",0.0);
  desc.add<double>("RMRCut",-999999.0);
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
   //Calculate R 

   TLorentzVector ja(hemispheres->at(0).x(),hemispheres->at(0).y(),hemispheres->at(0).z(),hemispheres->at(0).t());
   TLorentzVector jb(hemispheres->at(1).x(),hemispheres->at(1).y(),hemispheres->at(1).z(),hemispheres->at(1).t());

   if(ja.Pt() > 0.1) { // some good R combination was found
     
     ja.SetPtEtaPhiM(ja.Pt(),ja.Eta(),ja.Phi(),0.0);
     jb.SetPtEtaPhiM(jb.Pt(),jb.Eta(),jb.Phi(),0.0);
     
     if(ja.Pt() > jb.Pt()){
       TLorentzVector temp = ja;
       ja = jb;
       jb = temp;
     }
     
     double A = ja.P();
     double B = jb.P();
     double az = ja.Pz();
     double bz = jb.Pz();
     TVector3 jaT, jbT;
     jaT.SetXYZ(ja.Px(),ja.Py(),0.0);
     jbT.SetXYZ(jb.Px(),jb.Py(),0.0);
     double ATBT = (jaT+jbT).Mag2();

     double MR = sqrt((A+B)*(A+B)-(az+bz)*(az+bz)-
			(jbT.Dot(jbT)-jaT.Dot(jaT))*(jbT.Dot(jbT)-jaT.Dot(jaT))/(jaT+jbT).Mag2());

     double mybeta = (jbT.Dot(jbT)-jaT.Dot(jaT))/
       sqrt(ATBT*((A+B)*(A+B)-(az+bz)*(az+bz)));

     double mygamma = 1./sqrt(1.-mybeta*mybeta);

     //use gamma times MRstar
     MR *= mygamma;

     //now we can calculate MTR
     TVector3 met;
     met.SetPtEtaPhi((inputMet->front()).pt(),0.0,(inputMet->front()).phi());
     double MTR = sqrt(0.5*(met.Mag()*(ja.Pt()+jb.Pt()) - met.Dot(ja.Vect()+jb.Vect())));
     
     //filter events
     float R = float(MTR)/float(MR);
     if(MR>=min_MR_ && R>=min_R_
	&& ( (R*R - R_offset_)*(MR-MR_offset_) )>=R_MR_cut_) return true;
     
   }

   // filter decision
   return false;
}

DEFINE_FWK_MODULE(HLTRFilter);

