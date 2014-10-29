#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HLTrigger/JetMET/interface/HLTRFilter.h"
#include "DataFormats/Common/interface/Ref.h"

//
// constructors and destructor
//
HLTRFilter::HLTRFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputTag_    (iConfig.getParameter<edm::InputTag>("inputTag")),
  inputMetTag_ (iConfig.getParameter<edm::InputTag>("inputMetTag")),
  doMuonCorrection_(iConfig.getParameter<bool>         ("doMuonCorrection" )),
  min_R_       (iConfig.getParameter<double>       ("minR"   )),
  min_MR_      (iConfig.getParameter<double>       ("minMR"   )),
  DoRPrime_    (iConfig.getParameter<bool>       ("doRPrime"   )),
  accept_NJ_   (iConfig.getParameter<bool>       ("acceptNJ"   )),
  R_offset_    ((iConfig.existsAs<double>("R2Offset") ? iConfig.getParameter<double>("R2Offset"):0)),
  MR_offset_   ((iConfig.existsAs<double>("MROffset") ? iConfig.getParameter<double>("MROffset"):0)),
  R_MR_cut_    ((iConfig.existsAs<double>("RMRCut") ? iConfig.getParameter<double>("RMRCut"):-999999.))
  
{
   m_theInputToken = consumes<std::vector<math::XYZTLorentzVector>>(inputTag_);
   m_theMETToken = consumes<edm::View<reco::MET> >(inputMetTag_);
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

  //put a dummy METCollection into the event, holding values for MR and Rsq
  produces<reco::METCollection>();

}

HLTRFilter::~HLTRFilter()
{
}

void
HLTRFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltRHemisphere"));
  desc.add<edm::InputTag>("inputMetTag",edm::InputTag("hltMet"));
  desc.add<bool>("doMuonCorrection",false);
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
HLTRFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   // get hold of collection of objects
   Handle< vector<math::XYZTLorentzVector> > hemispheres;
   iEvent.getByToken (m_theInputToken,hemispheres);

   // get hold of the MET Collection
   edm::Handle<edm::View<reco::MET> > inputMet;
   iEvent.getByToken(m_theMETToken,inputMet);

   // check the the input collections are available
   if (not hemispheres.isValid() or not inputMet.isValid())
     return false;

   if(hemispheres->size() ==0){  // the Hemisphere Maker will produce an empty collection of hemispheres if the number of jets in the
     return accept_NJ_;   // event is greater than the maximum number of jets
   }

   //***********************************
   //Calculate R 

   int nMuons;
   switch(hemispheres->size()){
   case 2:
     nMuons=0; break;
   case 5:
     nMuons=1; break;
   case 10:
     nMuons=2; break;
   default:
     return false; //invalid hemisphere collection
   }

   //muons as MET
   TLorentzVector ja(hemispheres->at(0).x(),hemispheres->at(0).y(),hemispheres->at(0).z(),hemispheres->at(0).t());
   TLorentzVector jb(hemispheres->at(1).x(),hemispheres->at(1).y(),hemispheres->at(1).z(),hemispheres->at(1).t());

   std::vector<math::XYZTLorentzVector> muonVec;
   
   double MR = CalcMR(ja,jb);
   double R  = CalcR(MR,ja,jb,inputMet,muonVec);
   
   if(MR>=min_MR_ && R>=min_R_
      && ( (R*R - R_offset_)*(MR-MR_offset_) )>=R_MR_cut_) {
     addObjects(iEvent, filterproduct, MR, R*R);
     return true;
   }
   if(nMuons==0) {
     addObjects(iEvent, filterproduct, MR, R*R); 
     return false;  // if no muons and we get here, reject event
   }
   //Lead Muon as Jet
   ja.SetXYZT(hemispheres->at(3).x(),hemispheres->at(3).y(),hemispheres->at(3).z(),hemispheres->at(3).t());
   jb.SetXYZT(hemispheres->at(4).x(),hemispheres->at(4).y(),hemispheres->at(4).z(),hemispheres->at(4).t());
   muonVec.push_back(hemispheres->at(2)); // lead muon at position 2

   MR = CalcMR(ja,jb);
   R  = CalcR(MR,ja,jb,inputMet,muonVec);
   if(MR>=min_MR_ && R>=min_R_
      && ( (R*R - R_offset_)*(MR-MR_offset_) )>=R_MR_cut_){
       addObjects(iEvent, filterproduct, MR, R*R); 
       return true;
   }
   
   if(nMuons==1){
       addObjects(iEvent, filterproduct, MR, R*R); 
       return false;  // if no muons and we get here, reject event
   }

   muonVec.pop_back();
   //Second Muon as Jet
   ja.SetXYZT(hemispheres->at(6).x(),hemispheres->at(6).y(),hemispheres->at(6).z(),hemispheres->at(6).t());
   jb.SetXYZT(hemispheres->at(7).x(),hemispheres->at(7).y(),hemispheres->at(7).z(),hemispheres->at(7).t());
   muonVec.push_back(hemispheres->at(5)); // sublead muon at position 5

   MR = CalcMR(ja,jb);
   R  = CalcR(MR,ja,jb,inputMet,muonVec);
   if(MR>=min_MR_ && R>=min_R_
      && ( (R*R - R_offset_)*(MR-MR_offset_) )>=R_MR_cut_){
       addObjects(iEvent, filterproduct, MR, R*R); 
       return true;
   }

   ja.SetXYZT(hemispheres->at(8).x(),hemispheres->at(8).y(),hemispheres->at(8).z(),hemispheres->at(8).t());
   jb.SetXYZT(hemispheres->at(9).x(),hemispheres->at(9).y(),hemispheres->at(9).z(),hemispheres->at(9).t());
   muonVec.push_back(hemispheres->at(2)); // lead muon at position 2
   
   MR = CalcMR(ja,jb);
   R  = CalcR(MR,ja,jb,inputMet,muonVec);
   if(MR>=min_MR_ && R>=min_R_
      && ( (R*R - R_offset_)*(MR-MR_offset_) )>=R_MR_cut_){
       addObjects(iEvent, filterproduct, MR, R*R); 
       return true;   
   }

   // filter decision   
   addObjects(iEvent, filterproduct, MR, R*R); 
   return false;
}

void HLTRFilter::addObjects(edm::Event& iEvent, trigger::TriggerFilterObjectWithRefs & filterproduct, double MR, double Rsq) const{
    
    //create METCollection for storing MR and Rsq results
    std::auto_ptr<reco::METCollection> razorObject(new reco::METCollection());
    
    reco::MET::LorentzVector mrRsqP4(MR,Rsq,0,0);
    reco::MET::Point vtx(0,0,0);

    reco::MET mrRsq(mrRsqP4, vtx);
    razorObject->push_back(mrRsq);

    edm::RefProd<reco::METCollection > ref_before_put = iEvent.getRefBeforePut<reco::METCollection >();

    //put the METCollection into the event (necessary because of how addCollectionTag works...)
    iEvent.put(razorObject);
    edm::Ref<reco::METCollection> mrRsqRef(ref_before_put, 0);

    //add it to the trigger object collection
    if (saveTags()) filterproduct.addCollectionTag(edm::InputTag( *moduleLabel()));
    filterproduct.addObject(trigger::TriggerMET, mrRsqRef); //give it the ID of a MET object 
}

double 
HLTRFilter::CalcMR(TLorentzVector ja, TLorentzVector jb){
  if(ja.Pt()<=0.1) return -1;

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
  return MR*mygamma;  
}

double 
  HLTRFilter::CalcR(double MR, TLorentzVector ja, TLorentzVector jb, edm::Handle<edm::View<reco::MET> > inputMet, const std::vector<math::XYZTLorentzVector>& muons){
  //now we can calculate MTR
  TVector3 met;
  met.SetPtEtaPhi((inputMet->front()).pt(),0.0,(inputMet->front()).phi());
  
  std::vector<math::XYZTLorentzVector>::const_iterator muonIt;
  for(muonIt = muons.begin(); muonIt!=muons.end(); muonIt++){
    TVector3 tmp;
    tmp.SetPtEtaPhi(muonIt->pt(),0,muonIt->phi());
    met-=tmp;
  }

  double MTR = sqrt(0.5*(met.Mag()*(ja.Pt()+jb.Pt()) - met.Dot(ja.Vect()+jb.Vect())));
  
  //filter events
  return float(MTR)/float(MR); //R
  
}
DEFINE_FWK_MODULE(HLTRFilter);

