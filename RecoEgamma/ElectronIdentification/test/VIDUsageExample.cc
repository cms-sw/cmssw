/* This module shows how to access the electron ID results using the   
** VID Framework  
** It works for both AOD and miniAOD 
**
** Author: Sam Harper borrowing liberaly from Ilya Kravchenko and 
**         Lindsey Gray's examples
**
*/


#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"



class VIDUsageExample : public edm::stream::EDAnalyzer<> {

private:
  //we want this to work on mini-aod and aod, this means we dont know if we are getting
  //GsfElectrons or PAT electrons
  //so we try both (this could be a bit more sophosticated but lets keep things simple)
  edm::EDGetTokenT<reco::GsfElectronCollection> gsfEleToken_;
  edm::EDGetTokenT<edm::View<pat::Electron> > patEleToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > idDecisionMapToken_; //a bool true=passed ID, false = failed ID
  edm::EDGetTokenT<edm::ValueMap<unsigned> > firstIdCutFailedMapToken_; //the number of the first cut failed in the order they are defined in the PSet starting at zero (ie if you et,dEtaIn,dPhiIn,hadem cuts defined and it passed et,dEtaIn but failed dPhiIn, this number would be 2, in the case of no cuts failed it is #cuts
  edm::EDGetTokenT<std::string> idMD5NameToken_; //the md5sum of the ID you are using (E/gamma might ask you for this to verify you are running the right ID) 
  
  size_t nrPassID_;
  size_t nrFailID_;

public:
  explicit VIDUsageExample(const edm::ParameterSet& para);
  ~VIDUsageExample(){}
  
  virtual void analyze(const edm::Event& event,const edm::EventSetup& setup) override;
 
  void endStream() override {std::cout <<"nrPass "<<nrPassID_<<" nrFail "<<nrFailID_<<" (note this is all \"electrons\" so is not the ID efficiency)"<<std::endl;}
 
};

VIDUsageExample::VIDUsageExample(const edm::ParameterSet& para):
  gsfEleToken_(consumes<reco::GsfElectronCollection>(para.getParameter<edm::InputTag>("eles"))),
  patEleToken_(consumes<edm::View<pat::Electron> >(para.getParameter<edm::InputTag>("eles"))),
   
  idDecisionMapToken_(consumes<edm::ValueMap<bool> >(para.getParameter<edm::InputTag>("id"))),
  firstIdCutFailedMapToken_(consumes<edm::ValueMap<unsigned> >(para.getParameter<edm::InputTag>("id"))),
  idMD5NameToken_(consumes<std::string>(para.getParameter<edm::InputTag>("id"))),
  nrPassID_(0),
  nrFailID_(0)

{
  
}


void VIDUsageExample::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 
  edm::Handle<edm::View<pat::Electron> > patEles;
  iEvent.getByToken(patEleToken_,patEles);
  
  edm::Handle<reco::GsfElectronCollection> gsfEles;
  iEvent.getByToken(gsfEleToken_,gsfEles);

  edm::Handle<edm::ValueMap<bool> > idDecisionMap;
  iEvent.getByToken(idDecisionMapToken_,idDecisionMap);

  edm::Handle<edm::ValueMap<unsigned> > firstIdCutFailedMap;
  iEvent.getByToken(firstIdCutFailedMapToken_,firstIdCutFailedMap);

  edm::Handle<std::string> idMD5Name;
  iEvent.getByToken(idMD5NameToken_,idMD5Name);

  std::cout <<"md5sum of the ID "<<*idMD5Name<<std::endl;

  //note the copy/paste is only to make things clearer, its not a great way of solving this issue

  if(patEles.isValid()){ //we have pat electrons availible use them
    for(auto ele=patEles->begin();ele!=patEles->end();++ele){
      const edm::Ptr<pat::Electron> elePtr(patEles,ele-patEles->begin()); //value map is keyed of edm::Ptrs so we need to make one
      bool passID = (*idDecisionMap)[elePtr]; //a bool, true if it passed the ID, false if it didnt
      if(passID) {
	std::cout <<"pat ele passed ID"<<std::endl;
	nrPassID_++;
      }else{
	int firstCutFailedNr = (*firstIdCutFailedMap)[elePtr];
	std::cout <<"pat ele failed ID at cut #"<<firstCutFailedNr<<std::endl;
	nrFailID_++;
      }
    }
  }else if(gsfEles.isValid()){ //no pat electrons availible, fall back to GsfElectrons
    for(auto ele=gsfEles->begin();ele!=gsfEles->end();++ele){
      const edm::Ptr<reco::GsfElectron> elePtr(gsfEles,ele-gsfEles->begin()); //value map is keyed of edm::Ptrs so we need to make one
      bool passID = (*idDecisionMap)[elePtr]; //a bool, true if it passed the ID, false if it didnt
      if(passID){
	std::cout <<"gsf ele passed ID"<<std::endl;
	nrPassID_++;
      }else{
	int firstCutFailedNr = (*firstIdCutFailedMap)[elePtr];
	std::cout <<"gsf ele failed ID at cut #"<<firstCutFailedNr<<std::endl;
	nrFailID_++;
      }
    }
  }else{
    std::cout <<"no Gsf or PAT electrons found"<<std::endl;
  }

}

 



DEFINE_FWK_MODULE(VIDUsageExample);
