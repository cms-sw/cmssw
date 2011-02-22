#include <map>
#include <string>

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Math/VectorUtil.h"


class PatMCMatchingExtended : public edm::EDAnalyzer {

public:
  /// default constructor
  explicit PatMCMatchingExtended(const edm::ParameterSet&);
  /// default destructor
  ~PatMCMatchingExtended();
  
private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // simple map to contain all histograms; 
  // histograms are booked in the beginJob() 
  // method
  std::map<std::string,TH1F*> histContainer_; 

  // input tags  
  edm::InputTag muonSrc_;

  //counts how often a genParticle with different charge gives a match
  unsigned int diffCharge;

  //how many muons have no match
  unsigned int noMatch;

  //how many muons have no status 1 or 3 match, but decay in flight
  unsigned int decayInFlight;

  //count the number of muons in all events
  unsigned int numberMuons;

};


#include "DataFormats/PatCandidates/interface/Muon.h"


PatMCMatchingExtended::PatMCMatchingExtended(const edm::ParameterSet& iConfig):
  histContainer_(),
  muonSrc_(iConfig.getUntrackedParameter<edm::InputTag>("muonSrc"))
{
}

PatMCMatchingExtended::~PatMCMatchingExtended()
{
}

void
PatMCMatchingExtended::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get muon collection
  edm::Handle<edm::View<pat::Muon> > muons;
  iEvent.getByLabel(muonSrc_,muons);

  for(edm::View<pat::Muon>::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon){
    if(muon->genParticleById(0,1).isNonnull() ){
      histContainer_["DR_status1Match"]->Fill( ROOT::Math::VectorUtil::DeltaR(muon->p4() , (muon->genParticleById(0,1) )->p4() ) ); 
      histContainer_["DPt_status1Match"]->Fill(muon->pt() - (muon->genParticleById(0,1) )->pt() );
    }
    if(muon->genParticleById(0,3).isNonnull() ){
      histContainer_["DR_status3Match"]->Fill( ROOT::Math::VectorUtil::DeltaR(muon->p4() , (muon->genParticleById(0,3) )->p4() ) );
      histContainer_["DPt_status3Match"]->Fill(muon->pt() - (muon->genParticleById(0,3) )->pt() );
    }
    if(muon->genParticleById(0,-1).isNonnull() ){
      histContainer_["DR_defaultMatch"]->Fill( ROOT::Math::VectorUtil::DeltaR(muon->p4() , (muon->genParticleById(0,-1) )->p4() ) );
      histContainer_["DPt_defaultMatch"]->Fill(muon->pt() - (muon->genParticleById(0,-1) )->pt() );
    }
    if(muon->genParticleById(0,1).isNull() && muon->genParticleById(0,3).isNull() && muon->genParticleById(0,-1).isNull()) noMatch++;
    if(muon->genParticleById(0,1).isNull() && muon->genParticleById(0,3).isNull() && muon->genParticleById(0,-1).isNonnull())decayInFlight++;
    
    
    
    if( muon->genParticleById(-13,0, 1).isNonnull() ){
      diffCharge++;
      std::cout<<" DIFF CHARGE!!! charge gen: "<< muon->genParticleById(-13,0, true)->charge()<< " charge reco: "<< muon->charge()<<std::endl;
    }
    numberMuons++;  
  }
  
}

void 
PatMCMatchingExtended::beginJob()
{
  // register to the TFileService
  edm::Service<TFileService> fs;
  
  // book histograms:
  //DR
  histContainer_["DR_defaultMatch"  ]=fs->make<TH1F>("DR_defaultMatch",   "DR_defaultMatch",     100, 0,  0.02);
  histContainer_["DR_status1Match"  ]=fs->make<TH1F>("DR_status1Match",   "DR_status1Match",     100, 0,  0.02);
  histContainer_["DR_status3Match"  ]=fs->make<TH1F>("DR_status3Match",   "DR_status3Match",     100, 0,  0.02);
  //DPT
  histContainer_["DPt_defaultMatch"  ]=fs->make<TH1F>("DPt_defaultMatch",   "DPt_defaultMatch",     10, 0,  1.2);
  histContainer_["DPt_status1Match"  ]=fs->make<TH1F>("DPt_status1Match",   "DPt_status1Match",     10, 0,  1.2);
  histContainer_["DPt_status3Match"  ]=fs->make<TH1F>("DPt_status3Match",   "DPt_status3Match",     10, 0,  1.2);
  //some counters
  diffCharge=0;
  noMatch=0;
  decayInFlight=0;
  numberMuons=0;

}

void 
PatMCMatchingExtended::endJob() 
{
    std::cout<<"diffcharge: "<< diffCharge <<std::endl;
    std::cout<<"noMatch: "<<  noMatch <<std::endl;
    std::cout<<"decayInFlight: "<< decayInFlight <<std::endl;
    std::cout<<"numberMuons: "<< numberMuons <<std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatMCMatchingExtended);


