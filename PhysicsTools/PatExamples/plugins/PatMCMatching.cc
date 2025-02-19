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


class PatMCMatching : public edm::EDAnalyzer {

public:
  /// default constructor
  explicit PatMCMatching(const edm::ParameterSet&);
  /// default destructor
  ~PatMCMatching();
  
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
};


#include "DataFormats/PatCandidates/interface/Muon.h"


PatMCMatching::PatMCMatching(const edm::ParameterSet& iConfig):
  histContainer_(),
  muonSrc_(iConfig.getUntrackedParameter<edm::InputTag>("muonSrc"))
{
}

PatMCMatching::~PatMCMatching()
{
}

void
PatMCMatching::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get muon collection
  edm::Handle<edm::View<pat::Muon> > muons;
  iEvent.getByLabel(muonSrc_,muons);

  for(edm::View<pat::Muon>::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon){
  
      for(unsigned int i = 0 ; i < muon->genParticleRefs().size() ; ++i ){

	  switch( muon->genParticle(i)->status() ){	
	      case 1:
		  histContainer_["DR_status1Match"]->Fill( ROOT::Math::VectorUtil::DeltaR(muon->p4() , muon->genParticle(i)->p4() ) ); 
		  break;
	      case 3:
		  histContainer_["DR_status3Match"]->Fill( ROOT::Math::VectorUtil::DeltaR(muon->p4() , muon->genParticle(i)->p4() ) ); 
		  break;
	      default:
		  histContainer_["DR_defaultMatch"]->Fill( ROOT::Math::VectorUtil::DeltaR(muon->p4() , muon->genParticle(i)->p4() ) ); 
		  break;
	  }
      }
  }

}

void 
PatMCMatching::beginJob()
{
  // register to the TFileService
  edm::Service<TFileService> fs;
  
  // book histograms:
  histContainer_["DR_defaultMatch"  ]=fs->make<TH1F>("DR_defaultMatch",   "DR_defaultMatch", 100,  0.,  0.02);
  histContainer_["DR_status1Match"  ]=fs->make<TH1F>("DR_status1Match",   "DR_status1Match", 100,  0.,  0.02);
  histContainer_["DR_status3Match"  ]=fs->make<TH1F>("DR_status3Match",   "DR_status3Match", 100,  0.,  0.02);

}

void 
PatMCMatching::endJob() 
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatMCMatching);
