#include <map>
#include <string>

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class PatZjetsJetAnalyzer : public edm::EDAnalyzer {

public:
  explicit PatZjetsJetAnalyzer(const edm::ParameterSet&);
  ~PatZjetsJetAnalyzer();
  
private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // simple map to contain all histograms; 
  // histograms are booked in the beginJob() 
  // method
  std::map<std::string,TH1F*> histContainer_; 

  // input tags  
  edm::InputTag src_;
};

#include "DataFormats/PatCandidates/interface/Jet.h"

PatZjetsJetAnalyzer::PatZjetsJetAnalyzer(const edm::ParameterSet& iConfig):
  histContainer_(),
  src_(iConfig.getUntrackedParameter<edm::InputTag>("src"))
{
}

PatZjetsJetAnalyzer::~PatZjetsJetAnalyzer()
{
}

void
PatZjetsJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get electron collection
  edm::Handle<edm::View<pat::Jet> > jets;
  iEvent.getByLabel(src_,jets);

  // loop jets
  for(edm::View<pat::Jet>::const_iterator ijet=jets->begin(); ijet!=jets->end(); ++ijet){
    // fill simple histograms
    pat::Jet jet = ijet->correctedJet("had", "uds");
    histContainer_["pt"  ]->Fill( jet.pt () );
    histContainer_["eta" ]->Fill( jet.eta() );
    histContainer_["phi" ]->Fill( jet.phi() );
    histContainer_["emf" ]->Fill( jet.emEnergyFraction() );
    for(unsigned int i=0; i<jet.getCaloConstituents().size(); ++i){
      histContainer_["dEta"]->Fill( jet.getCaloConstituent(i)->eta()-jet.eta() );
    }
  }
}

void 
PatZjetsJetAnalyzer::beginJob()
{
  // register to the TFileService
  edm::Service<TFileService> fs;
  
  // book histograms:
  histContainer_["pt"  ]=fs->make<TH1F>("pt"   , "pt"   ,  150,   0.,  150.);
  histContainer_["eta" ]=fs->make<TH1F>("eta"  , "eta"  ,   50,   0.,    5.);
  histContainer_["phi" ]=fs->make<TH1F>("phi"  , "phi"  ,   60, 3.14,  3.14);
  histContainer_["emf" ]=fs->make<TH1F>("emf"  , "emf"  ,   40,   0.,    1.);
  histContainer_["dEta"]=fs->make<TH1F>("dEta" , "dEta" ,   40,   0.,    1.);
}

void 
PatZjetsJetAnalyzer::endJob() 
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatZjetsJetAnalyzer);
