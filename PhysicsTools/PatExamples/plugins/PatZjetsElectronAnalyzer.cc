#include <map>
#include <string>

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class PatZjetsElectronAnalyzer : public edm::EDAnalyzer {

public:
  explicit PatZjetsElectronAnalyzer(const edm::ParameterSet&);
  ~PatZjetsElectronAnalyzer();
  
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

#include "DataFormats/PatCandidates/interface/Electron.h"

PatZjetsElectronAnalyzer::PatZjetsElectronAnalyzer(const edm::ParameterSet& iConfig):
  histContainer_(),
  src_(iConfig.getUntrackedParameter<edm::InputTag>("src"))
{
}

PatZjetsElectronAnalyzer::~PatZjetsElectronAnalyzer()
{
}

void
PatZjetsElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get electron collection
  edm::Handle<edm::View<pat::Electron> > elecs;
  iEvent.getByLabel(src_,elecs);

  // loop electrons
  for(edm::View<pat::Electron>::const_iterator elec=elecs->begin(); elec!=elecs->end(); ++elec){
    // fill simple histograms
    histContainer_["pt"  ]->Fill( elec->pt () );
    histContainer_["eta" ]->Fill( elec->eta() );
    histContainer_["phi" ]->Fill( elec->phi() );
    histContainer_["iso" ]->Fill((elec->trackIso()+elec->caloIso())/elec->pt() );
    histContainer_["eop" ]->Fill( elec->eSeedClusterOverP() );
    histContainer_["clus"]->Fill( elec->e1x5()/elec->e5x5() );
    // fill enegry flow histogram for isolation
    for(int bin=1; bin<=histContainer_["dr"]->GetNbinsX(); ++bin){
      double lowerEdge = histContainer_["dr"]->GetBinLowEdge(bin);
      double upperEdge = histContainer_["dr"]->GetBinLowEdge(bin)+histContainer_["dr"]->GetBinWidth(bin);
      histContainer_["dr"]->Fill(histContainer_["dr"]->GetBinCenter(bin), elec->trackIsoDeposit()->depositWithin(upperEdge) - elec->trackIsoDeposit()->depositWithin(lowerEdge));
    }
    // fill electron id histograms
    if( elec->electronID("eidRobustLoose") > 0.5 )
      histContainer_["eIDs" ]->Fill(0);
    if( elec->electronID("eidRobustTight") > 0.5 )
      histContainer_["eIDs" ]->Fill(1);
    if( elec->electronID("eidLoose"      ) > 0.5 )
      histContainer_["eIDs" ]->Fill(2);
    if( elec->electronID("eidTight"      ) > 0.5 )
      histContainer_["eIDs" ]->Fill(3);
    if( elec->electronID("eidRobustHighEnergy") > 0.5 )
      histContainer_["eIDs" ]->Fill(4);
  }
}

void 
PatZjetsElectronAnalyzer::beginJob()
{
  // register to the TFileService
  edm::Service<TFileService> fs;
  
  // book histograms:
  histContainer_["pt"  ]=fs->make<TH1F>("pt"   , "pt"   ,  150,   0.,  150.);
  histContainer_["eta" ]=fs->make<TH1F>("eta"  , "eta"  ,   50,   0.,    5.);
  histContainer_["phi" ]=fs->make<TH1F>("phi"  , "phi"  ,   60, 3.14,  3.14);
  histContainer_["iso" ]=fs->make<TH1F>("iso"  , "iso"  ,   30,   0.,   10.);
  histContainer_["dr"  ]=fs->make<TH1F>("dr"   , "dr"   ,   40,   0.,    1.);
  histContainer_["eop" ]=fs->make<TH1F>("eop"  , "eop"  ,   40,   0.,    1.);
  histContainer_["clus"]=fs->make<TH1F>("clus" , "clus" ,   40,   0.,    1.);
  histContainer_["eIDs"]=fs->make<TH1F>("eIDs" , "eIDS" ,    5,   0.,    5.);
}

void 
PatZjetsElectronAnalyzer::endJob() 
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatZjetsElectronAnalyzer);
