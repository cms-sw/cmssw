// Authors: F. Ambroglini, L. Fano', F. Bechtel
#include <QCDAnalysis/UEAnalysis/interface/AnalysisRootpleProducerOnlyMC.h>
 
using namespace edm;
using namespace std;
using namespace reco;

class GreaterPt
{
public:
  bool operator()( const math::XYZTLorentzVector& a, const math::XYZTLorentzVector& b) 
  {
    return a.pt() > b.pt();
  }
};

class GenJetSort
{
public:
  bool operator()(const GenJet& a, const GenJet& b) 
  {
    return a.pt() > b.pt();
  }
};


void AnalysisRootpleProducerOnlyMC::store()
{
  AnalysisTree->Fill();

}

void AnalysisRootpleProducerOnlyMC::fillEventInfo(int e)
{
  EventKind = e;
}


AnalysisRootpleProducerOnlyMC::AnalysisRootpleProducerOnlyMC( const ParameterSet& pset )
{
  mcEvent = pset.getUntrackedParameter<InputTag>("MCEvent",std::string(""));
  genJetCollName = pset.getUntrackedParameter<InputTag>("GenJetCollectionName",std::string(""));
  chgJetCollName = pset.getUntrackedParameter<InputTag>("ChgGenJetCollectionName",std::string(""));
  chgGenPartCollName = pset.getUntrackedParameter<InputTag>("ChgGenPartCollectionName",std::string(""));
  gammaGenPartCollName = pset.getUntrackedParameter<InputTag>("GammaGenPartCollectionName",std::string(""));

  piG = acos(-1.);

}


void AnalysisRootpleProducerOnlyMC::beginJob( const EventSetup& )
{
  // use TFileService for output to root file 
  AnalysisTree = fs->make<TTree>("AnalysisTree","MBUE Analysis Tree ");
 
  // process type
  AnalysisTree->Branch("EventKind",&EventKind,"EventKind/I");
  
  // save TClonesArrays of TLorentzVectors
  // i.e. store 4-vectors of particles and jets

  MCGamma = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("MCGamma", "TClonesArray", &MCGamma, 128000, 0);

  MonteCarlo = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("MonteCarlo", "TClonesArray", &MonteCarlo, 128000, 0);
  
  InclusiveJet = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("InclusiveJet", "TClonesArray", &InclusiveJet, 128000, 0);

  ChargedJet = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("ChargedJet", "TClonesArray", &ChargedJet, 128000, 0);
}

  
void AnalysisRootpleProducerOnlyMC::analyze( const Event& e, const EventSetup& )
{

  e.getByLabel( mcEvent           , EvtHandle        ) ;
  e.getByLabel( chgGenPartCollName, CandHandleMC     );
  e.getByLabel( chgJetCollName    , ChgGenJetsHandle );
  e.getByLabel( genJetCollName    , GenJetsHandle    );
  e.getByLabel( gammaGenPartCollName , GammaHandleMC );
  
  const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
  
  EventKind = Evt->signal_process_id();
  
  std::vector<math::XYZTLorentzVector> GenPart;
  std::vector<GenJet> ChgGenJetContainer;
  std::vector<GenJet> GenJetContainer;
  std::vector<math::XYZTLorentzVector> GammaPart;
  
  GenPart.clear();
  ChgGenJetContainer.clear();
  GenJetContainer.clear();
  GammaPart.clear();
  
  MCGamma->Clear();
  ChargedJet->Clear();
  InclusiveJet->Clear();
  MonteCarlo->Clear();

  if (ChgGenJetsHandle->size()){

    for ( GenJetCollection::const_iterator it(ChgGenJetsHandle->begin()), itEnd(ChgGenJetsHandle->end());
	  it!=itEnd; ++it)
      {
	ChgGenJetContainer.push_back(*it);
      }
    
    std::stable_sort(ChgGenJetContainer.begin(),ChgGenJetContainer.end(),GenJetSort());
    
    std::vector<GenJet>::const_iterator it(ChgGenJetContainer.begin()), itEnd(ChgGenJetContainer.end());
    for ( int iChargedJet(0); it != itEnd; ++it, ++iChargedJet)
      {
	new((*ChargedJet)[iChargedJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
      }
  }
  
  if (GenJetsHandle->size()){
    
    for ( GenJetCollection::const_iterator it(GenJetsHandle->begin()), itEnd(GenJetsHandle->end());
	  it!=itEnd; ++it )
      {
	GenJetContainer.push_back(*it);
      }

    std::stable_sort(GenJetContainer.begin(),GenJetContainer.end(),GenJetSort());

    std::vector<GenJet>::const_iterator it(GenJetContainer.begin()), itEnd(GenJetContainer.end()); 
    for ( int iInclusiveJet(0); it != itEnd; ++it, ++iInclusiveJet)
      {
	new((*InclusiveJet)[iInclusiveJet]) TLorentzVector(it->px(), it->py(), it->pz(), it->energy());
      }
  }
  
   if (CandHandleMC->size()){

    for (vector<GenParticle>::const_iterator it(CandHandleMC->begin()), itEnd(CandHandleMC->end());
         it != itEnd;it++)
    {
      GenPart.push_back(it->p4());
    }

  std::stable_sort(GenPart.begin(),GenPart.end(),GreaterPt());

  std::vector<math::XYZTLorentzVector>::const_iterator it(GenPart.begin()), itEnd(GenPart.end());
  for( int iMonteCarlo(0); it != itEnd; ++it, ++iMonteCarlo )
    {
      new((*MonteCarlo)[iMonteCarlo]) TLorentzVector(it->Px(), it->Py(), it->Pz(), it->E());
    }
  }
 
   // cout << GammaHandleMC->size() << endl; //It's a test. It work for CandHandleMC
   if(GammaHandleMC->size()){
      for (vector<GenParticle>::const_iterator it(GammaHandleMC->begin()), itEnd(GammaHandleMC->end());
	   it != itEnd;it++)
	{
	  GammaPart.push_back(it->p4());
	}
    
      std::stable_sort(GammaPart.begin(),GammaPart.end(),GreaterPt());
      std::vector<math::XYZTLorentzVector>::const_iterator it(GammaPart.begin()), itEnd(GammaPart.end());
      for( int iMCGamma(0); it != itEnd; ++it, ++iMCGamma )
	{
	  new((*MCGamma)[iMCGamma]) TLorentzVector(it->Px(), it->Py(), it->Pz(), it->E());
	}

   }

  store();
}

void AnalysisRootpleProducerOnlyMC::endJob()
{
}

