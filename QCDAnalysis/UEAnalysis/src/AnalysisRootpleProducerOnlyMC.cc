// Authors: F. Ambroglini, L. Fano'
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

  NumberMCParticles=0;
  NumberInclusiveJet=0;
  NumberChargedJet=0;
}

void AnalysisRootpleProducerOnlyMC::fillEventInfo(int e)
{
  EventKind = e;
}

void AnalysisRootpleProducerOnlyMC::fillMCParticles(float p, float pt, float eta, float phi)
{
  MomentumMC[NumberMCParticles]=p;
  TransverseMomentumMC[NumberMCParticles]=pt;
  EtaMC[NumberMCParticles]=eta;
  PhiMC[NumberMCParticles]=phi;
  NumberMCParticles++;
}

void AnalysisRootpleProducerOnlyMC::fillInclusiveJet(float p, float pt, float eta, float phi)
{
  MomentumIJ[NumberInclusiveJet]=p;
  TransverseMomentumIJ[NumberInclusiveJet]=pt;
  EtaIJ[NumberInclusiveJet]=eta;
  PhiIJ[NumberInclusiveJet]=phi;
  NumberInclusiveJet++;
}

void AnalysisRootpleProducerOnlyMC::fillChargedJet(float p, float pt, float eta, float phi)
{
  MomentumCJ[NumberChargedJet]=p;
  TransverseMomentumCJ[NumberChargedJet]=pt;
  EtaCJ[NumberChargedJet]=eta;
  PhiCJ[NumberChargedJet]=phi;
  NumberChargedJet++;
}

AnalysisRootpleProducerOnlyMC::AnalysisRootpleProducerOnlyMC( const ParameterSet& pset )
{
  mcEventToken = consumes<edm::HepMCProduct>(pset.getUntrackedParameter<InputTag>("MCEvent",std::string("")));
  genJetCollToken = consumes<reco::GenJetCollection>(pset.getUntrackedParameter<InputTag>("GenJetCollectionName",std::string("")));
  chgJetCollToken = consumes<reco::GenJetCollection>(pset.getUntrackedParameter<InputTag>("ChgGenJetCollectionName",std::string("")));
  chgGenPartCollToken = consumes<std::vector<reco::GenParticle> >(pset.getUntrackedParameter<InputTag>("ChgGenPartCollectionName",std::string("")));

  piG = acos(-1.);
  NumberMCParticles=0;
  NumberInclusiveJet=0;
  NumberChargedJet=0;
}


void AnalysisRootpleProducerOnlyMC::beginJob()
{
  // use TFileService for output to root file
  AnalysisTree = fs->make<TTree>("AnalysisTree","MBUE Analysis Tree ");

  // process type
  AnalysisTree->Branch("EventKind",&EventKind,"EventKind/I");

  // store p, pt, eta, phi for particles and jets

  // GenParticles at hadron level
  AnalysisTree->Branch("NumberMCParticles",&NumberMCParticles,"NumberMCParticles/I");
  AnalysisTree->Branch("MomentumMC",MomentumMC,"MomentumMC[NumberMCParticles]/F");
  AnalysisTree->Branch("TransverseMomentumMC",TransverseMomentumMC,"TransverseMomentumMC[NumberMCParticles]/F");
  AnalysisTree->Branch("EtaMC",EtaMC,"EtaMC[NumberMCParticles]/F");
  AnalysisTree->Branch("PhiMC",PhiMC,"PhiMC[NumberMCParticles]/F");

  // GenJets
  AnalysisTree->Branch("NumberInclusiveJet",&NumberInclusiveJet,"NumberInclusiveJet/I");
  AnalysisTree->Branch("MomentumIJ",MomentumIJ,"MomentumIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("TrasverseMomentumIJ",TransverseMomentumIJ,"TransverseMomentumIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("EtaIJ",EtaIJ,"EtaIJ[NumberInclusiveJet]/F");
  AnalysisTree->Branch("PhiIJ",PhiIJ,"PhiIJ[NumberInclusiveJet]/F");

  // jets from charged GenParticles
  AnalysisTree->Branch("NumberChargedJet",&NumberChargedJet,"NumberChargedJet/I");
  AnalysisTree->Branch("MomentumCJ",MomentumCJ,"MomentumCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("TrasverseMomentumCJ",TransverseMomentumCJ,"TransverseMomentumCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("EtaCJ",EtaCJ,"EtaCJ[NumberChargedJet]/F");
  AnalysisTree->Branch("PhiCJ",PhiCJ,"PhiCJ[NumberChargedJet]/F");


  // alternative storage method:
  // save TClonesArrays of TLorentzVectors
  // i.e. store 4-vectors of particles and jets

  MonteCarlo = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("MonteCarlo", "TClonesArray", &MonteCarlo, 128000, 0);

  InclusiveJet = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("InclusiveJet", "TClonesArray", &InclusiveJet, 128000, 0);

  ChargedJet = new TClonesArray("TLorentzVector", 10000);
  AnalysisTree->Branch("ChargedJet", "TClonesArray", &ChargedJet, 128000, 0);
}


void AnalysisRootpleProducerOnlyMC::analyze( const Event& e, const EventSetup& )
{

  e.getByToken( mcEventToken       , EvtHandle        );
  e.getByToken( chgGenPartCollToken, CandHandleMC     );
  e.getByToken( chgJetCollToken    , ChgGenJetsHandle );
  e.getByToken( genJetCollToken    , GenJetsHandle    );

  const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;

  EventKind = Evt->signal_process_id();

  std::vector<math::XYZTLorentzVector> GenPart;
  std::vector<GenJet> ChgGenJetContainer;
  std::vector<GenJet> GenJetContainer;

  GenPart.clear();
  ChgGenJetContainer.clear();
  GenJetContainer.clear();

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
	fillChargedJet(it->p(),it->pt(),it->eta(),it->phi());
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
	fillInclusiveJet(it->p(),it->pt(),it->eta(),it->phi());
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
	fillMCParticles(it->P(),it->Pt(),it->Eta(),it->Phi());
	new((*MonteCarlo)[iMonteCarlo]) TLorentzVector(it->Px(), it->Py(), it->Pz(), it->E());
      }
  }

  store();
}

void AnalysisRootpleProducerOnlyMC::endJob()
{
}

