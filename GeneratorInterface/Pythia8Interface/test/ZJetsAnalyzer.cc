#include <iostream>
#include <fstream>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>

#include "GeneratorInterface/Pythia8Interface/test/analyserhepmc/LeptonAnalyserHepMC.h"
#include "GeneratorInterface/Pythia8Interface/test/analyserhepmc/JetInputHepMC.h"

struct ParticlePtGreater {
  double operator () (const HepMC::GenParticle *v1,
                                  const HepMC::GenParticle *v2)
  { return v1->momentum().perp() > v2->momentum().perp(); }
};


class ZJetsAnalyzer : public edm::EDAnalyzer
{

  public:
   
    //
    explicit ZJetsAnalyzer( const edm::ParameterSet& ) ;
    virtual ~ZJetsAnalyzer(); // no need to delete ROOT stuff
                              // as it'll be deleted upon closing TFile
      
    virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;
    virtual void beginJob() override;
    virtual void endRun( const edm::Run&, const edm::EventSetup& ) override;

  private:
   
    LeptonAnalyserHepMC LA;
    JetInputHepMC JetInput;
    fastjet::Strategy strategy;
    fastjet::RecombinationScheme recombScheme;
    fastjet::JetDefinition* jetDef;

    int icategories[6];

    TH1D*       fHist2muMass ;
     
}; 


using namespace edm;
using namespace std;


ZJetsAnalyzer::ZJetsAnalyzer( const ParameterSet& pset )
  : fHist2muMass(0)
{
// actually, pset is NOT in use - we keep it here just for illustratory putposes
}


ZJetsAnalyzer::~ZJetsAnalyzer()
{;}


void ZJetsAnalyzer::beginJob()
{
  
  Service<TFileService> fs;
  fHist2muMass = fs->make<TH1D>(  "Hist2muMass", "2-mu inv. mass", 100,  60., 120. ) ;
    
  double Rparam = 0.5;
  strategy = fastjet::Best;
  recombScheme = fastjet::E_scheme;
  jetDef = new fastjet::JetDefinition(fastjet::antikt_algorithm, Rparam,
                                      recombScheme, strategy);

  for (int ind=0; ind < 6; ind++) {icategories[ind]=0;}

  return ;
  
}


void ZJetsAnalyzer::endRun( const Run& r, const EventSetup& )
{
  ofstream testi("testi.dat");
  double val, errval;

  Handle< GenRunInfoProduct > genRunInfoProduct;
  r.getByLabel("generator", genRunInfoProduct );
  val = (double)genRunInfoProduct->crossSection();
  cout << endl;
  cout << "cross section = " << val << endl;
  cout << endl;

  errval = 0.;
  if(icategories[0] > 0) errval = val/sqrt( (double)(icategories[0]) );
  testi << "pythia8_test1  1   " << val << " " << errval << " " << endl;

  cout << endl;
  cout << " Events with at least 1 isolated lepton  :                     "
       << ((double)icategories[1])/((double)icategories[0]) << endl;
  cout << " Events with at least 2 isolated leptons :                     "
       << ((double)icategories[2])/((double)icategories[0]) << endl;
  cout << " Events with at least 2 isolated leptons and at least 1 jet  : "
       << ((double)icategories[3])/((double)icategories[0]) << endl;
  cout << " Events with at least 2 isolated leptons and at least 2 jets : "
       << ((double)icategories[4])/((double)icategories[0]) << endl;
  cout << endl;

  val = ((double)icategories[4])/((double)icategories[0]);
  errval = 0.;
  if(icategories[4] > 0) errval = val/sqrt((double)icategories[4]);
  testi << "pythia8_test1  2   " << val << " " << errval << " " << endl;

}


void ZJetsAnalyzer::analyze( const Event& e, const EventSetup& )
{
  
  icategories[0]++;

  // here's an example of accessing GenEventInfoProduct
  Handle< GenEventInfoProduct > GenInfoHandle;
  e.getByLabel( "generator", GenInfoHandle );
  double qScale = GenInfoHandle->qScale();
  double pthat = ( GenInfoHandle->hasBinningValues() ? 
                  (GenInfoHandle->binningValues())[0] : 0.0);
  cout << " qScale = " << qScale << " pthat = " << pthat << endl;
  //
  // this (commented out) code below just exemplifies how to access certain info 
  //
  //double evt_weight1 = GenInfoHandle->weights()[0]; // this is "stanrd Py6 evt weight;
                                                    // corresponds to PYINT1/VINT(97)
  //double evt_weight2 = GenInfoHandle->weights()[1]; // in case you run in CSA mode or otherwise
                                                    // use PYEVWT routine, this will be weight
						    // as returned by PYEVWT, i.e. PYINT1/VINT(99)
  //std::cout << " evt_weight1 = " << evt_weight1 << std::endl;
  //std::cout << " evt_weight2 = " << evt_weight2 << std::endl;
  //double weight = GenInfoHandle->weight();
  //std::cout << " as returned by the weight() method, integrated event weight = " << weight << std::endl;
  
  // here's an example of accessing particles in the event record (HepMCProduct)
  //
  Handle< HepMCProduct > EvtHandle ;
  
  // find initial (unsmeared, unfiltered,...) HepMCProduct
  //
  e.getByLabel( "generator", EvtHandle ) ;
  
  const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
  
  int nisolep = LA.nIsolatedLeptons(Evt);

  //cout << "Number of leptons = " << nisolep << endl;
  if(nisolep > 0) icategories[1]++;
  if(nisolep > 1) icategories[2]++;

  JetInputHepMC::ParticleVector jetInput = JetInput(Evt);
  std::sort(jetInput.begin(), jetInput.end(), ParticlePtGreater());

  //cout << "Size of jet input = " << jetInput.size() << endl;

  // Fastjet input
  std::vector <fastjet::PseudoJet> jfInput;
  jfInput.reserve(jetInput.size());
  for (JetInputHepMC::ParticleVector::const_iterator iter = jetInput.begin();
       iter != jetInput.end(); ++iter) {
    jfInput.push_back(fastjet::PseudoJet( (*iter)->momentum().px(),
                                          (*iter)->momentum().py(),
                                          (*iter)->momentum().pz(),
                                          (*iter)->momentum().e()  )  );
    jfInput.back().set_user_index(iter - jetInput.begin());
  }

  // Run Fastjet algorithm
  vector <fastjet::PseudoJet> inclusiveJets, sortedJets, cleanedJets;
  fastjet::ClusterSequence clustSeq(jfInput, *jetDef);

  // Extract inclusive jets sorted by pT (note minimum pT in GeV)
  inclusiveJets = clustSeq.inclusive_jets(20.0);
  sortedJets    = sorted_by_pt(inclusiveJets);

  //cout << "Size of jets = " << sortedJets.size() << endl;

  cleanedJets = LA.removeLeptonsFromJets(sortedJets, Evt);

  //cout << "Size of cleaned jets = " << cleanedJets.size() << endl;
  if(nisolep > 1) {
    if(cleanedJets.size() > 0) icategories[3]++;
    if(cleanedJets.size() > 1) icategories[4]++;
  }

  return ;
   
}


typedef ZJetsAnalyzer ZJetsTest;
DEFINE_FWK_MODULE(ZJetsTest);
