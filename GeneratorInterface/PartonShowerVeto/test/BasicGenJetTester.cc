#include <iostream>

#include "DataFormats/JetReco/interface/GenJetCollection.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"

class BasicGenJetTester : public edm::EDAnalyzer
{

   public:
   
      //
      explicit BasicGenJetTester( const edm::ParameterSet& ) ;
      virtual ~BasicGenJetTester() {} // no need to delete ROOT stuff
                                   // as it'll be deleted upon closing TFile
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob() ;
      virtual void beginRun( const edm::Run &, const edm::EventSetup& );
      virtual void endRun( const edm::Run&, const edm::EventSetup& ) ;
      virtual void endJob() ;

   private:
   
     double      fQCut;
     TH1F*       fNJets;
     TH1F*       fNJetsAboveQCut;
     TH1D*       fLeadingJetPt ;
     TH1D*       fLeadingJetEta;
     TH1D*       fNext2LeadingJetPt;
     TH1D*       fNext2LeadingJetEta;
     TH1D*       fLowestJetHt;
     TH1D*       fLowestJetEta;
     
}; 

using namespace edm;
using namespace reco;
using namespace std;

BasicGenJetTester::BasicGenJetTester( const ParameterSet& pset )
   : fNJets(0), fNJetsAboveQCut(0), 
     fLeadingJetPt(0), fLeadingJetEta(0), 
     fNext2LeadingJetPt(0), fNext2LeadingJetEta(0),
     fLowestJetHt(0), fLowestJetEta(0)
{

   fQCut = pset.getParameter<double>( "qcut" );
   
}

void BasicGenJetTester::beginJob()
{
  
   Service<TFileService> fs;
  
  fNJets          = fs->make<TH1F>( "NJets",         "Number of Jets (total)", 50,  0.,   50. );
  fNJetsAboveQCut = fs->make<TH1F>( "NJetsAboveQCut","Number of Jets (above qcut)", 10,  0.,   10. );
  fLeadingJetPt  = fs->make<TH1D>(  "LeadingJetPt",  "Leading Jet Pt", 100, 0., 250. );
  fLeadingJetEta = fs->make<TH1D>(  "LeadingJetEta", "Leading Jet Eta", 100, -5.0, 5.0 );
  fNext2LeadingJetPt  = fs->make<TH1D>( "Next2LeadingJetPt",  "Next to Leading Jet Pt", 100, 0., 250. );
  fNext2LeadingJetEta = fs->make<TH1D>( "Next2LeadingJetEta", "Next to Leading Jet Eta", 100, -5.0, 5.0 );
  fLowestJetHt  = fs->make<TH1D>(   "LowestJetHt",  "Ht (Lowest Jet above qcut)", 100, 0., 250. );
  fLowestJetEta = fs->make<TH1D>(   "LowestJetEta", "Lowest Jet Eta", 100, -5.0, 5.0 );

  return ;
  
}

void BasicGenJetTester::beginRun( const edm::Run& r, const edm::EventSetup& es )
{
   
   return ;

}

void BasicGenJetTester::analyze( const Event& e, const EventSetup& )
{
  
   // here's an example of accessing GenJetCollection
   //
   Handle< GenJetCollection > ak5GenJetHandle ;
   Handle< GenJetCollection > ak7GenJetHandle ;
  
   // find initial (unsmeared, unfiltered,...) HepMCProduct
   //
   e.getByLabel( "ak5GenJets", ak5GenJetHandle ) ;
   // e.getByLabel( "ak7GenJets", ak7GenJetHandle ) ;
     
   int NGenJets5 = ak5GenJetHandle->size();
   // int NGenJets7 = ak7GenJetHandle->size();
  
   if ( NGenJets5 <= 0 ) return;
   
   fNJets->Fill( (float)NGenJets5 );
   
   int NGenJets5AboveQCut = 0;
   GenJet GJet;
   
   for ( unsigned int idx=0; idx<ak5GenJetHandle->size(); ++idx )
   {
      GJet = (*ak5GenJetHandle)[idx];
      double pt  = GJet.pt();    //cout << ": pt=" << pt; 
      if ( pt < fQCut ) continue;
      NGenJets5AboveQCut++;      
   }

   if ( NGenJets5AboveQCut <= 0 ) return;

   fNJetsAboveQCut->Fill( (float)NGenJets5AboveQCut );
   
   // leading jet
   //
   GJet = (*ak5GenJetHandle)[0];
   fLeadingJetPt->Fill( GJet.pt() );
   fLeadingJetEta->Fill( GJet.eta() );

   if ( NGenJets5AboveQCut <= 1 ) return;
   
   // next-to-leading jet
   //
   GJet = (*ak5GenJetHandle)[1];
   fNext2LeadingJetPt->Fill( GJet.pt() );
   fNext2LeadingJetEta->Fill( GJet.eta() );

   if ( NGenJets5AboveQCut <= 2 ) return;

   // lowest jet (above qcut)
   //
   GJet = (*ak5GenJetHandle)[NGenJets5AboveQCut-1];
   fLowestJetHt->Fill( GJet.pt() );
   fLowestJetEta->Fill( GJet.eta() );
       
   return ;
   
}

void BasicGenJetTester::endRun( const edm::Run& r, const edm::EventSetup& )
{

   return;

}


void BasicGenJetTester::endJob()
{
   
   return ;
}
 
DEFINE_FWK_MODULE(BasicGenJetTester);
