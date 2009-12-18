
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h" 
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "JetMETCorrections/JetVertexAssociation/test/AnalysisJV.h"

#include "TFile.h"
#include "TH1.h"

class TFile;
class TH1D;
using namespace edm;
using namespace std;
using namespace reco;

AnalysisJV::AnalysisJV(const edm::ParameterSet& pset) :
  fOutputFileName( pset.getUntrackedParameter<string>("HistOutFile",std::string("jv_analysis.root"))){


}


AnalysisJV::~AnalysisJV()
{
 
}


void AnalysisJV::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   cout <<"----------------------------"<<endl;
   using namespace edm;
   typedef std::vector<double> ResultCollection1;
   typedef std::vector<bool> ResultCollection2;

   Handle<ResultCollection1> JV_alpha;
   iEvent.getByLabel("jetvertex","Var",JV_alpha);

   Handle<ResultCollection2> JV_jet_type;
   iEvent.getByLabel("jetvertex","JetType",JV_jet_type);

   Handle<CaloJetCollection> CaloIconeJetsHandle;
   iEvent.getByLabel( "iterativeCone5CaloJets", CaloIconeJetsHandle);

    if(CaloIconeJetsHandle->size()){
    ResultCollection1::const_iterator it_jv1 = JV_alpha->begin();
    ResultCollection2::const_iterator it_jv2 = JV_jet_type->begin();
      for(CaloJetCollection::const_iterator it=CaloIconeJetsHandle->begin();it!=CaloIconeJetsHandle->end();it++){

            if(*it_jv2)	cout<<"Jet: Et = "<<it->pt()<<" - true jet"<<endl;
            else cout<<"Jet: Et = "<<it->pt()<<" - 'fake' jet"<<endl;
              
            fHistAlpha->Fill(*it_jv1);
            it_jv1++;
            it_jv2++;

      }
    }

}



void AnalysisJV::beginJob(){

   fOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
   fHistAlpha   = new TH1D(  "HistAlpha"  , "", 30,  0., 1.5 ) ;

}

void AnalysisJV::endJob() {

  fOutputFile->Write() ;
  fOutputFile->Close() ;
  
  return ;
}

//define this as a plug-in
DEFINE_FWK_MODULE(AnalysisJV);
