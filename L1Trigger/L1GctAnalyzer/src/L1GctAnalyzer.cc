#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "L1Trigger/L1GctAnalyzer/interface/L1GctAnalyzer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctAnalyzer::L1GctAnalyzer(const edm::ParameterSet& iConfig) :
  m_histFileName(iConfig.getUntrackedParameter<string>( "histogramFileName", "testHisto.root" )),
  doBasicHist(false),
  doJetCheckHist(false),
  doMETCheckHist(false)
 {
  std::cout << "In L1GctAnalyzer ctor" << std::endl;
  // Read the options from the configuration file into local strings
  // These are then used in beginJob() to setup the histogramming

  // Get the list of histogram types
  vector<string> histogramModules = iConfig.getUntrackedParameter< vector<string> >( "histogramModules" );
  for (vector<string>::const_iterator module=histogramModules.begin(); module != histogramModules.end(); ++module) {
    if (*module == "basic") {
      doBasicHist = true;
    } else {

      // allow different types of histogrammer modules, each with its list of configuration parameters
      vector<string> histogramOptions = iConfig.getUntrackedParameter< vector<string> >(*module);
      if (*module == "jetCheck") {
	doJetCheckHist = true;
	jetCheckOptions = histogramOptions;
      } else if (*module == "mETCheck") {
	doMETCheckHist = true;
	mETCheckOptions = histogramOptions;
      } else {
	throw cms::Exception("L1GctAnalyzer setup error") << " invalid histogramModule argument, " << *module  
							  << " to L1GctAnalyzer in configuration file " << std::endl;
      }
    }

  }
}

L1GctAnalyzer::~L1GctAnalyzer()
{
 
   // Nothing to do, all tidying up done in endJob() 

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1GctAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get some GCT digis
   GctOutputData gctdata;

   gctdata.electrons.resize(2);
   gctdata.jets.resize(3);

   iEvent.getByLabel("gct","isoEm",   gctdata.electrons.at(0));
   iEvent.getByLabel("gct","nonIsoEm",gctdata.electrons.at(1));
   iEvent.getByLabel("gct","cenJets", gctdata.jets.at(0));
   iEvent.getByLabel("gct","forJets", gctdata.jets.at(1));
   iEvent.getByLabel("gct","tauJets", gctdata.jets.at(2));
   iEvent.getByLabel("gct",gctdata.etTotal);
   iEvent.getByLabel("gct",gctdata.etHad);
   iEvent.getByLabel("gct",gctdata.etMiss);

   if (doBasicHist) {
     basicHist->fillHistograms(gctdata);
   }

   // Get some reconstructed quantities and compare results

   if (doJetCheckHist) {
     Handle<GenJetCollection> genJets;
     for (unsigned i=0; i<jetCheckHist.size(); ++i) {
       string label=jetCheckOptions.at(i) + "GenJets";
       iEvent.getByLabel(label,genJets);
       jetCheckHist.at(i)->setInputProduct(genJets);
       jetCheckHist.at(i)->fillHistograms(gctdata);
     }
   }

   if (doMETCheckHist) {
     Handle<METCollection> genMET;
     for (unsigned i=0; i<mETCheckHist.size(); ++i) {
       string label=mETCheckOptions.at(i);
       iEvent.getByLabel(label,genMET);
       mETCheckHist.at(i)->setInputProduct(genMET);
       mETCheckHist.at(i)->fillHistograms(gctdata);
     }
   }

   Handle<SuperClusterCollection> SuperClusters;
   iEvent.getByLabel("correctedHybridSuperClusters","",SuperClusters);
   std::cout << "Size of SuperClusters is " << SuperClusters->size() << std::endl;
   if (SuperClusters->size() != 0) {
     int i=0;
     for (SuperClusterCollection::const_iterator sc=SuperClusters->begin(); sc != SuperClusters->end(); ++sc) {
       std::cout << "SuperCluster " << i << " energy " << sc->energy() << " eta " << sc->eta() << " phi " << sc->phi() << std::endl;
       ++i;
     }
   }
  
   iEvent.getByLabel("correctedIslandBarrelSuperClusters","",SuperClusters);
   std::cout << "Size of SuperClusters is " << SuperClusters->size() << std::endl;
   if (SuperClusters->size() != 0) {
     int i=0;
     for (SuperClusterCollection::const_iterator sc=SuperClusters->begin(); sc != SuperClusters->end(); ++sc) {
       std::cout << "SuperCluster " << i << " energy " << sc->energy() << " eta " << sc->eta() << " phi " << sc->phi() << std::endl;
       ++i;
     }
   }
  
   iEvent.getByLabel("correctedIslandEndcapSuperClusters","",SuperClusters);
   std::cout << "Size of SuperClusters is " << SuperClusters->size() << std::endl;
   if (SuperClusters->size() != 0) {
     int i=0;
     for (SuperClusterCollection::const_iterator sc=SuperClusters->begin(); sc != SuperClusters->end(); ++sc) {
       std::cout << "SuperCluster " << i << " energy " << sc->energy() << " eta " << sc->eta() << " phi " << sc->phi() << std::endl;
       ++i;
     }
   }
  
   for (unsigned i=0; i<gctdata.electrons.size(); ++i) {
     for (unsigned j=0; j<gctdata.electrons.at(i)->size(); ++j) {
       L1GctEmCand ele = gctdata.electrons.at(i)->at(j);
       std::cout << "gct electron energy " << ele.rank() << " eta " << ele.etaIndex() << " phi " << ele.phiIndex() << std::endl;
     }
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1GctAnalyzer::beginJob(const edm::EventSetup& iSetup)
{
  // Open rootfile for histograms
  m_file = new TFile( m_histFileName.c_str(), "RECREATE" );

  // Histogrammer for basic gct output quantities
  if (doBasicHist) {
    basicHist    = new L1GctBasicHistogrammer(m_file, "basicHistograms");
  }

  // Histogrammers for comparison with jet reconstruction
  if (doJetCheckHist) {
    for (vector<string>::const_iterator option=jetCheckOptions.begin(); option != jetCheckOptions.end(); ++option) {
      string directory = *option + "JetCheck";
      jetCheckHist.push_back(new L1GctJetCheckHistogrammer(m_file, directory));
    }
  }

  // Histogrammers for comparison with missing Et reconstruction
  if (doMETCheckHist) {
    for (vector<string>::const_iterator option=mETCheckOptions.begin(); option != mETCheckOptions.end(); ++option) {
      string directory = *option + "METCheck";
      mETCheckHist.push_back(new L1GctMETCheckHistogrammer(m_file, directory));
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1GctAnalyzer::endJob() {
  if (doBasicHist) { delete basicHist; }
  if (doJetCheckHist) {
    for (unsigned i=0; i<jetCheckHist.size(); ++i) {
      delete jetCheckHist.at(i);
    }
  }
  if (doMETCheckHist) {
    for (unsigned i=0; i<mETCheckHist.size(); ++i) {
      delete mETCheckHist.at(i);
    }
  }
  delete m_file;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1GctAnalyzer)
