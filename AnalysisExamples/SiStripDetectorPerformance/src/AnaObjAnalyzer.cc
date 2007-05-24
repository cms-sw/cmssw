#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnaObjAnalyzer.h"

#include "AnalysisExamples/AnalysisObjects/interface/AnalyzedCluster.h"
#include "AnalysisExamples/AnalysisObjects/interface/AnalyzedTrack.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AnaObjAnalyzer::AnaObjAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


AnaObjAnalyzer::~AnaObjAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
AnaObjAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace anaobj;

  Handle<AnalyzedClusterCollection> v_anaclu;
//  iEvent.getByLabel("modAnaObjProducer","AnalyzedClusters",v_anaclu);
  iEvent.getByType(v_anaclu);
  Handle<AnalyzedTrackCollection> v_anatk;
//  iEvent.getByLabel("modAnaObjProducer","AnalyzedTracks",v_anatk);
  iEvent.getByType(v_anatk);

  std::cout << "Total clusters in the event = " << v_anaclu->size() << std::endl;
  std::cout << "Total tracks in the event = " << v_anatk->size() << std::endl;

  AnalyzedClusterCollection::const_iterator anaclu_iter;
  for (anaclu_iter = v_anaclu->begin(); anaclu_iter != v_anaclu->end(); ++anaclu_iter) {
//    std::cout << "Cluster position = " << anaclu_iter->clusterpos << std::endl;
    std::cout << "map size = " << anaclu_iter->angle.size() << std::endl;
    if ( anaclu_iter->angle.size() > 0 ) {
      std::cout << "angle = " << anaclu_iter->angle.find( anaclu_iter->tk_id[0] )->second << std::endl;
    }
    edm::RefVector<std::vector<AnalyzedTrack> > vecRefTrack;
    vecRefTrack = anaclu_iter->GetTrackRefVec();
    std::cout << "size = " << vecRefTrack.size() << std::endl;
    edm::RefVector<std::vector<AnalyzedTrack> >::const_iterator tk_ref_iter;
    for( tk_ref_iter = vecRefTrack.begin(); tk_ref_iter != vecRefTrack.end(); ++tk_ref_iter ) {
      std::cout << "hitspertrack = " << (*tk_ref_iter)->hitspertrack << std::endl;

    }
  }

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
      ESHandle<SetupData> pSetup;
      iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
AnaObjAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
AnaObjAnalyzer::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(AnaObjAnalyzer);
