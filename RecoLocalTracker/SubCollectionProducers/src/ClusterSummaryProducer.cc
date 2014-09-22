#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterSummaryProducer.h"

ClusterSummaryProducer::ClusterSummaryProducer(const edm::ParameterSet& iConfig)
  : doStrips(iConfig.getParameter<bool>("doStrips")),
    doPixels(iConfig.getParameter<bool>("doPixels")),
    verbose(iConfig.getParameter<bool>("verbose"))
{
 
  pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"));
  stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("stripClusters"));

  ClusterSummary::CMSTracker maxEnum = ClusterSummary::STRIP;

  std::vector<std::string> wantedsubdets = iConfig.getParameter<std::vector<std::string> >("wantedSubDets");
  for(const auto& iS : wantedsubdets){

    ClusterSummary::CMSTracker subdet = ClusterSummary::NVALIDENUMS;
    for(int iN = 0; iN < ClusterSummary::NVALIDENUMS; ++iN)
      if(ClusterSummary::subDetNames[iN] == iS)
        subdet = ClusterSummary::CMSTracker(iN);
    if(subdet == ClusterSummary::NVALIDENUMS) throw cms::Exception( "No standard selection: ") << iS;

    selectors.push_back(ModuleSelection(DetIdSelector(ClusterSummary::subDetSelections[subdet]),subdet));
    if(subdet > maxEnum) maxEnum = subdet;
    if(verbose)moduleNames.push_back(ClusterSummary::subDetNames[subdet]);
  }


  std::vector<edm::ParameterSet> wantedusersubdets_ps = iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedUserSubDets");
  for(const auto& iS : wantedusersubdets_ps){
    ClusterSummary::CMSTracker subdet    = (ClusterSummary::CMSTracker)iS.getParameter<unsigned int>("detSelection");
    std::string                detname   = iS.getParameter<std::string>("detLabel");
    std::vector<std::string>   selection = iS.getParameter<std::vector<std::string> >("selection");

    if(subdet <=  ClusterSummary::NVALIDENUMS) throw cms::Exception( "Already predefined selection: ") << subdet;
    if(subdet >=  ClusterSummary::NTRACKERENUMS) throw cms::Exception( "Selection is out of range: ") << subdet;

    selectors.push_back(ModuleSelection(DetIdSelector(selection),subdet));
    if(subdet > maxEnum) maxEnum = subdet;
    if(verbose)moduleNames.push_back(detname);
  }

  cCluster = ClusterSummary(maxEnum + 1);
  produces<ClusterSummary>().setBranchAlias("trackerClusterSummary");
}

void
ClusterSummaryProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   cCluster.reset();
   std::vector<bool> selectedVector(selectors.size(),false);
   
   auto getSelections   =  [&] (const uint32_t detid ){
     for(unsigned int iS = 0; iS < selectors.size(); ++iS)
       selectedVector[iS] = selectors[iS].first.isSelected(detid);
   };
   auto fillSelections =  [&] (const int clusterSize, const float clusterCharge ){
     for(unsigned int iS = 0; iS < selectors.size(); ++iS){
       if(!selectedVector[iS]) continue;
       const ClusterSummary::CMSTracker  module   = selectors[iS].second;
       cCluster.addNClusByIndex     (module, 1 );
       cCluster.addClusSizeByIndex  (module, clusterSize );
       cCluster.addClusChargeByIndex(module, clusterCharge );
     }
   };

   //===================++++++++++++========================
   //                   For SiStrips
   //===================++++++++++++========================
   if (doStrips){
     edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
     iEvent.getByToken(stripClusters_, stripClusters);
     edmNew::DetSetVector<SiStripCluster>::const_iterator itClusters=stripClusters->begin();
     for(;itClusters!=stripClusters->end();++itClusters){
       getSelections(itClusters->id());
       for(edmNew::DetSet<SiStripCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){
         const ClusterVariables Summaryinfo(*cluster);
         fillSelections(Summaryinfo.clusterSize(),Summaryinfo.charge());
       }
     }
   }

   //===================++++++++++++========================
   //                   For SiPixels
   //===================++++++++++++========================
   if (doPixels){
     edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
     iEvent.getByToken(pixelClusters_, pixelClusters);
     edmNew::DetSetVector<SiPixelCluster>::const_iterator itClusters=pixelClusters->begin();
     for(;itClusters!=pixelClusters->end();++itClusters){
       getSelections(itClusters->id());
       for(edmNew::DetSet<SiPixelCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){
         fillSelections(cluster->size(),float(cluster->charge())/1000.);
       }
     }
   }

   //===================+++++++++++++========================
   //                   Fill Producer
   //===================+++++++++++++========================
   if(verbose){
     for(const auto& iS : selectors){
       const ClusterSummary::CMSTracker  module   = iS.second;
       edm::LogInfo("ClusterSummaryProducer") << "n" << moduleNames[module]   <<", avg size, avg charge = "
           << cCluster.getNClusByIndex     (module ) << ", "
           << cCluster.getClusSizeByIndex  (module )/cCluster.getNClusByIndex(module ) << ", "
           << cCluster.getClusChargeByIndex(module )/cCluster.getNClusByIndex(module)
           << std::endl;
     }
     std::cout << "-------------------------------------------------------" << std::endl;
   }

   //Put the filled class into the producer
   std::auto_ptr<ClusterSummary> result(new ClusterSummary (0) );
   //Cleanup empty selections
   result->copyNonEmpty(cCluster);
   iEvent.put( result );
}


void 
ClusterSummaryProducer::beginStream(edm::StreamID)
{
  if(!verbose) return;
  edm::LogInfo("ClusterSummaryProducer") << "+++++++++++++++++++++++++++++++ "  << std::endl << "Getting info on " ;
    for (const auto& iS : moduleNames ) { edm::LogInfo("ClusterSummaryProducer") << iS<< " " ;}
    edm::LogInfo("ClusterSummaryProducer")  << std::endl;
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterSummaryProducer);


