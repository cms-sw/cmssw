#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterSummaryProducer.h"

ClusterSummaryProducer::ClusterSummaryProducer(const edm::ParameterSet& iConfig)
  : doStrips(iConfig.getParameter<bool>("doStrips")),
    doPixels(iConfig.getParameter<bool>("doPixels")),
    verbose(iConfig.getParameter<bool>("verbose"))
{
 
  pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"));
  stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("stripClusters"));


  std::vector<edm::ParameterSet> wantedsubdets_ps = iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedSubDets");
  for(std::vector<edm::ParameterSet>::const_iterator wsdps = wantedsubdets_ps.begin();wsdps!=wantedsubdets_ps.end();++wsdps) {
    unsigned int             detsel    = wsdps->getParameter<unsigned int>("detSelection");
    std::string              detname   = wsdps->getParameter<std::string>("detLabel");
    std::vector<std::string> selection = wsdps->getParameter<std::vector<std::string> >("selection");

    if(ClusterSummary::checkSubDet(detsel)){
      pixelSelector.push_back(ModuleSelection(DetIdSelector(selection),(ClusterSummary::CMSTracker)detsel));
      if(verbose)pixelModuleNames.push_back(detname);
    } else{
      stripSelector.push_back(ModuleSelection(DetIdSelector(selection),(ClusterSummary::CMSTracker)detsel));
      if(verbose)stripModuleNames.push_back(detname);
    }
  }

  //register your products
  produces<ClusterSummary>().setBranchAlias("SummaryCollection");

}


void
ClusterSummaryProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   cCluster.prepareStorage();
   
   //===================++++++++++++========================
   //
   //                   For SiStrips
   //
   //===================++++++++++++========================
   if (doStrips){
     edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
     iEvent.getByToken(stripClusters_, stripClusters);
     std::vector<bool> selectedVector(stripSelector.size(),false);

     edmNew::DetSetVector<SiStripCluster>::const_iterator itClusters=stripClusters->begin();
     for(;itClusters!=stripClusters->end();++itClusters){

       for(unsigned int iS = 0; iS < stripSelector.size(); ++iS)
         selectedVector[iS] = stripSelector[iS].first.isSelected(itClusters->id());

       for(edmNew::DetSet<SiStripCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){
         const ClusterVariables Summaryinfo(*cluster);
         for(unsigned int iS = 0; iS < stripSelector.size(); ++iS){
           if(!selectedVector[iS]) continue;
           const ClusterSummary::CMSTracker  module   = stripSelector[iS].second;
           int modLocation = cCluster.getModuleLocation((int)module,false);
           if(modLocation < 0) {
             modLocation = cCluster.getNumberOfModules();
             cCluster.addModule( module ) ;
           }
           cCluster.addNClusByIndex     (modLocation, 1 );
           cCluster.addClusSizeByIndex  (modLocation, Summaryinfo.clusterSize() );
           cCluster.addClusChargeByIndex(modLocation, Summaryinfo.charge() );
         }
       }
     }
   }

   //===================++++++++++++========================
   //
   //                   For SiPixels
   //
   //===================++++++++++++========================
   if (doPixels){
     edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
     iEvent.getByToken(pixelClusters_, pixelClusters);
     std::vector<bool> selectedVector(pixelSelector.size(),false);

     edmNew::DetSetVector<SiPixelCluster>::const_iterator itClusters=pixelClusters->begin();
     for(;itClusters!=pixelClusters->end();++itClusters){

       for(unsigned int iS = 0; iS < pixelSelector.size(); ++iS)
         selectedVector[iS] = pixelSelector[iS].first.isSelected(itClusters->id());

       for(edmNew::DetSet<SiPixelCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){
         for(unsigned int iS = 0; iS < pixelSelector.size(); ++iS){
           if(!selectedVector[iS]) continue;
           const ClusterSummary::CMSTracker  module   = pixelSelector[iS].second;
           int modLocation = cCluster.getModuleLocation((int)module,false);
           if(modLocation < 0) {
             modLocation = cCluster.getNumberOfModules();
             cCluster.addModule( module ) ;
           }
           cCluster.addNClusByIndex     (modLocation, 1 );
           cCluster.addClusSizeByIndex  (modLocation, cluster->size());
           cCluster.addClusChargeByIndex(modLocation, float(cluster->charge())/1000. );
         }
       }
     }
   }

   //===================+++++++++++++========================
   //
   //                   Fill Producer
   //
   //===================+++++++++++++========================
   if(verbose){
     auto printMod =  [&] (std::vector<std::string>& modName, ModuleSelections& modSel){
       for(unsigned int iM = 0; iM < modName.size(); ++iM){
         int modLoc = cCluster.getModuleLocation(modSel[iM].second,false);
         if( modLoc<0 ) continue;
         std::cout << "n" << modName[iM]   <<", avg size, avg charge = "
             << cCluster.getNClusByIndex     (modLoc ) << ", "
             << cCluster.getClusSizeByIndex  (modLoc )/cCluster.getNClusByIndex(modLoc ) << ", "
             << cCluster.getClusChargeByIndex(modLoc )/cCluster.getNClusByIndex(modLoc)
             << std::endl;
       }
     };

     printMod(stripModuleNames,stripSelector);
     printMod(pixelModuleNames,pixelSelector);
     std::cout << "-------------------------------------------------------" << std::endl;
   }
   //Put the filled class into the producer
   std::auto_ptr<ClusterSummary> result(new ClusterSummary (cCluster) );
   iEvent.put( result );
}


void 
ClusterSummaryProducer::beginStream(edm::StreamID)
{
  if(!verbose) return;
  if (doStrips){
    std::cout << "+++++++++++++++++++++++++++++++ "  << std::endl;
    std::cout << "FOR STRIPS: "  << std::endl;
    std::cout << "Getting info on " ;
    for (unsigned int ii = 0; ii < stripModuleNames.size( ); ++ii) {
      std::cout << stripModuleNames[ii] << " " ;
    }
    std::cout << std::endl;
  }

  if (doPixels){
    std::cout << "FOR PIXELS: " << std::endl;
    std::cout << "Getting info on " ;
    for (unsigned int ii = 0; ii < pixelModuleNames.size( ); ++ii) {
      std::cout << pixelModuleNames[ii] << " " ;
    }
    std::cout << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++ "  << std::endl;
  }

}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterSummaryProducer);


