#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterSummaryProducer.h"

ClusterSummaryProducer::ClusterSummaryProducer(const edm::ParameterSet& iConfig)
  : stripModules(iConfig.getParameter<std::string>("stripModule")),
    pixelModules(iConfig.getParameter<std::string>("pixelModule")),
    stripVariables(iConfig.getParameter<std::string>("stripVariables")),
    pixelVariables(iConfig.getParameter<std::string>("pixelVariables")),
    doStrips(iConfig.getParameter<bool>("doStrips")),
    doPixels(iConfig.getParameter<bool>("doPixels")),
    verbose(iConfig.getParameter<bool>("verbose"))
{
 
  pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"));
  stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("stripClusters"));

  //register your products
  produces<ClusterSummary>().setBranchAlias("SummaryCollection");
  
  firstpass = true;
  firstpass_mod = true;
  firstpassPixel = true;
  firstpassPixel_mod = true;

}


void
ClusterSummaryProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   cCluster.ClearUserModules();
   
   //===================++++++++++++========================
   //
   //                   For SiStrips
   //
   //===================++++++++++++========================
   if (doStrips){
     edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
     iEvent.getByToken(stripClusters_, stripClusters);

     ModuleSelectionVect.clear();
     for ( std::vector<std::string>::iterator it=v_stripModuleTypes.begin() ; it < v_stripModuleTypes.end(); it++ ){
       ModuleSelectionVect.push_back ( new ClusterSummary::ModuleSelection(*it) );
     }
 
     firstpass_mod = true;
   
     int CurrMod = -1;
     //Loop over all the ModuleSelectors
     for ( unsigned int i = 0; i < ModuleSelectionVect.size(); i++){
     
       // Loop over the strip clusters
       edmNew::DetSetVector<SiStripCluster>::const_iterator itClusters=stripClusters->begin();
       for(;itClusters!=stripClusters->end();++itClusters){
	 uint32_t id = itClusters->id();
	 for(edmNew::DetSet<SiStripCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){

	   const ClusterVariables Summaryinfo(*cluster);
      
	   // For each ModuleSelector, check if the detID belongs to a desired module. If so, update the summary information for that module

	   std::pair<int, int> ModSelect = ModuleSelectionVect.at(i) -> IsStripSelected( id );
	   int mod_pair = ModSelect.first;
	   int mod_pair2 = ModSelect.second;

	   if ( mod_pair ){
	     if ( firstpass_mod ) {
	       CurrMod = mod_pair2;
	       cCluster.SetUserModules( mod_pair2 ) ;
	     }
	   
	     firstpass_mod = false;
	     int CurrModTmp = mod_pair2;

	     if ( CurrMod != CurrModTmp ) {
	       cCluster.SetUserModules( mod_pair2 ) ;
	       CurrMod = CurrModTmp;
	     }

	     cCluster.SetGenericVariable( "cHits", mod_pair2, 1 );
	     cCluster.SetGenericVariable( "cSize", mod_pair2, Summaryinfo.clusterSize() );
	     cCluster.SetGenericVariable( "cCharge", mod_pair2, Summaryinfo.charge() );

	   }
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

     ModuleSelectionVectPixels.clear();
     for ( std::vector<std::string>::iterator it=v_pixelModuleTypes.begin() ; it < v_pixelModuleTypes.end(); it++ ){
       ModuleSelectionVectPixels.push_back ( new ClusterSummary::ModuleSelection(*it) );
     }

     firstpassPixel_mod = true;

     int CurrModPixel = -1;
     //Loop over all the ModuleSelectors
     for ( unsigned int i = 0; i < ModuleSelectionVectPixels.size(); i++){

       // Loop over the pixel clusters
       edmNew::DetSetVector<SiPixelCluster>::const_iterator itClusters=pixelClusters->begin();
       for(;itClusters!=pixelClusters->end();++itClusters){
	 uint32_t detid = itClusters->detId();    
	 for(edmNew::DetSet<SiPixelCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){

	   // For each ModuleSelector, check if the detID belongs to a desired module. If so, update the summary information for that module
	 
	   std::pair<int, int> ModSelectPixel = ModuleSelectionVectPixels.at(i) -> IsPixelSelected( detid );
	   int mod_pair = ModSelectPixel.first;
	   int mod_pair2 = ModSelectPixel.second;
	   if ( mod_pair ){
	     if ( firstpassPixel_mod ) {
	       CurrModPixel = mod_pair2;
	       cCluster.SetUserModules( mod_pair2 ) ;
	     }
	     firstpassPixel_mod = false;
	     int CurrModTmp = mod_pair2;
	     if ( CurrModPixel != CurrModTmp ) {
	       cCluster.SetUserModules( mod_pair2 ) ;
	       CurrModPixel = CurrModTmp;
	     }

	     cCluster.SetGenericVariable( "pHits", mod_pair2, 1 );
	     cCluster.SetGenericVariable( "pSize", mod_pair2, cluster->size() );
	     cCluster.SetGenericVariable( "pCharge", mod_pair2, float(cluster->charge())/1000. );

	   }
	 }
       }
     }
   
   }


   //===================+++++++++++++========================
   //
   //                   Fill Producer
   //
   //===================+++++++++++++========================


   unsigned int n = 0;
   int n_pixel = 0;

   cCluster.PrepairGenericVariable( );
   
   std::vector<int> _mod = cCluster.GetUserModules( );
   for(std::vector<int>::iterator it = _mod.begin(); it != _mod.end(); ++it) {
     
     if ( n < v_stripModuleTypes.size() && doStrips ){
       if (verbose) std::cout << "n" << v_stripModuleTypes.at(n) <<", avg size, avg charge = "<< cCluster.GetGenericVariable( "cHits",*it ) << ", " << cCluster.GetGenericVariable( "cSize",*it )/cCluster.GetGenericVariable( "cHits",*it ) << ", "<< cCluster.GetGenericVariable( "cCharge",*it )/cCluster.GetGenericVariable( "cHits",*it )  << std::endl;
       delete ModuleSelectionVect[n];
     }
     else if (doPixels) {     
       if (verbose) {
	 std::cout << "n" << v_pixelModuleTypes.at(n_pixel) << ", avg size, avg charge = "<< cCluster.GetGenericVariable( "pHits",*it ) << ", " << cCluster.GetGenericVariable( "pSize",*it )/cCluster.GetGenericVariable( "pHits",*it ) << ", "<< cCluster.GetGenericVariable( "pCharge",*it )/cCluster.GetGenericVariable( "pHits",*it )  << std::endl;
       }
       delete ModuleSelectionVectPixels[n_pixel];
       ++n_pixel;
     } 
     ++n;
   }
   
   if (verbose) std::cout << "-------------------------------------------------------" << std::endl;
   
   firstpass = false;
   firstpassPixel = false;

   //Put the filled class into the producer
   std::auto_ptr<ClusterSummary> result(new ClusterSummary (cCluster) );
   iEvent.put( result );

   cCluster.ClearGenericVariable();

}


void 
ClusterSummaryProducer::beginStream(edm::StreamID)
{
   
  if (doStrips) decodeInput(v_stripModuleTypes,stripModules.c_str());
  if (doStrips) decodeInput(v_stripVariables,stripVariables.c_str());  
  if (doPixels) decodeInput(v_pixelModuleTypes,pixelModules.c_str());
  if (doPixels) decodeInput(v_pixelVariables,pixelVariables.c_str());  

  if (doStrips){
  if (verbose){
    std::cout << "+++++++++++++++++++++++++++++++ "  << std::endl;
    std::cout << "FOR STRIPS: "  << std::endl;
    std::cout << "Getting info on " ;
    for (unsigned int ii = 0; ii < v_stripModuleTypes.size( ); ++ii) {
      std::cout << v_stripModuleTypes[ii] << " " ;
    }
    std::cout << std::endl;
  }
  
  if (verbose) std::cout << "Getting info on strip variables " ;
  for (unsigned int ii = 0; ii < v_stripVariables.size( ); ++ii) {
    if (verbose) std::cout << v_stripVariables[ii] << " " ;
    v_userContent.push_back(v_stripVariables[ii]);
  }
  if (verbose) std::cout << std::endl;
  }
  
  if (doPixels){
  if (verbose){
    std::cout << "FOR PIXELS: " << std::endl;
    std::cout << "Getting info on " ;
    for (unsigned int ii = 0; ii < v_pixelModuleTypes.size( ); ++ii) {
      std::cout << v_pixelModuleTypes[ii] << " " ;
    }
    std::cout << std::endl;
  }
  
  if (verbose) std::cout << "Getting info on pixel variables " ;
  for (unsigned int ii = 0; ii < v_pixelVariables.size( ); ++ii) {
    if (verbose) std::cout << v_pixelVariables[ii] << " " ;
    v_userContent.push_back(v_pixelVariables[ii]);
  }
  if (verbose) std::cout << std::endl;
  if (verbose) std::cout << "+++++++++++++++++++++++++++++++ "  << std::endl;
  }

  //Create the summary info for output 
  cCluster.SetUserContent(v_userContent);
  cCluster.SetUserIterator();
}




void 
ClusterSummaryProducer::decodeInput(std::vector<std::string> & vec, std::string mod)
{

  // Define the Modules to get the summary info out of  
  std::string::size_type i = 0;
  std::string::size_type j = mod.find(',');

  if ( j == std::string::npos ){
    vec.push_back(mod);
  }
  else{
    while (j != std::string::npos) {
      vec.push_back(mod.substr(i, j-i));
      i = ++j;
      j = mod.find(',', j);
      if (j == std::string::npos)
	vec.push_back(mod.substr(i, mod.length( )));
    }
  }


}



#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterSummaryProducer);


