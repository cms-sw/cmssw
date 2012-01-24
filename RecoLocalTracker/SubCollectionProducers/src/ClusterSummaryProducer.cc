#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterSummaryProducer.h"

ClusterSummaryProducer::ClusterSummaryProducer(const edm::ParameterSet& iConfig)
  : ClustersLabel(iConfig.getParameter<edm::InputTag>("Clusters")),
    modules(iConfig.getParameter<std::string>("Module")),
    variables(iConfig.getParameter<std::string>("Variables"))
{
 
  //register your products
  produces<ClusterSummary>().setBranchAlias("SummaryCollection");
  
  verbose = true;
  firstpass = true;
  firstpass_mod = true;

}


void
ClusterSummaryProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
      
   edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
   iEvent.getByLabel(ClustersLabel, clusters);

   ModuleSelectionVect.clear();
   for ( std::vector<std::string>::iterator it=v_moduleTypes.begin() ; it < v_moduleTypes.end(); it++ ){
     ModuleSelectionVect.push_back ( new ClusterSummary::ModuleSelection(*it) );
   }
 
   firstpass_mod = true;

   cCluster.ClearUserModules();
   //cCluster.ClearUserVariables();
   cCluster.ClearAllVariables();

   int CurrMod = -1;
   //Loop over all the ModuleSelectors
   for ( unsigned int i = 0; i < ModuleSelectionVect.size(); i++){
     
     // Loop over the clusters
     edmNew::DetSetVector<SiStripCluster>::const_iterator itClusters=clusters->begin();
     for(;itClusters!=clusters->end();++itClusters){
       uint32_t id = itClusters->id();
       for(edmNew::DetSet<SiStripCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){

	 const ClusterVariables Summaryinfo(*cluster);
      
	 // For each ModuleSelector, check if the detID belongs to a desired module. If so, update the summary information for that module

	 std::pair<int, int> ModSelect = ModuleSelectionVect.at(i) -> IsSelected( id );
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
	  
	   cCluster.SetNType( mod_pair2 );
	   cCluster.SetClusterSize( mod_pair2, Summaryinfo.clusterSize() );
	   cCluster.SetClusterCharge( mod_pair2, Summaryinfo.charge() );

	   cCluster.SetGenericVariable( "cHits", mod_pair2, 1 );
	   cCluster.SetGenericVariable( "cSize", mod_pair2, Summaryinfo.clusterSize() );
	   cCluster.SetGenericVariable( "cCharge", mod_pair2, Summaryinfo.charge() );

	 }
       }
     }
   }


   int n = 0;
   std::vector<int> _mod = cCluster.GetUserModules( );
   for(std::vector<int>::iterator it = _mod.begin(); it != _mod.end(); ++it) {

     cCluster.SetUserVariables( *it );  
    
     if ( verbose ) std::cout << "n" << v_moduleTypes.at(n) <<", avg size, avg charge = "<< cCluster.GetNType( *it ) << ", " << cCluster.GetAverageClusterSize( *it ) << ", "<< cCluster.GetAverageClusterCharge( *it )  << std::endl;

     cCluster.ClearNType( *it );
     cCluster.ClearClusterSize( *it );
     cCluster.ClearClusterCharge( *it );

     delete ModuleSelectionVect[n];
     ++n;
     
   }
   firstpass = false;

   //Put the filled class into the producer
   std::auto_ptr<ClusterSummary> result(new ClusterSummary (cCluster) );
   iEvent.put( result );

   cCluster.ClearGenericVariable();

}


void 
ClusterSummaryProducer::beginJob()
{

  // Define the Modules to get the summary info out of  
  std::string mod = modules.c_str();
  std::string::size_type i = 0;
  std::string::size_type j = mod.find(',');

  // Define the Variables to get the summary info out of  
  std::string var = variables.c_str();
  std::string::size_type var_i = 0;
  std::string::size_type var_j = var.find(',');


  if ( j == std::string::npos ){
      v_moduleTypes.push_back(mod);
    }
  else{
    while (j != std::string::npos) {
      v_moduleTypes.push_back(mod.substr(i, j-i));
      i = ++j;
      j = mod.find(',', j);
      if (j == std::string::npos)
	v_moduleTypes.push_back(mod.substr(i, mod.length( )));
    }
  }


  if ( var_j == std::string::npos ){
      v_variables.push_back(var);
    }
  else{
    while (var_j != std::string::npos) {
      v_variables.push_back(var.substr(var_i, var_j-var_i));
      var_i = ++var_j;
      var_j = var.find(',', var_j);
      if (var_j == std::string::npos)
	v_variables.push_back(var.substr(var_i, var.length( )));
    }
  }

  
  std::cout << "Getting info on " ;
  for (unsigned int ii = 0; ii < v_moduleTypes.size( ); ++ii) {
    std::cout << v_moduleTypes[ii] << " " ;
  }
  std::cout << std::endl;

  std::cout << "Getting info on variables " ;
  for (unsigned int ii = 0; ii < v_variables.size( ); ++ii) {
    std::cout << v_variables[ii] << " " ;
    v_userContent.push_back(v_variables[ii]);
  }
  std::cout << std::endl;


  //Create the summary info for output 
  //v_userContent.push_back("hits");
  //v_userContent.push_back("size");
  //v_userContent.push_back("charge");
  cCluster.SetUserContent(v_userContent);
  cCluster.SetUserIterator();

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterSummaryProducer);
