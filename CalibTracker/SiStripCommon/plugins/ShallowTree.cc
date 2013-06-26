#include "CalibTracker/SiStripCommon/interface/ShallowTree.h"

#include "FWCore/Framework/interface/ConstProductRegistry.h" 
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "DataFormats/Provenance/interface/Selections.h"

#include <map>
#include "boost/foreach.hpp"
#include <TBranch.h>

void ShallowTree::
analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  BOOST_FOREACH( BranchConnector* connector, connectors)
    connector->connect(iEvent);
  tree->Fill();
}

template <class T>
void ShallowTree::TypedBranchConnector<T>::
connect(const edm::Event& iEvent) {
  edm::Handle<T> handle_;
  iEvent.getByLabel(ml, pin, handle_);
  object_ = *handle_;
}

template <class T> 
ShallowTree::TypedBranchConnector<T>::
TypedBranchConnector(edm::BranchDescription const* desc, 
		     std::string t, 
		     TTree * tree)
  :  ml( desc->moduleLabel() ),  
     pin( desc->productInstanceName() )
{
  object_ptr_ = &object_;  
  std::string s=pin+t;  
  if(t!="")  { tree->Branch(pin.c_str(),  object_ptr_, s.c_str() );}  //raw type
  else       { tree->Branch(pin.c_str(), &object_ptr_            );}  //vector<type>
}

void ShallowTree::
beginJob() {
  tree = fs->make<TTree>("tree", ""); 

  std::map<std::string, LEAFTYPE> leafmap;
  leafmap["bool"]      = BOOL;       leafmap["bools"]     = BOOL_V;
  leafmap["short int"] = SHORT;      leafmap["shorts"]    = SHORT_V;
  leafmap["ushort int"]= U_SHORT;    leafmap["ushorts"]   = U_SHORT_V;
  leafmap["int"]       = INT;        leafmap["ints"]      = INT_V;
  leafmap["uint"]      = U_INT;      leafmap["uints"]     = U_INT_V;
  leafmap["float"]     = FLOAT;      leafmap["floats"]    = FLOAT_V;
  leafmap["double"]    = DOUBLE;     leafmap["doubles"]   = DOUBLE_V;
  leafmap["lint"]      = LONG;       leafmap["longs"]     = LONG_V;
  leafmap["ulint"]     = U_LONG;     leafmap["ulongs"]    = U_LONG_V;
  leafmap["char"]      = CHAR;       leafmap["chars"]     = CHAR_V;
  leafmap["uchar"]     = U_CHAR;     leafmap["uchars"]    = U_CHAR_V;


  edm::Service<edm::ConstProductRegistry> reg;
  edm::Selections allBranches = reg->allBranchDescriptions();
  edm::ProductSelectorRules productSelectorRules_(pset, "outputCommands", "ShallowTree");
  edm::ProductSelector productSelector_;
  productSelector_.initialize(productSelectorRules_, allBranches);

  std::set<std::string> branchnames;

  BOOST_FOREACH( const edm::Selections::value_type& selection, allBranches) {
    if(productSelector_.selected(*selection)) {

      //Check for duplicate branch names
      if (branchnames.find( selection->productInstanceName()) != branchnames.end() ) {
	throw edm::Exception(edm::errors::Configuration)
	  << "More than one branch named: "
	  << selection->productInstanceName() << std::endl
	  << "Exception thrown from ShallowTree::beginJob" << std::endl;
      }
      else {
	branchnames.insert( selection->productInstanceName() );
      }

      //Create ShallowTree branch
      switch(leafmap.find( selection->friendlyClassName() )->second) {
      case BOOL     :  connectors.push_back( new TypedBranchConnector                      <bool>  (selection, "/O", tree) ); break;
      case BOOL_V   :  connectors.push_back( new TypedBranchConnector<std::vector          <bool> >(selection,   "", tree) ); break;
      case INT      :  connectors.push_back( new TypedBranchConnector                       <int>  (selection, "/I", tree) ); break;
      case INT_V    :  connectors.push_back( new TypedBranchConnector<std::vector           <int> >(selection,   "", tree) ); break;
      case U_INT    :  connectors.push_back( new TypedBranchConnector              <unsigned int>  (selection, "/i", tree) ); break;
      case U_INT_V  :  connectors.push_back( new TypedBranchConnector<std::vector  <unsigned int> >(selection,   "", tree) ); break;
      case SHORT    :  connectors.push_back( new TypedBranchConnector                     <short>  (selection, "/S", tree) ); break;
      case SHORT_V  :  connectors.push_back( new TypedBranchConnector<std::vector         <short> >(selection,   "", tree) ); break;
      case U_SHORT  :  connectors.push_back( new TypedBranchConnector            <unsigned short>  (selection, "/s", tree) ); break;
      case U_SHORT_V:  connectors.push_back( new TypedBranchConnector<std::vector<unsigned short> >(selection,   "", tree) ); break;
      case FLOAT    :  connectors.push_back( new TypedBranchConnector                     <float>  (selection, "/F", tree) ); break;
      case FLOAT_V  :  connectors.push_back( new TypedBranchConnector<std::vector         <float> >(selection,   "", tree) ); break;
      case DOUBLE   :  connectors.push_back( new TypedBranchConnector                    <double>  (selection, "/D", tree) ); break;
      case DOUBLE_V :  connectors.push_back( new TypedBranchConnector<std::vector        <double> >(selection,   "", tree) ); break;
      case LONG     :  connectors.push_back( new TypedBranchConnector                      <long>  (selection, "/L", tree) ); break;
      case LONG_V   :  connectors.push_back( new TypedBranchConnector<std::vector          <long> >(selection,   "", tree) ); break;
      case U_LONG   :  connectors.push_back( new TypedBranchConnector             <unsigned long>  (selection, "/l", tree) ); break;
      case U_LONG_V :  connectors.push_back( new TypedBranchConnector<std::vector <unsigned long> >(selection,   "", tree) ); break;
      case CHAR     :  connectors.push_back( new TypedBranchConnector                      <char>  (selection, "/B", tree) ); break;
      case CHAR_V   :  connectors.push_back( new TypedBranchConnector<std::vector          <char> >(selection,   "", tree) ); break;
      case U_CHAR   :  connectors.push_back( new TypedBranchConnector             <unsigned char>  (selection, "/b", tree) ); break;
      case U_CHAR_V :  connectors.push_back( new TypedBranchConnector<std::vector <unsigned char> >(selection,   "", tree) ); break;
      default: 
	{
	  std::string leafstring = "";
	  typedef std::pair<std::string, LEAFTYPE> pair_t;
	  BOOST_FOREACH( const pair_t& leaf, leafmap) 
	    leafstring+= "\t" + leaf.first + "\n";

	  throw edm::Exception(edm::errors::Configuration)
	    << "class ShallowTree does not handle leaves of type " << selection->className() << " like\n"
	    <<   selection->friendlyClassName()   << "_" 
	    <<   selection->moduleLabel()         << "_" 
	    <<   selection->productInstanceName() << "_"  
	    <<   selection->processName()         << std::endl
	    << "Valid leaf types are (friendlyClassName):\n"
	    <<   leafstring
	    << "Exception thrown from ShallowTree::beginJob\n";
	}
      }
    }
  }
}

