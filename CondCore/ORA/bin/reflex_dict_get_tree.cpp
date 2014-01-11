#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
//
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"

//
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>

#ifdef AP_NOT_FOR_RIGHT_NOW

static std::string const prefix("LCGReflex/");

edm::TypeWithDict resolvedType(const edm::TypeWithDict& typ){
  edm::TypeWithDict resolvedType = typ;
  while(resolvedType.IsTypedef()){
    resolvedType = resolvedType.ToType();
  }
  return resolvedType;
}

bool isSTLContainer( const std::string& contName ){
  if(  contName == "std::vector"              ||
       contName == "std::list"                ||
       contName == "std::set"                 ||
       contName == "std::multiset"            ||
       contName == "std::deque"               ||
       contName == "__gnu_cxx::hash_set"      ||
       contName == "__gnu_cxx::hash_multiset" ||
       contName == "std::map"                 ||
       contName == "std::multimap"            ||
       contName == "__gnu_cxx::hash_map"      ||
       contName == "__gnu_cxx::hash_multimap" ||
       contName == "std::stack"               ||
       contName == "std::queue"               ||
       contName == "std::bitset"              ||
       contName == "std::pair"                ||
       contName == "std::tuple"){
    return true;
  }
  return false;
}

bool isBasic( const edm::TypeWithDict& typ ){
  return ( typ.IsFundamental() || 
	   typ.IsEnum() || 
	   typ.Name(Reflex::SCOPED|Reflex::FINAL) == "std::string" ||
	   typ.Name(Reflex::SCOPED|Reflex::FINAL) == "std::basic_string<char>" );
}

void processType( const edm::TypeWithDict& t, 
		  std::map<std::string,std::pair<std::set<std::string>,std::set<std::string> > >& outList, 
		  std::set<std::string>& doneList ){
  edm::TypeWithDict objType = t;
  std::string className = objType.Name( Reflex::SCOPED|Reflex::FINAL );
  auto iD = doneList.find( className );
  if( iD != doneList.end() ) return;

  doneList.insert( className );

  std::set<std::string> bases;
  std::set<std::string> members;

  while( objType.isArray() ) objType = objType.ToType();
  if( isBasic( objType ) ) return;
  edm::TypeWithDictTemplate templ = objType.TemplateFamily();
  if ( templ ) {
    className = templ.Name(Reflex::SCOPED|Reflex::FINAL);
    if( isSTLContainer( className ) ) return;
  }

  for ( size_t i=0;i<objType.BaseSize();i++){
    Reflex::Base base = objType.BaseAt(i);
    edm::TypeWithDict baseType = resolvedType( base.ToType() );
    if( !baseType ) std::cout <<"Type for base "<<base.Name()<<" of class "<<className<<" is unkown and will be skipped."<<std::endl;
    //if( !baseType ) throw std::runtime_error("Type for one base is unknown");
    if( baseType ) processType( baseType, outList, doneList );

    std::string baseName = baseType.Name( Reflex::SCOPED|Reflex::FINAL);
    edm::TypeWithDictTemplate baseTempl = baseType.TemplateFamily();
    if ( baseTempl ) {
      baseName = baseTempl.Name(Reflex::SCOPED|Reflex::FINAL);
    }
    bases.insert( baseName );
  }
  for ( size_t i=0;i< objType.DataMemberSize();i++){
    Reflex::Member dataMember = objType.DataMemberAt(i);
    if ( dataMember.IsTransient() || dataMember.IsStatic() ) continue;
    edm::TypeWithDict dataMemberType = resolvedType( dataMember.TypeOf() );
    if( !dataMemberType ) std::cout <<"Type for data member "+dataMember.Name()+" of class "<<className<<" is unknown and will be skipped"<<std::endl;
    //if( !dataMemberType ) throw std::runtime_error("Type for data member "+dataMember.Name()+" is unknown");
    if( dataMemberType ) processType( dataMemberType, outList, doneList );
    members.insert( dataMember.Name() );
  }

  auto iOut = outList.find( className );
  if( iOut == outList.end() ) {
    outList.insert( std::make_pair( className, std::make_pair(bases,members) ) );
  }
} 

int main ( int argc, char *argv[] )
{
  if ( argc != 2 ) 
    std::cout<<"Usage: "<< argv[0] <<" <class list filename>"<<std::endl;
  else {

    edmplugin::PluginManager::Config config;
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    std::string line;
    std::ifstream inputFile ( argv[1] );
    if (inputFile.is_open()){
      std::map<std::string,std::pair<std::set<std::string>, std::set<std::string> > > outList;
      std::set<std::string> doneList;
      while ( getline (inputFile,line) ){
	if(line.empty()) continue;
	std::cout <<"Processing class "<<line << std::endl;
	edmplugin::PluginCapabilities::get()->load(prefix + line);
	edm::TypeWithDict t = edm::TypeWithDict::ByName( line );
	if( ! t ) throw std::runtime_error("Class "+line+" is not known by the dictionary");
	processType( t, outList, doneList );
      }
      inputFile.close();
      std::ofstream outputFile;
      outputFile.open( "classes.out" );
      outputFile <<"{"<<std::endl;
      bool outerFirst = true;
      for( auto c : outList ){
        if( !outerFirst ) { 
	  outputFile <<",";
	  outputFile << std::endl;
	  //outputFile << std::endl;
	}
	outputFile <<"  \""<< c.first <<"\":[["<<std::endl;
	bool innerFirst = true;
	for( auto b : c.second.first ){
	  if( !innerFirst ){ 
	    outputFile <<",";
	    outputFile<< std::endl;
	  }
	  outputFile << "    \""<< b << "\"";
	  innerFirst = false;
	}
        if(!innerFirst) outputFile <<std::endl;
        outputFile <<"  ], [";
	outputFile << std::endl;
	innerFirst = true;
	for( auto m : c.second.second ){
	  if( !innerFirst ){ 
	    outputFile <<",";
	    outputFile<< std::endl;
	  }
	  outputFile << "    \""<< m << "\"";
	  innerFirst = false;
	}
        if(!innerFirst) outputFile <<std::endl;
        outputFile <<"  ]]";
        outerFirst = false;
      }
      outputFile <<std::endl;
      outputFile << "}"<<std::endl;
      outputFile.close();
    }
  }
}
#else // AP_NOT_FOR_RIGHT_NOW

int main()
{
  return 0;
}

#endif // defined AP_NOT_FOR_RIGHT_NOW
