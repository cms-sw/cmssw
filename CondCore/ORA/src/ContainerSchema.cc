#include "CondCore/ORA/interface/Configuration.h"
#include "CondCore/ORA/interface/Exception.h"
#include "ContainerSchema.h"
#include "DatabaseSession.h"
#include "IDatabaseSchema.h"
#include "MappingToSchema.h"
#include "MappingDatabase.h"
#include "MappingGenerator.h"
#include "MappingRules.h"
#include "ClassUtils.h"
// externals
#include "RelationalAccess/ISchema.h"

namespace ora {

  void getTableHierarchyFromMappingElement( const MappingElement& source,
                                            std::map<std::string, std::set<std::string> >& tableList ){
    const std::string& tableName = source.tableName();
    std::map<std::string, std::set<std::string> >::iterator iTab = tableList.find( tableName );
    if( iTab ==tableList.end() ){
      std::set<std::string> dependencies;
      tableList.insert(std::make_pair( tableName, dependencies ) );
    }
    for( MappingElement::const_iterator iElem = source.begin();
         iElem != source.end(); iElem++ ){
      std::map<std::string, std::set<std::string> >::iterator iT = tableList.find( tableName );
      const std::string& innerTable = iElem->second.tableName();
      if( innerTable != tableName ){
        iT->second.insert( innerTable );
      }
      getTableHierarchyFromMappingElement( iElem->second, tableList );
    }
  }
  
  //void getTableHierarchyFromMapping( const MappingTree& source,
  //                                   std::map<std::string,
  //                                   std::set<std::string> >& tableList ){
  //  getTableHierarchyFromMappingElement( source.element(), tableList );
  //  for( std::map<std::string, MappingElement >::const_iterator iDepEl = source.dependentElements().begin();
  //       iDepEl != source.dependentElements().end(); iDepEl++ ){
  //    getTableHierarchyFromMappingElement( iDepEl->second, tableList );
  //  }
  //}

  void addFromTableHierarchy( const std::string& tableName,
                              std::map<std::string, std::set<std::string> >& tableList,
                              std::vector<std::string>& orderedList ){
    orderedList.push_back( tableName );
    std::map<std::string, std::set<std::string> >::const_iterator iDeps = tableList.find( tableName );
    if(iDeps != tableList.end() ){
      for( std::set<std::string>::const_iterator iDt = iDeps->second.begin();
           iDt != iDeps->second.end(); iDt++ ){
        addFromTableHierarchy( *iDt, tableList, orderedList );
      }
    }
  }

}

ora::ContainerSchema::ContainerSchema( int containerId,
                                       const std::string& containerName,
                                       const Reflex::Type& containerType,
                                       DatabaseSession& session ):
  m_containerId( containerId ),
  m_containerName( containerName ),
  m_className( containerType.Name( Reflex::SCOPED ) ),
  m_classDict( containerType ),
  m_session( session ),
  m_loaded( false ),
  m_containerSchemaSequences( session.schema() ),
  m_mapping(),
  m_dependentMappings(){
}

ora::ContainerSchema::ContainerSchema( int containerId,
                                       const std::string& containerName,
                                       const std::string& className,
                                       DatabaseSession& session ):
  m_containerId( containerId ),
  m_containerName( containerName ),
  m_className( className ),
  m_classDict(),
  m_session( session ),
  m_loaded( false ),
  m_containerSchemaSequences( session.schema() ),
  m_mapping(),
  m_dependentMappings(){
}

ora::ContainerSchema::~ContainerSchema(){
  for( std::map<std::string,MappingTree*>::iterator iDep = m_dependentMappings.begin();
       iDep != m_dependentMappings.end(); ++iDep ){
    delete iDep->second;
  }
}

void ora::ContainerSchema::initClassDict(){
  if( !m_classDict ) m_classDict = ClassUtils::lookupDictionary( m_className, false );
  if( !m_classDict ) throwException("Container class \""+m_className+"\" has not been found in the dictionary.",
                                    "ContainerSchema::initClassDict");
}

void ora::ContainerSchema::create(){

  initClassDict();
  std::string newMappingVersion = m_session.mappingDatabase().newMappingVersionForContainer( m_containerName );
  MappingGenerator mapGen( m_session.schema().storageSchema() );
  mapGen.createNewMapping( m_containerName, m_classDict, m_mapping );
  //if(!mapGen.createNewMapping( m_containerName, m_classDict, m_mapping )){
  //  throwException("Mapping generation failed.",
  //                 "ContainerSchema::create");
  //}
  m_mapping.setVersion( newMappingVersion );
  m_session.mappingDatabase().storeMapping( m_mapping );
  m_session.mappingDatabase().insertClassVersion( m_classDict, 0, m_containerId, newMappingVersion, true );
  MappingToSchema mapping2Schema( m_session.schema().storageSchema() );
  m_mapping.tables();
  mapping2Schema.create(  m_mapping );
  m_containerSchemaSequences.create( MappingRules::sequenceNameForContainer( m_containerName ) );
  for( std::map<std::string,MappingTree*>::iterator iDep = m_dependentMappings.begin();
       iDep != m_dependentMappings.end(); ++iDep ){
    m_containerSchemaSequences.create( MappingRules::sequenceNameForDependentClass( m_containerName, iDep->first ));
  }
  m_loaded = true;
}

void ora::ContainerSchema::drop(){

  std::set<std::string> containerMappingVersions;
  if( m_session.mappingDatabase().getMappingVersionsForContainer( m_containerId, containerMappingVersions )){
    std::map< std::string, std::set<std::string> > tableHierarchy;
    std::set<std::string> topLevelTables; // should be strictly only one!
    for( std::set<std::string>::const_iterator iV = containerMappingVersions.begin();
         iV!= containerMappingVersions.end(); ++iV ){
      MappingTree mapping;
      if( m_session.mappingDatabase().getMappingByVersion( *iV, mapping ) ){
        topLevelTables.insert( mapping.topElement().tableName() );
        getTableHierarchyFromMappingElement( mapping.topElement(), tableHierarchy );
      }
    }

    std::vector<std::string> orderedTableList;
    for(std::set<std::string>::const_iterator iMainT = topLevelTables.begin();
        iMainT != topLevelTables.end(); ++iMainT ){
      addFromTableHierarchy( *iMainT, tableHierarchy, orderedTableList );
    }
    for(std::vector<std::string>::reverse_iterator iTable = orderedTableList.rbegin();
        iTable != orderedTableList.rend(); iTable++ ){
      m_session.schema().storageSchema().dropIfExistsTable( *iTable );
    }
  }
  std::set<std::string> depClasses;
  m_session.mappingDatabase().getDependentClassesForContainer( m_containerId, depClasses );
      
  for( std::set<std::string>::const_iterator iM = containerMappingVersions.begin();
       iM != containerMappingVersions.end(); ++iM ){
    m_session.mappingDatabase().removeMapping( *iM );
  }
  m_containerSchemaSequences.erase( MappingRules::sequenceNameForContainer( m_containerName ));
  for(std::set<std::string>::const_iterator iDepCl = depClasses.begin();
      iDepCl != depClasses.end(); iDepCl++){
    m_containerSchemaSequences.erase( MappingRules::sequenceNameForDependentClass( m_containerName, *iDepCl ) );
  }
}

void ora::ContainerSchema::evolve(){
  MappingGenerator mapGen( m_session.schema().storageSchema() );
  // retrieve the base mapping
  MappingTree baseMapping;
  if( !m_session.mappingDatabase().getBaseMappingForContainer( m_classDict.Name(Reflex::SCOPED), m_containerId, baseMapping )){
    throwException("Base mapping has not been found in the database.",
                   "ContainerSchema::evolve");
  }
  mapGen.createNewMapping( m_containerName, m_classDict, baseMapping,  m_mapping );
  std::string newMappingVersion = m_session.mappingDatabase().newMappingVersionForContainer( m_containerName );
  m_mapping.setVersion( newMappingVersion );
  m_session.mappingDatabase().storeMapping( m_mapping );
  m_session.mappingDatabase().insertClassVersion( m_classDict, 0, m_containerId, newMappingVersion );
  MappingToSchema mapping2Schema( m_session.schema().storageSchema() );
  mapping2Schema.alter(  m_mapping  );
  m_loaded = true;
}

void ora::ContainerSchema::evolve( const Reflex::Type& dependentClass, MappingTree& baseMapping ){
  std::string className = dependentClass.Name(Reflex::SCOPED);
  MappingGenerator mapGen( m_session.schema().storageSchema() );
  std::map<std::string,MappingTree*>::iterator iDep =
    m_dependentMappings.insert( std::make_pair( className, new MappingTree ) ).first;
  if( baseMapping.className() != dependentClass.Name(Reflex::SCOPED) ){
    throwException("Provided base mapping does not map class \""+dependentClass.Name(Reflex::SCOPED)+"\".",
                   "ContainerSchema::evolve");    
  }
  mapGen.createNewDependentMapping( dependentClass, m_mapping, baseMapping, *iDep->second );
  std::string newMappingVersion = m_session.mappingDatabase().newMappingVersionForContainer( m_containerName );
  iDep->second->setVersion( newMappingVersion );
  m_session.mappingDatabase().storeMapping( *iDep->second );
  m_session.mappingDatabase().insertClassVersion( dependentClass, 1, m_containerId, newMappingVersion, false );
}

const Reflex::Type& ora::ContainerSchema::type(){
  return m_classDict;
}

ora::MappingTree& ora::ContainerSchema::mapping( bool writeEnabled ){
  initClassDict();
  if(!m_loaded ){
    if( !m_session.mappingDatabase().getMappingForContainer( m_classDict, m_containerId, m_mapping ) ){
      // if enabled, invoke the evolution
      if( writeEnabled && m_session.configuration().properties().getFlag( Configuration::automaticSchemaEvolution() )){
        evolve();
      }
    } else {
      m_loaded = true;
    }
    
  }
  if( m_mapping.topElement().find( m_className )==m_mapping.topElement().end() ){
    throwException( "Mapping for container class \""+m_className+"\" could not be loaded.",
                    "ContainerSchema::mapping");
  }
  return m_mapping;
}

bool ora::ContainerSchema::loadMappingForDependentClass( const Reflex::Type& dependentClassDict ){
  if( !dependentClassDict ) throwException("The dependent class has not been found in the dictionary.",
                                    "ContainerSchema::loadMappingForDependentClass");
  std::string className = dependentClassDict.Name(Reflex::SCOPED);
  std::map<std::string,MappingTree*>::iterator iDep = m_dependentMappings.find( className );
  if( iDep ==  m_dependentMappings.end() ){
    // not in cache, search the database...
    iDep = m_dependentMappings.insert( std::make_pair( className, new MappingTree ) ).first;
    if( ! m_session.mappingDatabase().getMappingForContainer( dependentClassDict, m_containerId, *iDep->second ) ){
      m_dependentMappings.erase( className );
      return false;
    }
  }
  return true;  
}

void ora::ContainerSchema::create( const Reflex::Type& dependentClassDict ){
  std::string className = dependentClassDict.Name(Reflex::SCOPED);
  std::map<std::string,MappingTree*>::iterator iDep =
    m_dependentMappings.insert( std::make_pair( className, new MappingTree ) ).first;
  MappingGenerator mapGen( m_session.schema().storageSchema() );
  MappingToSchema mapping2Schema( m_session.schema().storageSchema() );
  mapGen.createNewDependentMapping( dependentClassDict, m_mapping, *iDep->second );
  mapping2Schema.create(  *iDep->second  );
  std::string newMappingVersion = m_session.mappingDatabase().newMappingVersionForContainer( m_containerName );
  iDep->second->setVersion( newMappingVersion );
  m_session.mappingDatabase().storeMapping( *iDep->second );
  m_session.mappingDatabase().insertClassVersion( dependentClassDict, 1, m_containerId, newMappingVersion, true );
  m_containerSchemaSequences.create( MappingRules::sequenceNameForDependentClass( m_containerName, className ));
}

void ora::ContainerSchema::extend( const Reflex::Type& dependentClassDict ){
  std::string className = dependentClassDict.Name(Reflex::SCOPED);
  MappingTree baseMapping;
  if( !m_session.mappingDatabase().getBaseMappingForContainer( className,
                                                               m_containerId, baseMapping ) ){
    create( dependentClassDict );
  } else {
    evolve( dependentClassDict, baseMapping );
  }
}

bool ora::ContainerSchema::extendIfRequired( const Reflex::Type& dependentClassDict ){
  bool ret = false;
  if( ! loadMappingForDependentClass( dependentClassDict ) ){
    extend( dependentClassDict );
    ret = true;
  }
  return ret;
}

ora::MappingElement& ora::ContainerSchema::mappingForDependentClass( const Reflex::Type& dependentClassDict,
                                                                     bool writeEnabled ){
  std::string className = dependentClassDict.Name(Reflex::SCOPED);
  if( ! loadMappingForDependentClass( dependentClassDict ) ){
    if( writeEnabled ){
      // check if a base is available:
      MappingTree baseMapping;
      if( !m_session.mappingDatabase().getBaseMappingForContainer( className,
                                                                   m_containerId, baseMapping ) ){
        // mapping has to be generated from scratch
        if( m_session.configuration().properties().getFlag( Configuration::automaticDatabaseCreation()) ||
            m_session.configuration().properties().getFlag( Configuration::automaticContainerCreation() ) ){
          create( dependentClassDict );
        }
      } else {
        // evolve if allowed
        if( m_session.configuration().properties().getFlag( Configuration::automaticSchemaEvolution() )){
          evolve( dependentClassDict, baseMapping );
        }
      }
    }
  }
  std::map<std::string,MappingTree*>::iterator iDep = m_dependentMappings.find( className );
  if( iDep ==  m_dependentMappings.end() ){
    throwException( "Mapping for class \""+ className + "\" is not available in the database.",
                    "ContainerSchema::mappingForDependentClass");
  }
  return iDep->second->topElement();
}

bool ora::ContainerSchema::mappingForDependentClasses( std::vector<ora::MappingElement>& destination ){
  return m_session.mappingDatabase().getDependentMappingsForContainer( m_containerId, destination );
}

ora::Sequences& ora::ContainerSchema::containerSequences(){
  return m_containerSchemaSequences;
}

ora::IBlobStreamingService* ora::ContainerSchema::blobStreamingService(){
  return m_session.configuration().blobStreamingService();
}

ora::IReferenceHandler* ora::ContainerSchema::referenceHandler(){
  return m_session.configuration().referenceHandler();
}
  
const std::string& ora::ContainerSchema::mappingVersion(){
  return m_mapping.version();
}

int ora::ContainerSchema::containerId(){
  return m_containerId;
}

const std::string&  ora::ContainerSchema::containerName(){
  return m_containerName;
}

const std::string&  ora::ContainerSchema::className(){
  return m_className;
}

coral::ISchema& ora::ContainerSchema::storageSchema(){
  return m_session.schema().storageSchema();
}

ora::DatabaseSession& ora::ContainerSchema::dbSession(){
  return m_session;
}



