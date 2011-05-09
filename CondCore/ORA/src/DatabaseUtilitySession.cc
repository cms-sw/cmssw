#include "CondCore/ORA/interface/Exception.h"
#include "DatabaseUtilitySession.h"
#include "DatabaseSession.h"
#include "DatabaseContainer.h"
#include "MappingRules.h"
#include "MappingDatabase.h"
#include "MappingToSchema.h"
#include "IDatabaseSchema.h"
#include "MappingTree.h"
// externals
#include "Reflex/Type.h"

ora::DatabaseUtilitySession::DatabaseUtilitySession(  DatabaseSession& dbSession ):
  m_session( dbSession ){
}

ora::DatabaseUtilitySession::~DatabaseUtilitySession(){
}

std::set<std::string> ora::DatabaseUtilitySession::listMappingVersions( int containerId ){
  std::set<std::string> mappingList;
  m_session.mappingDatabase().getMappingVersionsForContainer( containerId, mappingList );
  return mappingList;
}

std::map<std::string,std::string> ora::DatabaseUtilitySession::listMappings( int containerId ){
  std::map<std::string,std::string> versionMap;
  m_session.mappingDatabase().getClassVersionListForContainer( containerId, versionMap );
  return versionMap;
}

bool ora::DatabaseUtilitySession::dumpMapping( const std::string& mappingVersion,
                                               std::ostream& outputStream ){
  MappingTree dest;
  if(m_session.mappingDatabase().getMappingByVersion( mappingVersion, dest )){
    dest.printXML( outputStream );
    return true;
  }
  return false;
}

ora::Handle<ora::DatabaseContainer> ora::DatabaseUtilitySession::importContainerSchema( const std::string& containerName,
                                                                                        DatabaseSession& sourceSession){
  if(!m_session.exists()){
    if( m_session.configuration().properties().getFlag( Configuration::automaticDatabaseCreation() )){
      m_session.create();
      m_session.open();
    } else {
      throwException( "ORA Database not found in \""+m_session.connectionString()+"\".",
                      "DatabaseUtilitySession::importContainerSchema");      
    }
  } else {
    m_session.open();
    if( existsContainer( containerName ) ){
      throwException( "A Container named \""+containerName+"\" already exists in the database.",
                      "DatabaseUtilitySession::importContainerSchema" );
    }
  }
  sourceSession.open();
  Sequences containerSchemaSequences( m_session.schema() );
  Handle<ora::DatabaseContainer> cont = sourceSession.containerHandle( containerName );
  // first create the container locally:
  Handle<ora::DatabaseContainer> newCont = m_session.addContainer( containerName, cont->className() );
  cont->className();
  MappingToSchema mapping2Schema( m_session.schema().storageSchema() );
  std::set<std::string> existingVersions = m_session.mappingDatabase().versions();
  std::set<std::string> baseVersions;
  // first create the cont base schema
  MappingTree baseMapping;
  if(!sourceSession.mappingDatabase().getBaseMappingForContainer( cont->className(), cont->id(), baseMapping )){
    throwException( "Base mapping for class \""+cont->className()+"\" has not been found in the database.",
                    "DatabaseUtilitySession::importContainerSchema");
  }
  std::set<std::string> classVersions;
  if(!sourceSession.mappingDatabase().getClassVersionListForMappingVersion( baseMapping.version(), classVersions )){
    throwException( "No class versions found for mapping \""+baseMapping.version()+"\".",
                    "DatabaseUtilitySession::importContainerSchema");
  }
  if( existingVersions.find( baseMapping.version() )!= existingVersions.end() ){
    throwException("Mapping version \""+baseMapping.version()+"\" for base mapping of class \""+cont->className()+"\" already exists in the database.","DatabaseUtilitySession::importContainerSchema");
  }
  if( !mapping2Schema.check( baseMapping ) ){
    throwException("Schema base for class \""+baseMapping.className()+"\" cannot be replicated, because some schema objects have been found with the same name.","DatabaseUtilitySession::importContainerSchema");    
  }
  baseVersions.insert( baseMapping.version() );
  existingVersions.insert( baseMapping.version() );
  m_session.mappingDatabase().storeMapping( baseMapping );
  bool first = true;
  for( std::set<std::string>::const_iterator iCv = classVersions.begin(); iCv != classVersions.end(); ++iCv ){
    m_session.mappingDatabase().insertClassVersion( cont->className(), *iCv , 0, newCont->id(), baseMapping.version(), first );
    first = false;
  }
  mapping2Schema.create( baseMapping );
  // ...and the main container sequence
  containerSchemaSequences.create( MappingRules::sequenceNameForContainer( containerName )); 
  // second create the base dependencies if any
  std::set<std::string> dependentClasses;
  sourceSession.mappingDatabase().getDependentClassesForContainer( cont->id(), dependentClasses );
  for( std::set<std::string>::const_iterator iCl = dependentClasses.begin(); iCl != dependentClasses.end(); ++iCl ){
    MappingTree baseDepMapping;
    if(!sourceSession.mappingDatabase().getBaseMappingForContainer( *iCl, cont->id(), baseDepMapping )){
      throwException( "Base mapping for class \""+*iCl+"\" has not been found in the database.",
                    "DatabaseUtilitySession::importContainerSchema");      
    }
    std::set<std::string> depClassVersions;
    if(!sourceSession.mappingDatabase().getClassVersionListForMappingVersion( baseDepMapping.version(), depClassVersions )){
      throwException( "No class versions found for mapping \""+baseDepMapping.version()+"\".",
                      "DatabaseUtilitySession::importContainerSchema");
    }
    if( existingVersions.find( baseDepMapping.version() )!= existingVersions.end() ){
      throwException("Mapping version \""+baseDepMapping.version()+"\" for base mapping of class \""+*iCl+"\" already exists in the database.","DatabaseUtilitySession::importContainerSchema");
    }
    if( !mapping2Schema.check( baseDepMapping ) ){
      throwException("Schema base for class \""+baseDepMapping.className()+"\" cannot be replicated, because some schema objects have been found with the same name.","DatabaseUtilitySession::importContainerSchema");
    }
    baseVersions.insert( baseDepMapping.version() );
    existingVersions.insert( baseDepMapping.version() );
    m_session.mappingDatabase().storeMapping( baseDepMapping );
    first = true;
    for( std::set<std::string>::const_iterator iCv = depClassVersions.begin(); iCv != depClassVersions.end(); ++iCv ){
      m_session.mappingDatabase().insertClassVersion( *iCl, *iCv , 1, newCont->id(), baseDepMapping.version(), first );
      first = false;
    }
    mapping2Schema.create( baseDepMapping );
    // create the dep classes sequences. 
    containerSchemaSequences.create( MappingRules::sequenceNameForDependentClass( containerName, *iCl ));
  }
  /// third evolve the schema for all the further versions involved
  std::set<std::string> allVersions;
  if(!sourceSession.mappingDatabase().getMappingVersionsForContainer( cont->id(), allVersions )){
    std::stringstream mess;
    mess << "No mapping versions found for container id="<<cont->id();
    throwException( mess.str(), "DatabaseUtilitySession::importContainerSchema");
  }
  for( std::set<std::string>::const_iterator iVer = allVersions.begin(); iVer != allVersions.end(); ++iVer ){
    // skip the bases
    if( baseVersions.find( *iVer )== baseVersions.end() ){
      MappingTree evMapping;
      if(!sourceSession.mappingDatabase().getMappingByVersion( *iVer, evMapping) ){
        throwException("Mapping version \""+*iVer+"\" has not been found in the database.",
                       "DatabaseUtilitySession::importContainerSchema");
      }
      std::set<std::string> cvs;
      if(!sourceSession.mappingDatabase().getClassVersionListForMappingVersion( evMapping.version(), cvs )){
        throwException( "No class versions found for mapping \""+evMapping.version()+"\".",
                        "DatabaseUtilitySession::importContainerSchema");
      }
      if( existingVersions.find( *iVer )!= existingVersions.end() ){
        throwException("Mapping version \""+*iVer+"\" for mapping of class \""+evMapping.className()+"\" already exists in the database.","DatabaseUtilitySession::importContainerSchema");
      }
      if( !mapping2Schema.check( evMapping ) ){
        throwException("Evolved schema for class \""+evMapping.className()+"\" cannot be replicated, because some schema objects have been found with the same name.","DatabaseUtilitySession::importContainerSchema");
      }
      m_session.mappingDatabase().storeMapping( evMapping );
      existingVersions.insert( evMapping.version() );
      int depIndex = 0;
      std::string className = evMapping.className();
      if( evMapping.className() != baseMapping.className() ){
        // dependencies
        depIndex = 1;
      }
      for( std::set<std::string>::const_iterator iCv = cvs.begin(); iCv != cvs.end(); ++iCv ){
        m_session.mappingDatabase().insertClassVersion( evMapping.className(), *iCv , depIndex, newCont->id(), evMapping.version() );
      }
      // then evolve the schema
      mapping2Schema.alter( evMapping );
    }
  }
  return newCont;
}

void ora::DatabaseUtilitySession::importContainerSchema( const std::string& sourceConnectionString,
                                                         const std::string& containerName ){
  DatabaseSession sourceSession( m_session.connectionPool() );
  sourceSession.connect(sourceConnectionString, true );
  sourceSession.startTransaction( true );
  importContainerSchema(containerName, sourceSession );
  sourceSession.commitTransaction();
}


bool ora::DatabaseUtilitySession::existsContainer( const std::string& containerName ){
  bool found = false;
  for( std::map<int, Handle<DatabaseContainer> >::const_iterator iC = m_session.containers().begin();
       iC != m_session.containers().end(); ++iC ){
    if( iC->second->name() == containerName ) {
      found = true;
      break;
    }
  }
  return found;
}

void ora::DatabaseUtilitySession::importContainer( const std::string& sourceConnectionString,
                                                   const std::string& containerName ){
  DatabaseSession sourceSession( m_session.connectionPool() );
  sourceSession.connect(sourceConnectionString, true );
  sourceSession.startTransaction( true );
  Handle<ora::DatabaseContainer> newCont = importContainerSchema(containerName, sourceSession );
  Handle<ora::DatabaseContainer> cont = sourceSession.containerHandle( containerName );
  Handle<IteratorBuffer> iterator = cont->iteratorBuffer();
  std::vector<void*> objects;
  const Reflex::Type& contType = cont->type();
  while( iterator->next() ){
    void* data = iterator->getItem();
    objects.push_back( data );
    newCont->insertItem( data, contType );
  }
  newCont->flush();
  for( std::vector<void*>::const_iterator iO = objects.begin(); iO != objects.end(); iO++ ){
    contType.Destruct( *iO );
  }
  sourceSession.commitTransaction();
}

void ora::DatabaseUtilitySession::eraseMapping( const std::string& mappingVersion ){
  if( !m_session.exists() ){
    throwException( "ORA Database not found in \""+m_session.connectionString()+"\".",
                    "DatabaseUtilitySession::eraseMapping");      

  }
  m_session.mappingDatabase().removeMapping( mappingVersion );
}

ora::Handle<ora::DatabaseContainer> ora::DatabaseUtilitySession::containerHandle( const std::string& name ){
  if( !m_session.exists() ){
    throwException( "ORA Database not found in \""+m_session.connectionString()+"\".",
                    "DatabaseUtilitySession::containerHandle");      

  }
  m_session.open();
  return m_session.containerHandle( name );
}

