#include "CondCore/ORA/interface/Exception.h"
#include "MappingDatabase.h"
#include "IDatabaseSchema.h"
#include "MappingTree.h"
#include "MappingRules.h"
//
#include <sstream>
// externals
#include "Reflex/Type.h"

std::string
ora::MappingDatabase::versionOfClass( const Reflex::Type& dictionary ){
  std::string className = dictionary.Name(Reflex::SCOPED);
  Reflex::PropertyList classProps = dictionary.Properties();
  std::string classVersion = MappingRules::defaultClassVersion(className);
  if(classProps.HasProperty(MappingRules::classVersionPropertyNameInDictionary())){
    classVersion = classProps.PropertyAsString(MappingRules::classVersionPropertyNameInDictionary());
  }
  return classVersion;
}

void ora::MappingDatabase::buildElement( MappingElement& parentElement,
                                          const std::string& scopeName,
                                          std::map<std::string, std::vector<MappingRawElement> >& innerElements ){
  std::map<std::string,std::vector<MappingRawElement> >::iterator iScope = innerElements.find(scopeName);
  if(iScope != innerElements.end()){
    for( std::vector<MappingRawElement>::const_iterator iM = iScope->second.begin();
         iM != iScope->second.end(); ++iM ){
      MappingElement& element = parentElement.appendSubElement( iM->elementType,
                                                                iM->variableName,
                                                                iM->variableType,
                                                                iM->tableName );
      element.setColumnNames(iM->columns);
      std::string nextScope(scopeName);
      nextScope.append("::").append(iM->variableName);
      buildElement( element, nextScope, innerElements );
    }
  }
  innerElements.erase(scopeName);
}

void ora::MappingDatabase::unfoldElement( const MappingElement& element, MappingRawData& destination ){
  int newElemId = m_mappingSequence.getNextId();
  MappingRawElement&  elem = destination.addElement( newElemId );
  elem.elementType = MappingElement::elementTypeAsString( element.elementType() );
  elem.scopeName = element.scopeName();
  if(elem.scopeName.empty()) elem.scopeName = MappingRawElement::emptyScope();
  elem.variableName = element.variableName();
  elem.variableType = element.variableType();
  elem.tableName = element.tableName();
  elem.columns = element.columnNames();
  for ( MappingElement::const_iterator iSubEl = element.begin();
        iSubEl != element.end(); ++iSubEl) {
    unfoldElement( iSubEl->second, destination );
  }
}

ora::MappingDatabase::MappingDatabase( ora::IDatabaseSchema& schema ):
  m_schema( schema ),
  m_mappingSequence( MappingRules::mappingSequenceName(), schema ),
  m_versions(),
  m_isLoaded( false ){
}


ora::MappingDatabase::~MappingDatabase(){
}

std::string
ora::MappingDatabase::newMappingVersionForContainer( const std::string& containerName ){
  if(!m_isLoaded){
    m_schema.mappingSchema().getVersionList( m_versions );
    m_isLoaded = true;
  }
  
  std::string newMappingVersion = "";
  for ( int iteration = 0;; ++iteration ) {
    newMappingVersion = MappingRules::newMappingVersionForContainer( containerName, iteration );
    bool found = false;
    for ( std::set<std::string>::reverse_iterator iVersion = m_versions.rbegin();
          iVersion != m_versions.rend(); ++iVersion ) {
      if ( *iVersion == newMappingVersion ) {
        found = true;
        break;
      }
    }
    if ( ! found ){
      m_versions.insert( newMappingVersion );
      break;
    }
    
  }
  return newMappingVersion;
}

std::string
ora::MappingDatabase::newMappingVersionForDependentClass( const std::string& containerName, const std::string& className ){
  if(!m_isLoaded){
    m_schema.mappingSchema().getVersionList( m_versions );
    m_isLoaded = true;
  }
  
  std::string newMappingVersion = "";
  for ( int iteration = 0;; ++iteration ) {
    newMappingVersion = MappingRules::newMappingVersionForDependentClass( containerName, className, iteration );
    bool found = false;
    for ( std::set<std::string>::reverse_iterator iVersion = m_versions.rbegin();
          iVersion != m_versions.rend(); ++iVersion ) {
      if ( *iVersion == newMappingVersion ) {
        found = true;
        break;
      }
    }
    if ( ! found ){
      m_versions.insert( newMappingVersion );
      break;
    }
    
  }
  return newMappingVersion;
}

bool ora::MappingDatabase::getMappingByVersion( const std::string& version, MappingTree& destination  ){
  bool ret = false;
  MappingRawData mapData;
  if(m_schema.mappingSchema().getMapping( version, mapData )){
    ret = true;
    MappingRawElement topLevelElement;
    bool topLevelFound = false;
    bool dependency = false;
    std::map<std::string, std::vector<MappingRawElement> > innerElements;
    for( std::map< int, MappingRawElement>::iterator iElem = mapData.elements.begin();
         iElem != mapData.elements.end(); iElem++ ){
      // first loading the top level elements
      if( iElem->second.scopeName == MappingRawElement::emptyScope() ){
        if( iElem->second.elementType == MappingElement::classMappingElementType() ||
            iElem->second.elementType == MappingElement::dependencyMappingElementType() ){
          if( topLevelFound ){
            throwException("Mapping inconsistent.More then one top level element found.",
                           "MappingDatabase::getMappingByVersion");
          }
          topLevelElement = iElem->second;
          if( topLevelElement.elementType == MappingElement::dependencyMappingElementType() ) dependency = true;
        } 
      } else {
        std::map<std::string, std::vector<MappingRawElement> >::iterator iN = innerElements.find( iElem->second.scopeName );
        if(iN==innerElements.end()){
          innerElements.insert( std::make_pair( iElem->second.scopeName, std::vector<MappingRawElement>(1,iElem->second) ) );
        } else {
          iN->second.push_back( iElem->second );
        }
      }
    }
    MappingElement& topElement = destination.setTopElement( topLevelElement.variableName,
                                                            topLevelElement.tableName,
                                                            dependency );
    topElement.setColumnNames( topLevelElement.columns);
    buildElement( topElement, topLevelElement.variableName, innerElements  );
    destination.setVersion( version );
  }
  return ret;
}

void ora::MappingDatabase::removeMapping( const std::string& version ){
  m_schema.mappingSchema().removeMapping( version );
}

bool
ora::MappingDatabase::getMappingForContainer( const Reflex::Type& containerClass,
                                              int containerId,
                                              MappingTree& destination ){

  bool ret = false;
  // The class parameters
  std::string className = containerClass.Name(Reflex::SCOPED);
  std::string classVersion = versionOfClass( containerClass );
  std::string classId = MappingRules::classId( className, classVersion );

  std::string version("");
  bool found = m_schema.mappingSchema().selectMappingVersion( classId, containerId, version );

  if( found ){
    ret = getMappingByVersion( version, destination );
    if( !ret ){
      throwException("Mapping version \""+version+"\" not found.",
                     "MappingDatabase::getMappingForContainer");
    }
    if( destination.className() != className ){
      throwException("Mapping inconsistency detected for version=\""+version+"\"",
                     "MappingDatabase::getMappingForContainer");
    }
  }
  return ret;
}

bool ora::MappingDatabase::getBaseMappingForContainer( const std::string& className,
                                                       int containerId,
                                                       MappingTree& destination  ){
  bool ret = false;
  std::string classId = MappingRules::baseIdForClass( className );
  std::string mappingVersion("");
  bool found = m_schema.mappingSchema().selectMappingVersion( classId, containerId, mappingVersion );

  if( found ){
    ret = getMappingByVersion( mappingVersion, destination );
    if( !ret ){
      throwException("Mapping version \""+mappingVersion+"\" not found.",
                     "MappingDatabase::getBaseMappingForContainer");
    }
    if( destination.className() != className ){
      throwException("Mapping inconsistency detected for version=\""+mappingVersion+"\"",
                     "MappingDatabase::getBaseMappingForContainer");
    }
  }
  return ret;
}

bool ora::MappingDatabase::getDependentMappingsForContainer( int containerId,
                                                             std::vector<MappingElement>& destination  ){
  bool ret = false;
  std::set<std::string> versions;
  if( m_schema.mappingSchema().getMappingVersionListForContainer( containerId, versions, true ) ){
    ret = true;
    for( std::set<std::string>::iterator iM = versions.begin();
         iM != versions.end(); ++iM ){
      MappingTree mapping;
      if( ! getMappingByVersion( *iM, mapping )){
        throwException("Mapping version \""+*iM+"\" not found.",
                       "MappingDatabase::getDependentMappingsForContainer");
        
      }
      destination.push_back( mapping.topElement() );
    }
  }
  return ret;
}

void ora::MappingDatabase::insertClassVersion( const Reflex::Type& dictionaryEntry,
                                               int depIndex,
                                               int containerId,
                                               const std::string& mappingVersion,
                                               bool asBase  ){
  std::string className = dictionaryEntry.Name( Reflex::SCOPED );
  std::string classVersion = versionOfClass( dictionaryEntry );
  std::string classId = MappingRules::classId( className, classVersion );
  m_schema.mappingSchema().insertClassVersion( className, classVersion, classId, depIndex, containerId, mappingVersion );
  if( asBase ){
    std::string baseId = MappingRules::baseIdForClass( className );
    m_schema.mappingSchema().insertClassVersion( className, MappingRules::baseClassVersion(), baseId, depIndex, containerId, mappingVersion );
  }
}

void ora::MappingDatabase::setMappingVersionForClass( const Reflex::Type& dictionaryEntry,
                                                      int containerId,
                                                      const std::string& mappingVersion,
                                                      bool dependency ){
  std::string className = dictionaryEntry.Name( Reflex::SCOPED );
  std::string classVersion = versionOfClass( dictionaryEntry );
  std::string classId = MappingRules::classId( className, classVersion );
  std::string mv("");
  bool found = m_schema.mappingSchema().selectMappingVersion( classId, containerId, mv );
  if( !found ){
    int depIndex = 0;
    if( dependency ) depIndex = 1;
    m_schema.mappingSchema().insertClassVersion( className, classVersion, classId, depIndex, containerId, mappingVersion );
  } else {
    m_schema.mappingSchema().setMappingVersion( classId, containerId, mappingVersion );
  }
}

void ora::MappingDatabase::storeMapping( const MappingTree& mapping ){
  MappingRawData rowMapping( mapping.version() );
  unfoldElement( mapping.topElement(), rowMapping );
  m_mappingSequence.sinchronize();
  m_schema.mappingSchema().storeMapping( rowMapping );  
}

bool ora::MappingDatabase::getMappingVersionsForContainer( int containerId, std::set<std::string>& versionList ){
  return m_schema.mappingSchema().getMappingVersionListForContainer( containerId, versionList );
}

void ora::MappingDatabase::clear(){
  m_versions.clear();
  m_isLoaded = false;
}

