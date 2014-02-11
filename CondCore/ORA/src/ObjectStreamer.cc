#include "CondCore/ORA/interface/Exception.h"
#include "ObjectStreamer.h"
#include "DataElement.h"
#include "MappingElement.h"
#include "ClassUtils.h"
#include "MappingRules.h"
// externals
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "RflxPropList.h"
#include "oraHelper.h"

namespace ora {

  bool isLoosePersistencyDataMember( const edm::MemberWithDict& dataMember ){
    std::string persistencyType("");
    Reflex::PropertyList memberProps = ora::helper::Properties(dataMember);
    if( memberProps.HasProperty(ora::MappingRules::persistencyPropertyNameInDictionary())){
       persistencyType = memberProps.PropertyAsString(ora::MappingRules::persistencyPropertyNameInDictionary());
    }
    return ora::MappingRules::isLooseOnWriting( persistencyType ) || ora::MappingRules::isLooseOnReading( persistencyType ) ;
  }

}

ora::ObjectStreamerBase::ObjectStreamerBase( const edm::TypeWithDict& objectType,
                                             MappingElement& mapping,
                                             ContainerSchema& contSchema ):
  m_streamerFactory( contSchema ),
  m_objectType( objectType ),
  m_mapping( mapping ){
}

ora::ObjectStreamerBase::~ObjectStreamerBase(){
}


void ora::ObjectStreamerBase::buildBaseDataMembers( DataElement& dataElement,
                                                    IRelationalData& relationalData,
                                                    const edm::TypeWithDict& objType,
                                                    RelationalBuffer* operationBuffer ){
  
  for ( unsigned int i=0;i<ora::helper::BaseSize(objType);i++){
    edm::BaseWithDict base = ora::helper::BaseAt(objType, i);
    edm::TypeWithDict baseType = ClassUtils::resolvedType( base.typeOf() );
    buildBaseDataMembers( dataElement, relationalData, baseType, operationBuffer );
    for ( unsigned int j=0;j<baseType.dataMemberSize();j++){
      edm::MemberWithDict dataMember = ora::helper::DataMemberAt(baseType, j);      
      DataElement& dataMemberElement = dataElement.addChild( dataMember.offset(), /*base.offsetFP()*/ base.offset() );
      // Ignore the transients and the statics (how to deal with non-const statics?)
      if ( dataMember.isTransient() || dataMember.isStatic() ) continue;
      // Get the member type and resolve possible typedef chains
      edm::TypeWithDict dataMemberType = ClassUtils::resolvedType( dataMember.typeOf() );
      if ( ! dataMemberType ) {
        throwException( "Missing dictionary information for data member \"" +
                        dataMember.name() + "\" of class \"" +
                        baseType.qualifiedName() + "\"",
                        "ObjectStreamerBase::buildBaseDataMembers" );
      }
      
      // check if the member is from a class in the inheritance tree
      edm::TypeWithDict declaringType = ClassUtils::resolvedType( dataMember.declaringType());
      std::string scope = declaringType.qualifiedName();
      // Get the data member name
      std::string dataMemberName = MappingRules::scopedVariableName( dataMember.name(), scope );
      // Retrieve the relevant mapping element
      MappingElement::iterator iDataMemberMapping = m_mapping.find( dataMemberName );
      if ( iDataMemberMapping != m_mapping.end() ) {
        MappingElement& dataMemberMapping = iDataMemberMapping->second;
	if( !ClassUtils::checkMappedType(dataMemberType,dataMemberMapping.variableType()) ){
	  throwException( "Data member \""+dataMemberName +"\" type \"" + dataMemberType.qualifiedName() +
			  "\" does not match with the expected type in the mapping \""+dataMemberMapping.variableType()+"\".",
			  "ObjectStreamerBase::buildBaseDataMembers" );
	}
        processDataMember( dataMemberElement, relationalData, dataMemberType, dataMemberMapping, operationBuffer );
      } else {
/*
        // FIXME: Commented out because ROOT6 does not yet support attributes on "field name=" lines.
        if( !isLoosePersistencyDataMember( dataMember ) ){
	  throwException( "Data member \"" + dataMemberName +
                          "\" not found in the mapping element of variable \""+m_mapping.variableName()+"\".",
                          "ObjectStreamerBase::buildBaseDataMembers" );
	}
*/
      }
    }
  }
  
}

bool ora::ObjectStreamerBase::buildDataMembers( DataElement& dataElement,
                                                IRelationalData& relationalData,
                                                RelationalBuffer* operationBuffer ){
  buildBaseDataMembers( dataElement, relationalData, m_objectType, operationBuffer );
    // Loop over the data members of the class.
  for ( unsigned int i=0;i<m_objectType.dataMemberSize();i++){

    edm::MemberWithDict dataMember = ora::helper::DataMemberAt(m_objectType, i);
    DataElement& dataMemberElement = dataElement.addChild( dataMember.offset(), 0 );

    edm::TypeWithDict declaringType = ClassUtils::resolvedType( dataMember.declaringType());
    if( declaringType != m_objectType ){
      continue;
    }
          
    // Ignore the transients and the statics (how to deal with non-const statics?)
    if ( dataMember.isTransient() || dataMember.isStatic() ) continue;

    // Get the member type and resolve possible typedef chains
    edm::TypeWithDict dataMemberType = ClassUtils::resolvedType( dataMember.typeOf() );
    if ( ! dataMemberType ) {
      throwException( "Missing dictionary information for data member \"" +
                      dataMember.name() + "\" of class \"" +
                      m_objectType.qualifiedName() + "\"",
                      "ObjectStreamerBase::buildDataMembers" );
    }
      
    // check if the member is from a class in the inheritance tree
    std::string scope("");
    // Get the data member name
    std::string dataMemberName = MappingRules::scopedVariableName( dataMember.name(), scope );
    
    // Retrieve the relevant mapping element
    MappingElement::iterator idataMemberMapping = m_mapping.find( dataMemberName );
    if ( idataMemberMapping != m_mapping.end() ) {
      MappingElement& dataMemberMapping = idataMemberMapping->second;
      if( !ClassUtils::checkMappedType(dataMemberType,dataMemberMapping.variableType())){
        throwException( "Data member  \""+dataMemberName +"\" type \"" + dataMemberType.qualifiedName() +
                        "\" does not match with the expected type in the mapping \""+dataMemberMapping.variableType()+"\".",
                        "ObjectStreamerBase::buildDataMembers" );
      }
      processDataMember( dataMemberElement, relationalData, dataMemberType, dataMemberMapping, operationBuffer );
    } else {
/*
      // FIXME: Commented out because ROOT6 does not yet support attributes on "field name=" lines.
      if(!isLoosePersistencyDataMember( dataMember ) ){
        throwException( "Data member \"" + dataMemberName +
                        "\" not found in the mapping element of variable \""+m_mapping.variableName()+"\".",
                        "ObjectStreamerBase::buildDataMembers" );
      }
*/
    }
  }
  return true;
}

ora::ObjectWriter::ObjectWriter( const edm::TypeWithDict& objectType,
                                 MappingElement& mapping,
                                 ContainerSchema& contSchema ):
  ObjectStreamerBase( objectType, mapping, contSchema ),
  m_writers(){
}
      
ora::ObjectWriter::~ObjectWriter(){
  for( std::vector< IRelationalWriter* >::iterator iW = m_writers.begin();
       iW != m_writers.end(); ++iW ){
    delete *iW;
  }
  m_writers.clear();
}

bool ora::ObjectWriter::build(DataElement& dataElement,
                              IRelationalData& relationalData,
                              RelationalBuffer& operationBuffer){
  return buildDataMembers( dataElement, relationalData, &operationBuffer );
}

void ora::ObjectWriter::setRecordId( const std::vector<int>& identity ){
  for( std::vector< IRelationalWriter* >::iterator iW = m_writers.begin();
       iW !=  m_writers.end(); ++iW ){
    (*iW)->setRecordId( identity );
  }  
}

/// Writes a data element
void ora::ObjectWriter::write( int oid,
                               const void* data ){
  for( std::vector< IRelationalWriter* >::iterator iW = m_writers.begin();
       iW !=  m_writers.end(); ++iW ){
    (*iW)->write( oid, data );
  }
}

void ora::ObjectWriter::processDataMember( DataElement& dataMemberElement,
                                           IRelationalData& relationalData,
                                           edm::TypeWithDict& dataMemberType,
                                           MappingElement& dataMemberMapping,
                                           RelationalBuffer* operationBuffer ){
  IRelationalWriter* dataMemberWriter = m_streamerFactory.newWriter( dataMemberType, dataMemberMapping );
  m_writers.push_back( dataMemberWriter );
  dataMemberWriter->build( dataMemberElement, relationalData, *operationBuffer );
}


ora::ObjectUpdater::ObjectUpdater( const edm::TypeWithDict& objectType,
                                   MappingElement& mapping,
                                   ContainerSchema& contSchema ):
  ObjectStreamerBase( objectType, mapping, contSchema ),
  m_updaters(){
}
      
ora::ObjectUpdater::~ObjectUpdater(){
  for( std::vector< IRelationalUpdater* >::iterator iU = m_updaters.begin();
       iU != m_updaters.end(); ++iU ){
    delete *iU;
  }
  m_updaters.clear();
}

bool ora::ObjectUpdater::build(DataElement& dataElement,
                               IRelationalData& relationalData,
                               RelationalBuffer& operationBuffer){
  return buildDataMembers( dataElement, relationalData, &operationBuffer  );
}

void ora::ObjectUpdater::setRecordId( const std::vector<int>& identity ){
  for( std::vector< IRelationalUpdater* >::iterator iU = m_updaters.begin();
       iU !=  m_updaters.end(); ++iU){
    (*iU)->setRecordId( identity );
  }  
}

/// Writes a data element
void ora::ObjectUpdater::update( int oid,
                                 const void* data ){
  for( std::vector< IRelationalUpdater* >::iterator iU = m_updaters.begin();
       iU !=  m_updaters.end(); ++iU ){
    (*iU)->update( oid, data );
  }
}

void ora::ObjectUpdater::processDataMember( DataElement& dataMemberElement,
                                            IRelationalData& relationalData,
                                            edm::TypeWithDict& dataMemberType,
                                            MappingElement& dataMemberMapping,
                                            RelationalBuffer* operationBuffer ){
  IRelationalUpdater* dataMemberUpdater = m_streamerFactory.newUpdater( dataMemberType, dataMemberMapping );
  m_updaters.push_back( dataMemberUpdater );
  dataMemberUpdater->build( dataMemberElement, relationalData, *operationBuffer );
}

ora::ObjectReader::ObjectReader( const edm::TypeWithDict& objectType,
                                 MappingElement& mapping,
                                 ContainerSchema& contSchema ):
  ObjectStreamerBase( objectType, mapping, contSchema ),
  m_readers(){
}
      
ora::ObjectReader::~ObjectReader(){
  for( std::vector< IRelationalReader* >::iterator iStr = m_readers.begin();
       iStr != m_readers.end(); ++iStr ){
    delete *iStr;
  }
  m_readers.clear();
}

bool ora::ObjectReader::build( DataElement& dataElement,
                               IRelationalData& relationalData){
  return buildDataMembers( dataElement, relationalData, 0 );
}

void ora::ObjectReader::select( int oid ){
  for( std::vector< IRelationalReader* >::iterator iDepReader = m_readers.begin();
       iDepReader !=  m_readers.end(); ++iDepReader ){
    (*iDepReader)->select( oid );
  }
}

void ora::ObjectReader::setRecordId( const std::vector<int>& identity ){
  for( std::vector< IRelationalReader* >::iterator iDepReader = m_readers.begin();
       iDepReader !=  m_readers.end(); ++iDepReader ){
    (*iDepReader)->setRecordId( identity );
  }  
}

/// Read a data element
void ora::ObjectReader::read( void* data ){
  for( std::vector< IRelationalReader* >::iterator iDepReader = m_readers.begin();
       iDepReader !=  m_readers.end(); ++iDepReader ){
    (*iDepReader)->read( data );
  }
}

void ora::ObjectReader::clear(){
  for( std::vector< IRelationalReader* >::iterator iDepReader = m_readers.begin();
       iDepReader !=  m_readers.end(); ++iDepReader ){
    (*iDepReader)->clear();
  }
}

void ora::ObjectReader::processDataMember( DataElement& dataMemberElement,
                                           IRelationalData& relationalData,
                                           edm::TypeWithDict& dataMemberType,
                                           MappingElement& dataMemberMapping,
                                           RelationalBuffer*){
  IRelationalReader* dataMemberReader = m_streamerFactory.newReader( dataMemberType, dataMemberMapping );
  m_readers.push_back( dataMemberReader );
  dataMemberReader->build( dataMemberElement, relationalData );
}


ora::ObjectStreamer::ObjectStreamer( const edm::TypeWithDict& objectType,
                                     MappingElement& mapping,
                                     ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::ObjectStreamer::~ObjectStreamer(){
}

ora::IRelationalWriter* ora::ObjectStreamer::newWriter(){
  return new ObjectWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::ObjectStreamer::newUpdater(){
  return new ObjectUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::ObjectStreamer::newReader(){
  return new ObjectReader( m_objectType, m_mapping, m_schema );
}

