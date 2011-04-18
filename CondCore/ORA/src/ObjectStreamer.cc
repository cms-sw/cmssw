#include "CondCore/ORA/interface/Exception.h"
#include "ObjectStreamer.h"
#include "DataElement.h"
#include "MappingElement.h"
#include "ClassUtils.h"
#include "MappingRules.h"
// externals
#include "Reflex/Base.h"
#include "Reflex/Member.h"

namespace ora {

  bool isLoosePersistencyDataMember( const Reflex::Member& dataMember ){
    std::string persistencyType("");
    Reflex::PropertyList memberProps = dataMember.Properties();
    if( memberProps.HasProperty(ora::MappingRules::persistencyPropertyNameInDictionary())){
       persistencyType = memberProps.PropertyAsString(ora::MappingRules::persistencyPropertyNameInDictionary());
    }
    return ora::MappingRules::isLooseOnWriting( persistencyType ) || ora::MappingRules::isLooseOnReading( persistencyType ) ;
  }

}

ora::ObjectStreamerBase::ObjectStreamerBase( const Reflex::Type& objectType,
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
                                                    const Reflex::Type& objType,
                                                    RelationalBuffer* operationBuffer ){
  
  for ( unsigned int i=0;i<objType.BaseSize();i++){
    Reflex::Base base = objType.BaseAt(i);
    Reflex::Type baseType = ClassUtils::resolvedType( base.ToType() );
    buildBaseDataMembers( dataElement, relationalData, baseType, operationBuffer );
    for ( unsigned int j=0;j<baseType.DataMemberSize();j++){
      Reflex::Member dataMember = baseType.DataMemberAt(j);      
      DataElement& dataMemberElement = dataElement.addChild( dataMember.Offset(), base.OffsetFP() );
      // Ignore the transients and the statics (how to deal with non-const statics?)
      if ( dataMember.IsTransient() || dataMember.IsStatic() ) continue;
      // Get the member type and resolve possible typedef chains
      Reflex::Type dataMemberType = ClassUtils::resolvedType( dataMember.TypeOf() );
      if ( ! dataMemberType ) {
        throwException( "Missing dictionary information for data member \"" +
                        dataMember.Name() + "\" of class \"" +
                        baseType.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                        "ObjectStreamerBase::buildBaseDataMembers" );
      }
      
      // check if the member is from a class in the inheritance tree
      Reflex::Type declaringType = ClassUtils::resolvedType( dataMember.DeclaringType());
      std::string scope = declaringType.Name(Reflex::SCOPED|Reflex::FINAL);
      // Get the data member name
      std::string dataMemberName = MappingRules::scopedVariableName( dataMember.Name(), scope );
      // Retrieve the relevant mapping element
      MappingElement::iterator iDataMemberMapping = m_mapping.find( dataMemberName );
      if ( iDataMemberMapping != m_mapping.end() ) {
        MappingElement& dataMemberMapping = iDataMemberMapping->second;
        processDataMember( dataMemberElement, relationalData, dataMemberType, dataMemberMapping, operationBuffer );
      } else {
        if( !isLoosePersistencyDataMember( dataMember ) ){
	  throwException( "Data member \"" + dataMemberName +
                          "\" not found in the mapping element of variable \""+m_mapping.variableName()+"\".",
                          "ObjectStreamerBase::buildBaseDataMembers" );
	}
      }
    }
  }
  
}

bool ora::ObjectStreamerBase::buildDataMembers( DataElement& dataElement,
                                                IRelationalData& relationalData,
                                                RelationalBuffer* operationBuffer ){
  buildBaseDataMembers( dataElement, relationalData, m_objectType, operationBuffer );
    // Loop over the data members of the class.
  for ( unsigned int i=0;i<m_objectType.DataMemberSize();i++){

    Reflex::Member dataMember = m_objectType.DataMemberAt(i);
    DataElement& dataMemberElement = dataElement.addChild( dataMember.Offset(), 0 );

    Reflex::Type declaringType = ClassUtils::resolvedType( dataMember.DeclaringType());
    if( declaringType != m_objectType ){
      continue;
    }
          
    // Ignore the transients and the statics (how to deal with non-const statics?)
    if ( dataMember.IsTransient() || dataMember.IsStatic() ) continue;

    // Get the member type and resolve possible typedef chains
    Reflex::Type dataMemberType = ClassUtils::resolvedType( dataMember.TypeOf() );
    if ( ! dataMemberType ) {
      throwException( "Missing dictionary information for data member \"" +
                      dataMember.Name() + "\" of class \"" +
                      m_objectType.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                      "ObjectStreamerBase::buildDataMembers" );
    }
      
    // check if the member is from a class in the inheritance tree
    std::string scope("");
    // Get the data member name
    std::string dataMemberName = MappingRules::scopedVariableName( dataMember.Name(), scope );
    
    // Retrieve the relevant mapping element
    MappingElement::iterator idataMemberMapping = m_mapping.find( dataMemberName );
    if ( idataMemberMapping != m_mapping.end() ) {
      MappingElement& dataMemberMapping = idataMemberMapping->second;
      processDataMember( dataMemberElement, relationalData, dataMemberType, dataMemberMapping, operationBuffer );
    } else {
      if(!isLoosePersistencyDataMember( dataMember ) ){
        throwException( "Data member \"" + dataMemberName +
                        "\" not found in the mapping element of variable \""+m_mapping.variableName()+"\".",
                        "ObjectStreamerBase::buildDataMembers" );
      }
    }
  }
  return true;
}

ora::ObjectWriter::ObjectWriter( const Reflex::Type& objectType,
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
                                           Reflex::Type& dataMemberType,
                                           MappingElement& dataMemberMapping,
                                           RelationalBuffer* operationBuffer ){
  IRelationalWriter* dataMemberWriter = m_streamerFactory.newWriter( dataMemberType, dataMemberMapping );
  m_writers.push_back( dataMemberWriter );
  dataMemberWriter->build( dataMemberElement, relationalData, *operationBuffer );
}


ora::ObjectUpdater::ObjectUpdater( const Reflex::Type& objectType,
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
                                            Reflex::Type& dataMemberType,
                                            MappingElement& dataMemberMapping,
                                            RelationalBuffer* operationBuffer ){
  IRelationalUpdater* dataMemberUpdater = m_streamerFactory.newUpdater( dataMemberType, dataMemberMapping );
  m_updaters.push_back( dataMemberUpdater );
  dataMemberUpdater->build( dataMemberElement, relationalData, *operationBuffer );
}

ora::ObjectReader::ObjectReader( const Reflex::Type& objectType,
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
                                           Reflex::Type& dataMemberType,
                                           MappingElement& dataMemberMapping,
                                           RelationalBuffer*){
  IRelationalReader* dataMemberReader = m_streamerFactory.newReader( dataMemberType, dataMemberMapping );
  m_readers.push_back( dataMemberReader );
  dataMemberReader->build( dataMemberElement, relationalData );
}


ora::ObjectStreamer::ObjectStreamer( const Reflex::Type& objectType,
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

