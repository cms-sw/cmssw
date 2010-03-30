#include "CondCore/ORA/interface/Exception.h"
#include "RelationalStreamerFactory.h"
#include "MappingElement.h"
#include "ClassUtils.h"
#include "PrimitiveStreamer.h"
#include "BlobStreamer.h"
#include "ObjectStreamer.h"
#include "STLContainerStreamer.h"
#include "CArrayStreamer.h"
#include "InlineCArrayStreamer.h"
#include "OraReferenceStreamer.h"
#include "OraPtrStreamer.h"
#include "PVectorStreamer.h"
#include "QueryableVectorStreamer.h"
#include "UniqueRefStreamer.h"
//
#include <memory>

ora::RelationalStreamerFactory::RelationalStreamerFactory( ContainerSchema& contSchema ):
  m_containerSchema( contSchema ){
}

ora::RelationalStreamerFactory::~RelationalStreamerFactory(){
}

ora::IRelationalStreamer* ora::RelationalStreamerFactory::newStreamer( const Reflex::Type& type,
                                                                       MappingElement& mapping ){
  IRelationalStreamer* newStreamer = 0;
  if ( mapping.elementType() == MappingElement::Primitive ) { // Primitives
    if( ! ClassUtils::isTypePrimitive( type ) ){
      throwException( "Mapped variable \"" + mapping.variableName() +
                      "\", declared as Primitive, is associated to non-primitive type \""+type.Name()+"\"",
                      "RelationalStreamerFactory::newStreamer" );
    }
    newStreamer = new PrimitiveStreamer( type, mapping );
  } else if ( mapping.elementType() == MappingElement::Blob ){
    newStreamer = new BlobStreamer( type, mapping, m_containerSchema );
  } else if ( mapping.elementType() == MappingElement::Object ){
    newStreamer = new ObjectStreamer( type, mapping, m_containerSchema );
  } else if ( mapping.elementType() == MappingElement::Array ){
    if ( !ClassUtils::isTypeContainer( type ) ) {
      throwException( "Mapped variable \"" + mapping.variableName() +" of type "+
                      mapping.variableType() +
                      "\", declared as Array, is associated to the non-container type \""+
                      type.Name()+"\".",
                      "RelationalStreamerFactory::newStreamer" );
    }
    newStreamer = new STLContainerStreamer( type, mapping, m_containerSchema );
  } else if ( mapping.elementType() == MappingElement::CArray ) { 
    if ( ! type.IsArray() ) {
      throwException( "Mapped variable \"" + mapping.variableName() +" of type "+
                      mapping.variableType() +
                      "\", declared as C-Array, is associated to the non-array type \""+
                      type.Name()+"\".",
                      "RelationalStreamerFactory::newStreamer" );
    }
    newStreamer = new CArrayStreamer( type, mapping, m_containerSchema );
  } else if ( mapping.elementType() == MappingElement::InlineCArray ) { 
    if ( ! type.IsArray() ) {
      throwException( "Mapped variable \"" + mapping.variableName() +" of type "+
                      mapping.variableType() +
                      "\", declared as Inline C-Array, is associated to the non-array type \""+
                      type.Name()+"\".",
                      "RelationalStreamerFactory::newStreamer" );
    }
    newStreamer = new InlineCArrayStreamer( type, mapping, m_containerSchema );
  } else if ( mapping.elementType() == MappingElement::OraArray ) {
    if ( ! ClassUtils::isTypePVector( type ) && ! ClassUtils::isTypeQueryableVector( type )) {
      throwException( "Mapped variable \"" + mapping.variableName() +" of type "+
                      mapping.variableType() +
                      "\", declared as OraArray, is associated to the non-array type \""+
                      type.Name()+"\".",
                      "RelationalStreamerFactory::newStreamer" );
    }
    if( ClassUtils::isTypePVector( type ) )
      newStreamer = new PVectorStreamer( type, mapping, m_containerSchema );
    else if ( ClassUtils::isTypeQueryableVector( type ))
      newStreamer = new QueryableVectorStreamer( type, mapping, m_containerSchema );
  } else if ( mapping.elementType() == MappingElement::OraPointer ) { 
    if ( ! ClassUtils::isTypeOraPointer( type )) {
      throwException( "Mapped variable \"" + mapping.variableName() +" of type "+
                      mapping.variableType() +
                      "\", declared as a OraPointer, is associated to the type \""+
                      type.Name()+"\".",
                      "RelationalStreamerFactory::newStreamer" );
    }
    newStreamer = new OraPtrStreamer( type, mapping, m_containerSchema );
  } else if ( mapping.elementType() == MappingElement::OraReference ) {
    if ( ! ClassUtils::isTypeOraReference( type )) {
      throwException( "Mapped variable \"" + mapping.variableName() +" of type "+
                      mapping.variableType() +
                      "\", declared as a OraReference, is associated to the type \""+
                      type.Name()+"\".",
                      "RelationalStreamerFactory::newStreamer" );
    }
    newStreamer = new OraReferenceStreamer( type, mapping, m_containerSchema );
  } else if ( mapping.elementType() == MappingElement::UniqueReference ) {
    if ( ! ClassUtils::isTypeUniqueReference( type )) {
      throwException( "Mapped variable \"" + mapping.variableName() +" of type "+
                      mapping.variableType() +
                      "\", declared as a UniqueReference, is associated to the type \""+
                      type.Name()+"\".",
                      "RelationalStreamerFactory::newStreamer" );
    }
    newStreamer = new UniqueRefStreamer( type, mapping, m_containerSchema );
  } else {
    throwException( "Cannot find a streamer suitable for mapped variable \"" + mapping.variableName() +" of type "+
                    mapping.variableType() +
                    "\".",
                    "RelationalStreamerFactory::newStreamer" );
  }

  return newStreamer;
}



ora::IRelationalWriter* ora::RelationalStreamerFactory::newWriter(const Reflex::Type& type,
                                                                  MappingElement& mapping ){
  std::auto_ptr<IRelationalStreamer> streamer( newStreamer( type, mapping ) );
  return streamer->newWriter();
}

ora::IRelationalUpdater* ora::RelationalStreamerFactory::newUpdater(const Reflex::Type& type,
                                                                    MappingElement& mapping ){
  std::auto_ptr<IRelationalStreamer> streamer( newStreamer( type, mapping ) );
  return streamer->newUpdater();
}

ora::IRelationalReader* ora::RelationalStreamerFactory::newReader(const Reflex::Type& type,
                                                                  MappingElement& mapping ){
  std::auto_ptr<IRelationalStreamer> streamer( newStreamer( type, mapping ) );
  return streamer->newReader();
}
