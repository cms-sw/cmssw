#ifndef INCLUDE_ORA_RELATIONALSTREAMERFACTORY_H
#define INCLUDE_ORA_RELATIONALSTREAMERFACTORY_H

namespace Reflex {
  class Type;
}

namespace ora {

  class ContainerSchema;
  class MappingElement;
  class IRelationalWriter;
  class IRelationalUpdater;
  class IRelationalReader;
  class IRelationalStreamer;

  class RelationalStreamerFactory {

    public:

    RelationalStreamerFactory( ContainerSchema& contSchema);

    ~RelationalStreamerFactory();

    IRelationalWriter* newWriter(const Reflex::Type& dataType,MappingElement& dataMapping );

    IRelationalUpdater* newUpdater(const Reflex::Type& dataType,MappingElement& dataMapping );

    IRelationalReader* newReader(const Reflex::Type& dataType,MappingElement& dataMapping );

    private:

    IRelationalStreamer* newStreamer( const Reflex::Type& dataType,MappingElement& dataMapping );

    private:
    ContainerSchema& m_containerSchema;
  };
}


#endif
    
