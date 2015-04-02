#ifndef INCLUDE_ORA_RELATIONALSTREAMERFACTORY_H
#define INCLUDE_ORA_RELATIONALSTREAMERFACTORY_H

namespace edm {
  class TypeWithDict;
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

    IRelationalWriter* newWriter(const edm::TypeWithDict& dataType,MappingElement& dataMapping );

    IRelationalUpdater* newUpdater(const edm::TypeWithDict& dataType,MappingElement& dataMapping );

    IRelationalReader* newReader(const edm::TypeWithDict& dataType,MappingElement& dataMapping );

    private:

    IRelationalStreamer* newStreamer( const edm::TypeWithDict& dataType,MappingElement& dataMapping );

    private:
    ContainerSchema& m_containerSchema;
  };
}


#endif
    
