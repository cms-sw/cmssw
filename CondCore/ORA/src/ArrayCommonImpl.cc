#include "ArrayCommonImpl.h"
#include "MappingElement.h"
#include "RelationalBuffer.h"
#include "RelationalOperation.h"
// externals
#include "CoralBase/Attribute.h"

void ora::deleteArrayElements( MappingElement& mapping,
                               int oid,
                               int fromIndex,
                               RelationalBuffer& buffer ){
  for ( MappingElement::iterator iMe = mapping.begin();
        iMe != mapping.end(); ++iMe ) {
    MappingElement::ElementType elementType = iMe->second.elementType();
    // add the InlineCArray (or change the algorithm...
    if ( elementType == MappingElement::Object ||
         elementType == MappingElement::Array ||
         elementType == MappingElement::OraArray ||
         elementType == MappingElement::CArray ) {
      deleteArrayElements( iMe->second, oid, fromIndex, buffer  );
    }
  }
  if ( mapping.elementType() == MappingElement::Object) return;

  std::string oidColumn = mapping.columnNames()[ 0 ];
  std::string indexColumn = mapping.columnNames()[ 1 ];
  DeleteOperation& deleteOperation = buffer.newDelete( mapping.tableName() );
  deleteOperation.addWhereId( oidColumn );
  deleteOperation.addWhereId( indexColumn, Ge );
  coral::AttributeList::iterator condDataIter = deleteOperation.whereData().begin();
  condDataIter->data<int>() = oid;
  ++condDataIter;
  condDataIter->data<int>() = fromIndex;
}


