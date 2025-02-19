#ifndef INCLUDE_ORA_ARRAYCOMMONIMPL_H
#define INCLUDE_ORA_ARRAYCOMMONIMPL_H

namespace ora {

  class MappingElement;
  class RelationalBuffer;
  
  union U_Primitives {
      bool b;
      char c;
      unsigned char uc;
      short s;
      unsigned short us;
      int i;
      unsigned int ui;
      long l;
      unsigned long ul;
      float f;
      double d;
      long double ld;
  };

  void deleteArrayElements( MappingElement& mapping, int oid, int fromIndex, RelationalBuffer& buffer);

}

#endif  


