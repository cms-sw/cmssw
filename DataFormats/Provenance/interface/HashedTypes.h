#ifndef DataFormats_CommonHashedTypes_h
#define DataFormats_CommonHashedTypes_h

// $Id: HashedTypes.h,v 1.3 2006/07/07 19:42:34 paterno Exp $
//

/// Declaration of the enum HashedTypes, used in defining several "id"
/// classes.

namespace edm
{
  enum HashedTypes 
    {
      ModuleDescriptionType,
      ParameterSetType,
      ProcessHistoryType,
      ProcessConfigurationType
    };		     
}

#endif // DataFormats_CommonHashedTypes_h
