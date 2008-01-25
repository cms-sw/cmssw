#ifndef DataFormats_CommonHashedTypes_h
#define DataFormats_CommonHashedTypes_h

// $Id: HashedTypes.h,v 1.1 2007/03/04 04:48:08 wmtan Exp $
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
      ProcessConfigurationType,
      EntryDescriptionType
    };		     
}

#endif // DataFormats_CommonHashedTypes_h
