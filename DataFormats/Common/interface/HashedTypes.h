#ifndef DataFormatsCommonHashedTypes_h
#define DataFormatsCommonHashedTypes_h

// $Id: HashedTypes.h,v 1.2 2006/07/06 18:34:05 wmtan Exp $
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

#endif // DataFormatsCommonHashedTypes_h
