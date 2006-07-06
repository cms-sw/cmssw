#ifndef DataFormatsCommonHashedTypes_h
#define DataFormatsCommonHashedTypes_h

// $Id: HashedTypes.h,v 1.1.2.2 2006/07/04 13:56:44 wmtan Exp $
//

/// Declaration of the enum HashedTypes, used in defining several "id"
/// classes.

namespace edm
{
  enum HashedTypes 
    {
      ModuleDescriptionType,
      ParameterSetType,
      ProcessHistoryType
    };		     
}

#endif // DataFormatsCommonHashedTypes_h
