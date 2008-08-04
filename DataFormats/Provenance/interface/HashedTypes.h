#ifndef DataFormats_Common_HashedTypes_h
#define DataFormats_Common_HashedTypes_h


/// Declaration of the enum HashedTypes, used in defining several "id"
/// classes.

namespace edm
{
  enum HashedTypes {
      ModuleDescriptionType,
      ParameterSetType,
      ProcessHistoryType,
      ProcessConfigurationType,
      EntryDescriptionType
  };		     
}

#endif // DataFormats_Common_HashedTypes_h
