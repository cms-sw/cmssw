#ifndef DataFormats_Portable_interface_PortableHostCollectionReadRules_h
#define DataFormats_Portable_interface_PortableHostCollectionReadRules_h

#include <TGenericClassInfo.h>
#include <TVirtualObject.h>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "FWCore/Utilities/interface/concatenate.h"
#include "FWCore/Utilities/interface/stringize.h"

// read function for PortableHostCollection, called for every event
template <typename T>
static void readPortableHostCollection_v1(char *target, TVirtualObject *from_buffer) {
  // extract the actual types
  using Collection = T;
  using Layout = typename Collection::Layout;

  // valid only for PortableHostCollection<T>
  static_assert(std::is_same_v<Collection, PortableHostCollection<Layout>>);

  // proxy for the object being read from file
  struct OnFile {
    Layout &layout_;
  };

  // address in memory of the buffer containing the object being read from file
  char *address = static_cast<char *>(from_buffer->GetObject());
  // offset of the "layout_" data member
  static ptrdiff_t layout_offset = from_buffer->GetClass()->GetDataMemberOffset("layout_");
  // reference to the Layout object being read from file
  OnFile onfile = {*(Layout *)(address + layout_offset)};

  // pointer to the Collection object being constructed in memory
  Collection *newObj = (Collection *)target;

  // move the data from the on-file layout to the newly constructed object
  Collection::ROOTReadStreamer(newObj, onfile.layout_);
}

// put set_PortableHostCollection_read_rules in the ROOT namespace to let it forward declare GenerateInitInstance
namespace ROOT {

  // set the read rules for PortableHostCollection<T>;
  // this is called only once, when the dictionary is loaded.
  template <typename T>
  static bool set_PortableHostCollection_read_rules(std::string const &type) {
    // forward declaration
    TGenericClassInfo *GenerateInitInstance(T const *);

    // build the read rules
    std::vector<ROOT::Internal::TSchemaHelper> readrules(1);
    ROOT::Internal::TSchemaHelper &rule = readrules[0];
    rule.fTarget = "buffer_,layout_,view_";
    rule.fSourceClass = type;
    rule.fSource = type + "::Layout layout_;";
    rule.fCode = type + "::ROOTReadStreamer(newObj, onfile.layout_)";
    rule.fVersion = "[1-]";
    rule.fChecksum = "";
    rule.fInclude = "";
    rule.fEmbed = false;
    rule.fFunctionPtr = reinterpret_cast<void *>(::readPortableHostCollection_v1<T>);
    rule.fAttributes = "";

    // set the read rules
    TGenericClassInfo *instance = GenerateInitInstance((T const *)nullptr);
    instance->SetReadRules(readrules);

    return true;
  }
}  // namespace ROOT

#define SET_PORTABLEHOSTCOLLECTION_READ_RULES(COLLECTION)                                                  \
  static bool EDM_CONCATENATE(set_PortableHostCollection_read_rules_done_at_, __LINE__) [[maybe_unused]] = \
      ROOT::set_PortableHostCollection_read_rules<COLLECTION>(EDM_STRINGIZE(COLLECTION))

#endif  // DataFormats_Portable_interface_PortableHostCollectionReadRules_h
