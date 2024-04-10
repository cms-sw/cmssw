#ifndef DataFormats_Portable_interface_PortableHostObjectReadRules_h
#define DataFormats_Portable_interface_PortableHostObjectReadRules_h

#include <TGenericClassInfo.h>
#include <TVirtualObject.h>

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "FWCore/Utilities/interface/concatenate.h"
#include "FWCore/Utilities/interface/stringize.h"

// read function for PortableHostObject, called for every event
template <typename T>
static void readPortableHostObject_v1(char *target, TVirtualObject *from_buffer) {
  // extract the actual types
  using Object = T;
  using Product = typename Object::Product;

  // valid only for PortableHostObject<T>
  static_assert(std::is_same_v<Object, PortableHostObject<Product>>);

  // proxy for the object being read from file
  struct OnFile {
    Product *product_;
  };

  // address in memory of the buffer containing the object being read from file
  char *address = static_cast<char *>(from_buffer->GetObject());
  // offset of the "product_" data member
  static ptrdiff_t product_offset = from_buffer->GetClass()->GetDataMemberOffset("product_");
  // pointer to the Product object being read from file
  OnFile onfile = {*(Product **)(address + product_offset)};

  // pointer to the Object object being constructed in memory
  Object *newObj = (Object *)target;

  // move the data from the on-file layout to the newly constructed object
  Object::ROOTReadStreamer(newObj, *onfile.product_);
}

// put set_PortableHostObject_read_rules in the ROOT namespace to let it forward declare GenerateInitInstance
namespace ROOT {

  // set the read rules for PortableHostObject<T>;
  // this is called only once, when the dictionary is loaded.
  template <typename T>
  static bool set_PortableHostObject_read_rules(std::string const &type) {
    // forward declaration
    TGenericClassInfo *GenerateInitInstance(T const *);

    // build the read rules
    std::vector<ROOT::Internal::TSchemaHelper> readrules(1);
    ROOT::Internal::TSchemaHelper &rule = readrules[0];
    rule.fTarget = "buffer_,product_";
    rule.fSourceClass = type;
    rule.fSource = type + "::Product* product_;";
    rule.fCode = type + "::ROOTReadStreamer(newObj, *onfile.product_)";
    rule.fVersion = "[1-]";
    rule.fChecksum = "";
    rule.fInclude = "";
    rule.fEmbed = false;
    rule.fFunctionPtr = reinterpret_cast<void *>(::readPortableHostObject_v1<T>);
    rule.fAttributes = "";

    // set the read rules
    TGenericClassInfo *instance = GenerateInitInstance((T const *)nullptr);
    instance->SetReadRules(readrules);

    return true;
  }
}  // namespace ROOT

#define SET_PORTABLEHOSTOBJECT_READ_RULES(OBJECT)                                                      \
  static bool EDM_CONCATENATE(set_PortableHostObject_read_rules_done_at_, __LINE__) [[maybe_unused]] = \
      ROOT::set_PortableHostObject_read_rules<OBJECT>(EDM_STRINGIZE(OBJECT))

#endif  // DataFormats_Portable_interface_PortableHostObjectReadRules_h
