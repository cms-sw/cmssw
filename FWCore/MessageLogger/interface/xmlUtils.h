#ifndef FWCore_MessageLogger_xmlUtils_h
#define FWCore_MessageLogger_xmlUtils_h

#include "tinyxml2.h"

#include <format>
#include <ostream>
#include <string_view>

/**
 * These are really an implementation detail of JobReport, but are in interface because of tests
 */
namespace edm::xml {
  template <typename T>
  void addElement(char const* elementName, T&& elementValue, std::string_view suffix, std::ostream& os) {
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLPrinter printer(nullptr, true);  // compact mode - no extra whitespace

    const char* cstr;
    std::string formatted;

    using DecayedT = std::decay_t<T>;
    if constexpr (std::is_same_v<DecayedT, std::string>) {
      cstr = elementValue.c_str();
    } else if constexpr (std::is_same_v<DecayedT, char const*> || std::is_same_v<DecayedT, char*>) {
      cstr = elementValue;
    } else {
      formatted = std::format("{}", elementValue);
      cstr = formatted.c_str();
    }

    // Memory of element is managed by doc, so we don't need to delete it
    auto* element = doc.NewElement(elementName);
    element->SetText(cstr);
    element->Accept(&printer);
    os << printer.CStr() << suffix;
  }
}  // namespace edm::xml

#endif
