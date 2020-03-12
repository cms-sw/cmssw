#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <iostream>
#include <memory>
#include <utility>
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

class DDCompactView;
class DDLElementRegistry;

DDXMLElement::DDXMLElement(DDLElementRegistry* myreg) : myRegistry_(myreg), attributes_(), text_(), autoClear_(false) {}

DDXMLElement::DDXMLElement(DDLElementRegistry* myreg, const bool& clearme)
    : myRegistry_(myreg), attributes_(), text_(), autoClear_(clearme) {}

// For pre-processing, after attributes are loaded.  Default, do nothing!
void DDXMLElement::preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {}

// This loads the attributes into the attributes_ std::vector.
void DDXMLElement::loadAttributes(const std::string& elemName,
                                  const std::vector<std::string>& names,
                                  const std::vector<std::string>& values,
                                  const std::string& nmspace,
                                  DDCompactView& cpv) {
  attributes_.resize(attributes_.size() + 1);
  DDXMLAttribute& tAttributes = attributes_.back();

  // adds attributes
  for (size_t i = 0; i < names.size(); ++i) {
    tAttributes.insert(std::make_pair(names[i], values[i]));
  }

  preProcessElement(elemName, nmspace, cpv);
}

// clear data.
void DDXMLElement::clear(void) {
  text_.clear();
  attributes_.clear();
  attributeAccumulator_.clear();
}

// Access to current attributes by name.
const std::string& DDXMLElement::getAttribute(const std::string& name) const {
  static const std::string ldef;
  if (!attributes_.empty())
    return get(name, attributes_.size() - 1);
  return ldef;
}

const DDXMLAttribute& DDXMLElement::getAttributeSet(size_t aIndex) const { return attributes_[aIndex]; }

const DDName DDXMLElement::getDDName(const std::string& defaultNS, const std::string& attname, size_t aIndex) {
  if (aIndex < attributes_.size() && attributes_[aIndex].find(attname) != attributes_[aIndex].end()) {
    std::string ns = defaultNS;
    // For the user to fully control namespaces they must provide for
    // all name attributes something of the form, for example:
    //        <Solid name="ns:name" ...
    // If defaultNS is "!" (magic I don't like) then find and set
    // the namespace properly.
    if (defaultNS == "!") {
      ns = "";
    }
    const std::string& name = attributes_[aIndex].find(attname)->second;
    std::string rn = name;
    size_t foundColon = name.find(':');
    if (foundColon != std::string::npos) {
      ns = name.substr(0, foundColon);
      rn = name.substr(foundColon + 1);
    }
    return DDName(rn, ns);
  }
  std::string msg = "DDXMLElement:getDDName failed.  It was asked to make ";
  msg += "a DDName using attribute: " + attname;
  msg += " in position: " + std::to_string(aIndex) + ".  There are ";
  msg += std::to_string(attributes_.size()) + " entries in the element.";
  throwError(msg);
  return DDName("justToCompile", "justToCompile");  // used to make sure it compiles
}

// Returns a specific value from the aIndex set of attributes.
const std::string& DDXMLElement::get(const std::string& name, const size_t aIndex) const {
  static const std::string sts;
  if (aIndex < attributes_.size()) {
    DDXMLAttribute::const_iterator it = attributes_[aIndex].find(name);
    if (attributes_[aIndex].end() == it) {
      return sts;
    } else
      return (it->second);
  }
  std::string msg = "DDXMLElement:get failed.  It was asked for attribute " + name;
  msg += " in position " + std::to_string(aIndex) + " when there are only ";
  msg += std::to_string(attributes_.size()) + " in the element storage.\n";
  throwError(msg);
  // meaningless...
  return sts;
}

// Returns a specific set of values as a std::vector of std::strings,
// given the attribute name.
std::vector<std::string> DDXMLElement::getVectorAttribute(const std::string& name) {
  //  The idea here is that the attributeAccumulator_ is a cache of
  //  on-the-fly generation from the std::vector<DDXMLAttribute> and the
  //  reason is simply to speed things up if it is requested more than once.
  std::vector<std::string> tv;
  AttrAccumType::const_iterator ita = attributeAccumulator_.find(name);
  if (ita != attributeAccumulator_.end()) {
    tv = attributeAccumulator_[name];
    if (tv.size() < attributes_.size()) {
      appendAttributes(tv, name);
    }
  } else {
    if (!attributes_.empty()) {
      appendAttributes(tv, name);
    } else {
      //      throw cms::Exception("DDException") << msg;
    }
  }
  return tv;
}

// Default do-nothing processElementBases.
void DDXMLElement::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  loadText(std::string());
  if (autoClear_)
    clear();
}

void DDXMLElement::loadText(const std::string& inText) { text_.emplace_back(inText); }

void DDXMLElement::appendText(const std::string& inText) {
  static const std::string cr("\n");
  if (!text_.empty()) {
    text_[text_.size() - 1] += cr;
    text_[text_.size() - 1] += inText;
  } else {
    std::string msg = "DDXMLElement::appendText could not append to non-existent text.";
    throwError(msg);
  }
}

const std::string DDXMLElement::getText(size_t tindex) const {
  if (tindex > text_.size()) {
    std::string msg = "DDXMLElement::getText tindex is greater than text_.size()).";
    throwError(msg);
  }
  return text_[tindex];
}

bool DDXMLElement::gotText(void) const {
  if (!text_.empty())
    return true;
  return false;
}

std::ostream& operator<<(std::ostream& os, const DDXMLElement& element) {
  element.stream(os);
  return os;
}

void DDXMLElement::stream(std::ostream& os) const {
  os << "Output of current element attributes:" << std::endl;
  for (const auto& attribute : attributes_) {
    for (DDXMLAttribute::const_iterator it = attribute.begin(); it != attribute.end(); ++it)
      os << it->first << " = " << it->second << "\t";
    os << std::endl;
  }
}

void DDXMLElement::appendAttributes(std::vector<std::string>& tv, const std::string& name) {
  for (size_t i = tv.size(); i < attributes_.size(); ++i) {
    DDXMLAttribute::const_iterator itnv = attributes_[i].find(name);
    if (itnv != attributes_[i].end())
      tv.emplace_back(itnv->second);
    else
      tv.emplace_back("");
  }
}

// Number of elements accumulated.
size_t DDXMLElement::size(void) const { return attributes_.size(); }

std::vector<DDXMLAttribute>::const_iterator DDXMLElement::begin(void) {
  myIter_ = attributes_.begin();
  return attributes_.begin();
}

std::vector<DDXMLAttribute>::const_iterator DDXMLElement::end(void) {
  myIter_ = attributes_.end();
  return attributes_.end();
}

std::vector<DDXMLAttribute>::const_iterator& DDXMLElement::operator++(int inc) {
  myIter_ = myIter_ + inc;
  return myIter_;
}

const std::string& DDXMLElement::parent(void) const { return parentElement_; }

void DDXMLElement::setParent(const std::string& pename) { parentElement_ = pename; }

void DDXMLElement::setSelf(const std::string& sename) { myElement_ = sename; }

bool DDXMLElement::isEmpty(void) const { return (attributes_.empty() ? true : false); }

void DDXMLElement::throwError(const std::string& keyMessage) const {
  std::string msg = keyMessage + "\n";
  msg += " Element " + myElement_ + "\n";

  throw cms::Exception("DDException") << msg;
}
