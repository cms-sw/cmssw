#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <DD4hep/Detector.h>
#include <DD4hep/DetectorTools.h>
#include <DD4hep/Volumes.h>
#include <XML/DocumentHandler.h>
#include <XML/XMLElements.h>

#include <iostream>

namespace cms {

  DDDetector::DDDetector(const std::string& tag, const std::string& fileName, bool bigXML) : m_tag(tag) {
    //We do not want to use any previously created TGeoManager but we do want to reset after we are done.
    auto oldGeoManager = gGeoManager;
    gGeoManager = nullptr;
    auto resetManager = [oldGeoManager](TGeoManager*) { gGeoManager = oldGeoManager; };
    std::unique_ptr<TGeoManager, decltype(resetManager)> sentry(oldGeoManager, resetManager);

    std::string tagStr(m_tag);
    bool makePayload = false;
    if (tagStr == "make-payload") {
      makePayload = true;
      tagStr = "";
    }
    m_description = &dd4hep::Detector::getInstance(tagStr);
    m_description->addExtension<cms::DDVectorsMap>(&m_vectors);
    //only validate if using XML
    auto parsingContext =
        new cms::DDParsingContext(*m_description, makePayload, not bigXML);  // Removed at end of constructor
    m_description->addExtension<cms::DDParsingContext>(parsingContext);
    m_description->addExtension<dd4hep::PartSelectionMap>(&m_partsels);
    m_description->addExtension<dd4hep::SpecParRegistry>(&m_specpars);
    m_description->setStdConditions("NTP");
    edm::LogVerbatim("Geometry") << "DDDetector::ctor Setting DD4hep STD conditions to NTP";
    if (bigXML)
      processXML(fileName);
    else
      process(fileName);
    if (makePayload == false)  // context no longer needed if not making payloads
      m_description->removeExtension<cms::DDParsingContext>();
  }

  void DDDetector::process(const std::string& fileName) {
    std::string name("DD4hep_CompactLoader");
    const char* files[] = {fileName.c_str(), nullptr};
    m_description->apply(name.c_str(), 2, (char**)files);
  }

  void DDDetector::processXML(const std::string& xml) {
    edm::LogVerbatim("Geometry") << "DDDetector::processXML process string size " << xml.size() << " with max_size "
                                 << xml.max_size();
    edm::LogVerbatim("Geometry") << "DDDetector::processXML XML string contents = " << xml.substr(0, 800);
    std::string name("DD4hep_XMLProcessor");
    dd4hep::xml::DocumentHolder doc(dd4hep::xml::DocumentHandler().parse(xml.c_str(), xml.length()));

    char* args[] = {(char*)doc.root().ptr(), nullptr};
    m_description->apply(name.c_str(), 1, (char**)args);
  }

  dd4hep::Volume DDDetector::worldVolume() const {
    assert(m_description);
    return m_description->worldVolume();
  }

  dd4hep::DetElement DDDetector::world() const {
    assert(m_description);
    return m_description->world();
  }

  TGeoManager& DDDetector::manager() const {
    assert(m_description);
    return m_description->manager();
  }

  dd4hep::DetElement DDDetector::findElement(const std::string& path) const {
    assert(m_description);
    return dd4hep::detail::tools::findElement(*m_description, path);
  }
}  // namespace cms
