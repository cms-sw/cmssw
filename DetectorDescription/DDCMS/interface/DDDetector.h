#ifndef DetectorDescription_DDCMS_DDDetector_h
#define DetectorDescription_DDCMS_DDDetector_h

#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"
#include <DD4hep/Detector.h>
#include <DD4hep/SpecParRegistry.h>
#include <string>

class TGeoManager;

namespace cms {
  class DDDetector {
  public:
    explicit DDDetector(const std::string&, const std::string&, bool bigXML = false, bool makePayload = false);
    DDDetector() = delete;

    cms::DDVectorsMap const& vectors() const { return m_vectors; }

    dd4hep::PartSelectionMap const& partsels() const { return m_partsels; }

    dd4hep::SpecParRegistry const& specpars() const { return m_specpars; }

    //! Handle to the world volume containing everything
    dd4hep::Volume worldVolume() const;

    //! Reference to the top-most (world) detector element
    dd4hep::DetElement world() const;

    //! The geometry manager of this instance
    TGeoManager& manager() const;

    //! Find DetElement as child of the top level volume by it's absolute path
    dd4hep::DetElement findElement(const std::string&) const;

    dd4hep::Detector const* description() const { return m_description; }

    DDParsingContext* m_context;

  private:
    void process(const std::string&);
    void processXML(const std::string&);

    dd4hep::Detector* m_description = nullptr;
    cms::DDVectorsMap m_vectors;
    dd4hep::PartSelectionMap m_partsels;
    dd4hep::SpecParRegistry m_specpars;
    const std::string m_tag;
  };
}  // namespace cms

#endif
