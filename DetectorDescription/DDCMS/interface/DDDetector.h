#ifndef DETECTOR_DESCRIPTION_DD_DETECTOR_H
#define DETECTOR_DESCRIPTION_DD_DETECTOR_H

#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include <DD4hep/Detector.h>
#include <string>

class TGeoManager;

namespace cms {
  class DDDetector {
  public:
    using Detector = dd4hep::Detector;
    using DetElement = dd4hep::DetElement;
    using HandleMap = dd4hep::Detector::HandleMap;
    using PlacedVolume = dd4hep::PlacedVolume;
    using Volume = dd4hep::Volume;

    explicit DDDetector(const std::string&, const std::string&, bool bigXML = false);
    DDDetector() = delete;

    DDVectorsMap const& vectors() const { return m_vectors; }

    DDPartSelectionMap const& partsels() const { return m_partsels; }

    DDSpecParRegistry const& specpars() const { return m_specpars; }

    //! Handle to the world volume containing everything
    Volume worldVolume() const;

    //! Access to the physical volume of the world detector element
    PlacedVolume worldPlacement() const;

    //! Reference to the top-most (world) detector element
    DetElement world() const;

    //! The map of sub-detectors
    const HandleMap& detectors() const;

    //! The geometry manager of this instance
    TGeoManager& manager() const;

    //! Find DetElement as child of the top level volume by it's absolute path
    DetElement findElement(const std::string&) const;

    Detector const* description() const { return m_description; }

  private:
    void process(const std::string&);
    void processXML(const std::string&);

    Detector* m_description = nullptr;
    DDVectorsMap m_vectors;
    DDPartSelectionMap m_partsels;
    DDSpecParRegistry m_specpars;
    const std::string m_tag;
  };
}  // namespace cms

#endif
