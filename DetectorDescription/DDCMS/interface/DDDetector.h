#ifndef DETECTOR_DESCRIPTION_DD_DETECTOR_H
#define DETECTOR_DESCRIPTION_DD_DETECTOR_H

#include <DD4hep/Detector.h>
#include <DD4hep/SpecParRegistry.h>
#include <string>

class TGeoManager;

namespace cms {
  class DDDetector {
  public:
    explicit DDDetector(const std::string&, const std::string&, bool bigXML = false);
    DDDetector() = delete;

    dd4hep::VectorsMap const& vectors() const { return m_vectors; }

    dd4hep::PartSelectionMap const& partsels() const { return m_partsels; }

    dd4hep::SpecParRegistry const& specpars() const { return m_specpars; }

    //! Handle to the world volume containing everything
    dd4hep::Volume worldVolume() const;

    //! Access to the physical volume of the world detector element
    dd4hep::PlacedVolume worldPlacement() const;

    //! Reference to the top-most (world) detector element
    dd4hep::DetElement world() const;

    //! The map of sub-detectors
    const dd4hep::Detector::HandleMap& detectors() const;

    //! The geometry manager of this instance
    TGeoManager& manager() const;

    //! Find DetElement as child of the top level volume by it's absolute path
    dd4hep::DetElement findElement(const std::string&) const;

    dd4hep::Detector const* description() const { return m_description; }

  private:
    void process(const std::string&);
    void processXML(const std::string&);

    dd4hep::Detector* m_description = nullptr;
    dd4hep::VectorsMap m_vectors;
    dd4hep::PartSelectionMap m_partsels;
    dd4hep::SpecParRegistry m_specpars;
    const std::string m_tag;
  };
}  // namespace cms

#endif
