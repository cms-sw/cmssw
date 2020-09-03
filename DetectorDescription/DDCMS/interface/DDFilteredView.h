#ifndef DETECTOR_DESCRIPTION_DD_FILTERED_VIEW_H
#define DETECTOR_DESCRIPTION_DD_FILTERED_VIEW_H

// -*- C++ -*-
//
// Package:    DetectorDescription/Core
// Class:      DDFilteredView
//
/**\class DDFilteredView

 Description: Filtered View of a Tree

 Implementation:
     Filter criteria is defined in XML
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 30 Jan 2019 09:24:30 GMT
//
//
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "DetectorDescription/DDCMS/interface/ExpandedNodes.h"
#include <DD4hep/Filter.h>
#include <DD4hep/SpecParRegistry.h>
#include <DD4hep/Volumes.h>
#include <memory>
#include <vector>

namespace cms {

  struct DDSolid {
    explicit DDSolid(dd4hep::Solid s) : solid_(s) {}
    dd4hep::Solid solid() const { return solid_; }
    dd4hep::Solid solidA() const;
    dd4hep::Solid solidB() const;
    const std::vector<double> parameters() const;

  private:
    dd4hep::Solid solid_;
  };

  class DDDetector;
  class DDCompactView;

  using Volume = dd4hep::Volume;
  using PlacedVolume = dd4hep::PlacedVolume;
  using ExpandedNodes = cms::ExpandedNodes;
  using Filter = dd4hep::Filter;
  using DDSpecPar = dd4hep::SpecPar;
  using DDSpecParRefs = dd4hep::SpecParRefs;
  using DDSpecParRegistry = dd4hep::SpecParRegistry;
  using DDVectorsMap = dd4hep::VectorsMap;
  using Iterator = TGeoIterator;
  using Node = TGeoNode;
  using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
  using RotationMatrix = ROOT::Math::Rotation3D;

  struct DDFilter {
    DDFilter(const std::string& attribute = "", const std::string& value = "")
        : m_attribute(attribute), m_value(value) {}
    const std::string& attribute() const { return m_attribute; }
    const std::string& value() const { return m_value; }

  private:
    const std::string m_attribute;
    const std::string m_value;
  };

  class DDFilteredView {
  public:
    using nav_type = std::vector<int>;

    DDFilteredView(const DDDetector*, const Volume);
    DDFilteredView(const DDCompactView&, const cms::DDFilter&);
    DDFilteredView() = delete;

    //! The numbering history of the current node
    const ExpandedNodes& history();

    //! The physical volume of the current node
    const PlacedVolume volume() const;

    //! The full path to the current node
    const std::string path() const;

    const std::vector<const Node*> geoHistory() const;

    //! The list of the volume copy numbers
    //  along the full path to the current node
    const std::vector<int> copyNos() const;

    template <typename... Ts>
    auto copyNumbers(Ts&&... ts) const -> decltype(copyNos(std::forward<Ts>(ts)...)) {
      return copyNos(std::forward<Ts>(ts)...);
    }

    //! The absolute translation of the current node
    // Return value is Double_t translation[3] with x, y, z elements.
    const Double_t* trans() const;
    const Translation translation() const;
    const Translation translation(const std::vector<Node*>&) const;

    //! The absolute rotation of the current node
    const Double_t* rot() const;
    const RotationMatrix rotation() const;
    void rot(dd4hep::Rotation3D& matrixOut) const;

    //! User specific data
    void mergedSpecifics(DDSpecParRefs const&);
    const cms::DDSpecParRefs specpars() const { return refs_; }

    //! set the current node to the first child
    bool firstChild();

    //! set the current node to the child in path
    std::vector<std::vector<Node*>> children(const std::string& path);

    //! set the current node to the next sibling
    bool nextSibling();

    //! set the current node to the next sub sibling
    bool sibling();

    //! count the number of children matching selection
    bool checkChild();

    //! set the current node to the parent node ...
    bool parent();

    //! set current node to the next node in the filtered tree
    bool next(int);

    //! set current node to the child node in the filtered tree
    void down();

    //! set current node to the parent node in the filtered tree
    void up();

    // Get shape pointer of current node.
    // Caller must check that current node matches desired type
    // before calling this function.

    template <class Shape>
    const Shape* getShapePtr() const {
      Volume currVol = node_->GetVolume();
      return (dynamic_cast<Shape*>(currVol->GetShape()));
    }

    // Shape of current node

    template <class Shape>
    bool isA() const {
      return dd4hep::isA<Shape>(solid());
    }

    bool isABox() const { return isA<dd4hep::Box>(); }

    bool isAConeSeg() const { return isA<dd4hep::ConeSegment>(); }

    bool isAPseudoTrap() const { return isA<dd4hep::PseudoTrap>(); }

    bool isATrapezoid() const { return isA<dd4hep::Trap>(); }

    bool isATruncTube() const { return isA<dd4hep::TruncatedTube>(); }

    bool isATubeSeg() const { return isA<dd4hep::Tube>(); }

    bool isASubtraction() const {
      return (isA<dd4hep::SubtractionSolid>() and not isA<dd4hep::TruncatedTube>() and not isA<dd4hep::PseudoTrap>());
    }

    dd4hep::Solid solid() const;

    // Name of current node
    std::string_view name() const;

    // Copy number of current node
    unsigned short copyNum() const;

    // Material name of current node
    std::string_view materialName() const;

    //! extract shape parameters
    const std::vector<double> parameters() const;

    const cms::DDSolidShape shape() const;

    // Convert new DD4hep shape id to an old DD one
    LegacySolidShape legacyShape(const cms::DDSolidShape shape) const;

    //! extract attribute value
    template <typename T>
    T get(const std::string&) const;

    //! extract attribute value in SpecPar
    template <typename T>
    T get(const std::string&, const std::string&) const;

    //! convert an attribute value from SpecPar
    //  without passing it through an evaluator,
    //  e.g. the original values have no units
    std::vector<double> get(const std::string&, const std::string&) const;

    std::string_view getString(const std::string&) const;

    //! return the stack of sibling numbers which indicates
    //  the current position in the DDFilteredView
    nav_type navPos() const;

    //! print Filter paths and selections
    void printFilter() const;

  private:
    bool accept(std::string_view);
    int nodeCopyNo(const std::string_view) const;
    std::vector<std::pair<std::string_view, int>> toNodeNames(const std::string&);
    bool match(const std::string&, const std::vector<std::pair<std::string_view, int>>&) const;
    const TClass* getShape() const;

    //! set the current node to the first sibling
    bool firstSibling();

    ExpandedNodes nodes_;
    std::vector<Iterator> it_;
    std::vector<std::unique_ptr<Filter>> filters_;
    Filter* currentFilter_ = nullptr;
    Node* node_ = nullptr;
    const DDSpecParRegistry* registry_;
    DDSpecParRefs refs_;
    int startLevel_;
  };
}  // namespace cms

#endif
