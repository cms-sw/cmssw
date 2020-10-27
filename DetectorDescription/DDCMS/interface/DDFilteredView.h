#ifndef DetectorDescription_DDCMS_DDFilteredView_h
#define DetectorDescription_DDCMS_DDFilteredView_h

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
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include <DD4hep/Filter.h>
#include <DD4hep/SpecParRegistry.h>
#include <DD4hep/Volumes.h>
#include <memory>
#include <tuple>
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
    bool nextChild();

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

    // Shape of current node
    dd4hep::Solid solid() const;

    // Name of current node
    std::string_view name() const;

    // Name of current node with namespace
    std::string_view fullName() const;

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
    T get(const std::string&);

    //! extract attribute value for current Node
    //  keep the maespace for comparison
    //  assume there are no regular expressions
    template <typename T>
    std::vector<T> getValuesNS(const std::string& key) {
      DDSpecParRefs refs;
      registry_->filter(refs, key);

      std::string path = this->path();
      for (const auto& specPar : refs) {
        for (const auto& part : specPar.second->paths) {
          bool flag(true);
          std::size_t from = 0;
          for (auto name : dd4hep::dd::split(part, "/")) {
            auto const& to = path.find(name, from);
            if (to == std::string::npos) {
              flag = false;
              break;
            } else {
              from = to;
            }
          }
          if (flag) {
            return specPar.second->value<std::vector<T>>(key);
          }
        }
      }
      return std::vector<T>();
    }

    //! extract another value from the same SpecPar
    //  call get<double> first to find a relevant one
    double getNextValue(const std::string&) const;

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

    //! get Iterator level
    const int level() const;

    //! transversed the DDFilteredView according
    //  to the given stack of sibling numbers
    bool goTo(const nav_type&);

    //! print Filter paths and selections
    void printFilter() const;

    //! find a current Node SpecPar that has at least
    //  one of the attributes
    template <class T, class... Ts>
    void findSpecPar(T const& first, Ts const&... rest) {
      currentSpecPar_ = find(first);
      if constexpr (sizeof...(rest) > 0) {
        // this line will only be instantiated if there are further
        // arguments. if rest... is empty, there will be no call to
        // findSpecPar(next).
        if (currentSpecPar_ == nullptr)
          findSpecPar(rest...);
      }
    }

  private:
    bool accept(std::string_view);
    int nodeCopyNo(const std::string_view) const;
    std::vector<std::pair<std::string, int>> toNodeNames(const std::string&);
    bool match(const std::string&, const std::vector<std::pair<std::string, int>>&) const;

    //! set the current node to the first sibling
    bool firstSibling();

    //! find a SpecPar in the registry by key
    //  doesn't really belong here, but allows
    //  speeding up avoiding the same search over
    //  the registry
    const DDSpecPar* find(const std::string&) const;
    void filter(DDSpecParRefs&, const std::string&) const;
    std::string_view front(const std::string_view) const;
    std::string_view back(const std::string_view) const;

    //! helper functions
    std::string_view nodeNameAt(int) const;
    const int nodeCopyNoAt(int) const;
    bool compareEqualName(const std::string_view, const std::string_view) const;
    std::tuple<std::string_view, std::string_view> alignNamespaces(std::string_view, std::string_view) const;
    bool compareEqualCopyNumber(const std::string_view, int) const;
    bool matchPath(const std::string_view) const;

    ExpandedNodes nodes_;
    std::vector<Iterator> it_;
    std::vector<std::unique_ptr<Filter>> filters_;
    Filter* currentFilter_ = nullptr;
    Node* node_ = nullptr;
    const DDSpecParRegistry* registry_;
    DDSpecParRefs refs_;
    const DDSpecPar* currentSpecPar_ = nullptr;
    int startLevel_;
  };
}  // namespace cms

//stream geoHistory
std::ostream& operator<<(std::ostream& os, const std::vector<const cms::Node*>& hst);

#endif
