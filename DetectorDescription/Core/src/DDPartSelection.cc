#include "DetectorDescription/Core/interface/Singleton.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/spirit/include/classic.hpp"

namespace boost {
  namespace spirit {
    namespace classic {}
  }  // namespace spirit
}  // namespace boost
using namespace boost::spirit::classic;

struct DDSelLevelCollector {
  std::string namespace_;
  std::string name_;
  int copyNo_;
  bool isCopyNoValid_;
  bool isChild_;
  std::vector<DDPartSelRegExpLevel>* p_;

  std::vector<DDPartSelRegExpLevel>* path(std::vector<DDPartSelRegExpLevel>* p = nullptr) {
    if (p) {
      p_ = p;
      namespace_ = "";
      name_ = "";
      copyNo_ = 0;
      isCopyNoValid_ = false;
      isChild_ = false;
    }
    return p_;
  }
};

void noNameSpace(char const* /*first*/, char const* /*last*/) {
  DDI::Singleton<DDSelLevelCollector>::instance().namespace_ = "";
}

/* Functor for the parser; it does not consume memory -
  pointers are only used to store references to memory
  managed elsewhere 
*/
struct DDSelLevelFtor {
  DDSelLevelFtor() : c_(DDI::Singleton<DDSelLevelCollector>::instance()) {}

  // parser calls this whenever a selection has been parsed ( //ns:nm[cn], /nm, //ns:nm, .... )
  void operator()(char const* /*first*/, char const* /*last*/) const {
    if (c_.path()) {
      if (c_.isCopyNoValid_ && c_.isChild_) {
        c_.path()->emplace_back(DDPartSelRegExpLevel(c_.namespace_, c_.name_, c_.copyNo_, ddchildposp));
        //edm::LogInfo("DDPartSelection")  << namespace_ << name_ << copyNo_ << ' ' << ddchildposp << std::endl;
      } else if (c_.isCopyNoValid_ && !c_.isChild_) {
        c_.path()->emplace_back(DDPartSelRegExpLevel(c_.namespace_, c_.name_, c_.copyNo_, ddanyposp));
        //      edm::LogInfo("DDPartSelection")  << namespace_ << name_ << copyNo_ << ' ' << ddanyposp << std::endl;
      } else if (!c_.isCopyNoValid_ && c_.isChild_) {
        c_.path()->emplace_back(DDPartSelRegExpLevel(c_.namespace_, c_.name_, c_.copyNo_, ddchildlogp));
        //      edm::LogInfo("DDPartSelection")  << namespace_ << name_ << copyNo_ << ' ' << ddchildlogp << std::endl;
      } else if (!c_.isCopyNoValid_ && !c_.isChild_) {
        c_.path()->emplace_back(DDPartSelRegExpLevel(c_.namespace_, c_.name_, c_.copyNo_, ddanylogp));
        //      edm::LogInfo("DDPartSelection")  << namespace_ << name_ << copyNo_ << ' ' << ddanylogp << std::endl;
      }
      c_.namespace_ = "";
      c_.name_ = "";
      c_.isCopyNoValid_ = false;
    }
  }

  DDSelLevelCollector& c_;
};

struct DDIsChildFtor {
  void operator()(char const* first, char const* last) const {
    DDSelLevelCollector& sl = DDI::Singleton<DDSelLevelCollector>::instance();
    if ((last - first) > 1)
      sl.isChild_ = false;
    if ((last - first) == 1)
      sl.isChild_ = true;
    //edm::LogInfo("DDPartSelection")  << "DDIsChildFtor  isChild=" << (last-first) << std::endl;
  }
};

struct DDNameSpaceFtor {
  void operator()(char const* first, char const* last) const {
    DDSelLevelCollector& sl = DDI::Singleton<DDSelLevelCollector>::instance();
    sl.namespace_.assign(first, last);
    // edm::LogInfo("DDPartSelection")  << "DDNameSpaceFtor singletonname=" << DDI::Singleton<DDSelLevelCollector>::instance().namespace_ << std::endl;
  }

  DDSelLevelFtor* selLevelFtor_;
};

struct DDNameFtor {
  void operator()(char const* first, char const* last) const {
    DDSelLevelCollector& sl = DDI::Singleton<DDSelLevelCollector>::instance();
    sl.name_.assign(first, last);
    // edm::LogInfo("DDPartSelection")  << "DDNameFtor singletonname=" << Singleton<DDSelLevelCollector>::instance().name_ << std::endl;
  }
};

struct DDCopyNoFtor {
  void operator()(int i) const {
    DDSelLevelCollector& sl = DDI::Singleton<DDSelLevelCollector>::instance();
    sl.copyNo_ = i;
    sl.isCopyNoValid_ = true;
    // edm::LogInfo("DDPartSelection")  << "DDCopyNoFtor ns=" << i;
  }
};

/** A boost::spirit parser for the <SpecPar path="xxx"> syntax */
struct SpecParParser : public grammar<SpecParParser> {
  template <typename ScannerT>
  struct definition {
    definition(SpecParParser const& /*self*/) {
      Selection  //= FirstStep[selLevelFtor()]
                 //>> *SelectionStep[selLevelFtor()]
          = +SelectionStep[selLevelFtor()];

      FirstStep = Descendant >> Part;

      Part = PartNameCopyNumber | PartName;

      PartNameCopyNumber = PartName >> CopyNumber;

      SelectionStep = NavigationalElement[isChildFtor()] >> Part;

      NavigationalElement = Descendant | Child;

      CopyNumber = ch_p('[') >> int_p[copyNoFtor()] >> ch_p(']');

      PartName = NameSpaceName | SimpleName[nameFtor()][&noNameSpace];

      SimpleName = +(alnum_p | ch_p('_') | ch_p('.') | ch_p('*'));

      NameSpaceName = SimpleName[nameSpaceFtor()] >> ':' >> SimpleName[nameFtor()];

      Descendant = ch_p('/') >> ch_p('/');

      Child = ch_p('/');
    }

    rule<ScannerT> Selection, FirstStep, Part, SelectionStep, NavigationalElement, CopyNumber, PartName,
        PartNameCopyNumber, NameSpaceName, SimpleName, Descendant, Child;

    rule<ScannerT> const& start() const { return Selection; }

    DDSelLevelFtor& selLevelFtor() { return DDI::Singleton<DDSelLevelFtor>::instance(); }

    DDNameFtor& nameFtor() {
      static DDNameFtor f_;
      return f_;
    }

    DDNameSpaceFtor& nameSpaceFtor() {
      static DDNameSpaceFtor f_;
      return f_;
    }

    DDIsChildFtor& isChildFtor() {
      static DDIsChildFtor f_;
      return f_;
    }

    DDCopyNoFtor& copyNoFtor() {
      static DDCopyNoFtor f_;
      return f_;
    }
  };
};

DDPartSelectionLevel::DDPartSelectionLevel(const DDLogicalPart& lp, int c, ddselection_type t)
    : lp_(lp), copyno_(c), selectionType_(t) {}

void DDTokenize2(const std::string& sel, std::vector<DDPartSelRegExpLevel>& path) {
  static SpecParParser parser;
  DDI::Singleton<DDSelLevelCollector>::instance().path(&path);
  bool result = parse(sel.c_str(), parser).full;
  if (!result) {
    edm::LogError("DDPartSelection") << "DDTokenize2() error in parsing of " << sel << std::endl;
  }
}

std::ostream& operator<<(std::ostream& o, const DDPartSelection& p) {
  DDPartSelection::const_iterator it(p.begin()), ed(p.end());
  for (; it != ed; ++it) {
    const DDPartSelectionLevel& lv = *it;
    switch (lv.selectionType_) {
      case ddanylogp:
        o << "//" << lv.lp_.ddname();
        break;
      case ddanyposp:
        o << "//" << lv.lp_.ddname() << '[' << lv.copyno_ << ']';
        break;
      case ddchildlogp:
        o << "/" << lv.lp_.ddname();
        break;
      case ddchildposp:
        o << "/" << lv.lp_.ddname() << '[' << lv.copyno_ << ']';
        break;
      default:
        o << "{Syntax ERROR}";
    }
  }
  return o;
}

std::ostream& operator<<(std::ostream& os, const std::vector<DDPartSelection>& v) {
  std::vector<DDPartSelection>::const_iterator it(v.begin()), ed(v.end());
  for (; it != (ed - 1); ++it) {
    os << *it << std::endl;
  }
  if (it != ed) {
    ++it;
    os << *it;
  }
  return os;
}

// explicit template instantiation.

template class DDI::Singleton<DDSelLevelFtor>;
//template class DDI::Singleton<DDI::Store<DDName, DDSelLevelCollector> >;
template class DDI::Singleton<DDSelLevelCollector>;
#include <DetectorDescription/Core/interface/Singleton.icc>
