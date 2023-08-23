#ifndef PhysicsTools_FWLite_ScannerHelpers_h
#define PhysicsTools_FWLite_ScannerHelpers_h

#include <string>
#include <FWCore/Reflection/interface/TypeWithDict.h>
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TGraph.h>

// AFAIK These includes are needed to build the dictionary
// but must be kept hidden if CINT sees this class
#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "CommonTools/Utils/interface/parser/ExpressionPtr.h"
#include "CommonTools/Utils/interface/parser/SelectorPtr.h"
#include "CommonTools/Utils/interface/parser/ExpressionBase.h"
#include "CommonTools/Utils/interface/parser/SelectorBase.h"
#endif

namespace helper {
  /** Class helper::Parser has collection of useful static methods related to StringParser that can be exported to CINT via dictionaries.
     *  It's mosly meant to be used through the helper::ScannerBase class. */
  class Parser {
  public:
    /// Empty constructor, necessary for Root, useless
    Parser() {}
    /// Parse an expression for a given object type (using lazy parsing when resolving methods)
    static reco::parser::ExpressionPtr makeExpression(const std::string &expr, const edm::TypeWithDict &type);
    /// Parse an expression for a given object type (using lazy parsing when resolving methods)
    static reco::parser::SelectorPtr makeSelector(const std::string &expr, const edm::TypeWithDict &type);
    /// Perform the type deduction form edm::Wrapper<C> to C::value_type and resolves typedefs
    static edm::TypeWithDict elementType(const edm::TypeWithDict &wrapperType);

    //--- we define also dictionaries for these two trivial functions that should be callable even by CINT
    //    because otherwise sometimes CINT crashes even on the creation and destruction of edm::ObjectWithDict
    /// Make a edm::ObjectWithDict(type, obj) and pass it to the selector
    static bool test(const reco::parser::SelectorPtr &sel, const edm::TypeWithDict type, const void *obj);
    /// Make a edm::ObjectWithDict(type, obj) and pass it to the expression
    static double eval(const reco::parser::ExpressionPtr &sel, const edm::TypeWithDict type, const void *obj);
  };

  /** Class helper::ScannerBase: tool to print or histogram proprieties of an object using the dictionary,
     *  The class is generic, but each instance is restricted to the type of the objects to inspect, fixed at construction time. */
  class ScannerBase {
  public:
    /// Empty constructor, necessary for Root, DO NOT USE
    ScannerBase() {}
    /// Constructor taking as argument the type of the individual object passed to the scanner
    ScannerBase(const edm::TypeWithDict &objType) : objType_(objType), cuts_(1), ignoreExceptions_(false) {}

    /// Add an expression to be evaluated on the objects
    /// Returns false if the parsing failed
    bool addExpression(const char *expr);
    /// Clear all the expressions
    void clearExpressions() { exprs_.clear(); }
    /// Number of valid expressions
    size_t numberOfExpressions() const { return exprs_.size(); }

    /// Set the default cut that is applied to the events
    bool setCut(const char *cut);
    /// Clear the default cut
    void clearCut();
    /// Add one extra cut that can be evaluated separately (as if it was an expression)
    bool addExtraCut(const char *cut);
    /// Clear all extra cuts ;
    void clearExtraCuts();
    /// Number of extra cuts
    size_t numberOfExtraCuts() const { return cuts_.size() - 1; }

    /// Check if the object passes the default cut (icut=0) or any extra cut (icut = 1 .. numberOfExtraCuts)
    /// Obj must point to an object of the type used to construct this ScannerBase
    bool test(const void *obj, size_t icut = 0) const;

    /// Evaluate one of the expressions set in this scanner
    /// Obj must point to an object of the type used to construct this ScannerBase
    double eval(const void *obj, size_t iexpr = 0) const;

    /// Print out in a single row all the expressions for this object
    /// Obj must point to an object of the type used to construct this ScannerBase
    void print(const void *obj) const;

    /// Fill the histogram with the first expression evaluated on the object, if it passes the default cut
    /// Obj must point to an object of the type used to construct this ScannerBase
    void fill1D(const void *obj, TH1 *hist) const;

    /// Fill the histogram with (x,y) equal to the first and second expressions evaluated on the object, if it passes the default cut
    /// Obj must point to an object of the type used to construct this ScannerBase
    void fill2D(const void *obj, TH2 *hist2d) const;

    /// Fill the graph with (x,y) equal to the first and second expressions evaluated on the object, if it passes the default cut
    /// Obj must point to an object of the type used to construct this ScannerBase
    void fillGraph(const void *obj, TGraph *graph) const;

    /// Fill the profile histogram with (x,y) equal to the first and second expressions evaluated on the object, if it passes the default cut
    /// Obj must point to an object of the type used to construct this ScannerBase
    void fillProf(const void *obj, TProfile *prof) const;

    /// If set to true, exceptions are silently ignored: test will return 'false', and 'eval' will return 0.
    /// If left to the default value, false, for each exception a printout is done.
    void setIgnoreExceptions(bool ignoreThem) { ignoreExceptions_ = ignoreThem; }

  private:
    edm::TypeWithDict objType_;
    std::vector<reco::parser::ExpressionPtr> exprs_;

    /// The first one is the default cut, the others are the extra ones
    std::vector<reco::parser::SelectorPtr> cuts_;

    /// See setIgnoreExceptions to find out what this means
    bool ignoreExceptions_;
  };
}  // namespace helper

#endif
