#ifndef Framework_Selector_h
#define Framework_Selector_h

/*----------------------------------------------------------------------
  
Classes for all "selector" objects, used to select
EDProducts based on information in the associated Provenance.

Developers who make their own Selector class should inherit
from SelectorBase.

Users can use the classes defined below

  ModuleLabelSelector
  ProcessNameSelector
  ProductInstanceNameSelector

Users can also use the class Selector, which can be constructed given a
logical expression formed from any other selectors, combined with &&
(the AND operator), || (the OR operator) or ! (the NOT operator).

For example, to select only products produced by a module with label
"mymodule" and made in the process "PROD", one can use:

  Selector s( ModuleLabelSelector("mymodule") && 
              ProcessNameSelector("PROD") );

If a module (EDProducter, EDFilter, EDAnalyzer, or OutputModule) is
to use such a selector, it is best to initialize it directly upon
construction of the module, rather than creating a new Selector instance
for every event.

----------------------------------------------------------------------*/

#include <string>

#include "boost/utility/enable_if.hpp"

#include "FWCore/Framework/interface/SelectorBase.h"
#include "FWCore/Utilities/interface/value_ptr.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

namespace edm 
{
  //------------------------------------------------------------------
  /// struct template has_match.
  /// Used to declare that a class has a match function.
  /// This is needed to work around a bug in GCC 3.2.x
  //------------------------------------------------------------------
  template <class T>
  struct has_match
  {
    static const bool value = boost::is_base_of<SelectorBase,T>::value;
  };

  template <>
  struct has_match<SelectorBase>
  {
    static const bool value = true;
  };

  //------------------------------------------------------------------
  //
  /// Class ProcessNameSelector.
  /// Selects EDProducts based upon process name.
  ///
  /// As a special case, a ProcessNameSelector created with the
  /// string "*" matches *any* process (and so is rather like having
  /// no ProcessNameSelector at all).
  //------------------------------------------------------------------

  class ProcessNameSelector : public SelectorBase 
  {
  public:
    ProcessNameSelector(const std::string& pn) :
    pn_(pn.empty() ? std::string("*") : pn)
      { }
    
    virtual bool doMatch(ConstBranchDescription const& p) const 
    {
      return (pn_=="*") || (p.processName() == pn_);
    }

    virtual ProcessNameSelector* clone() const
    {
      return new ProcessNameSelector(*this);
    }

    std::string const& name() const
    {
      return pn_;
    }

  private:
    std::string pn_;
  };

  //------------------------------------------------------------------
  //
  /// Class ProductInstanceNameSelector.
  /// Selects EDProducts based upon product instance name.
  //
  //------------------------------------------------------------------

  class ProductInstanceNameSelector : public SelectorBase
  {
  public:
    ProductInstanceNameSelector(const std::string& pin) :
      pin_(pin)
    { }
    
    virtual bool doMatch(ConstBranchDescription const& p) const 
    {
      return p.productInstanceName() == pin_;
    }

    virtual ProductInstanceNameSelector* clone() const
    {
      return new ProductInstanceNameSelector(*this);
    }
  private:
    std::string pin_;
  };

  //------------------------------------------------------------------
  //
  /// Class ModuleLabelSelector.
  /// Selects EDProducts based upon module label.
  //
  //------------------------------------------------------------------

  class ModuleLabelSelector : public SelectorBase
  {
  public:
    ModuleLabelSelector(const std::string& label) :
      label_(label)
    { }
    
    virtual bool doMatch(ConstBranchDescription const& p) const 
    {
      return p.moduleLabel() == label_;
    }

    virtual ModuleLabelSelector* clone() const
    {
      return new ModuleLabelSelector(*this);
    }
  private:
    std::string label_;
  };

  //------------------------------------------------------------------
  //
  /// Class MatchAllSelector.
  /// Dummy selector whose match function always returns true.
  //
  //------------------------------------------------------------------

  class MatchAllSelector : public SelectorBase
  {
  public:
    MatchAllSelector()
    { }
    
    virtual bool doMatch(ConstBranchDescription const& p) const 
    {
      return true;
    }

    virtual MatchAllSelector* clone() const
    {
      return new MatchAllSelector;
    }
  };

  //----------------------------------------------------------
  //
  // AndHelper template.
  // Used to form expressions involving && between other selectors.
  //
  //----------------------------------------------------------

  template <class A, class B>
  class AndHelper
  {
  public:
    AndHelper(A const& a, B const& b) : a_(a), b_(b) { }
    bool match(ConstBranchDescription const& p) const { return a_.match(p) && b_.match(p); }  
  private:
    A a_;
    B b_;
  };
  
  template <class A, class B>
  struct has_match<AndHelper<A,B> >
  {
    static const bool value = true;
  };
  
  template <class A, class B>
  typename boost::enable_if_c< has_match<A>::value && has_match<B>::value,
			       AndHelper<A,B> >::type
  operator&& (A const& a, B const& b)
  {
    return AndHelper<A,B>(a,b);
  }

  //----------------------------------------------------------
  //
  // OrHelper template.
  // Used to form expressions involving || between other selectors.
  //
  //----------------------------------------------------------

  template <class A, class B>
  class OrHelper
  {
  public:
    OrHelper(A const& a, B const& b) : a_(a), b_(b) { }
    bool match(ConstBranchDescription const& p) const { return a_.match(p) || b_.match(p); }  
  private:
    A a_;
    B b_;
  };
  
  template <class A, class B>
  struct has_match<OrHelper<A,B> >
  {
    static const bool value = true;
  };
  
  template <class A, class B>
  typename boost::enable_if_c< has_match<A>::value && has_match<B>::value,
			       OrHelper<A,B> >::type
  operator|| (A const& a, B const& b)
  {
    return OrHelper<A,B>(a,b);
  }


  //----------------------------------------------------------
  //
  // NotHelper template.
  // Used to form expressions involving ! acting on a selector.
  //
  //----------------------------------------------------------

  template <class A>
  class NotHelper
  {
  public:
    explicit NotHelper(A const& a) : a_(a) { }
    bool match(ConstBranchDescription const& p) const { return ! a_.match(p); }
  private:
    A a_;
  };
  
  template <class A>
  typename boost::enable_if_c< has_match<A>::value,
			       NotHelper<A> >::type
  operator! (A const& a)
  {
    return NotHelper<A>(a);
  }
  
  template <class A>
  struct has_match<NotHelper<A> >
  {
    static const bool value = true;
  };

  //----------------------------------------------------------
  //
  // ComposedSelectorWrapper template
  // Used to hold an expression formed from the various helpers.
  //
  //----------------------------------------------------------

  template <class T>
  class ComposedSelectorWrapper : public SelectorBase
  {
  public:
    typedef T wrapped_type;
    explicit ComposedSelectorWrapper(T const& t) : expression_(t) { }
    ~ComposedSelectorWrapper() {};
    virtual bool doMatch(ConstBranchDescription const& p) const { return expression_.match(p); }
    ComposedSelectorWrapper<T>* clone() const { return new ComposedSelectorWrapper<T>(*this); }
  private:
    wrapped_type expression_;
  };

  //----------------------------------------------------------
  //
  // Selector
  //
  //----------------------------------------------------------

  class Selector : public SelectorBase
  {
  public:
    template <class T> Selector(T const& expression);
    void swap(Selector& other);
    virtual ~Selector();
    virtual Selector* clone() const;

    virtual bool doMatch(ConstBranchDescription const& p) const;
    
  private:
    value_ptr<SelectorBase> sel_;
  };

  template <class T>
  Selector::Selector(T const& expression) :
    sel_(new ComposedSelectorWrapper<T>(expression))
  { }


}

#endif
