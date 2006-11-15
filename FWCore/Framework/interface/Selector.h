#ifndef Framework_Selector_h
#define Framework_Selector_h

/*----------------------------------------------------------------------
  
Selector: Base class for all "selector" objects, used to select
EDProducts based on information in the associated Provenance.

Developers who make their own Selectors should inherit from SelectorBase.

Users can use the classes

  ModuleDescriptionSelector
  ModuleLabelSelector
  ProcessNameSelector
  ProductInstanceNameSelector

Users can also use the class Selector, which can be construced given a
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

$Id: Selector.h,v 1.13 2006/10/23 23:50:34 chrjones Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <string>
#include <vector>

#include "boost/type_traits.hpp"
#include "boost/utility/enable_if.hpp"

#include "DataFormats/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/ProvenanceAccess.h"

#include "FWCore/Framework/interface/SelectorBase.h"
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
  /// Class ModuleDescriptionSelector.
  /// Selects EDProducts based upon full description of EDProducer.
  //
  //------------------------------------------------------------------

  class ModuleDescriptionSelector : public SelectorBase 
  {
  public:
    ModuleDescriptionSelector(const ModuleDescriptionID& mdid) :
      mdid_(mdid) 
    { }
    
    virtual bool doMatch(Provenance const& p) const 
    {
      return p.moduleDescriptionID() == mdid_;
    }

    virtual ModuleDescriptionSelector* clone() const
    {
      return new ModuleDescriptionSelector(*this);
    }

  private:
    ModuleDescriptionID mdid_;
  };

  //------------------------------------------------------------------
  //
  /// Class ProcessNameSelector.
  /// Selects EDProducts based upon process name.
  //
  //------------------------------------------------------------------

  class ProcessNameSelector : public SelectorBase 
  {
  public:
    ProcessNameSelector(const std::string& pn) :
      pn_(pn)
    { }
    
    virtual bool doMatch(Provenance const& p) const 
    {
      return p.processName() == pn_;
    }

    virtual ProcessNameSelector* clone() const
    {
      return new ProcessNameSelector(*this);
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
    
    virtual bool doMatch(Provenance const& p) const 
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
  /// Selects EDProducts based upon product instance name.
  //
  //------------------------------------------------------------------

  class ModuleLabelSelector : public SelectorBase
  {
  public:
    ModuleLabelSelector(const std::string& label) :
      label_(label)
    { }
    
    virtual bool doMatch(Provenance const& p) const 
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
    bool match(ProvenanceAccess const& p) const { return a_.match(p) && b_.match(p); }  
    bool match(Provenance const& p) const { return a_.match(p) && b_.match(p); }  
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
    bool match(ProvenanceAccess const& p) const { return a_.match(p) || b_.match(p); }  
    bool match(Provenance const& p) const { return a_.match(p) || b_.match(p); }  
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
    bool match(ProvenanceAccess const& p) const { return ! a_.match(p); }
    bool match(Provenance const& p) const { return ! a_.match(p); }
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
    virtual bool doMatch(Provenance const& p) const { return expression_.match(p); }
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
    Selector(Selector const& other);
    Selector& operator= (Selector const& other);
    void swap(Selector& other);
    virtual ~Selector();
    virtual Selector* clone() const;

    virtual bool doMatch(Provenance const& p) const;
    
  private:
    SelectorBase* sel_;
  };

  template <class T>
  Selector::Selector(T const& expression) :
    sel_(new ComposedSelectorWrapper<T>(expression))
  { }


}

#endif
