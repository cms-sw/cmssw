#ifndef PhysicsTools_MVAComputer_ProcessRegistry_h
#define PhysicsTools_MVAComputer_ProcessRegistry_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcessRegistry
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
//
// Refactoring for gcc 4.7.0 and higher
// by Gena Kukartsev following design and advice from Chris Jones
// January 2013
// 



#include <string>
#include "tbb/concurrent_unordered_map.h"



namespace PhysicsTools 
{

  // forward declaration
  template<class Base_t, class CalibBase_t, class Parent_t, class Instance_t, class Calibration_t> class ProcessRegistryImpl;



  /** \class ProcessRegistry
   *
   * \short Generic registry template for polymorphic processor implementations
   *
   * template parameters are: base class, calibration base class and a
   * pointer to a user-definable "parent type".
   * Template allows registration by name of a given base type using the factory.
   * The variable processors can register themselves with the registry of the
   * common base class.
   *
   ************************************************************/
  template<class Base_t, class CalibBase_t, class Parent_t>
    class ProcessRegistry {
    public:
    

#ifndef __GCCXML__

    // template alias to replace the former Registry class
    template<class Instance_t, class Calibration_t>
      using Registry = ProcessRegistryImpl<Base_t,CalibBase_t,Parent_t,Instance_t,Calibration_t>;
    
#endif  

    
    /** \class Factory
     *
     * \short Factory helper class to instantiate a processor.
     *
     * The common base class of a processor can inherit from this
     * helper class to provide a create() method to instantiate variable
     * processor instances.
     *
     ************************************************************/
    class Factory {
    public:
      static Base_t *create(const char *name,
			    const CalibBase_t *calib,
			    Parent_t *parent = 0);
    };
    
  protected:
    friend class Factory;
    
    /// instantiate registry and registers itself with \a name
      ProcessRegistry(const char *name) : name(name)
	{ registerProcess(name, this); }
	virtual ~ProcessRegistry() { unregisterProcess(name); }
	
	/// create an instance of \a name, given a calibration \a calib and parent \a parent
	  static Base_t *create(const char *name, const CalibBase_t *calib,
				Parent_t *parent);
	  
	  /// virtual method to implement by respective processor instance classes
	    virtual Base_t *instance(const char *name, const CalibBase_t *calib,
				     Parent_t *parent) const = 0;
	    
  private:
	    static void registerProcess(const char *name,
					const ProcessRegistry *process);
	    static void unregisterProcess(const char *name);
	    
	    typedef tbb::concurrent_unordered_map<std::string, const ProcessRegistry*> RegistryMap;
	    
	    /// return map of all registered processes, allocate if necessary
	      static RegistryMap *getRegistry();
	      
	      const char *name;
  }; // class ProcessRegistry
  
  
  /** \class ProcessRegistryImpl (formerly ProcessRegistry::Registry)
   *
   * \short template to generate a registry singleton for a type.
   *
   * Instantiating an instance of this type registers that class
   * with the registry of the base type and provides a factory that
   * calls the constructor of the instance type.
   *
   ************************************************************/
  template<class Base_t, class CalibBase_t, class Parent_t, class Instance_t, class Calibration_t> 
    class ProcessRegistryImpl : public ProcessRegistry<Base_t, CalibBase_t, Parent_t> {
  public:
    ProcessRegistryImpl<Base_t, CalibBase_t, Parent_t, Instance_t, Calibration_t>(const char *name) : ProcessRegistry<Base_t, CalibBase_t, Parent_t>(name){}
  protected:
    Base_t *instance(const char *name, const CalibBase_t *calib,
		     Parent_t *parent) const
      {
	return new Instance_t(name,
			      dynamic_cast<const Calibration_t*>(calib),
			      parent);
      }
  }; // class ProcessRegistryImpl


  
} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_ProcessRegistry_h
