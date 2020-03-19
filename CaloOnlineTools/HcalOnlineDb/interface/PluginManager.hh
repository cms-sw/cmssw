#ifndef PluginManager_hh_included
#define PluginManager_hh_included 1

#include <map>
#include <string>
#include <vector>

namespace hcal {

  /** \brief Base class for classes to be managed by the PluginManager.
      This class is provided to enable appropriate use of the dynamic_cast<> operator and
      to defince clearly the newInstance() operator of the plugin factory.

      \ingroup hcalBase
   */
  class Pluggable {
  public:
    /// A virtual destructor is required to establish the inheritance tree
    virtual ~Pluggable() = default;
  };

  /** \brief Class which is able to create instances of another class.

      The AbstractPluginFactory defines the interface of a Factory
      class used by the PluginManager.  In general, factories are
      defined in terms of an abstract base class (derived from
      Pluggable), and a concrete implementation class.  This
      architecture is used in the database interface, where
      ConfigurationDatabaseImpl is the base class and
      ConfigurationDatabaseImplMySQL is a concrete implementation.
      \ingroup hcalBase
  */
  class AbstractPluginFactory {
  public:
    virtual ~AbstractPluginFactory() = default;
    /** \brief Create a new instance of the class which this factory provides */
    virtual Pluggable* newInstance() = 0;
    /** \brief Get the name of the class which is considered the base for this factory's class.  
	In the example, this would be ConfigurationDatabaseImpl
    */
    virtual const char* getBaseClass() const = 0;
    /** \brief Get the name of the concrete class provided directly by this factory.
	In the example, this would be ConfigurationDatabaseImplMySQL
    */
    virtual const char* getDerivedClass() const = 0;
  };

  /** \brief Global resource for managing implementations of abstract interface classes.

  The PluginManager class is a central repository to register factory
  objects.  Each factory object can create objects of a class derived
  from Pluggable.  Such factories can be easily created from the
  PluginFactoryTemplate using the DECLARE_PLUGGABLE macro.  The user
  of the PluginManager should know what base class is required and may
  obtain a specific factory or get a list of all factories capable of
  supplying concrete objects derived from the specified base class.  
  \ingroup hcalBase
  */
  class PluginManager {
  public:
    /** \brief Register a new factory with the singleton plugin manager.  Each factory provides objects of a single class.
	\param baseClass Name of the abstract base class for objects provided by the factory.
	\param derivedClass Name of the concrete class for objects provided by the factory.
	\param factory Pointer to the factory object itself
    */
    static void registerFactory(const char* baseClass, const char* derivedClass, AbstractPluginFactory* factory);
    /** \brief Lookup a specific factory by base class and derived class name 
	\return NULL if the specified factory is not found
    */
    static AbstractPluginFactory* getFactory(const char* baseClass, const char* derivedClass);
    /** \brief Lookup all factories providing concrete objects from the specified base class
     */
    static void getFactories(const char* baseClass, std::vector<AbstractPluginFactory*>& factories);

  private:
    static std::map<std::string, std::map<std::string, AbstractPluginFactory*> >& factories();
  };

  /** \brief Templated generic plugin factory used by the DECLARE_PLUGGABLE macro 
      \ingroup hcalBase
   */
  template <class T>
  class PluginFactoryTemplate : public AbstractPluginFactory {
  public:
    PluginFactoryTemplate(const char* bc, const char* dc) : m_baseClass(bc), m_derivedClass(dc) {
      PluginManager::registerFactory(bc, dc, this);
    }
    ~PluginFactoryTemplate() override = default;
    Pluggable* newInstance() override { return new T; }
    const char* getBaseClass() const override { return m_baseClass; }
    const char* getDerivedClass() const override { return m_derivedClass; }

  private:
    const char* m_baseClass;
    const char* m_derivedClass;
  };

}  // namespace hcal

/** \def DECLARE_PLUGGABLE(BASECLASS,MYCLASS)
   \brief Utility macro which creates a static object specializing PluginFactoryTemplate for a specific class (base and concrete).
   \param BASECLASS Base class for the factory (not in quotes)
   \param MYCLASS Derived concrete class for the factory (not in quotes)

   \b Example:
   \code
   DECLARE_PLUGGABLE(hcal::ConfigurationDatabaseImpl,ConfigurationDatabaseImplMySQL)
   \endcode
   \ingroup hcalBase
*/
#define DECLARE_PLUGGABLE(BASECLASS, MYCLASS) \
  static hcal::PluginFactoryTemplate<MYCLASS> hcal_plugin_##MYCLASS(#BASECLASS, #MYCLASS);

#endif  // PluginManager_hh_included
