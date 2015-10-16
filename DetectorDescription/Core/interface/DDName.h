#ifndef DDName_h
#define DDName_h

#include <string>
#include <map>
#include <vector>

class DDCurrentNamespace;
class DDStreamer;

//! DDName is used to identify DDD entities uniquely.
/** A DDName consists of a \a name and a \a namespace. Both are represented as std::string.
*/
class DDName
{
  friend class DStreamer; // intrusive!
  
public:
  
  typedef DDCurrentNamespace CNS;
  typedef int id_type;
  typedef std::map<std::pair<std::string,std::string>,id_type> Registry;
  typedef std::vector<Registry::const_iterator> IdToName;
  
  //! Constructs a DDName with name \a name and assigns \a name to the namespace \a ns.
  DDName( const std::string & name,
          const std::string & ns);

  //! Creates a DDName with \a name in the current namespace defined in the singleton DDCurrentNamespace		
  DDName( const std::string & name );
  DDName( const char* name ); 
  DDName( const char* name, const char* ns );
  // DDName(pair<int,int>);
  DDName(id_type);
  //! register pre-defined ids
  static void defineId( const std::pair<std::string,std::string> &, id_type id);

  explicit DDName();
  
  //! true, if a DDName with given name and namespace (ns) already is registerd, otherwise false
  static bool exists(const std::string & name, const std::string & ns);
  //! Returns the \a name		
  const std::string & name() const;
  
  //! Returns the \a namespace
  const std::string & ns() const;
  
  /** Returns a string complete of the \a namespace and \a name separated by ":". 
      Most likely you want to use ns() and / or name() methods instead.
   */
  const std::string fullname() const { return ns() + ":" + name(); }

  // DEPRECATED!!!
  operator std::string() const { return ns() + ":" + name(); }

  id_type id() const { return id_;}
    
  bool operator<(const DDName & rhs) const { return id_ < rhs.id_; }
  bool operator==(const DDName & rhs) const { return id_ == rhs.id_; }

private:
  id_type id_;
    
  static Registry::iterator registerName(const std::pair<std::string,std::string> & s);  
};

//! DDNameInterface provides a common interface to DDD entities
/** DDLogicalPart, DDMaterial, DDSolids, ... all are uniquely identified by
    their class and the value of their associated DDName. 
    DDNameInterface provides read-access to their DDName.
*/
struct DDNameInterface
{
  virtual ~DDNameInterface() {}
  
  //! Returns the \a name without the \a namespace
  virtual const std::string & name() const=0;   // name without namespace
  
  //! Return the \a namespace 
  virtual const std::string & ns() const=0; // only the namespace 
  
  //! Return the DDName
  DDName ddname() const { return DDName(name(),ns()); } 
  
  virtual operator bool() const=0;
  
  //! \b don't \b use \b! 
  virtual int id() const=0; // dont't know if usefull ...
};

std::ostream & operator<<(std::ostream & os, const DDName & n); 

#endif
