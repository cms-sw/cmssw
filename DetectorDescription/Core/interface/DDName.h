#ifndef DETECTOR_DESCRIPTION_CORE_DDNAME_H
#define DETECTOR_DESCRIPTION_CORE_DDNAME_H

#include <iosfwd>
#include <map>
#include <string>
#include <utility>
#include <vector>

class DDCurrentNamespace;

//! DDName is used to identify DDD entities uniquely.
/** A DDName consists of a \a name and a \a namespace. Both are represented as std::string.
*/
class DDName
{
public:
  
  using id_type = int;
  using Registry = std::map<std::pair<std::string, std::string>, id_type>;
  using IdToName = std::vector<Registry::const_iterator>;
  
  //! Constructs a DDName with name \a name and assigns \a name to the namespace \a ns.
  DDName( const std::string & name,
          const std::string & ns );

  //! Creates a DDName with \a name in the current namespace defined in the singleton DDCurrentNamespace		
  DDName( const std::string & name );
  DDName( const char* name ); 
  DDName( const char* name, const char* ns );

  explicit DDName();
  
  //! Returns the \a name		
  const std::string & name() const;
  
  //! Returns the \a namespace
  const std::string & ns() const;
  
  /** Returns a string complete of the \a namespace and \a name separated by ":". 
      Most likely you want to use ns() and / or name() methods instead.
   */
  const std::string fullname() const { return ns() + ":" + name(); }

  id_type id() const { return id_; }
  
  bool operator<(const DDName & rhs) const { return id_ < rhs.id_; }
  bool operator==(const DDName & rhs) const { return id_ == rhs.id_; }

private:
  
  id_type id_;
    
  static Registry::iterator registerName( const std::pair<std::string, std::string> & s );
};

std::ostream & operator<<( std::ostream & os, const DDName & n ); 

#endif
