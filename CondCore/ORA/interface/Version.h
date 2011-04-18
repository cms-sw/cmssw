#ifndef INCLUDE_ORA_VERSION_H
#define INCLUDE_ORA_VERSION_H

//
#include <string>

namespace ora {

  class Version {
  public:
   
    static Version& poolSchemaVersion();
    static Version& thisSchemaVersion();
    static Version fromString( const std::string& versionString );

    public:
    Version();
    ~Version(){
    }
    Version( const Version& rhs );
    Version& operator=( const Version& rhs );

    bool operator==( const Version& rhs ) const;
    bool operator!=( const Version& rhs ) const;
    bool operator>( const Version& rhs ) const;
    bool operator<( const Version& rhs ) const;
    bool operator>=( const Version& rhs ) const;
    bool operator<=( const Version& rhs ) const;

    std::string toString() const;
    void toOutputStream( std::ostream& os ) const;    
    
  private:
    std::string m_label;
    int m_main;
    int m_release;
    int m_patch;
  };

}
inline std::ostream& operator << (std::ostream& os, const ora::Version& ver ){
  ver.toOutputStream(os);
  return os;
}

#endif
