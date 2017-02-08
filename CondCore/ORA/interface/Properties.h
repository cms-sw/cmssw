#ifndef INCLUDE_ORA_PROPERTIES_H
#define INCLUDE_ORA_PROPERTIES_H

//
#include <string>
#include <map>
#include <set>

namespace ora {

  class Properties {
    public:

    Properties();
    virtual ~Properties();
    bool setProperty(const std::string& propertyName, const std::string& propertyValue);
    bool removeProperty(const std::string& propertyName);
    void setFlag(const std::string& flagName);
    bool removeFlag(const std::string& flagName);

    bool hasProperty(const std::string& propertyName) const;
    std::string getProperty(const std::string& propertyName) const;
    bool getFlag(const std::string& flagName) const;

    private:

    std::map<std::string,std::string> m_properties;
    std::set<std::string> m_flags;
  };
  
}

#endif
