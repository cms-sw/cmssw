#ifndef CondCoreDBCommon_DecodingKey_H
#define CondCoreDBCommon_DecodingKey_H
#include <string>

namespace cond {

  class DecodingKey {

    public:

    DecodingKey();

    virtual ~DecodingKey(){}

    bool readFromFile(const std::string& keyFileName);

    bool readFromString(const std::string& content);

    const std::string& password() const;

    const std::string& dataFileName() const;

    static bool validatePassword(const std::string& password);
    
    static bool createFile(const std::string& password, const std::string& dataFileName, const std::string& keyFileName);

    private:

    std::string m_password;

    std::string m_dataFileName;
    
  };
}

inline
cond::DecodingKey::DecodingKey():m_password(""),m_dataFileName(""){
}

inline
const std::string& cond::DecodingKey::password() const{
  return m_password;
}

inline
const std::string& cond::DecodingKey::dataFileName() const{
  return m_dataFileName;
}

#endif //  CondCoreDBCommon_DecodingKey_H

  
