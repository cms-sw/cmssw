#ifndef CondCoreDBCommon_DecodingKey_H
#define CondCoreDBCommon_DecodingKey_H
#include <string>

namespace cond {

  class DecodingKey {

    public:

    DecodingKey();

    virtual ~DecodingKey(){}

    bool readUserKey(const std::string& keyFileName);

    bool readUserKeyString(const std::string& content);
    
    bool readFromFile(const std::string& password, const std::string& keyFileName);

    bool readFromString(const std::string& password, const std::string& content);

    const std::string& key() const;

    const std::string& dataSource() const;

    static bool validateKey(const std::string& key);

    static std::string getUserName();
    
    static bool createFile(const std::string& password, const std::string& key,
                           const std::string& dataSource, const std::string& keyFileName);

    private:

    std::string m_key;

    std::string m_dataSource;
    
  };
}

inline
cond::DecodingKey::DecodingKey():m_key(""),m_dataSource(""){
}

inline
const std::string& cond::DecodingKey::key() const{
  return m_key;
}

inline
const std::string& cond::DecodingKey::dataSource() const{
  return m_dataSource;
}

#endif //  CondCoreDBCommon_DecodingKey_H

  
