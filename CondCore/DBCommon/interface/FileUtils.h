#ifndef CondCoreDBCommon_FileUtils_H
#define CondCoreDBCommon_FileUtils_H
#include <string>

namespace cond {

  class FileReader {

    public:

    FileReader();

    virtual ~FileReader(){}

    bool read(const std::string& fileName);

    const std::string& content() const;

    private:

    std::string m_content;
  };
  
}

inline
cond::FileReader::FileReader():m_content(""){
}

inline
const std::string& cond::FileReader::content() const {
  return m_content;
}

#endif //  CondCoreDBCommon_FileUtils_H

