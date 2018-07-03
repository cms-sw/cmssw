#ifndef CondCore_CondDB_FileUtils_h
#define CondCore_CondDB_FileUtils_h
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

#endif // CondCore_CondDB_FileUtils_h

