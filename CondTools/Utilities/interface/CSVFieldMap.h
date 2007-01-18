#ifndef CSVFieldMap_h
#define CSVFieldMap_h
#include <vector>
#include <map>
#include <string>
#include <typeinfo>
class CSVFieldMap{
 public:
  CSVFieldMap(){}
  ~CSVFieldMap(){}
  void push_back(const std::string& fieldName, const std::string& fieldType);
  std::string fieldName( int idx ) const;
  const std::type_info& fieldType( int idx ) const;
  std::string fieldTypeName( int idx ) const;
  int size() const;
 private:
  std::vector< std::pair<std::string, std::string > > m_fieldMap;
};
#endif
