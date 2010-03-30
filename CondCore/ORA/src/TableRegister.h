#ifndef INCLUDE_ORA_TABLEREGISTER_H
#define INCLUDE_ORA_TABLEREGISTER_H

//
#include <string>
#include <set>
#include <map>

namespace ora {

  class TableRegister {
    
    public:

    TableRegister();

    virtual ~TableRegister();

    bool checkTable(const std::string& tableName);

    bool checkColumn(const std::string& tableName, const std::string& columnName);

    size_t numberOfColumns(const std::string& tableName);

    void insertTable(const std::string& tableName);

    bool insertColumn(const std::string& tableName, const std::string& columnName);

    //unsigned int nextDependencyIndex( const std::string& tableName );

    private:

    std::map<std::string,std::set<std::string> > m_register;

    const std::string* m_currentTable;

    std::set<std::string>* m_currentColumns;
    
  };

}

#endif
