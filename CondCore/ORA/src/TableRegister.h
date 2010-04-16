#ifndef INCLUDE_ORA_TABLEREGISTER_H
#define INCLUDE_ORA_TABLEREGISTER_H

//
#include <string>
#include <set>
#include <map>
#include <vector>

namespace coral{
  class ISchema;
}

namespace ora {

  class TableRegister {
    
    public:

    explicit TableRegister( coral::ISchema& schema );

    virtual ~TableRegister();

    bool checkTable(const std::string& tableName);

    bool checkColumn(const std::string& tableName, const std::string& columnName);

    size_t numberOfColumns(const std::string& tableName);

    void insertTable(const std::string& tableName);

    bool insertColumn(const std::string& tableName, const std::string& columnName );

    bool insertColumns(const std::string& tableName, const std::vector<std::string>& columns );

    private:

    void init();

    private:

    coral::ISchema& m_schema;

    bool m_init;

    std::map<std::string,std::set<std::string> > m_register;

    const std::string* m_currentTable;

    std::set<std::string>* m_currentColumns;

  };

}

#endif
