// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     OMDSReader
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Mar  2 01:46:46 CET 2008
// $Id: OMDSReader.cc,v 1.12 2010/02/16 21:56:37 wsun Exp $
//

// system include files
#include <set>
#include <iostream>

// user include files
#include "CondTools/L1Trigger/interface/OMDSReader.h"
#include "RelationalAccess/ITableDescription.h"
#include "RelationalAccess/IColumn.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

namespace l1t {

  //
  // constructors and destructor
  //
  OMDSReader::OMDSReader() : DataManager() {}

  OMDSReader::OMDSReader(const std::string& connectString, const std::string& authenticationPath)
      : DataManager(connectString, authenticationPath, true) {
    session.transaction().start(true);
  }

  void OMDSReader::connect(const std::string& connectString, const std::string& authenticationPath) {
    DataManager::connect(connectString, authenticationPath, true);
    session.transaction().start(true);
  }

  // OMDSReader::OMDSReader(const OMDSReader& rhs)
  // {
  //    // do actual copying here;
  // }

  OMDSReader::~OMDSReader() {}

  //
  // assignment operators
  //
  // const OMDSReader& OMDSReader::operator=(const OMDSReader& rhs)
  // {
  //   //An exception safe implementation is
  //   OMDSReader temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //

  //
  // const member functions
  //

  const OMDSReader::QueryResults OMDSReader::basicQuery(const std::vector<std::string>& columnNames,
                                                        const std::string& schemaName,
                                                        const std::string& tableName,
                                                        const std::string& conditionLHS,
                                                        const QueryResults conditionRHS,
                                                        const std::string& conditionRHSName) {
    coral::ISessionProxy& coralSession = session.coralSession();
    coral::ISchema& schema = schemaName.empty() ? coralSession.nominalSchema() : coralSession.schema(schemaName);

    coral::ITable& table = schema.tableHandle(tableName);

    // Pointer is deleted automatically at end of function.
    std::shared_ptr<coral::IQuery> query(table.newQuery());

    // Construct query
    std::vector<std::string>::const_iterator it = columnNames.begin();
    std::vector<std::string>::const_iterator end = columnNames.end();
    for (; it != end; ++it) {
      query->addToOutputList(*it);
    }

    // Only apply condition if RHS has one row.
    if (!conditionLHS.empty() && conditionRHS.numberRows() == 1) {
      if (!conditionRHSName.empty()) {
        // Assume all RHS types are strings.
        coral::AttributeList attList;
        attList.extend(conditionRHSName, typeid(std::string));
        std::string tmp;
        conditionRHS.fillVariable(conditionRHSName, tmp);
        attList[conditionRHSName].data<std::string>() = tmp;

        query->setCondition(conditionLHS + " = :" + conditionRHSName, attList);
      } else if (conditionRHS.columnNames().size() == 1)
      // check for only one column
      {
        query->setCondition(conditionLHS + " = :" + conditionRHS.columnNames().front(),
                            conditionRHS.attributeLists().front());
      }
    }

    coral::ICursor& cursor = query->execute();

    // Copy AttributeLists for external use because the cursor is deleted
    // when the query goes out of scope.
    std::vector<coral::AttributeList> atts;
    while (cursor.next()) {
      atts.push_back(cursor.currentRow());
    };

    return QueryResults(columnNames, atts);
  }

  const OMDSReader::QueryResults OMDSReader::basicQuery(const std::string& columnName,
                                                        const std::string& schemaName,
                                                        const std::string& tableName,
                                                        const std::string& conditionLHS,
                                                        const QueryResults conditionRHS,
                                                        const std::string& conditionRHSName) {
    std::vector<std::string> columnNames;
    columnNames.push_back(columnName);
    return basicQuery(columnNames, schemaName, tableName, conditionLHS, conditionRHS, conditionRHSName);
  }

  std::vector<std::string> OMDSReader::columnNames(const std::string& schemaName, const std::string& tableName) {
    coral::ISessionProxy& coralSession = session.coralSession();
    coral::ISchema& schema = schemaName.empty() ? coralSession.nominalSchema() : coralSession.schema(schemaName);

    coral::ITable& table = schema.tableHandle(tableName);
    const coral::ITableDescription& tableDesc = table.description();

    std::vector<std::string> names;
    int nCols = tableDesc.numberOfColumns();

    for (int i = 0; i < nCols; ++i) {
      const coral::IColumn& column = tableDesc.columnDescription(i);
      names.push_back(column.name());
    }

    return names;
  }

  // VIEW

  const OMDSReader::QueryResults OMDSReader::basicQueryView(const std::vector<std::string>& columnNames,
                                                            const std::string& schemaName,
                                                            const std::string& viewName,
                                                            const std::string& conditionLHS,
                                                            const QueryResults conditionRHS,
                                                            const std::string& conditionRHSName) {
    coral::ISessionProxy& coralSession = session.coralSession();
    coral::ISchema& schema = schemaName.empty() ? coralSession.nominalSchema() : coralSession.schema(schemaName);

    //    coral::IView& view = schema.viewHandle( viewName ) ;

    // Pointer is deleted automatically at end of function.
    coral::IQuery* query = schema.newQuery();
    ;

    // Construct query
    for (std::vector<std::string>::const_iterator constIt = columnNames.begin(); constIt != columnNames.end();
         ++constIt) {
      query->addToOutputList(*constIt);
    }

    query->addToTableList(viewName);

    // Only apply condition if RHS has one row.
    if (!conditionLHS.empty() && conditionRHS.numberRows() == 1) {
      if (!conditionRHSName.empty()) {
        // Assume all RHS types are strings.
        coral::AttributeList attList;
        attList.extend(conditionRHSName, typeid(std::string));
        std::string tmp;
        conditionRHS.fillVariable(conditionRHSName, tmp);
        attList[conditionRHSName].data<std::string>() = tmp;

        query->setCondition(conditionLHS + " = :" + conditionRHSName, attList);
      } else if (conditionRHS.columnNames().size() == 1)
      // check for only one column
      {
        query->setCondition(conditionLHS + " = :" + conditionRHS.columnNames().front(),
                            conditionRHS.attributeLists().front());
      }
    }

    coral::ICursor& cursor = query->execute();

    // Copy AttributeLists for external use because the cursor is deleted
    // when the query goes out of scope.
    std::vector<coral::AttributeList> atts;
    while (cursor.next()) {
      atts.push_back(cursor.currentRow());
    };

    delete query;

    //    // Run a wildcard query on the view
    //    coral::IQuery* query2 = workingSchema.newQuery();
    //    query2->addToTableList(V0);
    //    coral::ICursor& cursor2 = query2->execute();
    //    while ( cursor2.next() ) {
    //      cursor2.currentRow().toOutputStream( std::cout ) << std::endl;
    //    }
    //    delete query2;

    return QueryResults(columnNames, atts);
  }

  const OMDSReader::QueryResults OMDSReader::basicQueryView(const std::string& columnName,
                                                            const std::string& schemaName,
                                                            const std::string& viewName,
                                                            const std::string& conditionLHS,
                                                            const QueryResults conditionRHS,
                                                            const std::string& conditionRHSName) {
    std::vector<std::string> columnNames;
    columnNames.push_back(columnName);
    return basicQuery(columnNames, schemaName, viewName, conditionLHS, conditionRHS, conditionRHSName);
  }

  std::vector<std::string> OMDSReader::columnNamesView(const std::string& schemaName, const std::string& viewName) {
    coral::ISessionProxy& coralSession = session.coralSession();
    coral::ISchema& schema = schemaName.empty() ? coralSession.nominalSchema() : coralSession.schema(schemaName);

    std::set<std::string> views = schema.listViews();
    std::vector<std::string> names;

    if (schema.existsView(viewName)) {
      coral::IView& view = schema.viewHandle(viewName);

      int nCols = view.numberOfColumns();

      for (int i = 0; i < nCols; ++i) {
        const coral::IColumn& column = view.column(i);
        names.push_back(column.name());
      }

      return names;
    }

    return names;
  }

  //
  // static member functions
  //
}  // namespace l1t
