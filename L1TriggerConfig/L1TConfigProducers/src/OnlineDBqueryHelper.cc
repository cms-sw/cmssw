#include "OnlineDBqueryHelper.h"

std::map<std::string, std::string> l1t::OnlineDBqueryHelper::fetch(const std::vector<std::string> &queryColumns,
                                                                   const std::string &table,
                                                                   const std::string &key,
                                                                   l1t::OMDSReader &m_omdsReader) {
  if (queryColumns.empty() || table.length() == 0)
    return std::map<std::string, std::string>();

  l1t::OMDSReader::QueryResults queryResult =
      m_omdsReader.basicQuery(queryColumns, "CMS_TRG_L1_CONF", table, table + ".ID", m_omdsReader.singleAttribute(key));

  if (queryResult.queryFailed() || queryResult.numberRows() != 1)
    throw std::runtime_error(std::string("Cannot get ") + table + ".{" +
                             std::accumulate(std::next(queryColumns.begin()),
                                             queryColumns.end(),
                                             std::string(queryColumns[0]),
                                             [](const std::string &a, const std::string &b) { return a + "," + b; }) +
                             "} for ID = " + key);

  std::vector<std::string> retval(queryColumns.size());

  std::transform(queryColumns.begin(), queryColumns.end(), retval.begin(), [queryResult](const std::string &a) {
    std::string res;
    if (!queryResult.fillVariable(a, res))
      res = "";
    return res;
  });

  std::map<std::string, std::string> retvalMap;
  for (unsigned int i = 0; i < queryColumns.size(); i++)
    retvalMap.insert(make_pair(queryColumns[i], retval[i]));

  return retvalMap;
}
