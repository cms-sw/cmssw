#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/RunInfoEditor.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include "CondTools/RunInfo/interface/RunInfoUpdate.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include <iostream>

#include <sstream>

namespace cond {

  class RunInfoUtils : public cond::Utilities {
  public:
    RunInfoUtils();
    ~RunInfoUtils() override;
    int execute() override;
  };
}  // namespace cond

cond::RunInfoUtils::RunInfoUtils() : Utilities("conddb_import_runInfo") {
  addConnectOption("fromConnect", "f", "source connection string (optional, default=connect)");
  addConnectOption("connect", "c", "target connection string (required)");
  addAuthenticationOptions();
  addOption<std::string>("tag", "t", "source tag (required)");
  addOption<size_t>("max_entries", "m", "max entries to migrate (default=1000)");
}

cond::RunInfoUtils::~RunInfoUtils() {}

boost::posix_time::ptime parseTimeFromIsoString(const std::string& isoString) {
  boost::posix_time::time_input_facet* tif = new boost::posix_time::time_input_facet;
  tif->set_iso_extended_format();
  std::istringstream iss(isoString);
  iss.imbue(std::locale(std::locale::classic(), tif));
  boost::posix_time::ptime ret;
  iss >> ret;
  return ret;
}

int cond::RunInfoUtils::execute() {
  std::string connect = getOptionValue<std::string>("connect");
  std::string fromConnect = getOptionValue<std::string>("fromConnect");

  // this is mandatory
  std::string tag = getOptionValue<std::string>("tag");
  std::cout << "# Source tag is " << tag << std::endl;

  size_t max_entries = 1000;
  if (hasOptionValue("max_entries"))
    max_entries = getOptionValue<size_t>("max_entries");

  persistency::ConnectionPool connPool;
  if (hasOptionValue("authPath")) {
    connPool.setAuthenticationPath(getOptionValue<std::string>("authPath"));
  }
  if (hasDebug())
    connPool.setMessageVerbosity(coral::Debug);
  connPool.configure();

  persistency::Session session = connPool.createSession(connect, true);
  std::cout << "# Connecting to target database on " << connect << std::endl;
  persistency::Session sourceSession = connPool.createSession(fromConnect);
  std::cout << "# Connecting to source database on " << fromConnect << std::endl;
  RunInfoUpdate updater(session);
  persistency::TransactionScope tsc(session.transaction());
  tsc.start(false);
  sourceSession.transaction().start();
  updater.import(max_entries, tag, sourceSession);
  sourceSession.transaction().commit();
  tsc.commit();
  return 0;
}

int main(int argc, char** argv) {
  cond::RunInfoUtils utilities;
  return utilities.run(argc, argv);
}
