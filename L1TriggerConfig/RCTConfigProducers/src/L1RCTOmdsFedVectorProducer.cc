// -*- C++ -*-
//
// Package:    L1RCTOmdsFedVectorProducer
// Class:      L1RCTOmdsFedVectorProducer
//
/**\class L1RCTOmdsFedVectorProducer L1RCTOmdsFedVectorProducer.h L1TriggerConfig/L1RCTOmdsFedVectorProducer/src/L1RCTOmdsFedVectorProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jessica Lynn Leonard,32 4-C20,+41227674522,
//         Created:  Fri Sep  9 11:19:20 CEST 2011
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

// OMDS stuff
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IConnection.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralKernel/Context.h"

#include "CondCore/CondDB/interface/ConnectionPool.h"
// end OMDS stuff

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

//
// class declaration
//

class L1RCTOmdsFedVectorProducer : public edm::ESProducer {
public:
  L1RCTOmdsFedVectorProducer(const edm::ParameterSet&);
  ~L1RCTOmdsFedVectorProducer() override;

  using ReturnType = std::unique_ptr<RunInfo>;

  ReturnType produce(const RunInfoRcd&);

private:
  // ----------member data ---------------------------
  std::string connectionString;
  std::string authpath;
  std::string tableToRead;

  edm::ESGetToken<RunInfo, RunInfoRcd> token_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1RCTOmdsFedVectorProducer::L1RCTOmdsFedVectorProducer(const edm::ParameterSet& iConfig)
    : connectionString(iConfig.getParameter<std::string>("connectionString")),
      authpath(iConfig.getParameter<std::string>("authpath")),
      tableToRead(iConfig.getParameter<std::string>("tableToRead")) {
  //the following line is needed to tell the framework what
  // data is being produced
  token_ = setWhatProduced(this, "OmdsFedVector").consumes();

  //now do what ever other initialization is needed
}

L1RCTOmdsFedVectorProducer::~L1RCTOmdsFedVectorProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
L1RCTOmdsFedVectorProducer::ReturnType L1RCTOmdsFedVectorProducer::produce(const RunInfoRcd& iRecord) {
  //  std::cout << "ENTERING L1RCTOmdsFedVectorProducer::produce()" << std::endl;

  //  std::cout << "GETTING FED VECTOR FROM OMDS" << std::endl;

  // GETTING ALREADY-EXISTING RUNINFO OUT OF ES TO FIND OUT RUN NUMBER
  const RunInfo* summary = &iRecord.get(token_);
  int runNumber = summary->m_run;

  // CREATING NEW RUNINFO WHICH WILL GET NEW FED VECTOR AND BE RETURNED
  auto pRunInfo = std::make_unique<RunInfo>();

  // DO THE DATABASE STUFF

  //make connection object
  cond::persistency::ConnectionPool connection;

  //set in configuration object authentication path
  connection.setAuthenticationPath(authpath);
  connection.configure();

  //create session object from connection
  cond::persistency::Session session = connection.createSession(connectionString, true);

  session.transaction().start(true);  // (true=readOnly)

  coral::ISchema& schema = session.coralSession().schema("CMS_RUNINFO");

  //condition
  coral::AttributeList conditionData;
  conditionData.extend<int>("n_run");
  conditionData[0].data<int>() = runNumber;

  std::string columnToRead_val = "VALUE";

  std::string tableToRead_fed = "RUNSESSION_STRING";
  coral::IQuery* queryV = schema.newQuery();
  queryV->addToTableList(tableToRead);
  queryV->addToTableList(tableToRead_fed);
  queryV->addToOutputList(tableToRead_fed + "." + columnToRead_val, columnToRead_val);
  //queryV->addToOutputList(tableToRead + "." + columnToRead, columnToRead);
  //condition
  std::string condition =
      tableToRead + ".RUNNUMBER=:n_run AND " + tableToRead +
      ".NAME='CMS.LVL0:FED_ENABLE_MASK' AND RUNSESSION_PARAMETER.ID = RUNSESSION_STRING.RUNSESSION_PARAMETER_ID";
  //std::string condition = tableToRead + ".runnumber=:n_run AND " + tableToRead + ".name='CMS.LVL0:FED_ENABLE_MASK'";
  queryV->setCondition(condition, conditionData);
  coral::ICursor& cursorV = queryV->execute();
  std::string fed;
  if (cursorV.next()) {
    //cursorV.currentRow().toOutputStream(std::cout) << std::endl;
    const coral::AttributeList& row = cursorV.currentRow();
    fed = row[columnToRead_val].data<std::string>();
  } else {
    fed = "null";
  }
  //std::cout << "string fed emask == " << fed << std::endl;
  delete queryV;

  std::replace(fed.begin(), fed.end(), '%', ' ');
  std::stringstream stream(fed);
  for (;;) {
    std::string word;
    if (!(stream >> word)) {
      break;
    }
    std::replace(word.begin(), word.end(), '&', ' ');
    std::stringstream ss(word);
    int fedNumber;
    int val;
    ss >> fedNumber >> val;
    //std::cout << "fed:: " << fed << "--> val:: " << val << std::endl;
    //val bit 0 represents the status of the SLINK, but 5 and 7 means the SLINK/TTS is ON but NA or BROKEN (see mail of alex....)
    if ((val & 0001) == 1 && (val != 5) && (val != 7))
      pRunInfo->m_fed_in.push_back(fedNumber);
  }
  //   std::cout << "feds in run:--> ";
  //   std::copy(pRunInfo->m_fed_in.begin(), pRunInfo->m_fed_in.end(), std::ostream_iterator<int>(std::cout, ", "));
  //   std::cout << std::endl;
  /*
    for (size_t i =0; i<pRunInfo->m_fed_in.size() ; ++i)
    {
    std::cout << "fed in run:--> " << pRunInfo->m_fed_in[i] << std::endl;
    }
  */

  return pRunInfo;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RCTOmdsFedVectorProducer);
