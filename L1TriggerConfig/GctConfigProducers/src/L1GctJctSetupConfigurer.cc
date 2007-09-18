#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "L1TriggerConfig/GctConfigProducers/interface/L1GctJctSetupConfigurer.h"

#include <string>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctJctSetupConfigurer::L1GctJctSetupConfigurer(const std::vector<edm::ParameterSet>& iConfig) :
  m_jetCounterCuts()
{

  // ------------------------------------------------------------------------------------------
  // Read jet counter setup info from config file
  for (unsigned j=0; j<iConfig.size(); j++) {
    m_jetCounterCuts.push_back(addJetCounter(iConfig.at(j)));
  }

  if (m_jetCounterCuts.size() > L1GctJetCounterSetup::MAX_JET_COUNTERS) {
    edm::LogError("L1GctJetCounterSetup") << "Too many cuts specified, "
    << "maximum allowed number of cuts is " << L1GctJetCounterSetup::MAX_JET_COUNTERS << std::endl;
  }
                                 
}


L1GctJctSetupConfigurer::~L1GctJctSetupConfigurer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
    

// ------------ methods called to produce the data  ------------

L1GctJctSetupConfigurer::JctSetupReturnType 
L1GctJctSetupConfigurer::produceJctSetup()
{
   boost::shared_ptr<L1GctJetCounterSetup> pL1GctJetCounterSetup=
     boost::shared_ptr<L1GctJetCounterSetup> (new L1GctJetCounterSetup(m_jetCounterCuts));

   return pL1GctJetCounterSetup;
}

//
//--------------------------------------------------------------------------

L1GctJetCounterSetup::cutsListForJetCounter
L1GctJctSetupConfigurer::addJetCounter(const edm::ParameterSet& iConfig)
{
  std::vector<std::string> cutDescList = iConfig.getParameter< std::vector<std::string> >("cutDescriptionList");

  L1GctJetCounterSetup::cutsListForJetCounter cutsList;
  unsigned numberOfCuts = cutDescList.size();

  for (unsigned cutNo=0; cutNo<numberOfCuts; cutNo++) {
    cutsList.push_back(parseDescriptor(cutDescList.at(cutNo)));
  }

  return cutsList;

}

L1GctJetCounterSetup::cutDescription
L1GctJctSetupConfigurer::parseDescriptor(const std::string& desc) const
{
  using namespace std;
  L1GctJetCounterSetup::cutDescription cut;
  istringstream ss(desc);

  // The string has to consist of pieces separated by underscore characters.
  // Go through and find the pieces.
  // The input string should be of the form JC_(cutType)_(cutValue1)_(cutValue2),
  // with values being omitted if not required
  string nextToken;
  getline(ss, nextToken, '_');
  if (nextToken=="JC") {
    if (getline(ss, nextToken, '_')) {
      cut.cutType = descToCutType(nextToken);

      if (!ss.eof()) {
        getline(ss, nextToken, '_');
        cut.cutValue1 = descToCutValue(nextToken);
      } else {
        cut.cutValue1=0;
      }

      if (!ss.eof()) {
        getline(ss, nextToken, '_');
        cut.cutValue2 = descToCutValue(nextToken);
      } else {
        cut.cutValue2=0;
      }

    } else {
      edm::LogError("L1GctJetCounterSetup") << "Error reading jet counter configuration file\n"
      << "Can't read cut description" << endl;
    }
  } else {
    edm::LogError("L1GctJetCounterSetup") << "Error reading jet counter configuration file\n"
    << "Cut description must start with \"JC\" " << endl;
  }

  return cut;
}

L1GctJetCounterSetup::validCutType
L1GctJctSetupConfigurer::descToCutType(const std::string& token) const
{
  if (token=="minRank")    { return L1GctJetCounterSetup::minRank; }
  if (token=="maxRank")    { return L1GctJetCounterSetup::maxRank; }
  if (token=="centralEta") { return L1GctJetCounterSetup::centralEta; }
  if (token=="forwardEta") { return L1GctJetCounterSetup::forwardEta; }
  if (token=="phiWindow")  { return L1GctJetCounterSetup::phiWindow; }

  if (token!="nullCutType") {
    edm::LogWarning("L1GctJetCounterSetup") << "Reading jet counter configuration file\n"
                                            << "Unrecognised cut type option: " << token
                                            << "; a null counter will be used" << std::endl;
  }

  return L1GctJetCounterSetup::nullCutType;
}

unsigned
L1GctJctSetupConfigurer::descToCutValue(const std::string& token) const
{
  unsigned result=0;
  std::istringstream ss(token);
  ss >> result;
  if (!ss.eof()) {
    edm::LogError("L1GctJetCounterSetup") << "Error reading jet counter configuration file\n"
    << "Expecting numeric cut value, found: " << token << std::endl;
  }
  return result;
}

