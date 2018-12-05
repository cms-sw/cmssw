// Original Author:  Jan Kašpar

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsCollection.h"
#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Loads optical functions from ROOT files.
 **/
class CTPPSOpticalFunctionsESSource: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
  public:
    CTPPSOpticalFunctionsESSource(const edm::ParameterSet &);

    ~CTPPSOpticalFunctionsESSource() {};

    edm::ESProducts<std::unique_ptr<LHCOpticalFunctionsCollection>> produce(const CTPPSOpticsRcd &);

  private:
    double m_xangle1, m_xangle2;
    std::string m_fileName1, m_fileName2;

    struct RPInfo
    {
      std::string dirName;
      double scoringPlaneZ;
    };

    std::unordered_map<unsigned int, RPInfo> m_rpInfo;

    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSOpticalFunctionsESSource::CTPPSOpticalFunctionsESSource(const edm::ParameterSet& conf) :
  m_xangle1(conf.getParameter<double>("xangle1")),
  m_xangle2(conf.getParameter<double>("xangle2")),
  m_fileName1(conf.getParameter<edm::FileInPath>("fileName1").fullPath()),
  m_fileName2(conf.getParameter<edm::FileInPath>("fileName2").fullPath())
{
  for (const auto &pset : conf.getParameter<std::vector<edm::ParameterSet>>("scoringPlanes"))
  {
    const unsigned int rpId = pset.getParameter<unsigned int>("rpId");
    const std::string dirName = pset.getParameter<std::string>("dirName");
    const double z = pset.getParameter<double>("z");
    const RPInfo entry = {dirName, z};
    m_rpInfo.emplace(rpId, entry);
  }

  setWhatProduced(this);
  findingRecord<CTPPSOpticsRcd>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSOpticalFunctionsESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
  const edm::IOVSyncValue& iosv, edm::ValidityInterval& oValidity)
{
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

//----------------------------------------------------------------------------------------------------

edm::ESProducts< std::unique_ptr<LHCOpticalFunctionsCollection> >
  CTPPSOpticalFunctionsESSource::produce(const CTPPSOpticsRcd &)
{
  // fill the output
  auto output = std::make_unique<LHCOpticalFunctionsCollection>();

  output->m_xangle1 = m_xangle1;
  output->m_xangle2 = m_xangle2;

  for (const auto &p : m_rpInfo)
  {
    LHCOpticalFunctionsSet fcn1(m_fileName1, p.second.dirName, p.second.scoringPlaneZ);
    fcn1.initializeSplines();
    output->m_functions1.emplace(p.first, std::move(fcn1));

    LHCOpticalFunctionsSet fcn2(m_fileName2, p.second.dirName, p.second.scoringPlaneZ);
    fcn2.initializeSplines();
    output->m_functions2.emplace(p.first, std::move(fcn2));
  }

  // commit the output
  return edm::es::products(std::move(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSOpticalFunctionsESSource);
