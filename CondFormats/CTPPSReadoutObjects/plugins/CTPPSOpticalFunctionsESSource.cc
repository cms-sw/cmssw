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
    ~CTPPSOpticalFunctionsESSource() override = default;

    edm::ESProducts<std::unique_ptr<LHCOpticalFunctionsCollection>> produce(const CTPPSOpticsRcd &);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;

    double m_xangle1, m_xangle2;
    std::string m_fileName1, m_fileName2;

    struct RPInfo
    {
      std::string dirName;
      double scoringPlaneZ;
    };
    std::unordered_map<unsigned int, RPInfo> m_rpInfo;
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

edm::ESProducts<std::unique_ptr<LHCOpticalFunctionsCollection> >
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

void
CTPPSOpticalFunctionsESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<double>("xangle1", 185.)->setComment("half crossing angle for sector 45");
  desc.add<edm::FileInPath>("fileName1", edm::FileInPath())->setComment("optical functions input file for sector 45");
  desc.add<double>("xangle2", 185.)->setComment("half crossing angle for sector 56");
  desc.add<edm::FileInPath>("fileName2", edm::FileInPath())->setComment("optical functions input file for sector 56");

  //--- information about scoring planes
  edm::ParameterSetDescription sp_desc;
  sp_desc.add<unsigned int>("rpId")->setComment("associated detector DetId");
  sp_desc.add<std::string>("dirName")->setComment("associated path to the optical functions file");
  sp_desc.add<double>("z")->setComment("longitudinal position at scoring plane/detector");
  std::vector<edm::ParameterSet> sp;
  desc.addVPSet("scoringPlanes", sp_desc, sp)->setComment("list of sensitive planes/detectors stations");

  descriptions.add("ctppsOpticalFunctionsESSource", desc);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSOpticalFunctionsESSource);

