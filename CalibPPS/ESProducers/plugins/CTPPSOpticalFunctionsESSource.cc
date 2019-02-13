// Original Author:  Jan Kašpar

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSetCollection.h"
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

    edm::ESProducts<std::unique_ptr<LHCOpticalFunctionsSetCollection>> produce(const CTPPSOpticsRcd &);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;

    struct FileInfo
    {
      double xangle;
      std::string fileName;
    };
    std::vector<FileInfo> m_fileInfo;

    struct RPInfo
    {
      std::string dirName;
      double scoringPlaneZ;
    };
    std::unordered_map<unsigned int, RPInfo> m_rpInfo;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSOpticalFunctionsESSource::CTPPSOpticalFunctionsESSource(const edm::ParameterSet& conf)
{
  for (const auto &pset : conf.getParameter<std::vector<edm::ParameterSet>>("opticalFunctions"))
  {
    const double &xangle = pset.getParameter<double>("xangle");
    const std::string &fileName = pset.getParameter<edm::FileInPath>("fileName").fullPath();
    m_fileInfo.push_back({xangle, fileName});
  }

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

edm::ESProducts<std::unique_ptr<LHCOpticalFunctionsSetCollection> >
CTPPSOpticalFunctionsESSource::produce(const CTPPSOpticsRcd &)
{
  // fill the output
  auto output = std::make_unique<LHCOpticalFunctionsSetCollection>();

  for (const auto &fi : m_fileInfo)
  {
    std::unordered_map<unsigned int, LHCOpticalFunctionsSet> xa_data;

    for (const auto &rpi : m_rpInfo)
    {
      LHCOpticalFunctionsSet fcn(fi.fileName, rpi.second.dirName, rpi.second.scoringPlaneZ);
      xa_data.emplace(rpi.first, std::move(fcn));
    }

    output->emplace(fi.xangle, xa_data);
  }

  // commit the output
  return edm::es::products(std::move(output));
}

//----------------------------------------------------------------------------------------------------

void
CTPPSOpticalFunctionsESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription of_desc;
  of_desc.add<double>("xangle")->setComment("half crossing angle value in urad");
  of_desc.add<edm::FileInPath>("fileName")->setComment("ROOT file with optical functions");
  std::vector<edm::ParameterSet> of;
  desc.addVPSet("opticalFunctions", of_desc, of)->setComment("list of optical functions at different crossing angles");

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

