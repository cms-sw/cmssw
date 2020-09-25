// Original Author:  Jan Kašpar

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/JamesRandom.h"

#include "TFile.h"
#include "TH2D.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Provides LHCInfo data necessary for CTPPS reconstruction (and direct simulation).
 **/
class CTPPSLHCInfoRandomXangleESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CTPPSLHCInfoRandomXangleESSource(const edm::ParameterSet &);
  edm::ESProducts<std::unique_ptr<LHCInfo>> produce(const LHCInfoRcd &);
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;

  std::string m_label;

  unsigned int m_generateEveryNEvents;

  double m_beamEnergy;
  double m_betaStar;

  std::unique_ptr<CLHEP::HepRandomEngine> m_engine;

  template <typename T>
  struct BinData {
    double min, max;
    T profile;
  };

  std::vector<BinData<std::pair<double,double>>> xangleBetaStarBins;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSLHCInfoRandomXangleESSource::CTPPSLHCInfoRandomXangleESSource(const edm::ParameterSet &conf)
    : m_label(conf.getParameter<std::string>("label")),

      m_generateEveryNEvents(conf.getParameter<unsigned int>("generateEveryNEvents")),

      m_beamEnergy(conf.getParameter<double>("beamEnergy")),

      m_engine(new CLHEP::HepJamesRandom(conf.getParameter<unsigned int>("seed"))) {
  const auto &xangleBetaStarHistogramFile = conf.getParameter<std::string>("xangleBetaStarHistogramFile");
  const auto &xangleBetaStarHistogramObject = conf.getParameter<std::string>("xangleBetaStarHistogramObject");

  TFile *f_in = TFile::Open(xangleBetaStarHistogramFile.c_str());
  if (!f_in)
    throw cms::Exception("PPS") << "Cannot open input file '" << xangleBetaStarHistogramFile << "'.";

  TH2D *h_xangle_beta_star = (TH2D *)f_in->Get(xangleBetaStarHistogramObject.c_str());
  if (!h_xangle_beta_star)
    throw cms::Exception("PPS") << "Cannot load input object '" << xangleBetaStarHistogramObject << "'.";
  h_xangle_beta_star->SetDirectory(0);
  delete f_in;

  double sum = 0.;
  for (int bi = 1; bi <= h_xangle_beta_star->GetNcells(); ++bi){
    double val=h_xangle_beta_star->GetBinContent(bi);
    sum+=val;
  }

  double cw=0;
  for (int x = 1; x <= h_xangle_beta_star->GetNbinsX(); ++x) 
    for (int y = 1; y <= h_xangle_beta_star->GetNbinsY(); ++y) {
      double sample=h_xangle_beta_star->GetBinContent(h_xangle_beta_star->GetBin(x,y));
      if(sample>=1){
        sample/=sum;
        xangleBetaStarBins.push_back({cw,cw+sample,std::pair<double,double>(h_xangle_beta_star->GetXaxis()->GetBinCenter(x),h_xangle_beta_star->GetYaxis()->GetBinCenter(y))});
        cw+=sample;
      }
    }
  delete h_xangle_beta_star;

  setWhatProduced(this, m_label);
  findingRecord<LHCInfoRcd>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoRandomXangleESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("label", "")->setComment("label of the LHCInfo record");

  desc.add<unsigned int>("seed", 1)->setComment("random seed");

  desc.add<unsigned int>("generateEveryNEvents", 1)->setComment("how often to generate new xangle");

  desc.add<std::string>("xangleBetaStarHistogramFile", "")->setComment("ROOT file with xangle distribution");
  desc.add<std::string>("xangleBetaStarHistogramObject", "")->setComment("xangle distribution object in the ROOT file");

  desc.add<double>("beamEnergy", 0.)->setComment("beam energy");

  descriptions.add("ctppsLHCInfoRandomXangleESSource", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoRandomXangleESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                                                      const edm::IOVSyncValue &iosv,
                                                      edm::ValidityInterval &oValidity) {
  edm::EventID beginEvent = iosv.eventID();
  edm::EventID endEvent(beginEvent.run(), beginEvent.luminosityBlock(), beginEvent.event() + m_generateEveryNEvents);
  oValidity = edm::ValidityInterval(edm::IOVSyncValue(beginEvent), edm::IOVSyncValue(endEvent));
}

//----------------------------------------------------------------------------------------------------

edm::ESProducts<std::unique_ptr<LHCInfo>> CTPPSLHCInfoRandomXangleESSource::produce(const LHCInfoRcd &) {
  auto output = std::make_unique<LHCInfo>();

  double xangle=0, beta = 0.;
  const double u = CLHEP::RandFlat::shoot(m_engine.get(), 0., 1.);
  for (const auto &d : xangleBetaStarBins) {
    if (d.min <= u && u <= d.max) {
      xangle = d.profile.first;
      beta=d.profile.second;
      break;
    }
  }

  output->setEnergy(m_beamEnergy);
  output->setCrossingAngle(xangle);
  output->setBetaStar(beta);


  return edm::es::products(std::move(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSLHCInfoRandomXangleESSource);
