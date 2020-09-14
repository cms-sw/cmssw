// -*- C++ -*-
//
// Package:    Tools/SiPixelGainCalibScaler
// Class:      SiPixelGainCalibScaler
//
/**\class SiPixelGainCalibScaler SiPixelGainCalibScaler.cc Tools/SiPixelGainCalibScaler/plugins/SiPixelGainCalibScaler.cc

 Description: Scales Pixel Gain Payloads by applying the VCal offset and slopes.

 Implementation:
     Makes use of trick to loop over all IOVs in a tag by running on all the runs with EmptySource and just access DB once the IOV has changed via ESWatcher mechanism
*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 16 Jul 2020 10:36:21 GMT
//
//

// system include files
#include <memory>

// user include files
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

namespace gainScale {
  struct VCalInfo {
  private:
    double m_conversionFactor;
    double m_conversionFactorL1;
    double m_offset;
    double m_offsetL1;

  public:
    // default constructor
    VCalInfo() : m_conversionFactor(0.), m_conversionFactorL1(0.), m_offset(0.), m_offsetL1(0.) {}

    // initialize
    void init(double conversionFactor, double conversionFactorL1, double offset, double offsetL1) {
      m_conversionFactor = conversionFactor;
      m_conversionFactorL1 = conversionFactorL1;
      m_offset = offset;
      m_offsetL1 = offsetL1;
    }

    void printAllInfo() {
      edm::LogVerbatim("SiPixelGainCalibScaler") << " conversion factor      : " << m_conversionFactor << "\n"
                                                 << " conversion factor (L1) : " << m_conversionFactorL1 << "\n"
                                                 << " offset                 : " << m_offset << "\n"
                                                 << " offset            (L1) : " << m_offsetL1 << "\n";
    }

    double getConversionFactor() { return m_conversionFactor; }
    double getConversionFactorL1() { return m_conversionFactorL1; }
    double getOffset() { return m_offset; }
    double getOffsetL1() { return m_offsetL1; }
    virtual ~VCalInfo() {}
  };
}  // namespace gainScale

class SiPixelGainCalibScaler : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelGainCalibScaler(const edm::ParameterSet&);
  ~SiPixelGainCalibScaler() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  template <class tokenType, class PayloadType>
  void computeAndStorePalyoads(const edm::EventSetup& iSetup, const tokenType& token);

  // ----------member data ---------------------------
  const std::string recordName_;
  const bool isForHLT_;
  const bool verbose_;
  const std::vector<edm::ParameterSet> m_parameters;

  gainScale::VCalInfo phase0VCal;
  gainScale::VCalInfo phase1VCal;

  edm::ESGetToken<SiPixelGainCalibrationForHLT, SiPixelGainCalibrationForHLTRcd> gainHLTCalibToken_;
  edm::ESGetToken<SiPixelGainCalibrationOffline, SiPixelGainCalibrationOfflineRcd> gainOfflineCalibToken_;

  edm::ESWatcher<SiPixelGainCalibrationForHLTRcd> pixelHLTGainWatcher_;
  edm::ESWatcher<SiPixelGainCalibrationOfflineRcd> pixelOfflineGainWatcher_;
};

//
// constructors and destructor
//
SiPixelGainCalibScaler::SiPixelGainCalibScaler(const edm::ParameterSet& iConfig)
    : recordName_(iConfig.getParameter<std::string>("record")),
      isForHLT_(iConfig.getParameter<bool>("isForHLT")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      m_parameters(iConfig.getParameter<std::vector<edm::ParameterSet> >("parameters")) {
  gainHLTCalibToken_ = esConsumes<SiPixelGainCalibrationForHLT, SiPixelGainCalibrationForHLTRcd>();
  gainOfflineCalibToken_ = esConsumes<SiPixelGainCalibrationOffline, SiPixelGainCalibrationOfflineRcd>();

  for (auto& thePSet : m_parameters) {
    const unsigned int phase(thePSet.getParameter<unsigned int>("phase"));
    switch (phase) {
      case 0: {
        phase0VCal.init(thePSet.getParameter<double>("conversionFactor"),
                        thePSet.getParameter<double>("conversionFactorL1"),
                        thePSet.getParameter<double>("offset"),
                        thePSet.getParameter<double>("offsetL1"));
        break;
      }
      case 1: {
        phase1VCal.init(thePSet.getParameter<double>("conversionFactor"),
                        thePSet.getParameter<double>("conversionFactorL1"),
                        thePSet.getParameter<double>("offset"),
                        thePSet.getParameter<double>("offsetL1"));
        break;
      }
      default:
        throw cms::Exception("LogicError") << "Unrecongnized phase: " << phase << ". Exiting!";
    }
  }
}

SiPixelGainCalibScaler::~SiPixelGainCalibScaler() {}

//
// member functions
//

// ------------ method called for each event  ------------
void SiPixelGainCalibScaler::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  int run = iEvent.id().run();
  bool hasPixelHLTGainIOV = pixelHLTGainWatcher_.check(iSetup);
  bool hasPixelOfflineGainIOV = pixelOfflineGainWatcher_.check(iSetup);

  if ((hasPixelHLTGainIOV && isForHLT_) || (hasPixelOfflineGainIOV && !isForHLT_)) {
    edm::LogPrint("SiPixelGainCalibScaler") << " Pixel Gains have a new IOV for run: " << run << std::endl;

    if (isForHLT_) {
      computeAndStorePalyoads<edm::ESGetToken<SiPixelGainCalibrationForHLT, SiPixelGainCalibrationForHLTRcd>,
                              SiPixelGainCalibrationForHLT>(iSetup, gainHLTCalibToken_);
    } else {
      computeAndStorePalyoads<edm::ESGetToken<SiPixelGainCalibrationOffline, SiPixelGainCalibrationOfflineRcd>,
                              SiPixelGainCalibrationOffline>(iSetup, gainOfflineCalibToken_);
    }
  }  // if new IOV
}

// ------------ template method to construct the payloads  ------------
template <class tokenType, class PayloadType>
void SiPixelGainCalibScaler::computeAndStorePalyoads(const edm::EventSetup& iSetup, const tokenType& token) {
  gainScale::VCalInfo myVCalInfo;

  //=======================================================
  // Retrieve geometry information
  //=======================================================
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  edm::LogInfo("SiPixelGainCalibScaler") << "There are: " << pDD->dets().size() << " detectors";

  // switch on the phase1
  if ((pDD->isThere(GeomDetEnumerators::P1PXB)) || (pDD->isThere(GeomDetEnumerators::P1PXEC))) {
    myVCalInfo = phase1VCal;
    edm::LogInfo("SiPixelGainCalibScaler") << " ==> This is a phase1 IOV";
  } else {
    myVCalInfo = phase0VCal;
    edm::LogInfo("SiPixelGainCalibScaler") << " ==> This is a phase0 IOV";
  }

  myVCalInfo.printAllInfo();

  // if need the ESHandle to check if the SetupData was there or not
  auto payload = iSetup.getHandle(token);
  std::vector<uint32_t> detids;
  payload->getDetIds(detids);

  float mingain = payload->getGainLow();
  float maxgain = (payload->getGainHigh()) * myVCalInfo.getConversionFactorL1();
  float minped = payload->getPedLow();
  float maxped = payload->getPedHigh() * 1.10;

  auto SiPixelGainCalibration_ = new PayloadType(minped, maxped, mingain, maxgain);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* tTopo = tTopoHandle.product();

  //const char* path_toTopologyXML = "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
  //TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

  for (const auto& d : detids) {
    bool isLayer1 = false;
    int subid = DetId(d).subdetId();
    if (subid == PixelSubdetector::PixelBarrel) {
      auto layer = tTopo->pxbLayer(DetId(d));
      if (layer == 1) {
        isLayer1 = true;
      }
    }

    std::vector<char> theSiPixelGainCalibration;

    auto range = payload->getRange(d);
    int numberOfRowsToAverageOver = payload->getNumberOfRowsToAverageOver();
    int ncols = payload->getNCols(d);
    int nRocsInRow = (range.second - range.first) / ncols / numberOfRowsToAverageOver;
    unsigned int nRowsForHLT = 1;
    int nrows = std::max((payload->getNumberOfRowsToAverageOver() * nRocsInRow),
                         nRowsForHLT);  // dirty trick to make it work for the HLT payload

    auto rangeAndCol = payload->getRangeAndNCols(d);
    bool isDeadColumn;
    bool isNoisyColumn;

    if (verbose_) {
      edm::LogVerbatim("SiPixelGainCalibScaler")
          << "NCOLS: " << payload->getNCols(d) << " " << rangeAndCol.second << " NROWS:" << nrows
          << ", RANGES: " << rangeAndCol.first.second - rangeAndCol.first.first
          << ", Ratio: " << float(rangeAndCol.first.second - rangeAndCol.first.first) / rangeAndCol.second << std::endl;
    }

    for (int col = 0; col < ncols; col++) {
      for (int row = 0; row < nrows; row++) {
        float gain = payload->getGain(col, row, rangeAndCol.first, rangeAndCol.second, isDeadColumn, isNoisyColumn);
        float ped = payload->getPed(col, row, rangeAndCol.first, rangeAndCol.second, isDeadColumn, isNoisyColumn);

        if (verbose_)
          edm::LogInfo("SiPixelGainCalibScaler") << "pre-change gain: " << gain << " pede:" << ped << std::endl;

        //
        // From here https://github.com/cms-sw/cmssw/blob/master/CalibTracker/SiPixelESProducers/src/SiPixelGainCalibrationForHLTService.cc#L20-L47
        //
        // vcal = ADC * DBgain - DBped * DBgain
        // electrons = vcal * conversionFactor + offset
        //
        // follows:
        // electrons = (ADC*DBgain – DBped*DBgain)*conversionFactor + offset
        // electrons = ADC*conversionFactor*DBgain - conversionFactor*DBped*DBgain + offset
        //
        // this should equal the new equation:
        //
        // electrons = ADC*DBgain' - DBPed' * DBgain'
        //
        // So equating piece by piece:
        //
        // DBgain' = conversionFactor*DBgain
        // DBped' = (conversionFactor*DBped*Dbgain – offset)/(conversionFactor*DBgain)
        //        = DBped - offset/DBgain'
        //

        if (isLayer1) {
          gain = gain * myVCalInfo.getConversionFactorL1();
          ped = ped - myVCalInfo.getOffsetL1() / gain;
        } else {
          gain = gain * myVCalInfo.getConversionFactor();
          ped = ped - myVCalInfo.getOffset() / gain;
        }

        if (verbose_)
          edm::LogInfo("SiPixelGainCalibScaler") << "post-change gain: " << gain << " pede:" << ped << std::endl;

        if constexpr (std::is_same_v<PayloadType, SiPixelGainCalibrationForHLT>) {
          SiPixelGainCalibration_->setData(ped, gain, theSiPixelGainCalibration, false, false);
        } else {
          SiPixelGainCalibration_->setDataPedestal(ped, theSiPixelGainCalibration);
          if ((row + 1) % numberOfRowsToAverageOver == 0) {  // fill the column average after every ROC!
            SiPixelGainCalibration_->setDataGain(gain, numberOfRowsToAverageOver, theSiPixelGainCalibration);
          }
        }
      }  // loop on rows
    }    // loop on columns

    typename PayloadType::Range outrange(theSiPixelGainCalibration.begin(), theSiPixelGainCalibration.end());
    if (!SiPixelGainCalibration_->put(d, outrange, ncols))
      edm::LogError("SiPixelGainCalibScaler") << "[SiPixelGainCalibScaler::analyze] detid already exists" << std::endl;
  }  // loop on DetIds

  // Write into DB
  edm::LogInfo(" --- writing to DB!");
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("db service unavailable");
    return;
  } else {
    edm::LogInfo("DB service OK");
  }

  try {
    if (mydbservice->isNewTagRequest(recordName_)) {
      mydbservice->createNewIOV<PayloadType>(
          SiPixelGainCalibration_, mydbservice->beginOfTime(), mydbservice->endOfTime(), recordName_);
    } else {
      mydbservice->appendSinceTime<PayloadType>(SiPixelGainCalibration_, mydbservice->currentTime(), recordName_);
    }
    edm::LogInfo(" --- all OK");
  } catch (const cond::Exception& er) {
    edm::LogError("SiPixelGainCalibScaler") << er.what() << std::endl;
  } catch (const std::exception& er) {
    edm::LogError("SiPixelGainCalibScaler") << "caught std::exception " << er.what() << std::endl;
  } catch (...) {
    edm::LogError("SiPixelGainCalibScaler") << "Funny error" << std::endl;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SiPixelGainCalibScaler::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiPixelGainCalibScaler::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPixelGainCalibScaler::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("record", "SiPixelGainCalibrationForHLTRcd");
  desc.add<bool>("isForHLT", true);

  edm::ParameterSetDescription vcalInfos;
  vcalInfos.add<unsigned int>("phase");
  vcalInfos.add<double>("conversionFactor");
  vcalInfos.add<double>("conversionFactorL1");
  vcalInfos.add<double>("offset");
  vcalInfos.add<double>("offsetL1");

  std::vector<edm::ParameterSet> tmp;
  tmp.reserve(2);
  {
    edm::ParameterSet phase0VCal;
    phase0VCal.addParameter<unsigned int>("phase", 0);
    phase0VCal.addParameter<double>("conversionFactor", 65.);
    phase0VCal.addParameter<double>("conversionFactorL1", 65.);
    phase0VCal.addParameter<double>("offset", -414.);
    phase0VCal.addParameter<double>("offsetL1", -414.);
    tmp.push_back(phase0VCal);
  }
  {
    edm::ParameterSet phase1VCal;
    phase1VCal.addParameter<unsigned int>("phase", 1);
    phase1VCal.addParameter<double>("conversionFactor", 47.);
    phase1VCal.addParameter<double>("conversionFactorL1", 50.);
    phase1VCal.addParameter<double>("offset", -60.);
    phase1VCal.addParameter<double>("offsetL1", -670.);
    tmp.push_back(phase1VCal);
  }
  desc.addVPSet("parameters", vcalInfos, tmp);

  desc.addUntracked<bool>("verbose", false);

  descriptions.add("siPixelGainCalibScaler", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelGainCalibScaler);
