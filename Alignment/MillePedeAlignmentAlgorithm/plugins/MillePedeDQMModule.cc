/**
 * @package   Alignment/MillePedeAlignmentAlgorithm
 * @file      MillePedeDQMModule.cc
 *
 * @author    Max Stark (max.stark@cern.ch)
 * @date      Feb 19, 2016
 */

/*** header-file ***/
#include "Alignment/MillePedeAlignmentAlgorithm/plugins/MillePedeDQMModule.h"

/*** ROOT objects ***/
#include "TH1F.h"

/*** Core framework functionality ***/
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*** Alignment ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerPluginFactory.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

/*** Necessary Framework infrastructure ***/
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "DataFormats/Alignment/interface/AlignmentToken.h"

MillePedeDQMModule ::MillePedeDQMModule(const edm::ParameterSet& config)
    : tTopoToken_(esConsumes<edm::Transition::BeginRun>()),
      gDetToken_(esConsumes<edm::Transition::BeginRun>()),
      ptpToken_(esConsumes<edm::Transition::BeginRun>()),
      ptitpToken_(esConsumes<edm::Transition::BeginRun>()),
      aliThrToken_(esConsumes<edm::Transition::BeginRun>()),
      geomToken_(esConsumes<edm::Transition::BeginRun>()),
      mpReaderConfig_(config.getParameter<edm::ParameterSet>("MillePedeFileReader")),
      isHG_(mpReaderConfig_.getParameter<bool>("isHG")) {
  consumes<AlignmentToken, edm::InProcess>(config.getParameter<edm::InputTag>("alignmentTokenSrc"));
}

MillePedeDQMModule ::~MillePedeDQMModule() {}

//=============================================================================
//===   INTERFACE IMPLEMENTATION                                            ===
//=============================================================================

void MillePedeDQMModule ::bookHistograms(DQMStore::IBooker& booker) {
  edm::LogInfo("MillePedeDQMModule") << "Booking histograms";

  booker.cd();
  if (!isHG_) {
    booker.setCurrentFolder("AlCaReco/SiPixelAli/");
    h_xPos = booker.book1D("Xpos", "Alignment fit #DeltaX;;#mum", 36, 0., 36.);
    h_xRot = booker.book1D("Xrot", "Alignment fit #Delta#theta_{X};;#murad", 36, 0., 36.);
    h_yPos = booker.book1D("Ypos", "Alignment fit #DeltaY;;#mum", 36, 0., 36.);
    h_yRot = booker.book1D("Yrot", "Alignment fit #Delta#theta_{Y};;#murad", 36, 0., 36.);
    h_zPos = booker.book1D("Zpos", "Alignment fit #DeltaZ;;#mum", 36, 0., 36.);
    h_zRot = booker.book1D("Zrot", "Alignment fit #Delta#theta_{Z};;#murad", 36, 0., 36.);
    statusResults = booker.book2D("statusResults", "Status of SiPixelAli PCL workflow;;", 6, 0., 6., 1, 0., 1.);
  } else {
    booker.setCurrentFolder("AlCaReco/SiPixelAliHG/");

    layerVec = {{"Layer1", pixelTopologyMap_->getPXBLadders(1)},
                {"Layer2", pixelTopologyMap_->getPXBLadders(2)},
                {"Layer3", pixelTopologyMap_->getPXBLadders(3)},
                {"Layer4", pixelTopologyMap_->getPXBLadders(4)},
                {"Disk-3", pixelTopologyMap_->getPXFBlades(-3) * 2},
                {"Disk-2", pixelTopologyMap_->getPXFBlades(-2) * 2},
                {"Disk-1", pixelTopologyMap_->getPXFBlades(-1) * 2},
                {"Disk1", pixelTopologyMap_->getPXFBlades(1) * 2},
                {"Disk2", pixelTopologyMap_->getPXFBlades(2) * 2},
                {"Disk3", pixelTopologyMap_->getPXFBlades(3) * 2}};

    for (const auto& layer : layerVec) {
      h_xPos_HG[layer.first] = booker.book1D("Xpos_HG_" + layer.first,
                                             "Alignment fit #DeltaX for " + layer.first + ";;#mum",
                                             layer.second,
                                             0.,
                                             layer.second);
      h_xRot_HG[layer.first] = booker.book1D("Xrot_HG_" + layer.first,
                                             "Alignment fit #Delta#theta_{X} for " + layer.first + ";;#murad",
                                             layer.second,
                                             0.,
                                             layer.second);
      h_yPos_HG[layer.first] = booker.book1D("Ypos_HG_" + layer.first,
                                             "Alignment fit #DeltaY for " + layer.first + ";;#mum",
                                             layer.second,
                                             0.,
                                             layer.second);
      h_yRot_HG[layer.first] = booker.book1D("Yrot_HG_" + layer.first,
                                             "Alignment fit #Delta#theta_{Y} for " + layer.first + ";;#murad",
                                             layer.second,
                                             0.,
                                             layer.second);
      h_zPos_HG[layer.first] = booker.book1D("Zpos_HG_" + layer.first,
                                             "Alignment fit #DeltaZ for " + layer.first + ";;#mum",
                                             layer.second,
                                             0.,
                                             layer.second);
      h_zRot_HG[layer.first] = booker.book1D("Zrot_HG_" + layer.first,
                                             "Alignment fit #Delta#theta_{Z} for " + layer.first + ";;#murad",
                                             layer.second,
                                             0.,
                                             layer.second);
    }

    statusResults =
        booker.book2D("statusResults", "Fraction threshold check for SiPixelAliHG PCL;;", 6, 0., 6., 10, 0., 10.);
  }

  binariesAvalaible = booker.bookInt("BinariesFound");
  exitCode = booker.bookString("PedeExitCode", "");

  booker.cd();
}

void MillePedeDQMModule ::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter&) {
  bookHistograms(booker);
  if (mpReader_) {
    mpReader_->read();
  } else {
    throw cms::Exception("LogicError") << "@SUB=MillePedeDQMModule::dqmEndJob\n"
                                       << "Try to read MillePede results before initializing MillePedeFileReader";
  }
  if (!isHG_) {
    fillExpertHistos();
    fillStatusHisto(statusResults);
  } else {
    fillExpertHistos_HG();
    fillStatusHistoHG(statusResults);
  }
  binariesAvalaible->Fill(mpReader_->binariesAmount());
  auto theResults = mpReader_->getResults();
  std::string exitCodeStr = theResults.getExitMessage();
  exitCode->Fill(exitCodeStr);
}

//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

void MillePedeDQMModule ::beginRun(const edm::Run&, const edm::EventSetup& setup) {
  if (!setupChanged(setup))
    return;

  const TrackerTopology* const tTopo = &setup.getData(tTopoToken_);
  const GeometricDet* geometricDet = &setup.getData(gDetToken_);
  const PTrackerParameters* ptp = &setup.getData(ptpToken_);
  const PTrackerAdditionalParametersPerDet* ptitp = &setup.getData(ptitpToken_);
  const TrackerGeometry* geom = &setup.getData(geomToken_);

  pixelTopologyMap_ = std::make_shared<PixelTopologyMap>(geom, tTopo);

  // take the thresholds from DB
  const auto& thresholds_ = &setup.getData(aliThrToken_);

  auto myThresholds = std::make_shared<AlignPCLThresholdsHG>();
  myThresholds->setAlignPCLThresholds(thresholds_->getNrecords(), thresholds_->getThreshold_Map());
  myThresholds->setFloatMap(thresholds_->getFloatMap());

  TrackerGeomBuilderFromGeometricDet builder;

  const auto trackerGeometry = builder.build(geometricDet, ptitp, *ptp, tTopo);
  tracker_ = std::make_unique<AlignableTracker>(trackerGeometry, tTopo);

  const std::string labelerPlugin{"PedeLabeler"};
  edm::ParameterSet labelerConfig{};
  labelerConfig.addUntrackedParameter("plugin", labelerPlugin);
  labelerConfig.addUntrackedParameter("RunRangeSelection", edm::VParameterSet{});

  std::shared_ptr<PedeLabelerBase> pedeLabeler{PedeLabelerPluginFactory::get()->create(
      labelerPlugin, PedeLabelerBase::TopLevelAlignables(tracker_.get(), nullptr, nullptr), labelerConfig)};

  mpReader_ = std::make_unique<MillePedeFileReader>(
      mpReaderConfig_, pedeLabeler, std::shared_ptr<const AlignPCLThresholdsHG>(myThresholds), pixelTopologyMap_);
}

void MillePedeDQMModule ::fillStatusHisto(MonitorElement* statusHisto) {
  TH2F* histo_status = statusHisto->getTH2F();
  auto theResults = mpReader_->getResults();
  theResults.print();
  histo_status->SetBinContent(1, 1, theResults.getDBUpdated());
  histo_status->GetXaxis()->SetBinLabel(1, "DB updated");
  histo_status->SetBinContent(2, 1, theResults.exceedsCutoffs());
  histo_status->GetXaxis()->SetBinLabel(2, "significant movement");
  histo_status->SetBinContent(3, 1, theResults.getDBVetoed());
  histo_status->GetXaxis()->SetBinLabel(3, "DB update vetoed");
  histo_status->SetBinContent(4, 1, !theResults.exceedsThresholds());
  histo_status->GetXaxis()->SetBinLabel(4, "within max movement");
  histo_status->SetBinContent(5, 1, !theResults.exceedsMaxError());
  histo_status->GetXaxis()->SetBinLabel(5, "within max error");
  histo_status->SetBinContent(6, 1, !theResults.belowSignificance());
  histo_status->GetXaxis()->SetBinLabel(6, "above significance");
}

void MillePedeDQMModule ::fillStatusHistoHG(MonitorElement* statusHisto) {
  TH2F* histo_status = statusHisto->getTH2F();
  auto& theResults = mpReader_->getResultsHG();
  histo_status->GetXaxis()->SetBinLabel(1, "#DeltaX");
  histo_status->GetXaxis()->SetBinLabel(2, "#Delta#theta_{X}");
  histo_status->GetXaxis()->SetBinLabel(3, "#DeltaY");
  histo_status->GetXaxis()->SetBinLabel(4, "#Delta#theta_{Y}");
  histo_status->GetXaxis()->SetBinLabel(5, "#DeltaZ");
  histo_status->GetXaxis()->SetBinLabel(6, "#Delta#theta_{Z}");

  int i = 0;
  for (const auto& result : theResults) {
    histo_status->GetYaxis()->SetBinLabel(i + 1, result.first.data());
    for (std::size_t j = 0; j < result.second.size(); ++j) {
      histo_status->SetBinContent(j + 1, i + 1, result.second[j]);
    }
    i++;
  }
}

void MillePedeDQMModule ::fillExpertHistos() {
  std::array<double, SIZE_INDEX> Xcut_, sigXcut_, maxMoveXcut_, maxErrorXcut_;
  std::array<double, SIZE_INDEX> tXcut_, sigtXcut_, maxMovetXcut_, maxErrortXcut_;

  std::array<double, SIZE_INDEX> Ycut_, sigYcut_, maxMoveYcut_, maxErrorYcut_;
  std::array<double, SIZE_INDEX> tYcut_, sigtYcut_, maxMovetYcut_, maxErrortYcut_;

  std::array<double, SIZE_INDEX> Zcut_, sigZcut_, maxMoveZcut_, maxErrorZcut_;
  std::array<double, SIZE_INDEX> tZcut_, sigtZcut_, maxMovetZcut_, maxErrortZcut_;

  auto myMap = mpReader_->getThresholdMap();

  std::vector<std::string> alignablesList;
  for (auto it = myMap.begin(); it != myMap.end(); ++it) {
    alignablesList.push_back(it->first);
  }

  for (auto& alignable : alignablesList) {
    int detIndex = getIndexFromString(alignable);

    Xcut_[detIndex] = myMap[alignable].getXcut();
    sigXcut_[detIndex] = myMap[alignable].getSigXcut();
    maxMoveXcut_[detIndex] = myMap[alignable].getMaxMoveXcut();
    maxErrorXcut_[detIndex] = myMap[alignable].getErrorXcut();

    Ycut_[detIndex] = myMap[alignable].getYcut();
    sigYcut_[detIndex] = myMap[alignable].getSigYcut();
    maxMoveYcut_[detIndex] = myMap[alignable].getMaxMoveYcut();
    maxErrorYcut_[detIndex] = myMap[alignable].getErrorYcut();

    Zcut_[detIndex] = myMap[alignable].getZcut();
    sigZcut_[detIndex] = myMap[alignable].getSigZcut();
    maxMoveZcut_[detIndex] = myMap[alignable].getMaxMoveZcut();
    maxErrorZcut_[detIndex] = myMap[alignable].getErrorZcut();

    tXcut_[detIndex] = myMap[alignable].getThetaXcut();
    sigtXcut_[detIndex] = myMap[alignable].getSigThetaXcut();
    maxMovetXcut_[detIndex] = myMap[alignable].getMaxMoveThetaXcut();
    maxErrortXcut_[detIndex] = myMap[alignable].getErrorThetaXcut();

    tYcut_[detIndex] = myMap[alignable].getThetaYcut();
    sigtYcut_[detIndex] = myMap[alignable].getSigThetaYcut();
    maxMovetYcut_[detIndex] = myMap[alignable].getMaxMoveThetaYcut();
    maxErrortYcut_[detIndex] = myMap[alignable].getErrorThetaYcut();

    tZcut_[detIndex] = myMap[alignable].getThetaZcut();
    sigtZcut_[detIndex] = myMap[alignable].getSigThetaZcut();
    maxMovetZcut_[detIndex] = myMap[alignable].getMaxMoveThetaZcut();
    maxErrortZcut_[detIndex] = myMap[alignable].getErrorThetaZcut();
  }

  fillExpertHisto(h_xPos, Xcut_, sigXcut_, maxMoveXcut_, maxErrorXcut_, mpReader_->getXobs(), mpReader_->getXobsErr());
  fillExpertHisto(
      h_xRot, tXcut_, sigtXcut_, maxMovetXcut_, maxErrortXcut_, mpReader_->getTXobs(), mpReader_->getTXobsErr());

  fillExpertHisto(h_yPos, Ycut_, sigYcut_, maxMoveYcut_, maxErrorYcut_, mpReader_->getYobs(), mpReader_->getYobsErr());
  fillExpertHisto(
      h_yRot, tYcut_, sigtYcut_, maxMovetYcut_, maxErrortYcut_, mpReader_->getTYobs(), mpReader_->getTYobsErr());

  fillExpertHisto(h_zPos, Zcut_, sigZcut_, maxMoveZcut_, maxErrorZcut_, mpReader_->getZobs(), mpReader_->getZobsErr());
  fillExpertHisto(
      h_zRot, tZcut_, sigtZcut_, maxMovetZcut_, maxErrortZcut_, mpReader_->getTZobs(), mpReader_->getTZobsErr());
}

void MillePedeDQMModule ::fillExpertHisto(MonitorElement* histo,
                                          const std::array<double, SIZE_INDEX>& cut,
                                          const std::array<double, SIZE_INDEX>& sigCut,
                                          const std::array<double, SIZE_INDEX>& maxMoveCut,
                                          const std::array<double, SIZE_INDEX>& maxErrorCut,
                                          const std::array<double, SIZE_LG_STRUCTS>& obs,
                                          const std::array<double, SIZE_LG_STRUCTS>& obsErr) {
  TH1F* histo_0 = histo->getTH1F();

  double max_ = *std::max_element(maxMoveCut.begin(), maxMoveCut.end());

  histo_0->SetMinimum(-(max_));
  histo_0->SetMaximum(max_);

  //  Schematics of the bin contents
  //
  //  XX XX XX XX XX XX    OO OO OO OO    II II II II
  // |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
  // | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|  ...
  //
  // |-----------------|  |-----------|  |-----------|
  // |observed movement|  |thresholds1|  |thresholds2|

  for (size_t i = 0; i < obs.size(); ++i) {
    // fist obs.size() bins for observed movements
    histo_0->SetBinContent(i + 1, obs[i]);
    histo_0->SetBinError(i + 1, obsErr[i]);

    // then at bin 8,8+5,8+10,... for cutoffs
    // 5 bins is the space allocated for the 4 other thresholds + 1 empty separation bin
    histo_0->SetBinContent(8 + i * 5, cut[i]);

    // then at bin 9,9+5,9+10,... for significances
    histo_0->SetBinContent(9 + i * 5, sigCut[i]);

    // then at bin 10,10+5,10+10,... for maximum movements
    histo_0->SetBinContent(10 + i * 5, maxMoveCut[i]);

    // then at bin 11,11+5,11+10,... for maximum errors
    histo_0->SetBinContent(11 + i * 5, maxErrorCut[i]);
  }
}

void MillePedeDQMModule ::fillExpertHistos_HG() {
  std::array<double, SIZE_INDEX> Xcut_, sigXcut_, maxMoveXcut_, maxErrorXcut_;
  std::array<double, SIZE_INDEX> tXcut_, sigtXcut_, maxMovetXcut_, maxErrortXcut_;

  std::array<double, SIZE_INDEX> Ycut_, sigYcut_, maxMoveYcut_, maxErrorYcut_;
  std::array<double, SIZE_INDEX> tYcut_, sigtYcut_, maxMovetYcut_, maxErrortYcut_;

  std::array<double, SIZE_INDEX> Zcut_, sigZcut_, maxMoveZcut_, maxErrorZcut_;
  std::array<double, SIZE_INDEX> tZcut_, sigtZcut_, maxMovetZcut_, maxErrortZcut_;

  auto myMap = mpReader_->getThresholdMap();

  std::vector<std::string> alignablesList;
  for (auto it = myMap.begin(); it != myMap.end(); ++it) {
    alignablesList.push_back(it->first);
  }

  for (auto& alignable : alignablesList) {
    int detIndex = getIndexFromString(alignable);

    Xcut_[detIndex] = myMap[alignable].getXcut();
    sigXcut_[detIndex] = myMap[alignable].getSigXcut();
    maxMoveXcut_[detIndex] = myMap[alignable].getMaxMoveXcut();
    maxErrorXcut_[detIndex] = myMap[alignable].getErrorXcut();

    Ycut_[detIndex] = myMap[alignable].getYcut();
    sigYcut_[detIndex] = myMap[alignable].getSigYcut();
    maxMoveYcut_[detIndex] = myMap[alignable].getMaxMoveYcut();
    maxErrorYcut_[detIndex] = myMap[alignable].getErrorYcut();

    Zcut_[detIndex] = myMap[alignable].getZcut();
    sigZcut_[detIndex] = myMap[alignable].getSigZcut();
    maxMoveZcut_[detIndex] = myMap[alignable].getMaxMoveZcut();
    maxErrorZcut_[detIndex] = myMap[alignable].getErrorZcut();

    tXcut_[detIndex] = myMap[alignable].getThetaXcut();
    sigtXcut_[detIndex] = myMap[alignable].getSigThetaXcut();
    maxMovetXcut_[detIndex] = myMap[alignable].getMaxMoveThetaXcut();
    maxErrortXcut_[detIndex] = myMap[alignable].getErrorThetaXcut();

    tYcut_[detIndex] = myMap[alignable].getThetaYcut();
    sigtYcut_[detIndex] = myMap[alignable].getSigThetaYcut();
    maxMovetYcut_[detIndex] = myMap[alignable].getMaxMoveThetaYcut();
    maxErrortYcut_[detIndex] = myMap[alignable].getErrorThetaYcut();

    tZcut_[detIndex] = myMap[alignable].getThetaZcut();
    sigtZcut_[detIndex] = myMap[alignable].getSigThetaZcut();
    maxMovetZcut_[detIndex] = myMap[alignable].getMaxMoveThetaZcut();
    maxErrortZcut_[detIndex] = myMap[alignable].getErrorThetaZcut();
  }

  fillExpertHisto_HG(
      h_xPos_HG, Xcut_, sigXcut_, maxMoveXcut_, maxErrorXcut_, mpReader_->getXobs_HG(), mpReader_->getXobsErr_HG());
  fillExpertHisto_HG(h_xRot_HG,
                     tXcut_,
                     sigtXcut_,
                     maxMovetXcut_,
                     maxErrortXcut_,
                     mpReader_->getTXobs_HG(),
                     mpReader_->getTXobsErr_HG());

  fillExpertHisto_HG(
      h_yPos_HG, Ycut_, sigYcut_, maxMoveYcut_, maxErrorYcut_, mpReader_->getYobs_HG(), mpReader_->getYobsErr_HG());
  fillExpertHisto_HG(h_yRot_HG,
                     tYcut_,
                     sigtYcut_,
                     maxMovetYcut_,
                     maxErrortYcut_,
                     mpReader_->getTYobs_HG(),
                     mpReader_->getTYobsErr_HG());

  fillExpertHisto_HG(
      h_zPos_HG, Zcut_, sigZcut_, maxMoveZcut_, maxErrorZcut_, mpReader_->getZobs_HG(), mpReader_->getZobsErr_HG());
  fillExpertHisto_HG(h_zRot_HG,
                     tZcut_,
                     sigtZcut_,
                     maxMovetZcut_,
                     maxErrortZcut_,
                     mpReader_->getTZobs_HG(),
                     mpReader_->getTZobsErr_HG());
}

void MillePedeDQMModule ::fillExpertHisto_HG(std::map<std::string, MonitorElement*>& histo_map,
                                             const std::array<double, SIZE_INDEX>& cut,
                                             const std::array<double, SIZE_INDEX>& sigCut,
                                             const std::array<double, SIZE_INDEX>& maxMoveCut,
                                             const std::array<double, SIZE_INDEX>& maxErrorCut,
                                             const std::array<double, SIZE_HG_STRUCTS>& obs,
                                             const std::array<double, SIZE_HG_STRUCTS>& obsErr) {
  int currentStart = 0;

  for (const auto& layer : layerVec) {
    TH1F* histo_0 = histo_map[layer.first]->getTH1F();

    for (int i = currentStart; i < (currentStart + layer.second); ++i) {
      // first obs.size() bins for observed movements
      int bin = i - currentStart + 1;
      histo_0->SetBinContent(bin, obs[i]);
      histo_0->SetBinError(bin, obsErr[i]);
    }
    currentStart += layer.second;
  }
}

bool MillePedeDQMModule ::setupChanged(const edm::EventSetup& setup) {
  bool changed{false};

  if (watchIdealGeometryRcd_.check(setup))
    changed = true;
  if (watchTrackerTopologyRcd_.check(setup))
    changed = true;
  if (watchPTrackerParametersRcd_.check(setup))
    changed = true;

  return changed;
}

int MillePedeDQMModule ::getIndexFromString(const std::string& alignableId) {
  if (alignableId == "TPBHalfBarrelXminus") {
    return 3;
  } else if (alignableId == "TPBHalfBarrelXplus") {
    return 2;
  } else if (alignableId == "TPEHalfCylinderXminusZminus") {
    return 1;
  } else if (alignableId == "TPEHalfCylinderXplusZminus") {
    return 0;
  } else if (alignableId == "TPEHalfCylinderXminusZplus") {
    return 5;
  } else if (alignableId == "TPEHalfCylinderXplusZplus") {
    return 4;
  } else if (alignableId.rfind("TPBLadder", 0) == 0) {
    return 6;
  } else if (alignableId.rfind("TPEPanel", 0) == 0) {
    return 7;
  } else {
    throw cms::Exception("LogicError") << "@SUB=MillePedeDQMModule::getIndexFromString\n"
                                       << "Retrieving conversion for not supported Alignable partition" << alignableId;
  }
}
