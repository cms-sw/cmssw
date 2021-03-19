#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/Definitions.h"
#include "CLHEP/Vector/RotationInterfaces.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformationFactory.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackerGeometryCompare.h"
#include "TFile.h"
#include "CLHEP/Vector/ThreeVector.h"

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <iostream>
#include <fstream>
#include <sstream>

TrackerGeometryCompare::TrackerGeometryCompare(const edm::ParameterSet& cfg)
    : cpvTokenDDD_(esConsumes()),
      cpvTokenDD4Hep_(esConsumes()),
      topoToken_(esConsumes()),
      geomDetToken_(esConsumes()),
      ptpToken_(esConsumes()),
      pixQualityToken_(esConsumes()),
      stripQualityToken_(esConsumes()),
      referenceTracker(nullptr),
      dummyTracker(nullptr),
      currentTracker(nullptr),
      theSurveyIndex(0),
      theSurveyValues(nullptr),
      theSurveyErrors(nullptr),
      levelStrings_(cfg.getUntrackedParameter<std::vector<std::string> >("levels")),
      fromDD4hep_(cfg.getUntrackedParameter<bool>("fromDD4hep")),
      writeToDB_(cfg.getUntrackedParameter<bool>("writeToDB")),
      commonTrackerLevel_(align::invalid),
      moduleListFile_(nullptr),
      moduleList_(0),
      inputRootFile1_(nullptr),
      inputRootFile2_(nullptr),
      inputTree01_(nullptr),
      inputTree02_(nullptr),
      inputTree11_(nullptr),
      inputTree12_(nullptr),
      m_nBins_(10000),
      m_rangeLow_(-.1),
      m_rangeHigh_(.1),
      firstEvent_(true),
      m_vtkmap_(13) {
  moduleListName_ = cfg.getUntrackedParameter<std::string>("moduleList");

  //input is ROOT
  inputFilename1_ = cfg.getUntrackedParameter<std::string>("inputROOTFile1");
  inputFilename2_ = cfg.getUntrackedParameter<std::string>("inputROOTFile2");
  inputTreenameAlign_ = cfg.getUntrackedParameter<std::string>("treeNameAlign");
  inputTreenameDeform_ = cfg.getUntrackedParameter<std::string>("treeNameDeform");

  //output file
  filename_ = cfg.getUntrackedParameter<std::string>("outputFile");

  weightBy_ = cfg.getUntrackedParameter<std::string>("weightBy");
  setCommonTrackerSystem_ = cfg.getUntrackedParameter<std::string>("setCommonTrackerSystem");
  detIdFlag_ = cfg.getUntrackedParameter<bool>("detIdFlag");
  detIdFlagFile_ = cfg.getUntrackedParameter<std::string>("detIdFlagFile");
  weightById_ = cfg.getUntrackedParameter<bool>("weightById");
  weightByIdFile_ = cfg.getUntrackedParameter<std::string>("weightByIdFile");

  // if want to use, make id cut list
  if (detIdFlag_) {
    std::ifstream fin;
    fin.open(detIdFlagFile_.c_str());

    while (!fin.eof() && fin.good()) {
      uint32_t id;
      fin >> id;
      detIdFlagVector_.push_back(id);
    }
    fin.close();
  }

  // turn weightByIdFile into weightByIdVector
  if (weightById_) {
    std::ifstream inFile;
    inFile.open(weightByIdFile_.c_str());
    int ctr = 0;
    while (!inFile.eof()) {
      ctr++;
      unsigned int listId;
      inFile >> listId;
      inFile.ignore(256, '\n');

      weightByIdVector_.push_back(listId);
    }
    inFile.close();
  }

  //root configuration
  theFile_ = new TFile(filename_.c_str(), "RECREATE");
  alignTree_ = new TTree("alignTree",
                         "alignTree");  //,"id:level:mid:mlevel:sublevel:x:y:z:r:phi:a:b:c:dx:dy:dz:dr:dphi:da:db:dc");
  alignTree_->Branch("id", &id_, "id/I");
  alignTree_->Branch("badModuleQuality", &badModuleQuality_, "badModuleQuality/I");
  alignTree_->Branch("inModuleList", &inModuleList_, "inModuleList/I");
  alignTree_->Branch("level", &level_, "level/I");
  alignTree_->Branch("mid", &mid_, "mid/I");
  alignTree_->Branch("mlevel", &mlevel_, "mlevel/I");
  alignTree_->Branch("sublevel", &sublevel_, "sublevel/I");
  alignTree_->Branch("x", &xVal_, "x/F");
  alignTree_->Branch("y", &yVal_, "y/F");
  alignTree_->Branch("z", &zVal_, "z/F");
  alignTree_->Branch("r", &rVal_, "r/F");
  alignTree_->Branch("phi", &phiVal_, "phi/F");
  alignTree_->Branch("eta", &etaVal_, "eta/F");
  alignTree_->Branch("alpha", &alphaVal_, "alpha/F");
  alignTree_->Branch("beta", &betaVal_, "beta/F");
  alignTree_->Branch("gamma", &gammaVal_, "gamma/F");
  alignTree_->Branch("dx", &dxVal_, "dx/F");
  alignTree_->Branch("dy", &dyVal_, "dy/F");
  alignTree_->Branch("dz", &dzVal_, "dz/F");
  alignTree_->Branch("dr", &drVal_, "dr/F");
  alignTree_->Branch("dphi", &dphiVal_, "dphi/F");
  alignTree_->Branch("dalpha", &dalphaVal_, "dalpha/F");
  alignTree_->Branch("dbeta", &dbetaVal_, "dbeta/F");
  alignTree_->Branch("dgamma", &dgammaVal_, "dgamma/F");
  alignTree_->Branch("du", &duVal_, "du/F");
  alignTree_->Branch("dv", &dvVal_, "dv/F");
  alignTree_->Branch("dw", &dwVal_, "dw/F");
  alignTree_->Branch("da", &daVal_, "da/F");
  alignTree_->Branch("db", &dbVal_, "db/F");
  alignTree_->Branch("dg", &dgVal_, "dg/F");
  alignTree_->Branch("useDetId", &useDetId_, "useDetId/I");
  alignTree_->Branch("detDim", &detDim_, "detDim/I");
  alignTree_->Branch("surW", &surWidth_, "surW/F");
  alignTree_->Branch("surL", &surLength_, "surL/F");
  alignTree_->Branch("surRot", &surRot_, "surRot[9]/D");
  alignTree_->Branch("identifiers", &identifiers_, "identifiers[6]/I");
  alignTree_->Branch("type", &type_, "type/I");
  alignTree_->Branch("surfDeform", &surfDeform_, "surfDeform[13]/D");

  for (std::vector<TrackerMap>::iterator it = m_vtkmap_.begin(); it != m_vtkmap_.end(); ++it) {
    it->setPalette(1);
    it->addPixel(true);
  }

  edm::Service<TFileService> fs;
  TFileDirectory subDir_All = fs->mkdir("AllSubdetectors");
  TFileDirectory subDir_PXB = fs->mkdir("PixelBarrel");
  TFileDirectory subDir_PXF = fs->mkdir("PixelEndcap");
  for (int ii = 0; ii < 13; ++ii) {
    std::stringstream histname0;
    histname0 << "SurfDeform_Par_" << ii;
    m_h1_[histname0.str()] = subDir_All.make<TH1D>(
        (histname0.str()).c_str(), (histname0.str()).c_str(), m_nBins_, m_rangeLow_, m_rangeHigh_);

    std::stringstream histname1;
    histname1 << "SurfDeform_PixelBarrel_Par_" << ii;
    m_h1_[histname1.str()] = subDir_PXB.make<TH1D>(
        (histname1.str()).c_str(), (histname1.str()).c_str(), m_nBins_, m_rangeLow_, m_rangeHigh_);

    std::stringstream histname2;
    histname2 << "SurfDeform_PixelEndcap_Par_" << ii;
    m_h1_[histname2.str()] = subDir_PXF.make<TH1D>(
        (histname2.str()).c_str(), (histname2.str()).c_str(), m_nBins_, m_rangeLow_, m_rangeHigh_);
  }
}

void TrackerGeometryCompare::beginJob() { firstEvent_ = true; }

void TrackerGeometryCompare::endJob() {
  int iname(0);
  for (std::vector<TrackerMap>::iterator it = m_vtkmap_.begin(); it != m_vtkmap_.end(); ++it) {
    std::stringstream mapname;
    mapname << "TkMap_SurfDeform" << iname << ".png";
    it->save(true, 0, 0, mapname.str());
    mapname.str(std::string());
    mapname.clear();
    mapname << "TkMap_SurfDeform" << iname << ".pdf";
    it->save(true, 0, 0, mapname.str());
    ++iname;
  }

  theFile_->cd();
  alignTree_->Write();
  theFile_->Close();
}

void TrackerGeometryCompare::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  if (firstEvent_) {
    //Retrieve tracker topology from geometry
    const TrackerTopology* const tTopo = &iSetup.getData(topoToken_);

    //upload the ROOT geometries
    createROOTGeometry(iSetup);

    //setting the levels being used in the geometry comparator
    edm::LogInfo("TrackerGeometryCompare") << "levels: " << levelStrings_.size();
    for (const auto& level : levelStrings_) {
      m_theLevels.push_back(currentTracker->objectIdProvider().stringToId(level));
      edm::LogInfo("TrackerGeometryCompare") << "level: " << level;
      edm::LogInfo("TrackerGeometryCompare")
          << "structure type: " << currentTracker->objectIdProvider().stringToId(level);
    }

    //set common tracker system first
    // if setting the tracker common system
    if (setCommonTrackerSystem_ != "NONE") {
      setCommonTrackerSystem();
    }

    //compare the goemetries
    compareGeometries(referenceTracker, currentTracker, tTopo, iSetup);
    compareSurfaceDeformations(inputTree11_, inputTree12_);

    //write out ntuple
    //might be better to do within output module

    if (writeToDB_) {
      Alignments* myAlignments = currentTracker->alignments();
      AlignmentErrorsExtended* myAlignmentErrorsExtended = currentTracker->alignmentErrors();

      // 2. Store alignment[Error]s to DB
      edm::Service<cond::service::PoolDBOutputService> poolDbService;
      // Call service
      if (!poolDbService.isAvailable())  // Die if not available
        throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

      poolDbService->writeOne<Alignments>(&(*myAlignments), poolDbService->beginOfTime(), "TrackerAlignmentRcd");
      poolDbService->writeOne<AlignmentErrorsExtended>(
          &(*myAlignmentErrorsExtended), poolDbService->beginOfTime(), "TrackerAlignmentErrorExtendedRcd");
    }

    firstEvent_ = false;
  }
}

void TrackerGeometryCompare::createROOTGeometry(const edm::EventSetup& iSetup) {
  int inputRawId1, inputRawId2;
  double inputX1, inputY1, inputZ1, inputX2, inputY2, inputZ2;
  double inputAlpha1, inputBeta1, inputGamma1, inputAlpha2, inputBeta2, inputGamma2;

  //Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(topoToken_);

  // Fill module IDs from file into a list
  moduleListFile_.open(moduleListName_);
  if (moduleListFile_.is_open()) {
    std::string line;
    while (!moduleListFile_.eof()) {
      std::getline(moduleListFile_, line);
      moduleList_.push_back(std::atoi(line.c_str()));
    }
  } else {
    edm::LogInfo("TrackerGeometryCompare") << "Error: Module list not found! Please verify that given list exists!";
  }

  //declare alignments
  Alignments* alignments1 = new Alignments();
  AlignmentErrorsExtended* alignmentErrors1 = new AlignmentErrorsExtended();
  if (inputFilename1_ != "IDEAL") {
    inputRootFile1_ = new TFile(inputFilename1_.c_str());
    TTree* inputTree01_ = (TTree*)inputRootFile1_->Get(inputTreenameAlign_.c_str());
    inputTree01_->SetBranchAddress("rawid", &inputRawId1);
    inputTree01_->SetBranchAddress("x", &inputX1);
    inputTree01_->SetBranchAddress("y", &inputY1);
    inputTree01_->SetBranchAddress("z", &inputZ1);
    inputTree01_->SetBranchAddress("alpha", &inputAlpha1);
    inputTree01_->SetBranchAddress("beta", &inputBeta1);
    inputTree01_->SetBranchAddress("gamma", &inputGamma1);

    int nEntries1 = inputTree01_->GetEntries();
    //fill alignments
    for (int i = 0; i < nEntries1; ++i) {
      inputTree01_->GetEntry(i);
      CLHEP::Hep3Vector translation1(inputX1, inputY1, inputZ1);
      CLHEP::HepEulerAngles eulerangles1(inputAlpha1, inputBeta1, inputGamma1);
      uint32_t detid1 = inputRawId1;
      AlignTransform transform1(translation1, eulerangles1, detid1);
      alignments1->m_align.push_back(transform1);

      //dummy errors
      CLHEP::HepSymMatrix clhepSymMatrix(3, 0);
      AlignTransformErrorExtended transformError(clhepSymMatrix, detid1);
      alignmentErrors1->m_alignError.push_back(transformError);
    }

    // to get the right order, sort by rawId
    std::sort(alignments1->m_align.begin(), alignments1->m_align.end());
    std::sort(alignmentErrors1->m_alignError.begin(), alignmentErrors1->m_alignError.end());
  }
  //------------------
  Alignments* alignments2 = new Alignments();
  AlignmentErrorsExtended* alignmentErrors2 = new AlignmentErrorsExtended();
  if (inputFilename2_ != "IDEAL") {
    inputRootFile2_ = new TFile(inputFilename2_.c_str());
    TTree* inputTree02_ = (TTree*)inputRootFile2_->Get(inputTreenameAlign_.c_str());
    inputTree02_->SetBranchAddress("rawid", &inputRawId2);
    inputTree02_->SetBranchAddress("x", &inputX2);
    inputTree02_->SetBranchAddress("y", &inputY2);
    inputTree02_->SetBranchAddress("z", &inputZ2);
    inputTree02_->SetBranchAddress("alpha", &inputAlpha2);
    inputTree02_->SetBranchAddress("beta", &inputBeta2);
    inputTree02_->SetBranchAddress("gamma", &inputGamma2);

    int nEntries2 = inputTree02_->GetEntries();
    //fill alignments
    for (int i = 0; i < nEntries2; ++i) {
      inputTree02_->GetEntry(i);
      CLHEP::Hep3Vector translation2(inputX2, inputY2, inputZ2);
      CLHEP::HepEulerAngles eulerangles2(inputAlpha2, inputBeta2, inputGamma2);
      uint32_t detid2 = inputRawId2;
      AlignTransform transform2(translation2, eulerangles2, detid2);
      alignments2->m_align.push_back(transform2);

      //dummy errors
      CLHEP::HepSymMatrix clhepSymMatrix(3, 0);
      AlignTransformErrorExtended transformError(clhepSymMatrix, detid2);
      alignmentErrors2->m_alignError.push_back(transformError);
    }

    // to get the right order, sort by rawId
    std::sort(alignments2->m_align.begin(), alignments2->m_align.end());
    std::sort(alignmentErrors2->m_alignError.begin(), alignmentErrors2->m_alignError.end());
  }

  //accessing the initial geometry
  if (!fromDD4hep_) {
    edm::ESTransientHandle<DDCompactView> cpv = iSetup.getTransientHandle(cpvTokenDDD_);
  } else {
    edm::ESTransientHandle<cms::DDCompactView> cpv = iSetup.getTransientHandle(cpvTokenDD4Hep_);
  }

  const GeometricDet* theGeometricDet = &iSetup.getData(geomDetToken_);
  const PTrackerParameters* ptp = &iSetup.getData(ptpToken_);
  TrackerGeomBuilderFromGeometricDet trackerBuilder;

  //reference tracker
  TrackerGeometry* theRefTracker = trackerBuilder.build(theGeometricDet, *ptp, tTopo);
  if (inputFilename1_ != "IDEAL") {
    GeometryAligner aligner1;
    aligner1.applyAlignments<TrackerGeometry>(
        &(*theRefTracker), &(*alignments1), &(*alignmentErrors1), AlignTransform());
  }
  referenceTracker = new AlignableTracker(&(*theRefTracker), tTopo);
  //referenceTracker->setSurfaceDeformation(surfDef1, true) ;

  int inputRawid1;
  int inputRawid2;
  int inputDtype1, inputDtype2;
  std::vector<double> inputDpar1;
  std::vector<double> inputDpar2;
  std::vector<double>* p_inputDpar1 = &inputDpar1;
  std::vector<double>* p_inputDpar2 = &inputDpar2;

  const auto& comp1 = referenceTracker->deepComponents();

  SurfaceDeformation* surfDef1;
  if (inputFilename1_ != "IDEAL") {
    TTree* inputTree11_ = (TTree*)inputRootFile1_->Get(inputTreenameDeform_.c_str());
    inputTree11_->SetBranchAddress("irawid", &inputRawid1);
    inputTree11_->SetBranchAddress("dtype", &inputDtype1);
    inputTree11_->SetBranchAddress("dpar", &p_inputDpar1);

    unsigned int nEntries11 = inputTree11_->GetEntries();
    edm::LogInfo("TrackerGeometryCompare") << " nentries11 = " << nEntries11;
    for (unsigned int iEntry = 0; iEntry < nEntries11; ++iEntry) {
      inputTree11_->GetEntry(iEntry);

      surfDef1 = SurfaceDeformationFactory::create(inputDtype1, inputDpar1);

      if (int(comp1[iEntry]->id()) == inputRawid1) {
        comp1[iEntry]->setSurfaceDeformation(surfDef1, true);
      }
    }
  }

  //currernt tracker
  TrackerGeometry* theCurTracker = trackerBuilder.build(&*theGeometricDet, *ptp, tTopo);
  if (inputFilename2_ != "IDEAL") {
    GeometryAligner aligner2;
    aligner2.applyAlignments<TrackerGeometry>(
        &(*theCurTracker), &(*alignments2), &(*alignmentErrors2), AlignTransform());
  }
  currentTracker = new AlignableTracker(&(*theCurTracker), tTopo);

  const auto& comp2 = currentTracker->deepComponents();

  SurfaceDeformation* surfDef2;
  if (inputFilename2_ != "IDEAL") {
    TTree* inputTree12_ = (TTree*)inputRootFile2_->Get(inputTreenameDeform_.c_str());
    inputTree12_->SetBranchAddress("irawid", &inputRawid2);
    inputTree12_->SetBranchAddress("dtype", &inputDtype2);
    inputTree12_->SetBranchAddress("dpar", &p_inputDpar2);

    unsigned int nEntries12 = inputTree12_->GetEntries();
    edm::LogInfo("TrackerGeometryCompare") << " nentries12 = " << nEntries12;
    for (unsigned int iEntry = 0; iEntry < nEntries12; ++iEntry) {
      inputTree12_->GetEntry(iEntry);

      surfDef2 = SurfaceDeformationFactory::create(inputDtype2, inputDpar2);

      if (int(comp2[iEntry]->id()) == inputRawid2) {
        comp2[iEntry]->setSurfaceDeformation(surfDef2, true);
      }
    }
  }

  delete alignments1;
  delete alignmentErrors1;
  delete alignments2;
  delete alignmentErrors2;
}

void TrackerGeometryCompare::compareSurfaceDeformations(TTree* refTree, TTree* curTree) {
  if (inputFilename1_ != "IDEAL" && inputFilename2_ != "IDEAL") {
    int inputRawid1;
    int inputRawid2;
    int inputSubdetid1, inputSubdetid2;
    int inputDtype1, inputDtype2;
    std::vector<double> inputDpar1;
    std::vector<double> inputDpar2;
    std::vector<double>* p_inputDpar1 = &inputDpar1;
    std::vector<double>* p_inputDpar2 = &inputDpar2;

    TTree* refTree = (TTree*)inputRootFile1_->Get(inputTreenameDeform_.c_str());
    refTree->SetBranchAddress("irawid", &inputRawid1);
    refTree->SetBranchAddress("subdetid", &inputSubdetid1);
    refTree->SetBranchAddress("dtype", &inputDtype1);
    refTree->SetBranchAddress("dpar", &p_inputDpar1);

    TTree* curTree = (TTree*)inputRootFile2_->Get(inputTreenameDeform_.c_str());
    curTree->SetBranchAddress("irawid", &inputRawid2);
    curTree->SetBranchAddress("subdetid", &inputSubdetid2);
    curTree->SetBranchAddress("dtype", &inputDtype2);
    curTree->SetBranchAddress("dpar", &p_inputDpar2);

    unsigned int nEntries11 = refTree->GetEntries();
    unsigned int nEntries12 = curTree->GetEntries();

    if (nEntries11 != nEntries12) {
      edm::LogError("TrackerGeometryCompare") << " Surface deformation parameters in two geometries differ!\n";
      return;
    }

    for (unsigned int iEntry = 0; iEntry < nEntries12; ++iEntry) {
      refTree->GetEntry(iEntry);
      curTree->GetEntry(iEntry);
      for (int ii = 0; ii < 13; ++ii) {
        surfDeform_[ii] = -1.0;
      }
      for (int npar = 0; npar < int(inputDpar2.size()); ++npar) {
        if (inputRawid1 == inputRawid2) {
          surfDeform_[npar] = inputDpar2.at(npar) - inputDpar1.at(npar);
          std::stringstream histname0;
          histname0 << "SurfDeform_Par_" << npar;
          if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
            m_h1_[histname0.str()]->Fill(surfDeform_[npar]);
          if (inputSubdetid1 == 1 && inputSubdetid2 == 1) {
            std::stringstream histname1;
            histname1 << "SurfDeform_PixelBarrel_Par_" << npar;
            if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
              m_h1_[histname1.str()]->Fill(surfDeform_[npar]);
          }
          if (inputSubdetid1 == 2 && inputSubdetid2 == 2) {
            std::stringstream histname2;
            histname2 << "SurfDeform_PixelEndcap_Par_" << npar;
            if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
              m_h1_[histname2.str()]->Fill(surfDeform_[npar]);
          }
          (m_vtkmap_.at(npar)).fill_current_val(inputRawid1, surfDeform_[npar]);
        }
      }
    }

  } else if (inputFilename1_ == "IDEAL" && inputFilename2_ != "IDEAL") {
    int inputRawid2;
    int inputSubdetid2;
    int inputDtype2;
    std::vector<double> inputDpar2;
    std::vector<double>* p_inputDpar2 = &inputDpar2;

    TTree* curTree = (TTree*)inputRootFile2_->Get(inputTreenameDeform_.c_str());
    curTree->SetBranchAddress("irawid", &inputRawid2);
    curTree->SetBranchAddress("subdetid", &inputSubdetid2);
    curTree->SetBranchAddress("dtype", &inputDtype2);
    curTree->SetBranchAddress("dpar", &p_inputDpar2);

    unsigned int nEntries12 = curTree->GetEntries();

    for (unsigned int iEntry = 0; iEntry < nEntries12; ++iEntry) {
      curTree->GetEntry(iEntry);
      for (int ii = 0; ii < 12; ++ii) {
        surfDeform_[ii] = -1.0;
      }
      for (int npar = 0; npar < int(inputDpar2.size()); ++npar) {
        surfDeform_[npar] = inputDpar2.at(npar);
        std::stringstream histname0;
        histname0 << "SurfDeform_Par_" << npar;
        if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
          m_h1_[histname0.str()]->Fill(surfDeform_[npar]);
        if (inputSubdetid2 == 1) {
          std::stringstream histname1;
          histname1 << "SurfDeform_PixelBarrel_Par_" << npar;
          if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
            m_h1_[histname1.str()]->Fill(surfDeform_[npar]);
        }
        if (inputSubdetid2 == 2) {
          std::stringstream histname2;
          histname2 << "SurfDeform_PixelEndcap_Par_" << npar;
          if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
            m_h1_[histname2.str()]->Fill(surfDeform_[npar]);
        }
        (m_vtkmap_.at(npar)).fill_current_val(inputRawid2, surfDeform_[npar]);
      }
    }

  } else if (inputFilename1_ != "IDEAL" && inputFilename2_ == "IDEAL") {
    int inputRawid1;
    int inputSubdetid1;
    int inputDtype1;
    std::vector<double> inputDpar1;
    std::vector<double>* p_inputDpar1 = &inputDpar1;

    TTree* refTree = (TTree*)inputRootFile1_->Get(inputTreenameDeform_.c_str());
    refTree->SetBranchAddress("irawid", &inputRawid1);
    refTree->SetBranchAddress("subdetid", &inputSubdetid1);
    refTree->SetBranchAddress("dtype", &inputDtype1);
    refTree->SetBranchAddress("dpar", &p_inputDpar1);

    unsigned int nEntries11 = refTree->GetEntries();

    for (unsigned int iEntry = 0; iEntry < nEntries11; ++iEntry) {
      refTree->GetEntry(iEntry);
      for (int ii = 0; ii < 12; ++ii) {
        surfDeform_[ii] = -1.0;
      }
      for (int npar = 0; npar < int(inputDpar1.size()); ++npar) {
        surfDeform_[npar] = -inputDpar1.at(npar);
        std::stringstream histname0;
        histname0 << "SurfDeform_Par_" << npar;
        if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
          m_h1_[histname0.str()]->Fill(surfDeform_[npar]);
        if (inputSubdetid1 == 1) {
          std::stringstream histname1;
          histname1 << "SurfDeform_PixelBarrel_Par_" << npar;
          if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
            m_h1_[histname1.str()]->Fill(surfDeform_[npar]);
        }
        if (inputSubdetid1 == 2) {
          std::stringstream histname2;
          histname2 << "SurfDeform_PixelEndcap_Par_" << npar;
          if (TMath::Abs(surfDeform_[npar]) > (m_rangeHigh_ - m_rangeLow_) / (10. * m_nBins_))
            m_h1_[histname2.str()]->Fill(surfDeform_[npar]);
        }
        (m_vtkmap_.at(npar)).fill_current_val(inputRawid1, surfDeform_[npar]);
      }
    }

  } else if (inputFilename1_ == "IDEAL" && inputFilename2_ == "IDEAL") {
    edm::LogInfo("TrackerGeometryCompare") << ">>>> Comparing IDEAL with IDEAL: nothing to do! <<<<\n";
  }

  return;
}

void TrackerGeometryCompare::compareGeometries(Alignable* refAli,
                                               Alignable* curAli,
                                               const TrackerTopology* tTopo,
                                               const edm::EventSetup& iSetup) {
  using namespace align;

  const auto& refComp = refAli->components();
  const auto& curComp = curAli->components();

  unsigned int nComp = refComp.size();
  //only perform for designate levels
  bool useLevel = false;
  for (unsigned int i = 0; i < m_theLevels.size(); ++i) {
    if (refAli->alignableObjectId() == m_theLevels[i])
      useLevel = true;
  }

  //another added level for difference between det and detunit
  //if ((refAli->alignableObjectId()==2)&&(nComp == 1)) useLevel = false;

  //coordinate matching, etc etc
  if (useLevel) {
    DetId detid(refAli->id());

    CLHEP::Hep3Vector Rtotal, Wtotal, lRtotal, lWtotal;
    Rtotal.set(0., 0., 0.);
    Wtotal.set(0., 0., 0.);
    lRtotal.set(0., 0., 0.);
    lWtotal.set(0., 0., 0.);

    bool converged = false;

    AlgebraicVector diff, check;

    for (int i = 0; i < 100; i++) {
      // Get differences between alignments for rotations and translations
      // both local and global
      diff = align::diffAlignables(refAli, curAli, weightBy_, weightById_, weightByIdVector_);

      // 'diffAlignables' returns 'refAli - curAli' for translations and 'curAli - refAli' for rotations.
      // The plan is to unify this at some point, but a simple change of the sign for one of them was postponed
      // to do some further checks to understand the rotations better
      //Updated July 2018: as requested the sign in the translations has been changed to match the one in rotations. A test was done to change the diffAlignables function and solve the issue there, but proved quite time consuming. To unify the sign convention in the least amount of time the choice was made to change the sign here.
      CLHEP::Hep3Vector dR(-diff[0], -diff[1], -diff[2]);
      CLHEP::Hep3Vector dW(diff[3], diff[4], diff[5]);
      CLHEP::Hep3Vector dRLocal(-diff[6], -diff[7], -diff[8]);
      CLHEP::Hep3Vector dWLocal(diff[9], diff[10], diff[11]);

      // Translations
      Rtotal += dR;
      lRtotal += dRLocal;

      //Rotations
      CLHEP::HepRotation rot(Wtotal.unit(), Wtotal.mag());
      CLHEP::HepRotation drot(dW.unit(), dW.mag());
      rot *= drot;
      Wtotal.set(rot.axis().x() * rot.delta(), rot.axis().y() * rot.delta(), rot.axis().z() * rot.delta());

      CLHEP::HepRotation rotLocal(lWtotal.unit(), lWtotal.mag());
      CLHEP::HepRotation drotLocal(dWLocal.unit(), dWLocal.mag());
      rotLocal *= drotLocal;
      lWtotal.set(rotLocal.axis().x() * rotLocal.delta(),
                  rotLocal.axis().y() * rotLocal.delta(),
                  rotLocal.axis().z() * rotLocal.delta());

      // Move current alignable by shift and check if difference
      // is smaller than tolerance value
      // if true, break the loop
      align::moveAlignable(curAli, diff);
      float tolerance = 1e-7;
      check = align::diffAlignables(refAli, curAli, weightBy_, weightById_, weightByIdVector_);
      align::GlobalVector checkR(check[0], check[1], check[2]);
      align::GlobalVector checkW(check[3], check[4], check[5]);
      if ((checkR.mag() < tolerance) && (checkW.mag() < tolerance)) {
        converged = true;
        break;
      }
    }

    // give an exception if difference has not fallen below tolerance level
    // i.e. method has not converged
    if (!converged) {
      edm::LogInfo("TrackerGeometryCompare")
          << "Tolerance Exceeded!(alObjId: " << refAli->alignableObjectId()
          << ", rawId: " << refAli->geomDetId().rawId() << ", subdetId: " << detid.subdetId() << "): " << diff << check;
      throw cms::Exception("Tolerance in TrackerGeometryCompare exceeded");
    }

    AlgebraicVector TRtot(12);
    // global
    TRtot(1) = Rtotal.x();
    TRtot(2) = Rtotal.y();
    TRtot(3) = Rtotal.z();
    TRtot(4) = Wtotal.x();
    TRtot(5) = Wtotal.y();
    TRtot(6) = Wtotal.z();
    // local
    TRtot(7) = lRtotal.x();
    TRtot(8) = lRtotal.y();
    TRtot(9) = lRtotal.z();
    TRtot(10) = lWtotal.x();
    TRtot(11) = lWtotal.y();
    TRtot(12) = lWtotal.z();

    fillTree(refAli, TRtot, tTopo, iSetup);
  }

  // another added level for difference between det and detunit
  for (unsigned int i = 0; i < nComp; ++i)
    compareGeometries(refComp[i], curComp[i], tTopo, iSetup);
}

void TrackerGeometryCompare::setCommonTrackerSystem() {
  edm::LogInfo("TrackerGeometryCompare") << "Setting Common Tracker System....";

  // DM_534??AlignableObjectId dummy;
  // DM_534??_commonTrackerLevel = dummy.nameToType(_setCommonTrackerSystem);
  commonTrackerLevel_ = currentTracker->objectIdProvider().stringToId(setCommonTrackerSystem_);

  diffCommonTrackerSystem(referenceTracker, currentTracker);

  align::EulerAngles dOmega(3);
  dOmega[0] = TrackerCommonR_.x();
  dOmega[1] = TrackerCommonR_.y();
  dOmega[2] = TrackerCommonR_.z();
  align::RotationType rot = align::toMatrix(dOmega);
  align::GlobalVector theR = TrackerCommonT_;

  edm::LogInfo("TrackerGeometryCompare") << "what we get from overlaying the pixels..." << theR << ", " << rot;

  //transform to the Tracker System
  align::PositionType trackerCM = currentTracker->globalPosition();
  align::GlobalVector cmDiff(
      trackerCM.x() - TrackerCommonCM_.x(), trackerCM.y() - TrackerCommonCM_.y(), trackerCM.z() - TrackerCommonCM_.z());

  edm::LogInfo("TrackerGeometryCompare") << "Pixel CM: " << TrackerCommonCM_ << ", tracker CM: " << trackerCM;

  //adjust translational difference factoring in different rotational CM
  //needed because rotateInGlobalFrame is about CM of alignable, not Tracker
  const align::GlobalVector::BasicVectorType& lpvgf = cmDiff.basicVector();
  align::GlobalVector moveV(rot.multiplyInverse(lpvgf) - lpvgf);
  align::GlobalVector theRprime(theR + moveV);

  AlgebraicVector TrackerCommonTR(6);
  TrackerCommonTR(1) = theRprime.x();
  TrackerCommonTR(2) = theRprime.y();
  TrackerCommonTR(3) = theRprime.z();
  TrackerCommonTR(4) = TrackerCommonR_.x();
  TrackerCommonTR(5) = TrackerCommonR_.y();
  TrackerCommonTR(6) = TrackerCommonR_.z();

  edm::LogInfo("TrackerGeometryCompare") << "and after the transformation: " << TrackerCommonTR;

  align::moveAlignable(currentTracker, TrackerCommonTR);
}

void TrackerGeometryCompare::diffCommonTrackerSystem(Alignable* refAli, Alignable* curAli) {
  const auto& refComp = refAli->components();
  const auto& curComp = curAli->components();

  unsigned int nComp = refComp.size();
  //only perform for designate levels
  bool useLevel = false;
  if (refAli->alignableObjectId() == commonTrackerLevel_)
    useLevel = true;

  //useLevel = false;
  if (useLevel) {
    CLHEP::Hep3Vector Rtotal, Wtotal;
    Rtotal.set(0., 0., 0.);
    Wtotal.set(0., 0., 0.);

    AlgebraicVector diff = align::diffAlignables(refAli, curAli, weightBy_, weightById_, weightByIdVector_);
    CLHEP::Hep3Vector dR(diff[0], diff[1], diff[2]);
    Rtotal += dR;
    CLHEP::Hep3Vector dW(diff[3], diff[4], diff[5]);
    CLHEP::HepRotation rot(Wtotal.unit(), Wtotal.mag());
    CLHEP::HepRotation drot(dW.unit(), dW.mag());
    rot *= drot;
    Wtotal.set(rot.axis().x() * rot.delta(), rot.axis().y() * rot.delta(), rot.axis().z() * rot.delta());

    TrackerCommonT_ = align::GlobalVector(Rtotal.x(), Rtotal.y(), Rtotal.z());
    TrackerCommonR_ = align::GlobalVector(Wtotal.x(), Wtotal.y(), Wtotal.z());
    TrackerCommonCM_ = curAli->globalPosition();

  } else {
    for (unsigned int i = 0; i < nComp; ++i)
      diffCommonTrackerSystem(refComp[i], curComp[i]);
  }
}

void TrackerGeometryCompare::fillTree(Alignable* refAli,
                                      const AlgebraicVector& diff,
                                      const TrackerTopology* tTopo,
                                      const edm::EventSetup& iSetup) {
  //Get bad modules
  const SiPixelQuality* SiPixelModules = &iSetup.getData(pixQualityToken_);
  const SiStripQuality* SiStripModules = &iSetup.getData(stripQualityToken_);

  id_ = refAli->id();

  badModuleQuality_ = 0;
  //check if module has a bad quality tag
  if (SiPixelModules->IsModuleBad(id_)) {
    badModuleQuality_ = 1;
  }
  if (SiStripModules->IsModuleBad(id_)) {
    badModuleQuality_ = 1;
  }

  //check if module is in a given list of bad/untouched etc. modules
  inModuleList_ = 0;
  for (unsigned int i = 0; i < moduleList_.size(); i++) {
    if (moduleList_[i] == id_) {
      inModuleList_ = 1;
      break;
    }
  }

  level_ = refAli->alignableObjectId();
  //need if ali has no mother
  if (refAli->mother()) {
    mid_ = refAli->mother()->geomDetId().rawId();
    mlevel_ = refAli->mother()->alignableObjectId();
  } else {
    mid_ = -1;
    mlevel_ = -1;
  }
  DetId detid(id_);
  sublevel_ = detid.subdetId();
  fillIdentifiers(sublevel_, id_, tTopo);
  xVal_ = refAli->globalPosition().x();
  yVal_ = refAli->globalPosition().y();
  zVal_ = refAli->globalPosition().z();
  align::GlobalVector vec(xVal_, yVal_, zVal_);
  rVal_ = vec.perp();
  phiVal_ = vec.phi();
  etaVal_ = vec.eta();
  align::RotationType rot = refAli->globalRotation();
  align::EulerAngles eulerAngles = align::toAngles(rot);
  alphaVal_ = eulerAngles[0];
  betaVal_ = eulerAngles[1];
  gammaVal_ = eulerAngles[2];
  // global
  dxVal_ = diff[0];
  dyVal_ = diff[1];
  dzVal_ = diff[2];
  // local
  duVal_ = diff[6];
  dvVal_ = diff[7];
  dwVal_ = diff[8];
  //...TODO...
  align::GlobalVector g(dxVal_, dyVal_, dzVal_);
  //getting dR and dPhi
  align::GlobalVector vRef(xVal_, yVal_, zVal_);
  align::GlobalVector vCur(xVal_ + dxVal_, yVal_ + dyVal_, zVal_ + dzVal_);
  drVal_ = vCur.perp() - vRef.perp();
  dphiVal_ = vCur.phi() - vRef.phi();
  // global
  dalphaVal_ = diff[3];
  dbetaVal_ = diff[4];
  dgammaVal_ = diff[5];
  // local
  daVal_ = diff[9];
  dbVal_ = diff[10];
  dgVal_ = diff[11];

  //detIdFlag
  if (refAli->alignableObjectId() == align::AlignableDetUnit) {
    if (detIdFlag_) {
      if ((passIdCut(refAli->id())) || (passIdCut(refAli->mother()->id()))) {
        useDetId_ = 1;
      } else {
        useDetId_ = 0;
      }
    }
  }
  // det module dimension
  if (refAli->alignableObjectId() == align::AlignableDetUnit) {
    if (refAli->mother()->alignableObjectId() != align::AlignableDet)
      detDim_ = 1;
    else if (refAli->mother()->alignableObjectId() == align::AlignableDet)
      detDim_ = 2;
  } else
    detDim_ = 0;

  surWidth_ = refAli->surface().width();
  surLength_ = refAli->surface().length();
  align::RotationType rt = refAli->globalRotation();
  surRot_[0] = rt.xx();
  surRot_[1] = rt.xy();
  surRot_[2] = rt.xz();
  surRot_[3] = rt.yx();
  surRot_[4] = rt.yy();
  surRot_[5] = rt.yz();
  surRot_[6] = rt.zx();
  surRot_[7] = rt.zy();
  surRot_[8] = rt.zz();

  //Fill
  alignTree_->Fill();
}

void TrackerGeometryCompare::surveyToTracker(AlignableTracker* ali,
                                             Alignments* alignVals,
                                             AlignmentErrorsExtended* alignErrors) {
  //getting the right alignables for the alignment record
  auto detPB = ali->pixelHalfBarrelGeomDets();
  auto detPEC = ali->pixelEndcapGeomDets();
  auto detTIB = ali->innerBarrelGeomDets();
  auto detTID = ali->TIDGeomDets();
  auto detTOB = ali->outerBarrelGeomDets();
  auto detTEC = ali->endcapGeomDets();

  align::Alignables allGeomDets;
  std::copy(detPB.begin(), detPB.end(), std::back_inserter(allGeomDets));
  std::copy(detPEC.begin(), detPEC.end(), std::back_inserter(allGeomDets));
  std::copy(detTIB.begin(), detTIB.end(), std::back_inserter(allGeomDets));
  std::copy(detTID.begin(), detTID.end(), std::back_inserter(allGeomDets));
  std::copy(detTOB.begin(), detTOB.end(), std::back_inserter(allGeomDets));
  std::copy(detTEC.begin(), detTEC.end(), std::back_inserter(allGeomDets));

  align::Alignables rcdAlis;
  for (const auto& i : allGeomDets) {
    if (i->components().size() == 1) {
      rcdAlis.push_back(i);
    } else if (i->components().size() > 1) {
      rcdAlis.push_back(i);
      const auto& comp = i->components();
      for (const auto& j : comp)
        rcdAlis.push_back(j);
    }
  }

  //turning them into alignments
  for (const auto& k : rcdAlis) {
    const SurveyDet* surveyInfo = k->survey();
    const align::PositionType& pos(surveyInfo->position());
    align::RotationType rot(surveyInfo->rotation());
    CLHEP::Hep3Vector clhepVector(pos.x(), pos.y(), pos.z());
    CLHEP::HepRotation clhepRotation(
        CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz()));
    AlignTransform transform(clhepVector, clhepRotation, k->id());
    AlignTransformErrorExtended transformError(CLHEP::HepSymMatrix(3, 1), k->id());
    alignVals->m_align.push_back(transform);
    alignErrors->m_alignError.push_back(transformError);
  }

  // to get the right order, sort by rawId
  std::sort(alignVals->m_align.begin(), alignVals->m_align.end());
  std::sort(alignErrors->m_alignError.begin(), alignErrors->m_alignError.end());
}

void TrackerGeometryCompare::addSurveyInfo(Alignable* ali) {
  const auto& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i)
    addSurveyInfo(comp[i]);

  const SurveyError& error = theSurveyErrors->m_surveyErrors[theSurveyIndex];

  if (ali->geomDetId().rawId() != error.rawId() || ali->alignableObjectId() != error.structureType()) {
    throw cms::Exception("DatabaseError") << "Error reading survey info from DB. Mismatched id!";
  }

  const CLHEP::Hep3Vector& pos = theSurveyValues->m_align[theSurveyIndex].translation();
  const CLHEP::HepRotation& rot = theSurveyValues->m_align[theSurveyIndex].rotation();

  AlignableSurface surf(
      align::PositionType(pos.x(), pos.y(), pos.z()),
      align::RotationType(rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz()));

  surf.setWidth(ali->surface().width());
  surf.setLength(ali->surface().length());

  ali->setSurvey(new SurveyDet(surf, error.matrix()));

  ++theSurveyIndex;
}

bool TrackerGeometryCompare::passIdCut(uint32_t id) {
  bool pass = false;
  int nEntries = detIdFlagVector_.size();

  for (int i = 0; i < nEntries; i++) {
    if (detIdFlagVector_[i] == id)
      pass = true;
  }

  return pass;
}

void TrackerGeometryCompare::fillIdentifiers(int subdetlevel, int rawid, const TrackerTopology* tTopo) {
  switch (subdetlevel) {
    case 1: {
      identifiers_[0] = tTopo->pxbModule(rawid);
      identifiers_[1] = tTopo->pxbLadder(rawid);
      identifiers_[2] = tTopo->pxbLayer(rawid);
      identifiers_[3] = 999;
      identifiers_[4] = 999;
      identifiers_[5] = 999;
      break;
    }
    case 2: {
      identifiers_[0] = tTopo->pxfModule(rawid);
      identifiers_[1] = tTopo->pxfPanel(rawid);
      identifiers_[2] = tTopo->pxfBlade(rawid);
      identifiers_[3] = tTopo->pxfDisk(rawid);
      identifiers_[4] = tTopo->pxfSide(rawid);
      identifiers_[5] = 999;
      break;
    }
    case 3: {
      identifiers_[0] = tTopo->tibModule(rawid);
      identifiers_[1] = tTopo->tibStringInfo(rawid)[0];
      identifiers_[2] = tTopo->tibStringInfo(rawid)[1];
      identifiers_[3] = tTopo->tibStringInfo(rawid)[2];
      identifiers_[4] = tTopo->tibLayer(rawid);
      identifiers_[5] = 999;
      break;
    }
    case 4: {
      identifiers_[0] = tTopo->tidModuleInfo(rawid)[0];
      identifiers_[1] = tTopo->tidModuleInfo(rawid)[1];
      identifiers_[2] = tTopo->tidRing(rawid);
      identifiers_[3] = tTopo->tidWheel(rawid);
      identifiers_[4] = tTopo->tidSide(rawid);
      identifiers_[5] = 999;
      break;
    }
    case 5: {
      identifiers_[0] = tTopo->tobModule(rawid);
      identifiers_[1] = tTopo->tobRodInfo(rawid)[0];
      identifiers_[2] = tTopo->tobRodInfo(rawid)[1];
      identifiers_[3] = tTopo->tobLayer(rawid);
      identifiers_[4] = 999;
      identifiers_[5] = 999;
      break;
    }
    case 6: {
      identifiers_[0] = tTopo->tecModule(rawid);
      identifiers_[1] = tTopo->tecRing(rawid);
      identifiers_[2] = tTopo->tecPetalInfo(rawid)[0];
      identifiers_[3] = tTopo->tecPetalInfo(rawid)[1];
      identifiers_[4] = tTopo->tecWheel(rawid);
      identifiers_[5] = tTopo->tecSide(rawid);
      break;
    }
    default: {
      edm::LogInfo("TrackerGeometryCompare") << "Error: bad subdetid!!";
      break;
    }
  }
}

DEFINE_FWK_MODULE(TrackerGeometryCompare);
