#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CLHEP/Vector/RotationInterfaces.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
// The following looks generic enough to use
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"

#include "Alignment/MuonAlignment/interface/MuonAlignmentInputXML.h"
#include "Alignment/MuonAlignment/interface/MuonAlignment.h"

#include "MuonGeometryArrange.h"
#include "TFile.h"
#include "TLatex.h"
#include "TArrow.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TH2F.h"
#include "CLHEP/Vector/ThreeVector.h"

// Database
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <fstream>

MuonGeometryArrange::MuonGeometryArrange(const edm::ParameterSet& cfg)
    : theSurveyIndex(0),
      _levelStrings(cfg.getUntrackedParameter<std::vector<std::string> >("levels")),
      _writeToDB(false),
      _commonMuonLevel(align::invalid),
      firstEvent_(true),
      idealInputLabel1("MuonGeometryArrangeLabel1"),
      idealInputLabel2("MuonGeometryArrangeLabel2"),
      idealInputLabel2a("MuonGeometryArrangeLabel2a"),
      geomIdeal("MuonGeometryArrangeGeomIdeal"),
      dtGeomToken1_(esConsumes(edm::ESInputTag("", idealInputLabel1))),
      cscGeomToken1_(esConsumes(edm::ESInputTag("", idealInputLabel1))),
      gemGeomToken1_(esConsumes(edm::ESInputTag("", idealInputLabel1))),
      dtGeomToken2_(esConsumes(edm::ESInputTag("", idealInputLabel2))),
      cscGeomToken2_(esConsumes(edm::ESInputTag("", idealInputLabel2))),
      gemGeomToken2_(esConsumes(edm::ESInputTag("", idealInputLabel2))),
      dtGeomToken3_(esConsumes(edm::ESInputTag("", idealInputLabel2a))),
      cscGeomToken3_(esConsumes(edm::ESInputTag("", idealInputLabel2a))),
      gemGeomToken3_(esConsumes(edm::ESInputTag("", idealInputLabel2a))),
      dtGeomIdealToken_(esConsumes(edm::ESInputTag("", geomIdeal))),
      cscGeomIdealToken_(esConsumes(edm::ESInputTag("", geomIdeal))),
      gemGeomIdealToken_(esConsumes(edm::ESInputTag("", geomIdeal))) {
  referenceMuon = nullptr;
  currentMuon = nullptr;
  // Input is XML
  _inputXMLCurrent = cfg.getUntrackedParameter<std::string>("inputXMLCurrent");
  _inputXMLReference = cfg.getUntrackedParameter<std::string>("inputXMLReference");

  //input is ROOT
  _inputFilename1 = cfg.getUntrackedParameter<std::string>("inputROOTFile1");
  _inputFilename2 = cfg.getUntrackedParameter<std::string>("inputROOTFile2");
  _inputTreename = cfg.getUntrackedParameter<std::string>("treeName");

  //output file
  _filename = cfg.getUntrackedParameter<std::string>("outputFile");

  _weightBy = cfg.getUntrackedParameter<std::string>("weightBy");
  _detIdFlag = cfg.getUntrackedParameter<bool>("detIdFlag");
  _detIdFlagFile = cfg.getUntrackedParameter<std::string>("detIdFlagFile");
  _weightById = cfg.getUntrackedParameter<bool>("weightById");
  _weightByIdFile = cfg.getUntrackedParameter<std::string>("weightByIdFile");
  _endcap = cfg.getUntrackedParameter<int>("endcapNumber");
  _station = cfg.getUntrackedParameter<int>("stationNumber");
  _ring = cfg.getUntrackedParameter<int>("ringNumber");

  // if want to use, make id cut list
  if (_detIdFlag) {
    std::ifstream fin;
    fin.open(_detIdFlagFile.c_str());

    while (!fin.eof() && fin.good()) {
      uint32_t id;
      fin >> id;
      _detIdFlagVector.push_back(id);
    }
    fin.close();
  }

  // turn weightByIdFile into weightByIdVector
  unsigned int lastID = 999999999;
  if (_weightById) {
    std::ifstream inFile;
    inFile.open(_weightByIdFile.c_str());
    while (!inFile.eof()) {
      unsigned int listId;
      inFile >> listId;
      inFile.ignore(256, '\n');
      if (listId != lastID) {
        _weightByIdVector.push_back(listId);
      }
      lastID = listId;
    }
    inFile.close();
  }

  //root configuration
  _theFile = new TFile(_filename.c_str(), "RECREATE");
  _alignTree = new TTree("alignTree", "alignTree");
  _alignTree->Branch("id", &_id, "id/I");
  _alignTree->Branch("level", &_level, "level/I");
  _alignTree->Branch("mid", &_mid, "mid/I");
  _alignTree->Branch("mlevel", &_mlevel, "mlevel/I");
  _alignTree->Branch("sublevel", &_sublevel, "sublevel/I");
  _alignTree->Branch("x", &_xVal, "x/F");
  _alignTree->Branch("y", &_yVal, "y/F");
  _alignTree->Branch("z", &_zVal, "z/F");
  _alignTree->Branch("r", &_rVal, "r/F");
  _alignTree->Branch("phi", &_phiVal, "phi/F");
  _alignTree->Branch("eta", &_etaVal, "eta/F");
  _alignTree->Branch("alpha", &_alphaVal, "alpha/F");
  _alignTree->Branch("beta", &_betaVal, "beta/F");
  _alignTree->Branch("gamma", &_gammaVal, "gamma/F");
  _alignTree->Branch("dx", &_dxVal, "dx/F");
  _alignTree->Branch("dy", &_dyVal, "dy/F");
  _alignTree->Branch("dz", &_dzVal, "dz/F");
  _alignTree->Branch("dr", &_drVal, "dr/F");
  _alignTree->Branch("dphi", &_dphiVal, "dphi/F");
  _alignTree->Branch("dalpha", &_dalphaVal, "dalpha/F");
  _alignTree->Branch("dbeta", &_dbetaVal, "dbeta/F");
  _alignTree->Branch("dgamma", &_dgammaVal, "dgamma/F");
  _alignTree->Branch("ldx", &_ldxVal, "ldx/F");
  _alignTree->Branch("ldy", &_ldyVal, "ldy/F");
  _alignTree->Branch("ldz", &_ldzVal, "ldz/F");
  _alignTree->Branch("ldr", &_ldrVal, "ldr/F");
  _alignTree->Branch("ldphi", &_ldphiVal, "ldphi/F");
  _alignTree->Branch("useDetId", &_useDetId, "useDetId/I");
  _alignTree->Branch("detDim", &_detDim, "detDim/I");
  _alignTree->Branch("rotx", &_rotxVal, "rotx/F");
  _alignTree->Branch("roty", &_rotyVal, "roty/F");
  _alignTree->Branch("rotz", &_rotzVal, "rotz/F");
  _alignTree->Branch("drotx", &_drotxVal, "drotx/F");
  _alignTree->Branch("droty", &_drotyVal, "droty/F");
  _alignTree->Branch("drotz", &_drotzVal, "drotz/F");
  _alignTree->Branch("surW", &_surWidth, "surW/F");
  _alignTree->Branch("surL", &_surLength, "surL/F");
  _alignTree->Branch("surRot", &_surRot, "surRot[9]/D");

  _mgacollection.clear();
}
//////////////////////////////////////////////////
void MuonGeometryArrange::endHist() {
  // Unpack the list and create ntuples here.

  int size = _mgacollection.size();
  if (size <= 0)
    return;  // nothing to do here.
  std::vector<float> xp(size + 1);
  std::vector<float> yp(size + 1);
  int i;
  float minV, maxV;
  int minI, maxI;

  minV = 99999999.;
  maxV = -minV;
  minI = 9999999;
  maxI = -minI;
  TGraph* grx = nullptr;
  TH2F* dxh = nullptr;

  // for position plots:
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].phipos < minI)
      minI = _mgacollection[i].phipos;
    if (_mgacollection[i].phipos > maxI)
      maxI = _mgacollection[i].phipos;
    xp[i] = _mgacollection[i].phipos;
  }
  if (minI >= maxI)
    return;                     // can't do anything?
  xp[size] = xp[size - 1] + 1;  // wraparound point

  if (1 < minI)
    minI = 1;
  if (size > maxI)
    maxI = size;
  maxI++;  // allow for wraparound to show neighbors
  int sizeI = maxI + 1 - minI;
  float smi = minI - 1;
  float sma = maxI + 1;

  // Dx plot

  for (i = 0; i < size; i++) {
    if (_mgacollection[i].ldx < minV)
      minV = _mgacollection[i].ldx;
    if (_mgacollection[i].ldx > maxV)
      maxV = _mgacollection[i].ldx;
    yp[i] = _mgacollection[i].ldx;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delX_vs_position",
            "Local #delta X vs position",
            "GdelX_vs_position",
            "#delta x in cm",
            xp.data(),
            yp.data(),
            size);
  // Dy plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].ldy < minV)
      minV = _mgacollection[i].ldy;
    if (_mgacollection[i].ldy > maxV)
      maxV = _mgacollection[i].ldy;
    yp[i] = _mgacollection[i].ldy;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delY_vs_position",
            "Local #delta Y vs position",
            "GdelY_vs_position",
            "#delta y in cm",
            xp.data(),
            yp.data(),
            size);

  // Dz plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].dz < minV)
      minV = _mgacollection[i].dz;
    if (_mgacollection[i].dz > maxV)
      maxV = _mgacollection[i].dz;
    yp[i] = _mgacollection[i].dz;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delZ_vs_position",
            "Local #delta Z vs position",
            "GdelZ_vs_position",
            "#delta z in cm",
            xp.data(),
            yp.data(),
            size);

  // Dphi plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].dphi < minV)
      minV = _mgacollection[i].dphi;
    if (_mgacollection[i].dphi > maxV)
      maxV = _mgacollection[i].dphi;
    yp[i] = _mgacollection[i].dphi;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delphi_vs_position",
            "#delta #phi vs position",
            "Gdelphi_vs_position",
            "#delta #phi in radians",
            xp.data(),
            yp.data(),
            size);

  // Dr plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].dr < minV)
      minV = _mgacollection[i].dr;
    if (_mgacollection[i].dr > maxV)
      maxV = _mgacollection[i].dr;
    yp[i] = _mgacollection[i].dr;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delR_vs_position",
            "#delta R vs position",
            "GdelR_vs_position",
            "#delta R in cm",
            xp.data(),
            yp.data(),
            size);

  // Drphi plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    float ttemp = _mgacollection[i].r * _mgacollection[i].dphi;
    if (ttemp < minV)
      minV = ttemp;
    if (ttemp > maxV)
      maxV = ttemp;
    yp[i] = ttemp;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delRphi_vs_position",
            "R #delta #phi vs position",
            "GdelRphi_vs_position",
            "R #delta #phi in cm",
            xp.data(),
            yp.data(),
            size);

  // Dalpha plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].dalpha < minV)
      minV = _mgacollection[i].dalpha;
    if (_mgacollection[i].dalpha > maxV)
      maxV = _mgacollection[i].dalpha;
    yp[i] = _mgacollection[i].dalpha;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delalpha_vs_position",
            "#delta #alpha vs position",
            "Gdelalpha_vs_position",
            "#delta #alpha in rad",
            xp.data(),
            yp.data(),
            size);

  // Dbeta plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].dbeta < minV)
      minV = _mgacollection[i].dbeta;
    if (_mgacollection[i].dbeta > maxV)
      maxV = _mgacollection[i].dbeta;
    yp[i] = _mgacollection[i].dbeta;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delbeta_vs_position",
            "#delta #beta vs position",
            "Gdelbeta_vs_position",
            "#delta #beta in rad",
            xp.data(),
            yp.data(),
            size);

  // Dgamma plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].dgamma < minV)
      minV = _mgacollection[i].dgamma;
    if (_mgacollection[i].dgamma > maxV)
      maxV = _mgacollection[i].dgamma;
    yp[i] = _mgacollection[i].dgamma;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delgamma_vs_position",
            "#delta #gamma vs position",
            "Gdelgamma_vs_position",
            "#delta #gamma in rad",
            xp.data(),
            yp.data(),
            size);

  // Drotx plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].drotx < minV)
      minV = _mgacollection[i].drotx;
    if (_mgacollection[i].drotx > maxV)
      maxV = _mgacollection[i].drotx;
    yp[i] = _mgacollection[i].drotx;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delrotX_vs_position",
            "#delta rotX vs position",
            "GdelrotX_vs_position",
            "#delta rotX in rad",
            xp.data(),
            yp.data(),
            size);

  // Droty plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].droty < minV)
      minV = _mgacollection[i].droty;
    if (_mgacollection[i].droty > maxV)
      maxV = _mgacollection[i].droty;
    yp[i] = _mgacollection[i].droty;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delrotY_vs_position",
            "#delta rotY vs position",
            "GdelrotY_vs_position",
            "#delta rotY in rad",
            xp.data(),
            yp.data(),
            size);

  // Drotz plot
  minV = 99999999.;
  maxV = -minV;
  for (i = 0; i < size; i++) {
    if (_mgacollection[i].drotz < minV)
      minV = _mgacollection[i].drotz;
    if (_mgacollection[i].drotz > maxV)
      maxV = _mgacollection[i].drotz;
    yp[i] = _mgacollection[i].drotz;
  }
  yp[size] = yp[0];  // wraparound point

  makeGraph(sizeI,
            smi,
            sma,
            minV,
            maxV,
            dxh,
            grx,
            "delrotZ_vs_position",
            "#delta rotZ vs position",
            "GdelrotZ_vs_position",
            "#delta rotZ in rad",
            xp.data(),
            yp.data(),
            size);

  // Vector plots
  // First find the maximum length of sqrt(dx*dx+dy*dy):  we'll have to
  // scale these for visibility
  maxV = -99999999.;
  float ttemp, rtemp;
  float maxR = -9999999.;
  for (i = 0; i < size; i++) {
    ttemp = sqrt(_mgacollection[i].dx * _mgacollection[i].dx + _mgacollection[i].dy * _mgacollection[i].dy);
    rtemp = sqrt(_mgacollection[i].x * _mgacollection[i].x + _mgacollection[i].y * _mgacollection[i].y);
    if (ttemp > maxV)
      maxV = ttemp;
    if (rtemp > maxR)
      maxR = rtemp;
  }

  // Don't try to scale rediculously small values
  float smallestVcm = .001;  // 10 microns
  if (maxV < smallestVcm)
    maxV = smallestVcm;
  float scale = 0.;
  float lside = 1.1 * maxR;
  if (lside <= 0)
    lside = 100.;
  if (maxV > 0) {
    scale = .09 * lside / maxV;
  }  // units of pad length!
  char scalename[50];
  int ret = snprintf(scalename, 50, "#delta #bar{x}   length =%f cm", maxV);
  // If ret<=0 we don't want to print the scale!

  if (ret > 0) {
    dxh = new TH2F("vecdrplot", scalename, 80, -lside, lside, 80, -lside, lside);
  } else {
    dxh = new TH2F("vecdrplot", "delta #bar{x} Bad scale", 80, -lside, lside, 80, -lside, lside);
  }
  dxh->GetXaxis()->SetTitle("x in cm");
  dxh->GetYaxis()->SetTitle("y in cm");
  dxh->SetStats(kFALSE);
  dxh->Draw();
  TArrow* arrow;
  for (i = 0; i < size; i++) {
    ttemp = sqrt(_mgacollection[i].dx * _mgacollection[i].dx + _mgacollection[i].dy * _mgacollection[i].dy);
    //     ttemp=ttemp*scale;
    float nx = _mgacollection[i].x + scale * _mgacollection[i].dx;
    float ny = _mgacollection[i].y + scale * _mgacollection[i].dy;
    arrow = new TArrow(_mgacollection[i].x, _mgacollection[i].y, nx, ny);  // ttemp*.3*.05, "->");
    arrow->SetLineWidth(2);
    arrow->SetArrowSize(ttemp * .2 * .05 / maxV);
    arrow->SetLineColor(1);
    arrow->SetLineStyle(1);
    arrow->Paint();
    dxh->GetListOfFunctions()->Add(static_cast<TObject*>(arrow));
    //     arrow->Draw();
    //     arrow->Write();
  }
  dxh->Write();

  _theFile->Write();
  _theFile->Close();
}
//////////////////////////////////////////////////
void MuonGeometryArrange::makeGraph(int sizeI,
                                    float smi,
                                    float sma,
                                    float minV,
                                    float maxV,
                                    TH2F* dxh,
                                    TGraph* grx,
                                    const char* name,
                                    const char* title,
                                    const char* titleg,
                                    const char* axis,
                                    const float* xp,
                                    const float* yp,
                                    int size) {
  if (minV >= maxV || smi >= sma || sizeI <= 1 || xp == nullptr || yp == nullptr)
    return;
  // out of bounds, bail
  float diff = maxV - minV;
  float over = .05 * diff;
  double ylo = minV - over;
  double yhi = maxV + over;
  double dsmi, dsma;
  dsmi = smi;
  dsma = sma;
  dxh = new TH2F(name, title, sizeI + 2, dsmi, dsma, 50, ylo, yhi);
  dxh->GetXaxis()->SetTitle("Position around ring");
  dxh->GetYaxis()->SetTitle(axis);
  dxh->SetStats(kFALSE);
  dxh->Draw();
  grx = new TGraph(size, xp, yp);
  grx->SetName(titleg);
  grx->SetTitle(title);
  grx->SetMarkerColor(2);
  grx->SetMarkerStyle(3);
  grx->GetXaxis()->SetLimits(dsmi, dsma);
  grx->GetXaxis()->SetTitle("position number");
  grx->GetYaxis()->SetLimits(ylo, yhi);
  grx->GetYaxis()->SetTitle(axis);
  grx->Draw("A*");
  grx->Write();
  return;
}
//////////////////////////////////////////////////
void MuonGeometryArrange::beginJob() { firstEvent_ = true; }

//////////////////////////////////////////////////
void MuonGeometryArrange::createROOTGeometry(const edm::EventSetup& iSetup) {}
//////////////////////////////////////////////////
void MuonGeometryArrange::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  if (firstEvent_) {
    MuonAlignmentInputXML inputMethod1(_inputXMLCurrent,
                                       &iSetup.getData(dtGeomToken1_),
                                       &iSetup.getData(cscGeomToken1_),
                                       &iSetup.getData(gemGeomToken1_),
                                       &iSetup.getData(dtGeomToken1_),
                                       &iSetup.getData(cscGeomToken1_),
                                       &iSetup.getData(gemGeomToken1_));
    inputAlign1 = new MuonAlignment(iSetup, inputMethod1);
    inputAlign1->fillGapsInSurvey(0, 0);
    MuonAlignmentInputXML inputMethod2(_inputXMLReference,
                                       &iSetup.getData(dtGeomToken2_),
                                       &iSetup.getData(cscGeomToken2_),
                                       &iSetup.getData(gemGeomToken2_),
                                       &iSetup.getData(dtGeomToken1_),
                                       &iSetup.getData(cscGeomToken1_),
                                       &iSetup.getData(gemGeomToken1_));
    inputAlign2 = new MuonAlignment(iSetup, inputMethod2);
    inputAlign2->fillGapsInSurvey(0, 0);
    MuonAlignmentInputXML inputMethod2a(_inputXMLReference,
                                        &iSetup.getData(dtGeomToken3_),
                                        &iSetup.getData(cscGeomToken3_),
                                        &iSetup.getData(gemGeomToken3_),
                                        &iSetup.getData(dtGeomToken1_),
                                        &iSetup.getData(cscGeomToken1_),
                                        &iSetup.getData(gemGeomToken1_));
    inputAlign2a = new MuonAlignment(iSetup, inputMethod2a);
    inputAlign2a->fillGapsInSurvey(0, 0);

    inputGeometry1 = static_cast<Alignable*>(inputAlign1->getAlignableMuon());
    inputGeometry2 = static_cast<Alignable*>(inputAlign2->getAlignableMuon());
    auto inputGeometry2Copy2 = inputAlign2a->getAlignableMuon();

    //setting the levels being used in the geometry comparator
    edm::LogInfo("MuonGeometryArrange") << "levels: " << _levelStrings.size();
    for (const auto& level : _levelStrings) {
      theLevels.push_back(inputGeometry2Copy2->objectIdProvider().stringToId(level));
      edm::LogInfo("MuonGeometryArrange") << "level: " << level;
    }

    //compare the goemetries
    compare(inputGeometry1, inputGeometry2, inputGeometry2Copy2);

    //write out ntuple
    //might be better to do within output module
    _theFile->cd();
    _alignTree->Write();
    endHist();
    //   _theFile->Close();

    firstEvent_ = false;
  }
}

/////////////////////////////////////////////////
void MuonGeometryArrange::compare(Alignable* refAli, Alignable* curAli, Alignable* curAliCopy2) {
  // First sanity
  if (refAli == nullptr) {
    return;
  }
  if (curAli == nullptr) {
    return;
  }

  const auto& refComp = refAli->components();
  const auto& curComp = curAli->components();
  const auto& curComp2 = curAliCopy2->components();
  compareGeometries(refAli, curAli, curAliCopy2);

  int nComp = refComp.size();
  for (int i = 0; i < nComp; i++) {
    compare(refComp[i], curComp[i], curComp2[i]);
  }
  return;
}

//////////////////////////////////////////////////
void MuonGeometryArrange::compareGeometries(Alignable* refAli, Alignable* curAli, Alignable* curCopy) {
  // First sanity
  if (refAli == nullptr) {
    return;
  }
  if (curAli == nullptr) {
    return;
  }
  // Is this the Ring we want to align?  If so it will contain the
  // chambers specified in the configuration file
  if (!isMother(refAli))
    return;  // Not the desired alignable object
             // But... There are granddaughters involved--and I don't want to monkey with
             // the layers of the chambers.  So, if the mother of this is also an approved
             // mother, bail.
  if (isMother(refAli->mother()))
    return;
  const auto& refComp = refAli->components();
  const auto& curComp = curCopy->components();
  if (refComp.size() != curComp.size()) {
    return;
  }
  // GlobalVectors is a vector of GlobalVector which is a 3D vector
  align::GlobalVectors originalVectors;
  align::GlobalVectors currentVectors;
  align::GlobalVectors originalRelativeVectors;
  align::GlobalVectors currentRelativeVectors;

  int nComp = refComp.size();
  int nUsed = 0;
  // Use the total displacements here:
  CLHEP::Hep3Vector TotalX, TotalL;
  TotalX.set(0., 0., 0.);
  TotalL.set(0., 0., 0.);
  //  CLHEP::Hep3Vector* Rsubtotal, Wsubtotal, DRsubtotal, DWsubtotal;
  std::vector<CLHEP::Hep3Vector> Positions;
  std::vector<CLHEP::Hep3Vector> DelPositions;

  double xrcenter = 0.;
  double yrcenter = 0.;
  double zrcenter = 0.;
  double xccenter = 0.;
  double yccenter = 0.;
  double zccenter = 0.;

  bool useIt;
  // Create the "center" for the reference alignment chambers, and
  // load a vector of their centers
  for (int ich = 0; ich < nComp; ich++) {
    useIt = true;
    if (_weightById) {
      if (!align::readModuleList(curComp[ich]->id(), curComp[ich]->id(), _weightByIdVector))
        useIt = false;
    }
    if (!useIt)
      continue;
    align::GlobalVectors curVs;
    align::createPoints(&curVs, refComp[ich], _weightBy, _weightById, _weightByIdVector);
    align::GlobalVector pointsCM = align::centerOfMass(curVs);
    originalVectors.push_back(pointsCM);
    nUsed++;
    xrcenter += pointsCM.x();
    yrcenter += pointsCM.y();
    zrcenter += pointsCM.z();
  }
  xrcenter = xrcenter / nUsed;
  yrcenter = yrcenter / nUsed;
  zrcenter = zrcenter / nUsed;

  // Create the "center" for the current alignment chambers, and
  // load a vector of their centers
  for (int ich = 0; ich < nComp; ich++) {
    useIt = true;
    if (_weightById) {
      if (!align::readModuleList(curComp[ich]->id(), curComp[ich]->id(), _weightByIdVector))
        useIt = false;
    }
    if (!useIt)
      continue;
    align::GlobalVectors curVs;
    align::createPoints(&curVs, curComp[ich], _weightBy, _weightById, _weightByIdVector);
    align::GlobalVector pointsCM = align::centerOfMass(curVs);
    currentVectors.push_back(pointsCM);

    xccenter += pointsCM.x();
    yccenter += pointsCM.y();
    zccenter += pointsCM.z();
  }
  xccenter = xccenter / nUsed;
  yccenter = yccenter / nUsed;
  zccenter = zccenter / nUsed;

  // OK, now load the <very approximate> vectors from the ring "centers"
  align::GlobalVector CCur(xccenter, yccenter, zccenter);
  align::GlobalVector CRef(xrcenter, yrcenter, zrcenter);
  int nCompR = currentVectors.size();
  for (int ich = 0; ich < nCompR; ich++) {
    originalRelativeVectors.push_back(originalVectors[ich] - CRef);
    currentRelativeVectors.push_back(currentVectors[ich] - CCur);
  }

  // All right.  Now let the hacking begin.
  // First out of the gate let's try using the raw values and see what
  // diffRot does for us.

  align::RotationType rtype3 = align::diffRot(currentRelativeVectors, originalRelativeVectors);

  align::EulerAngles angles(3);
  angles = align::toAngles(rtype3);

  for (int ich = 0; ich < nComp; ich++) {
    if (_weightById) {
      if (!align::readModuleList(curComp[ich]->id(), curComp[ich]->id(), _weightByIdVector))
        continue;
    }
    CLHEP::Hep3Vector Rtotal, Wtotal;
    Rtotal.set(0., 0., 0.);
    Wtotal.set(0., 0., 0.);
    for (int i = 0; i < 100; i++) {
      AlgebraicVector diff =
          align::diffAlignables(refComp[ich], curComp[ich], _weightBy, _weightById, _weightByIdVector);
      CLHEP::Hep3Vector dR(diff[0], diff[1], diff[2]);
      Rtotal += dR;
      CLHEP::Hep3Vector dW(diff[3], diff[4], diff[5]);
      CLHEP::HepRotation rot(Wtotal.unit(), Wtotal.mag());
      CLHEP::HepRotation drot(dW.unit(), dW.mag());
      rot *= drot;
      Wtotal.set(rot.axis().x() * rot.delta(), rot.axis().y() * rot.delta(), rot.axis().z() * rot.delta());
      align::moveAlignable(curComp[ich], diff);
      float tolerance = 1e-7;
      AlgebraicVector check =
          align::diffAlignables(refComp[ich], curComp[ich], _weightBy, _weightById, _weightByIdVector);
      align::GlobalVector checkR(check[0], check[1], check[2]);
      align::GlobalVector checkW(check[3], check[4], check[5]);
      DetId detid(refComp[ich]->id());
      if ((checkR.mag() > tolerance) || (checkW.mag() > tolerance)) {
        //	 edm::LogInfo("CompareGeoms") << "Tolerance Exceeded!(alObjId: "
        //       << refAli->alignableObjectId()
        //	 << ", rawId: " << refComp[ich]->geomDetId().rawId()
        //	 << ", subdetId: "<< detid.subdetId() << "): " << diff;
      } else {
        TotalX += Rtotal;
        break;
      }  // end of else
    }    // end of for on int i
  }      // end of for on ich

  // At this point we should have a total displacement and total L
  TotalX = TotalX / nUsed;

  // Now start again!
  AlgebraicVector change(6);
  change(1) = TotalX.x();
  change(2) = TotalX.y();
  change(3) = TotalX.z();

  change(4) = angles[0];
  change(5) = angles[1];
  change(6) = angles[2];
  align::moveAlignable(curAli, change);  // move as a chunk

  // Now get the components again.  They should be in new locations
  const auto& curComp2 = curAli->components();

  for (int ich = 0; ich < nComp; ich++) {
    CLHEP::Hep3Vector Rtotal, Wtotal;
    Rtotal.set(0., 0., 0.);
    Wtotal.set(0., 0., 0.);
    if (_weightById) {
      if (!align::readModuleList(curComp[ich]->id(), curComp[ich]->id(), _weightByIdVector))
        continue;
    }

    for (int i = 0; i < 100; i++) {
      AlgebraicVector diff =
          align::diffAlignables(refComp[ich], curComp2[ich], _weightBy, _weightById, _weightByIdVector);
      CLHEP::Hep3Vector dR(diff[0], diff[1], diff[2]);
      Rtotal += dR;
      CLHEP::Hep3Vector dW(diff[3], diff[4], diff[5]);
      CLHEP::HepRotation rot(Wtotal.unit(), Wtotal.mag());
      CLHEP::HepRotation drot(dW.unit(), dW.mag());
      rot *= drot;
      Wtotal.set(rot.axis().x() * rot.delta(), rot.axis().y() * rot.delta(), rot.axis().z() * rot.delta());
      align::moveAlignable(curComp2[ich], diff);
      float tolerance = 1e-7;
      AlgebraicVector check =
          align::diffAlignables(refComp[ich], curComp2[ich], _weightBy, _weightById, _weightByIdVector);
      align::GlobalVector checkR(check[0], check[1], check[2]);
      align::GlobalVector checkW(check[3], check[4], check[5]);
      if ((checkR.mag() > tolerance) || (checkW.mag() > tolerance)) {
      } else {
        break;
      }
    }  // end of for on int i
    AlgebraicVector TRtot(6);
    TRtot(1) = Rtotal.x();
    TRtot(2) = Rtotal.y();
    TRtot(3) = Rtotal.z();
    TRtot(4) = Wtotal.x();
    TRtot(5) = Wtotal.y();
    TRtot(6) = Wtotal.z();
    fillTree(refComp[ich], TRtot);
  }  // end of for on ich
}

//////////////////////////////////////////////////

void MuonGeometryArrange::fillTree(Alignable* refAli, const AlgebraicVector& diff) {
  _id = refAli->id();
  _level = refAli->alignableObjectId();
  //need if ali has no mother
  if (refAli->mother()) {
    _mid = refAli->mother()->geomDetId().rawId();
    _mlevel = refAli->mother()->alignableObjectId();
  } else {
    _mid = -1;
    _mlevel = -1;
  }
  DetId detid(_id);
  _sublevel = detid.subdetId();
  int ringPhiPos = -99;
  if (detid.det() == DetId::Muon && detid.subdetId() == MuonSubdetId::CSC) {
    CSCDetId cscId(refAli->geomDetId());
    ringPhiPos = cscId.chamber();
  }
  _xVal = refAli->globalPosition().x();
  _yVal = refAli->globalPosition().y();
  _zVal = refAli->globalPosition().z();
  align::GlobalVector vec(_xVal, _yVal, _zVal);
  _rVal = vec.perp();
  _phiVal = vec.phi();
  _etaVal = vec.eta();
  align::RotationType rot = refAli->globalRotation();
  align::EulerAngles eulerAngles = align::toAngles(rot);
  _rotxVal = atan2(rot.yz(), rot.zz());
  float ttt = -rot.xz();
  if (ttt > 1.)
    ttt = 1.;
  if (ttt < -1.)
    ttt = -1.;
  _rotyVal = asin(ttt);
  _rotzVal = atan2(rot.xy(), rot.xx());
  _alphaVal = eulerAngles[0];
  _betaVal = eulerAngles[1];
  _gammaVal = eulerAngles[2];
  _dxVal = diff[0];
  _dyVal = diff[1];
  _dzVal = diff[2];
  //getting dR and dPhi
  align::GlobalVector vRef(_xVal, _yVal, _zVal);
  align::GlobalVector vCur(_xVal - _dxVal, _yVal - _dyVal, _zVal - _dzVal);
  _drVal = vCur.perp() - vRef.perp();
  _dphiVal = vCur.phi() - vRef.phi();

  _dalphaVal = diff[3];
  _dbetaVal = diff[4];
  _dgammaVal = diff[5];
  _drotxVal = -999.;
  _drotyVal = -999.;
  _drotzVal = -999.;

  align::EulerAngles deuler(3);
  deuler(1) = _dalphaVal;
  deuler(2) = _dbetaVal;
  deuler(3) = _dgammaVal;
  align::RotationType drot = align::toMatrix(deuler);
  double xx = rot.xx();
  double xy = rot.xy();
  double xz = rot.xz();
  double yx = rot.yx();
  double yy = rot.yy();
  double yz = rot.yz();
  double zx = rot.zx();
  double zy = rot.zy();
  double zz = rot.zz();
  double detrot = (zz * yy - zy * yz) * xx + (-zz * yx + zx * yz) * xy + (zy * yx - zx * yy) * xz;
  detrot = 1 / detrot;
  double ixx = (zz * yy - zy * yz) * detrot;
  double ixy = (-zz * xy + zy * xz) * detrot;
  double ixz = (yz * xy - yy * xz) * detrot;
  double iyx = (-zz * yx + zx * yz) * detrot;
  double iyy = (zz * xx - zx * xz) * detrot;
  double iyz = (-yz * xx + yx * xz) * detrot;
  double izx = (zy * yx - zx * yy) * detrot;
  double izy = (-zy * xx + zx * xy) * detrot;
  double izz = (yy * xx - yx * xy) * detrot;
  align::RotationType invrot(ixx, ixy, ixz, iyx, iyy, iyz, izx, izy, izz);
  align::RotationType prot = rot * drot * invrot;
  //	align::RotationType prot = rot*drot;
  float protx;  //, proty, protz;
  protx = atan2(prot.yz(), prot.zz());
  _drotxVal = protx;  //_rotxVal-protx; //atan2(drot.yz(), drot.zz());
  ttt = -prot.xz();
  if (ttt > 1.)
    ttt = 1.;
  if (ttt < -1.)
    ttt = -1.;
  _drotyVal = asin(ttt);                    // -_rotyVal;
  _drotzVal = atan2(prot.xy(), prot.xx());  // - _rotzVal;
                                            // Above does not account for 2Pi wraparounds!
                                            // Prior knowledge:  these are supposed to be small rotations.  Therefore:
  if (_drotxVal > 3.141592656)
    _drotxVal = -6.2831853072 + _drotxVal;
  if (_drotxVal < -3.141592656)
    _drotxVal = 6.2831853072 + _drotxVal;
  if (_drotyVal > 3.141592656)
    _drotyVal = -6.2831853072 + _drotyVal;
  if (_drotyVal < -3.141592656)
    _drotyVal = 6.2831853072 + _drotyVal;
  if (_drotzVal > 3.141592656)
    _drotzVal = -6.2831853072 + _drotzVal;
  if (_drotzVal < -3.141592656)
    _drotzVal = 6.2831853072 + _drotzVal;

  _ldxVal = -999.;
  _ldyVal = -999.;
  _ldxVal = -999.;
  _ldrVal = -999.;
  _ldphiVal = -999;  // set fake

  //	if(refAli->alignableObjectId() == align::AlignableDetUnit){
  align::GlobalVector dV(_dxVal, _dyVal, _dzVal);
  align::LocalVector pointL = refAli->surface().toLocal(dV);
  //align::LocalVector pointL = (refAli->mother())->surface().toLocal(dV);
  _ldxVal = pointL.x();
  _ldyVal = pointL.y();
  _ldzVal = pointL.z();
  _ldphiVal = pointL.phi();
  _ldrVal = pointL.perp();
  //	}
  //detIdFlag
  if (refAli->alignableObjectId() == align::AlignableDetUnit) {
    if (_detIdFlag) {
      if ((passIdCut(refAli->id())) || (passIdCut(refAli->mother()->id()))) {
        _useDetId = 1;
      } else {
        _useDetId = 0;
      }
    }
  }
  // det module dimension
  if (refAli->alignableObjectId() == align::AlignableDetUnit) {
    if (refAli->mother()->alignableObjectId() != align::AlignableDet) {
      _detDim = 1;
    } else if (refAli->mother()->alignableObjectId() == align::AlignableDet) {
      _detDim = 2;
    }
  } else
    _detDim = 0;

  _surWidth = refAli->surface().width();
  _surLength = refAli->surface().length();
  align::RotationType rt = refAli->globalRotation();
  _surRot[0] = rt.xx();
  _surRot[1] = rt.xy();
  _surRot[2] = rt.xz();
  _surRot[3] = rt.yx();
  _surRot[4] = rt.yy();
  _surRot[5] = rt.yz();
  _surRot[6] = rt.zx();
  _surRot[7] = rt.zy();
  _surRot[8] = rt.zz();

  MGACollection holdit;
  holdit.id = _id;
  holdit.level = _level;
  holdit.mid = _mid;
  holdit.mlevel = _mlevel;
  holdit.sublevel = _sublevel;
  holdit.x = _xVal;
  holdit.y = _yVal;
  holdit.z = _zVal;
  holdit.r = _rVal;
  holdit.phi = _phiVal;
  holdit.eta = _etaVal;
  holdit.alpha = _alphaVal;
  holdit.beta = _betaVal;
  holdit.gamma = _gammaVal;
  holdit.dx = _dxVal;
  holdit.dy = _dyVal;
  holdit.dz = _dzVal;
  holdit.dr = _drVal;
  holdit.dphi = _dphiVal;
  holdit.dalpha = _dalphaVal;
  holdit.dbeta = _dbetaVal;
  holdit.dgamma = _dgammaVal;
  holdit.useDetId = _useDetId;
  holdit.detDim = _detDim;
  holdit.surW = _surWidth;
  holdit.surL = _surLength;
  holdit.ldx = _ldxVal;
  holdit.ldy = _ldyVal;
  holdit.ldz = _ldzVal;
  holdit.ldr = _ldrVal;
  holdit.ldphi = _ldphiVal;
  holdit.rotx = _rotxVal;
  holdit.roty = _rotyVal;
  holdit.rotz = _rotzVal;
  holdit.drotx = _drotxVal;
  holdit.droty = _drotyVal;
  holdit.drotz = _drotzVal;
  for (int i = 0; i < 9; i++) {
    holdit.surRot[i] = _surRot[i];
  }
  holdit.phipos = ringPhiPos;
  _mgacollection.push_back(holdit);

  //Fill
  _alignTree->Fill();
}

//////////////////////////////////////////////////
bool MuonGeometryArrange::isMother(Alignable* ali) {
  // Is this the mother ring?
  if (ali == nullptr)
    return false;  // elementary sanity
  const auto& aliComp = ali->components();

  int size = aliComp.size();
  if (size <= 0)
    return false;  // no subcomponents

  for (int i = 0; i < size; i++) {
    if (checkChosen(aliComp[i]))
      return true;  // A ring has CSC chambers
  }                 // as subcomponents
  return false;     // 1'st layer of subcomponents weren't CSC chambers
}
//////////////////////////////////////////////////

bool MuonGeometryArrange::checkChosen(Alignable* ali) {
  // Check whether the item passed satisfies the criteria given.
  if (ali == nullptr)
    return false;  // elementary sanity
                   // Is this in the CSC section?  If not, bail.  Later may extend.
  if (ali->geomDetId().det() != DetId::Muon || ali->geomDetId().subdetId() != MuonSubdetId::CSC)
    return false;
  // If it is a CSC alignable, then check that the station, etc are
  // those requested.
  // One might think of aligning more than a single ring at a time,
  // by using a vector of ring numbers.  I don't see the sense in
  // trying to align more than one station at a time for comparison.
  CSCDetId cscId(ali->geomDetId());
#ifdef jnbdebug
  std::cout << "JNB " << ali->id() << " " << cscId.endcap() << " " << cscId.station() << " " << cscId.ring() << " "
            << cscId.chamber() << "   " << _endcap << " " << _station << " " << _ring << "\n"
            << std::flush;
#endif
  if (cscId.endcap() == _endcap && cscId.station() == _station && cscId.ring() == _ring) {
    return true;
  }
  return false;
}
//////////////////////////////////////////////////

bool MuonGeometryArrange::passChosen(Alignable* ali) {
  // Check to see if this contains CSC components of the appropriate ring
  // Ring will contain N Alignables which represent chambers, each of which
  // in turn contains M planes.  For our purposes we don't care about the
  // planes.
  // Hmm.  Interesting question:  Do I want to try to fit the chamber as
  // such, or use the geometry?
  // I want to fit the chamber, so I'll try to use its presence as the marker.
  // What specifically identifies a chamber as a chamber, and not as a layer?
  // The fact that it has layers as sub components, or the fact that it is
  // the first item with a non-zero ID breakdown?  Pick the latter.
  //
  if (ali == nullptr)
    return false;
  if (checkChosen(ali))
    return true;  // If this is one of the desired
                  // CSC chambers, accept it
  const auto& aliComp = ali->components();

  int size = aliComp.size();
  if (size <= 0)
    return false;  // no subcomponents

  for (int i = 0; i < size; i++) {
    if (checkChosen(aliComp[i]))
      return true;  // A ring has CSC chambers
  }                 // as subcomponents
  return false;     // 1'st layer of subcomponents weren't CSC chambers
}
//////////////////////////////////////////////////
bool MuonGeometryArrange::passIdCut(uint32_t id) {
  bool pass = false;
  DetId detid(id);
  //	if(detid.det()==DetId::Muon && detid.subdetId()== MuonSubdetId::CSC){
  //	   CSCDetId cscId(refAli->geomDetId());
  //	   if(cscId.layer()!=1) return false;		// ONLY FIRST LAYER!
  //	}
  int nEntries = _detIdFlagVector.size();

  for (int i = 0; i < nEntries; i++) {
    if (_detIdFlagVector[i] == id)
      pass = true;
  }

  return pass;
}

//////////////////////////////////////////////////
DEFINE_FWK_MODULE(MuonGeometryArrange);
