
// -*- C++ -*-
//
// Package:    DefaultApeTree
// Class:      DefaultApeTree
//
/**\class ApeTreeCreateDefault ApeTreeCreateDefault.cc Alignment/APEEstimation/src/ApeTreeCreateDefault.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marius Teroerde (code from ApeEstimator.cc and
//                   ApeEstimatorSummary.cc)
//         Created:  Tue Nov 14 11:43 CET 2017
//
//

// system include files
#include <fstream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "CLHEP/Matrix/SymMatrix.h"

#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"

//...............
#include "Alignment/APEEstimation/interface/TrackerSectorStruct.h"
#include "Alignment/APEEstimation/interface/ReducedTrackerTreeVariables.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "TString.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TMath.h"
//
// class declaration
//

class ApeTreeCreateDefault : public edm::one::EDAnalyzer<> {
public:
  explicit ApeTreeCreateDefault(const edm::ParameterSet&);
  ~ApeTreeCreateDefault() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void sectorBuilder();
  bool checkIntervalsForSectors(const unsigned int sectorCounter, const std::vector<double>&) const;
  bool checkModuleIds(const unsigned int, const std::vector<unsigned int>&) const;
  bool checkModuleBools(const bool, const std::vector<unsigned int>&) const;
  bool checkModuleDirections(const int, const std::vector<int>&) const;
  bool checkModulePositions(const float, const std::vector<double>&) const;

  // ----------member data ---------------------------
  const edm::ESGetToken<AlignmentErrorsExtended, TrackerAlignmentErrorExtendedRcd> alignmentErrorToken_;
  const std::string resultFile_;
  const std::string trackerTreeFile_;
  const std::vector<edm::ParameterSet> sectors_;

  std::map<unsigned int, TrackerSectorStruct> m_tkSector_;
  std::map<unsigned int, ReducedTrackerTreeVariables> m_tkTreeVar_;
  unsigned int noSectors;
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
ApeTreeCreateDefault::ApeTreeCreateDefault(const edm::ParameterSet& iConfig)
    : alignmentErrorToken_(esConsumes()),
      resultFile_(iConfig.getParameter<std::string>("resultFile")),
      trackerTreeFile_(iConfig.getParameter<std::string>("trackerTreeFile")),
      sectors_(iConfig.getParameter<std::vector<edm::ParameterSet>>("sectors")) {}

ApeTreeCreateDefault::~ApeTreeCreateDefault() {}

//
// member functions
//
void ApeTreeCreateDefault::sectorBuilder() {
  // Same procedure as in ApeEstimator.cc
  TFile* tkTreeFile(TFile::Open(trackerTreeFile_.c_str()));
  if (tkTreeFile) {
    edm::LogInfo("SectorBuilder") << "TrackerTreeFile OK";
  } else {
    edm::LogError("SectorBuilder") << "TrackerTreeFile not found";
    return;
  }
  TTree* tkTree(nullptr);
  tkTreeFile->GetObject("TrackerTreeGenerator/TrackerTree/TrackerTree", tkTree);
  if (tkTree) {
    edm::LogInfo("SectorBuilder") << "TrackerTree OK";
  } else {
    edm::LogError("SectorBuilder") << "TrackerTree not found in file";
    return;
  }

  unsigned int rawId(999), subdetId(999), layer(999), side(999), half(999), rod(999), ring(999), petal(999), blade(999),
      panel(999), outerInner(999), module(999), nStrips(999);
  bool isDoubleSide(false), isRPhi(false), isStereo(false);
  int uDirection(999), vDirection(999), wDirection(999);
  float posR(999.F), posPhi(999.F), posEta(999.F), posX(999.F), posY(999.F), posZ(999.F);

  tkTree->SetBranchAddress("RawId", &rawId);
  tkTree->SetBranchAddress("SubdetId", &subdetId);
  tkTree->SetBranchAddress("Layer", &layer);
  tkTree->SetBranchAddress("Side", &side);
  tkTree->SetBranchAddress("Half", &half);
  tkTree->SetBranchAddress("Rod", &rod);
  tkTree->SetBranchAddress("Ring", &ring);
  tkTree->SetBranchAddress("Petal", &petal);
  tkTree->SetBranchAddress("Blade", &blade);
  tkTree->SetBranchAddress("Panel", &panel);
  tkTree->SetBranchAddress("OuterInner", &outerInner);
  tkTree->SetBranchAddress("Module", &module);
  tkTree->SetBranchAddress("NStrips", &nStrips);
  tkTree->SetBranchAddress("IsDoubleSide", &isDoubleSide);
  tkTree->SetBranchAddress("IsRPhi", &isRPhi);
  tkTree->SetBranchAddress("IsStereo", &isStereo);
  tkTree->SetBranchAddress("UDirection", &uDirection);
  tkTree->SetBranchAddress("VDirection", &vDirection);
  tkTree->SetBranchAddress("WDirection", &wDirection);
  tkTree->SetBranchAddress("PosR", &posR);
  tkTree->SetBranchAddress("PosPhi", &posPhi);
  tkTree->SetBranchAddress("PosEta", &posEta);
  tkTree->SetBranchAddress("PosX", &posX);
  tkTree->SetBranchAddress("PosY", &posY);
  tkTree->SetBranchAddress("PosZ", &posZ);

  int nModules(tkTree->GetEntries());
  TrackerSectorStruct allSectors;

  //Loop over all Sectors
  unsigned int sectorCounter(0);
  std::vector<edm::ParameterSet> v_sectorDef(sectors_);
  edm::LogInfo("SectorBuilder") << "There are " << v_sectorDef.size() << " Sectors defined";

  for (auto const& parSet : v_sectorDef) {
    ++sectorCounter;
    const std::string& sectorName(parSet.getParameter<std::string>("name"));
    std::vector<unsigned int> v_rawId(parSet.getParameter<std::vector<unsigned int>>("rawId")),
        v_subdetId(parSet.getParameter<std::vector<unsigned int>>("subdetId")),
        v_layer(parSet.getParameter<std::vector<unsigned int>>("layer")),
        v_side(parSet.getParameter<std::vector<unsigned int>>("side")),
        v_half(parSet.getParameter<std::vector<unsigned int>>("half")),
        v_rod(parSet.getParameter<std::vector<unsigned int>>("rod")),
        v_ring(parSet.getParameter<std::vector<unsigned int>>("ring")),
        v_petal(parSet.getParameter<std::vector<unsigned int>>("petal")),
        v_blade(parSet.getParameter<std::vector<unsigned int>>("blade")),
        v_panel(parSet.getParameter<std::vector<unsigned int>>("panel")),
        v_outerInner(parSet.getParameter<std::vector<unsigned int>>("outerInner")),
        v_module(parSet.getParameter<std::vector<unsigned int>>("module")),
        v_nStrips(parSet.getParameter<std::vector<unsigned int>>("nStrips")),
        v_isDoubleSide(parSet.getParameter<std::vector<unsigned int>>("isDoubleSide")),
        v_isRPhi(parSet.getParameter<std::vector<unsigned int>>("isRPhi")),
        v_isStereo(parSet.getParameter<std::vector<unsigned int>>("isStereo"));
    std::vector<int> v_uDirection(parSet.getParameter<std::vector<int>>("uDirection")),
        v_vDirection(parSet.getParameter<std::vector<int>>("vDirection")),
        v_wDirection(parSet.getParameter<std::vector<int>>("wDirection"));
    std::vector<double> v_posR(parSet.getParameter<std::vector<double>>("posR")),
        v_posPhi(parSet.getParameter<std::vector<double>>("posPhi")),
        v_posEta(parSet.getParameter<std::vector<double>>("posEta")),
        v_posX(parSet.getParameter<std::vector<double>>("posX")),
        v_posY(parSet.getParameter<std::vector<double>>("posY")),
        v_posZ(parSet.getParameter<std::vector<double>>("posZ"));

    if (!this->checkIntervalsForSectors(sectorCounter, v_posR) ||
        !this->checkIntervalsForSectors(sectorCounter, v_posPhi) ||
        !this->checkIntervalsForSectors(sectorCounter, v_posEta) ||
        !this->checkIntervalsForSectors(sectorCounter, v_posX) ||
        !this->checkIntervalsForSectors(sectorCounter, v_posY) ||
        !this->checkIntervalsForSectors(sectorCounter, v_posZ)) {
      continue;
    }

    TrackerSectorStruct tkSector;
    tkSector.name = sectorName;

    ReducedTrackerTreeVariables tkTreeVar;

    //Loop over all Modules
    for (int module = 0; module < nModules; ++module) {
      tkTree->GetEntry(module);

      if (sectorCounter == 1) {
        tkTreeVar.subdetId = subdetId;
        tkTreeVar.nStrips = nStrips;
        tkTreeVar.uDirection = uDirection;
        tkTreeVar.vDirection = vDirection;
        tkTreeVar.wDirection = wDirection;
        m_tkTreeVar_[rawId] = tkTreeVar;
      }
      //Check if modules from Sector builder equal those from TrackerTree
      if (!this->checkModuleIds(rawId, v_rawId))
        continue;
      if (!this->checkModuleIds(subdetId, v_subdetId))
        continue;
      if (!this->checkModuleIds(layer, v_layer))
        continue;
      if (!this->checkModuleIds(side, v_side))
        continue;
      if (!this->checkModuleIds(half, v_half))
        continue;
      if (!this->checkModuleIds(rod, v_rod))
        continue;
      if (!this->checkModuleIds(ring, v_ring))
        continue;
      if (!this->checkModuleIds(petal, v_petal))
        continue;
      if (!this->checkModuleIds(blade, v_blade))
        continue;
      if (!this->checkModuleIds(panel, v_panel))
        continue;
      if (!this->checkModuleIds(outerInner, v_outerInner))
        continue;
      if (!this->checkModuleIds(module, v_module))
        continue;
      if (!this->checkModuleIds(nStrips, v_nStrips))
        continue;
      if (!this->checkModuleBools(isDoubleSide, v_isDoubleSide))
        continue;
      if (!this->checkModuleBools(isRPhi, v_isRPhi))
        continue;
      if (!this->checkModuleBools(isStereo, v_isStereo))
        continue;
      if (!this->checkModuleDirections(uDirection, v_uDirection))
        continue;
      if (!this->checkModuleDirections(vDirection, v_vDirection))
        continue;
      if (!this->checkModuleDirections(wDirection, v_wDirection))
        continue;
      if (!this->checkModulePositions(posR, v_posR))
        continue;
      if (!this->checkModulePositions(posPhi, v_posPhi))
        continue;
      if (!this->checkModulePositions(posEta, v_posEta))
        continue;
      if (!this->checkModulePositions(posX, v_posX))
        continue;
      if (!this->checkModulePositions(posY, v_posY))
        continue;
      if (!this->checkModulePositions(posZ, v_posZ))
        continue;

      tkSector.v_rawId.push_back(rawId);
      bool moduleSelected(false);
      for (auto const& i_rawId : allSectors.v_rawId) {
        if (rawId == i_rawId)
          moduleSelected = true;
      }
      if (!moduleSelected)
        allSectors.v_rawId.push_back(rawId);
    }

    // Stops you from combining pixel and strip detector into one sector
    bool isPixel(false);
    bool isStrip(false);
    for (auto const& i_rawId : tkSector.v_rawId) {
      switch (m_tkTreeVar_[i_rawId].subdetId) {
        case PixelSubdetector::PixelBarrel:
        case PixelSubdetector::PixelEndcap:
          isPixel = true;
          break;
        case StripSubdetector::TIB:
        case StripSubdetector::TOB:
        case StripSubdetector::TID:
        case StripSubdetector::TEC:
          isStrip = true;
          break;
      }
    }

    if (isPixel && isStrip) {
      edm::LogError("SectorBuilder")
          << "Incorrect Sector Definition: there are pixel and strip modules within one sector"
          << "\n... sector selection is not applied, sector " << sectorCounter << " is not built";
      continue;
    }
    tkSector.isPixel = isPixel;

    m_tkSector_[sectorCounter] = tkSector;
    edm::LogInfo("SectorBuilder") << "There are " << tkSector.v_rawId.size() << " Modules in Sector " << sectorCounter;
  }
  noSectors = sectorCounter;
  return;
}

// Checking methods copied from ApeEstimator.cc
bool ApeTreeCreateDefault::checkIntervalsForSectors(const unsigned int sectorCounter,
                                                    const std::vector<double>& v_id) const {
  if (v_id.empty())
    return true;
  if (v_id.size() % 2 == 1) {
    edm::LogError("SectorBuilder")
        << "Incorrect Sector Definition: Position Vectors need even number of arguments (Intervals)"
        << "\n... sector selection is not applied, sector " << sectorCounter << " is not built";
    return false;
  }
  int entry(0);
  double intervalBegin(999.);
  for (auto const& i_id : v_id) {
    ++entry;
    if (entry % 2 == 1)
      intervalBegin = i_id;
    if (entry % 2 == 0 && intervalBegin > i_id) {
      edm::LogError("SectorBuilder") << "Incorrect Sector Definition (Position Vector Intervals): \t" << intervalBegin
                                     << " is bigger than " << i_id << " but is expected to be smaller"
                                     << "\n... sector selection is not applied, sector " << sectorCounter
                                     << " is not built";
      return false;
    }
  }
  return true;
}

bool ApeTreeCreateDefault::checkModuleIds(const unsigned int id, const std::vector<unsigned int>& v_id) const {
  if (v_id.empty())
    return true;
  for (auto const& i_id : v_id) {
    if (id == i_id)
      return true;
  }
  return false;
}

bool ApeTreeCreateDefault::checkModuleBools(const bool id, const std::vector<unsigned int>& v_id) const {
  if (v_id.empty())
    return true;
  for (auto const& i_id : v_id) {
    if (1 == i_id && id)
      return true;
    if (2 == i_id && !id)
      return true;
  }
  return false;
}

bool ApeTreeCreateDefault::checkModuleDirections(const int id, const std::vector<int>& v_id) const {
  if (v_id.empty())
    return true;
  for (auto const& i_id : v_id) {
    if (id == i_id)
      return true;
  }
  return false;
}

bool ApeTreeCreateDefault::checkModulePositions(const float id, const std::vector<double>& v_id) const {
  if (v_id.empty())
    return true;
  int entry(0);
  double intervalBegin(999.);
  for (auto const& i_id : v_id) {
    ++entry;
    if (entry % 2 == 1)
      intervalBegin = i_id;
    if (entry % 2 == 0 && id >= intervalBegin && id < i_id)
      return true;
  }
  return false;
}

// ------------ method called to for each event  ------------
void ApeTreeCreateDefault::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Same procedure as in ApeEstimatorSummary.cc minus reading of baseline tree

  // Load APEs from the GT and write them to root files similar to the ones from calculateAPE()
  const AlignmentErrorsExtended* alignmentErrors = &iSetup.getData(alignmentErrorToken_);

  // Set up root file for default APE values
  const std::string defaultFileName(resultFile_);
  TFile* defaultFile = new TFile(defaultFileName.c_str(), "RECREATE");

  // Naming in the root files has to be iterTreeX to be consistent for the plotting tool
  TTree* defaultTreeX(nullptr);
  TTree* defaultTreeY(nullptr);
  defaultFile->GetObject("iterTreeX;1", defaultTreeX);
  defaultFile->GetObject("iterTreeY;1", defaultTreeY);
  // The same for TTree containing the names of the sectors (no additional check, since always handled exactly as defaultTree)
  TTree* sectorNameTree(nullptr);
  defaultFile->GetObject("nameTree;1", sectorNameTree);

  edm::LogInfo("DefaultAPETree") << "APE Tree is being created";
  defaultTreeX = new TTree("iterTreeX", "Tree for default APE x values from GT");
  defaultTreeY = new TTree("iterTreeY", "Tree for default APE y values from GT");
  sectorNameTree = new TTree("nameTree", "Tree with names of sectors");

  // Assign the information stored in the trees to arrays
  std::vector<double*> a_defaultSectorX;
  std::vector<double*> a_defaultSectorY;

  std::vector<std::string*> a_sectorName;
  for (auto const& i_sector : m_tkSector_) {
    const unsigned int iSector(i_sector.first);
    const bool pixelSector(i_sector.second.isPixel);

    a_defaultSectorX.push_back(new double(-99.));
    a_defaultSectorY.push_back(new double(-99.));
    a_sectorName.push_back(new std::string(i_sector.second.name));

    std::stringstream ss_sector;
    std::stringstream ss_sectorSuffixed;
    ss_sector << "Ape_Sector_" << iSector;

    ss_sectorSuffixed << ss_sector.str() << "/D";
    defaultTreeX->Branch(ss_sector.str().c_str(), &(*a_defaultSectorX[iSector - 1]), ss_sectorSuffixed.str().c_str());

    if (pixelSector) {
      defaultTreeY->Branch(ss_sector.str().c_str(), &(*a_defaultSectorY[iSector - 1]), ss_sectorSuffixed.str().c_str());
    }
    sectorNameTree->Branch(ss_sector.str().c_str(), &(*a_sectorName[iSector - 1]), 32000, 00);
  }

  // Loop over sectors for getting default APE

  for (auto& i_sector : m_tkSector_) {
    double defaultApeX(0.);
    double defaultApeY(0.);
    unsigned int nModules(0);
    for (auto const& i_rawId : i_sector.second.v_rawId) {
      std::vector<AlignTransformErrorExtended> alignErrors = alignmentErrors->m_alignError;
      for (auto const& i_alignError : alignErrors) {
        if (i_rawId == i_alignError.rawId()) {
          CLHEP::HepSymMatrix errMatrix = i_alignError.matrix();
          defaultApeX += errMatrix[0][0];
          defaultApeY += errMatrix[1][1];
          nModules++;
        }
      }
    }
    *a_defaultSectorX[i_sector.first - 1] = defaultApeX / nModules;
    *a_defaultSectorY[i_sector.first - 1] = defaultApeY / nModules;
  }

  sectorNameTree->Fill();
  sectorNameTree->Write("nameTree");
  defaultTreeX->Fill();
  defaultTreeX->Write("iterTreeX");
  defaultTreeY->Fill();
  defaultTreeY->Write("iterTreeY");

  defaultFile->Close();
  delete defaultFile;
  for (unsigned int i = 0; i < a_defaultSectorX.size(); i++) {
    delete a_defaultSectorX[i];
    delete a_defaultSectorY[i];
    delete a_sectorName[i];
  }
}

// ------------ method called once each job just before starting event loop  ------------
void ApeTreeCreateDefault::beginJob() { this->sectorBuilder(); }

// ------------ method called once each job just after ending the event loop  ------------
void ApeTreeCreateDefault::endJob() {}

void ApeTreeCreateDefault::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription sector;

  std::vector<unsigned> emptyUnsignedIntVector;
  std::vector<int> emptyIntVector;
  std::vector<double> emptyDoubleVector;
  sector.add<std::string>("name", "default");
  sector.add<std::vector<unsigned>>("rawId", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("subdetId", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("layer", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("side", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("half", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("rod", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("ring", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("petal", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("blade", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("panel", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("outerInner", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("module", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("nStrips", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("isDoubleSide", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("isRPhi", emptyUnsignedIntVector);
  sector.add<std::vector<unsigned>>("isStereo", emptyUnsignedIntVector);
  sector.add<std::vector<int>>("uDirection", emptyIntVector);
  sector.add<std::vector<int>>("vDirection", emptyIntVector);
  sector.add<std::vector<int>>("wDirection", emptyIntVector);
  sector.add<std::vector<double>>("posR", emptyDoubleVector);
  sector.add<std::vector<double>>("posPhi", emptyDoubleVector);
  sector.add<std::vector<double>>("posEta", emptyDoubleVector);
  sector.add<std::vector<double>>("posX", emptyDoubleVector);
  sector.add<std::vector<double>>("posY", emptyDoubleVector);
  sector.add<std::vector<double>>("posZ", emptyDoubleVector);

  desc.add<std::string>("resultFile", "defaultAPE.root");
  desc.add<std::string>("trackerTreeFile");
  desc.addVPSet("sectors", sector);

  descriptions.add("apeTreeCreateDefault", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ApeTreeCreateDefault);
