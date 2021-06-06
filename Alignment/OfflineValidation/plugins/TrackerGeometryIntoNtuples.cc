// -*- C++ -*-
//
// Package:    TrackerGeometryIntoNtuples
// Class:      TrackerGeometryIntoNtuples
//
/**\class TrackerGeometryIntoNtuples TrackerGeometryIntoNtuples.cc 
 
 Description: Takes a set of alignment constants and turns them into a ROOT file
 
 Implementation:
 <Notes on implementation>
 */
//
// Original class TrackerGeometryIntoNtuples.cc
// Original Author:  Nhan Tran
//         Created:  Mon Jul 16m 16:56:34 CDT 2007
// $Id: TrackerGeometryIntoNtuples.cc,v 1.14 2012/12/02 22:13:12 devdatta Exp $
//
// 26 May 2012
// ***********
// *********** Modified to add tracker module surface deformations ***********
//
//

// system include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <algorithm>
#include "TTree.h"
#include "TFile.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"

#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"

#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "Geometry/CommonTopologies/interface/GeometryAligner.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// To access kinks and bows
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

#include "CLHEP/Matrix/SymMatrix.h"

//
// class decleration
//

class TrackerGeometryIntoNtuples : public edm::one::EDAnalyzer<> {
public:
  explicit TrackerGeometryIntoNtuples(const edm::ParameterSet&);
  ~TrackerGeometryIntoNtuples() override;

private:
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void addBranches();

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> aliToken_;
  const edm::ESGetToken<AlignmentErrorsExtended, TrackerAlignmentErrorExtendedRcd> aliErrorToken_;
  const edm::ESGetToken<AlignmentSurfaceDeformations, TrackerSurfaceDeformationRcd> surfDefToken_;
  const edm::ESGetToken<Alignments, GlobalPositionRcd> gprToken_;

  //std::vector<AlignTransform> m_align;
  AlignableTracker* theCurrentTracker;

  uint32_t m_rawid;
  double m_x, m_y, m_z;
  double m_alpha, m_beta, m_gamma;
  int m_subdetid;
  double m_xx, m_xy, m_yy, m_xz, m_yz, m_zz;
  int m_dNpar;
  double m_d1, m_d2, m_d3;
  int m_dtype;
  //std::vector<double>m_dpar;
  std::vector<double>* mp_dpar;

  // Deformation parameters: stored in same tree as the alignment parameters
  UInt_t numDeformationValues_;
  enum { kMaxNumPar = 20 };  // slighly above 'two bowed surfaces' limit
  Float_t deformationValues_[kMaxNumPar];

  TTree* m_tree;
  TTree* m_treeDeformations;
  TTree* m_treeErrors;
  std::string m_outputFile;
  std::string m_outputTreename;
  TFile* m_file;
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
TrackerGeometryIntoNtuples::TrackerGeometryIntoNtuples(const edm::ParameterSet& iConfig)
    : topoToken_(esConsumes()),
      geomDetToken_(esConsumes()),
      ptpToken_(esConsumes()),
      aliToken_(esConsumes()),
      aliErrorToken_(esConsumes()),
      surfDefToken_(esConsumes()),
      gprToken_(esConsumes()),
      theCurrentTracker(nullptr),
      m_rawid(0),
      m_x(0.),
      m_y(0.),
      m_z(0.),
      m_alpha(0.),
      m_beta(0.),
      m_gamma(0.),
      m_subdetid(0),
      m_xx(0.),
      m_xy(0.),
      m_yy(0.),
      m_xz(0.),
      m_yz(0.),
      m_zz(0.),
      m_dNpar(0),
      m_d1(0.),
      m_d2(0.),
      m_d3(0.),
      m_dtype(0),
      mp_dpar(nullptr) {
  m_outputFile = iConfig.getUntrackedParameter<std::string>("outputFile");
  m_outputTreename = iConfig.getUntrackedParameter<std::string>("outputTreename");
  m_file = new TFile(m_outputFile.c_str(), "RECREATE");
  m_tree = new TTree(m_outputTreename.c_str(), m_outputTreename.c_str());
  m_treeDeformations = new TTree("alignTreeDeformations", "alignTreeDeformations");
  //char errorTreeName[256];
  //snprintf(errorTreeName, sizeof(errorTreeName), "%sErrors", m_outputTreename);
  //m_treeErrors = new TTree(errorTreeName,errorTreeName);
  m_treeErrors = new TTree("alignTreeErrors", "alignTreeErrors");
}

TrackerGeometryIntoNtuples::~TrackerGeometryIntoNtuples() { delete theCurrentTracker; }

//
// member functions
//

// ------------ method called to for each event  ------------
void TrackerGeometryIntoNtuples::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(topoToken_);

  edm::LogInfo("beginJob") << "Begin Job";

  //accessing the initial geometry
  const GeometricDet* theGeometricDet = &iSetup.getData(geomDetToken_);
  const PTrackerParameters* ptp = &iSetup.getData(ptpToken_);

  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  //currernt tracker
  TrackerGeometry* theCurTracker = trackerBuilder.build(theGeometricDet, *ptp, tTopo);

  //build the tracker
  const Alignments* alignments = &iSetup.getData(aliToken_);
  const AlignmentErrorsExtended* alignmentErrors = &iSetup.getData(aliErrorToken_);
  const AlignmentSurfaceDeformations* surfaceDeformations = &iSetup.getData(surfDefToken_);

  //apply the latest alignments
  const Alignments* globalPositionRcd = &iSetup.getData(gprToken_);
  GeometryAligner aligner;
  aligner.applyAlignments<TrackerGeometry>(&(*theCurTracker),
                                           alignments,
                                           alignmentErrors,
                                           align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Tracker)));
  aligner.attachSurfaceDeformations<TrackerGeometry>(&(*theCurTracker), &(*surfaceDeformations));

  theCurrentTracker = new AlignableTracker(&(*theCurTracker), tTopo);

  Alignments* theAlignments = theCurrentTracker->alignments();
  //AlignmentErrorsExtended* theAlignmentErrorsExtended = theCurrentTracker->alignmentErrors();

  //alignments
  addBranches();
  for (std::vector<AlignTransform>::const_iterator i = theAlignments->m_align.begin();
       i != theAlignments->m_align.end();
       ++i) {
    m_rawid = i->rawId();
    CLHEP::Hep3Vector translation = i->translation();
    m_x = translation.x();
    m_y = translation.y();
    m_z = translation.z();

    CLHEP::HepRotation rotation = i->rotation();
    m_alpha = rotation.getPhi();
    m_beta = rotation.getTheta();
    m_gamma = rotation.getPsi();
    m_tree->Fill();

    //DetId detid(m_rawid);
    //if (detid.subdetId() > 2){
    //
    //std::cout << " panel: " << tTopo->pxfPanel( m_rawid ) << ", module: " << tTopo->pxfModule( m_rawid ) << std::endl;
    //if ((tTopo->pxfPanel( m_rawid ) == 1) && (tTopo->pxfModule( m_rawid ) == 4)) std::cout << m_rawid << ", ";
    //std::cout << m_rawid << std::setprecision(9) <<  " " << m_x << " " << m_y << " " << m_z;
    //std::cout << std::setprecision(9) << " " << m_alpha << " " << m_beta << " " << m_gamma << std::endl;
    //}
  }

  delete theAlignments;

  std::vector<AlignTransformErrorExtended> alignErrors = alignmentErrors->m_alignError;
  for (std::vector<AlignTransformErrorExtended>::const_iterator i = alignErrors.begin(); i != alignErrors.end(); ++i) {
    m_rawid = i->rawId();
    CLHEP::HepSymMatrix errMatrix = i->matrix();
    DetId detid(m_rawid);
    m_subdetid = detid.subdetId();
    m_xx = errMatrix[0][0];
    m_xy = errMatrix[0][1];
    m_xz = errMatrix[0][2];
    m_yy = errMatrix[1][1];
    m_yz = errMatrix[1][2];
    m_zz = errMatrix[2][2];
    m_treeErrors->Fill();
  }

  // Get GeomDetUnits for the current tracker
  auto const& detUnits = theCurTracker->detUnits();
  int detUnit(0);
  //\\for (unsigned int iDet = 0; iDet < detUnits.size(); ++iDet) {
  for (auto iunit = detUnits.begin(); iunit != detUnits.end(); ++iunit) {
    DetId detid = (*iunit)->geographicalId();
    m_rawid = detid.rawId();
    m_subdetid = detid.subdetId();

    ++detUnit;
    //\\GeomDetUnit* geomDetUnit = detUnits.at(iDet) ;
    auto geomDetUnit = *iunit;

    // Get SurfaceDeformation for this GeomDetUnit
    if (geomDetUnit->surfaceDeformation()) {
      std::vector<double> surfaceDeformParams = (geomDetUnit->surfaceDeformation())->parameters();
      //edm::LogInfo("surfaceDeformParamsSize") << " surfaceDeformParams size  = " << surfaceDeformParams.size() << std::endl ;
      m_dNpar = surfaceDeformParams.size();
      m_dtype = (geomDetUnit->surfaceDeformation())->type();
      m_d1 = surfaceDeformParams.at(0);
      m_d2 = surfaceDeformParams.at(1);
      m_d3 = surfaceDeformParams.at(2);
      mp_dpar->clear();
      for (std::vector<double>::const_iterator it = surfaceDeformParams.begin(); it != surfaceDeformParams.end();
           ++it) {
        mp_dpar->push_back((*it));
        //edm::LogInfo("surfaceDeformParamsContent") << " surfaceDeformParam = " << (*it) << std::endl ;
      }
      m_treeDeformations->Fill();
    }
  }

  //write out
  m_file->cd();
  m_tree->Write();
  m_treeDeformations->Write();
  m_treeErrors->Write();
  m_file->Close();
}

void TrackerGeometryIntoNtuples::addBranches() {
  m_tree->Branch("rawid", &m_rawid, "rawid/I");
  m_tree->Branch("x", &m_x, "x/D");
  m_tree->Branch("y", &m_y, "y/D");
  m_tree->Branch("z", &m_z, "z/D");
  m_tree->Branch("alpha", &m_alpha, "alpha/D");
  m_tree->Branch("beta", &m_beta, "beta/D");
  m_tree->Branch("gamma", &m_gamma, "gamma/D");

  m_treeDeformations->Branch("irawid", &m_rawid, "irawid/I");
  m_treeDeformations->Branch("subdetid", &m_subdetid, "subdetid/I");
  m_treeDeformations->Branch("dNpar", &m_dNpar, "dNpar/I");
  //m_treeDeformations->Branch("d1", &m_d1, "d1/D");
  //m_treeDeformations->Branch("d2", &m_d2, "d2/D");
  //m_treeDeformations->Branch("d3", &m_d3, "d3/D");
  m_treeDeformations->Branch("dtype", &m_dtype);
  m_treeDeformations->Branch("dpar", "std::vector<double>", &mp_dpar);

  m_treeErrors->Branch("rawid", &m_rawid, "rawid/I");
  m_treeErrors->Branch("subdetid", &m_subdetid, "subdetid/I");
  m_treeErrors->Branch("xx", &m_xx, "xx/D");
  m_treeErrors->Branch("yy", &m_yy, "yy/D");
  m_treeErrors->Branch("zz", &m_zz, "zz/D");
  m_treeErrors->Branch("xy", &m_xy, "xy/D");
  m_treeErrors->Branch("xz", &m_xz, "xz/D");
  m_treeErrors->Branch("yz", &m_yz, "yz/D");

  //m_tree->Branch("NumDeform",    &numDeformationValues_, "NumDeform/i");
  //m_tree->Branch("DeformValues", deformationValues_,     "DeformValues[NumDeform]/F");
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerGeometryIntoNtuples);
