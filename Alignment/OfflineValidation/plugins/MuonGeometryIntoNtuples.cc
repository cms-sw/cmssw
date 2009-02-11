// -*- C++ -*-
//
// Package:    MuonGeometryIntoNtuples
// Class:      MuonGeometryIntoNtuples
// 
/**\class MuonGeometryIntoNtuples MuonGeometryIntoNtuples.cc Alignment/MuonGeometryIntoNtuples/src/MuonGeometryIntoNtuples.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Mon Jul 16 16:56:34 CDT 2007
// $Id: MuonGeometryIntoNtuples.cc,v 1.6 2008/02/21 12:03:16 flucke Exp $
//
//

// system include files
#include <memory>
#include <map>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h" 
#include "CondFormats/Alignment/interface/AlignmentErrors.h" 
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorRcd.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

//
// class decleration
//

class MuonGeometryIntoNtuples : public edm::EDAnalyzer {
   public:
      explicit MuonGeometryIntoNtuples(const edm::ParameterSet&);
      ~MuonGeometryIntoNtuples();


   private:
      virtual void beginJob(const edm::EventSetup &iSetup);
      virtual void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
      virtual void endJob() ;

      void getOriginals(Alignable *ali);
      void addBranches(TTree* tree);
      void recursiveWalk(Alignable *ali);
      int getId(Alignable *ali);

      // ----------member data ---------------------------

      AlignableObjectId m_alignableObjectId;
      enum { kMaxChildren = 256 };

      std::map<int, GlobalPoint> m_originalPositions;
      std::map<int, align::RotationType> m_originalRotations;

      bool m_dtApplyAlignment, m_cscApplyAlignment;
      bool m_dtFromSurvey, m_cscFromSurvey;
      std::string m_dtLabel, m_cscLabel;
      bool m_doMisalignmentScenario;
      edm::ParameterSet m_misalignmentScenario;

      bool m_book_dtWheels, m_book_dtStations, m_book_dtChambers, m_book_dtSuperLayers, m_book_dtLayers,
	 m_book_cscStations, m_book_cscChambers, m_book_cscLayers;
      
      TTree *m_dtWheels, *m_dtStations, *m_dtChambers, *m_dtSuperLayers, *m_dtLayers,
	 *m_cscStations, *m_cscChambers, *m_cscLayers;

      Int_t m_rawid;
      Char_t m_structa, m_structb, m_structc, m_slayer, m_layer;
      Float_t m_x, m_y, m_z;
      Float_t m_xhatx, m_xhaty, m_xhatz, m_yhatx, m_yhaty, m_yhatz, m_zhatx, m_zhaty, m_zhatz;
      Float_t m_dx, m_dy, m_dz;
      Float_t m_dxx, m_dxy, m_dxz, m_dyx, m_dyy, m_dyz, m_dzx, m_dzy, m_dzz;
      Int_t m_parent, m_numChildren, m_children[kMaxChildren];
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
MuonGeometryIntoNtuples::MuonGeometryIntoNtuples(const edm::ParameterSet& iConfig)
{
   m_dtApplyAlignment = iConfig.getParameter<bool>("DTApplyAlignment");
   m_cscApplyAlignment = iConfig.getParameter<bool>("CSCApplyAlignment");
   m_dtFromSurvey = iConfig.getParameter<bool>("DTFromSurveyRcd");
   m_cscFromSurvey = iConfig.getParameter<bool>("CSCFromSurveyRcd");
   m_dtLabel = iConfig.getParameter<std::string>("DTAlignmentLabel");
   m_cscLabel = iConfig.getParameter<std::string>("CSCAlignmentLabel");
   m_doMisalignmentScenario = iConfig.getParameter<bool>("ApplyMisalignmentScenario");
   m_misalignmentScenario = iConfig.getParameter<edm::ParameterSet>("MisalignmentScenario");

   m_book_dtWheels = iConfig.getUntrackedParameter<bool>("DTWheels", true);
   m_book_dtStations = iConfig.getUntrackedParameter<bool>("DTStations", true);
   m_book_dtChambers = iConfig.getUntrackedParameter<bool>("DTChambers", true);
   m_book_dtSuperLayers = iConfig.getUntrackedParameter<bool>("DTSuperLayers", true);
   m_book_dtLayers = iConfig.getUntrackedParameter<bool>("DTLayers", true);
   m_book_cscStations = iConfig.getUntrackedParameter<bool>("CSCStations", true);
   m_book_cscChambers = iConfig.getUntrackedParameter<bool>("CSCChambers", true);
   m_book_cscLayers = iConfig.getUntrackedParameter<bool>("CSCLayers", true);
}


MuonGeometryIntoNtuples::~MuonGeometryIntoNtuples()
{}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MuonGeometryIntoNtuples::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{}


// ------------ method called once each job just before starting event loop  ------------
void 
MuonGeometryIntoNtuples::beginJob(const edm::EventSetup& iSetup)
{
   edm::LogWarning("MuonGeometryIntoNtuples") << "begin";

   // Build ideal geometry
   edm::ESHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get(cpv);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   DTGeometryBuilderFromDDD DTGeometryBuilder;
   CSCGeometryBuilderFromDDD CSCGeometryBuilder;

   // Before 1_7_X
//    DTGeometry *dtGeometry = DTGeometryBuilder.build(&(*cpv), *mdc);
//    CSCGeometry *cscGeometry = CSCGeometryBuilder.build(&(*cpv), *mdc);

   // 1_7_X and later
   DTGeometry *dtGeometry = DTGeometryBuilder.build(&(*cpv), *mdc);
   boost::shared_ptr<CSCGeometry> boost_cscGeometry;
   CSCGeometryBuilder.build(boost_cscGeometry, &(*cpv), *mdc);
   CSCGeometry *cscGeometry = &(*boost_cscGeometry);

   edm::LogWarning("MuonGeometryIntoNtuples") << "built ideal geometry";

   AlignableMuon originals(dtGeometry, cscGeometry);
   getOriginals(&originals);

   edm::LogWarning("MuonGeometryIntoNtuples") << "recorded original positions/orientations";

   // Apply a (mis)aligned geometry

   GeometryAligner aligner;
   edm::ESHandle<Alignments> dtAlignments, cscAlignments;
   edm::ESHandle<AlignmentErrors> dtAlignmentErrors, cscAlignmentErrors;

   edm::ESHandle<Alignments> globalPositionRcd;
   iSetup.get<MuonGeometryRecord>().getRecord<GlobalPositionRcd>().get(globalPositionRcd);

   if (m_dtApplyAlignment) {
      if (m_dtFromSurvey) {
	 edm::LogWarning("MuonGeometryIntoNtuples") << "applying (mis)alignment from DTSurveyRcd \"" << m_dtLabel << "\"";
	 iSetup.get<DTSurveyRcd>().get(m_dtLabel, dtAlignments);
	 edm::ESHandle<SurveyErrors> surveyErrors;
	 iSetup.get<DTSurveyErrorRcd>().get(m_dtLabel, surveyErrors);

	 AlignmentErrors alignmentErrors;
	 for (std::vector<SurveyError>::const_iterator iter = surveyErrors->m_surveyErrors.begin();  iter != surveyErrors->m_surveyErrors.end();  ++iter) {
	    align::ErrorMatrix matrix6by6 = iter->matrix();  // start from 0,0
	    CLHEP::HepSymMatrix matrix3by3(3);               // start from 1,1
	    for (int i = 0;  i < 3;  i++) {
	       for (int j = 0;  j < 3;  j++) {
		  matrix3by3(i+1, j+1) = matrix6by6(i, j);
	       }
	    }
	    alignmentErrors.m_alignError.push_back(AlignTransformError(matrix3by3, iter->rawId()));
	 }
	 aligner.applyAlignments<DTGeometry>(dtGeometry, &(*dtAlignments), &alignmentErrors,
					     align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
      }
      else {
	 edm::LogWarning("MuonGeometryIntoNtuples") << "applying (mis)alignment from DTAlignmentRcd \"" << m_dtLabel << "\"";
	 iSetup.get<DTAlignmentRcd>().get(m_dtLabel, dtAlignments);
	 iSetup.get<DTAlignmentErrorRcd>().get(m_dtLabel, dtAlignmentErrors);
	 aligner.applyAlignments<DTGeometry>(dtGeometry, &(*dtAlignments), &(*dtAlignmentErrors),
					     align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
      }
      edm::LogWarning("MuonGeometryIntoNtuples") << "done!";
   }
   else {
      edm::LogWarning("MuonGeometryIntoNtuples") << "NOT applying a DT alignment or misalignment";
   }

   if (m_cscApplyAlignment) {
      if (m_cscFromSurvey) {
	 edm::LogWarning("MuonGeometryIntoNtuples") << "applying (mis)alignment from CSCSurveyRcd \"" << m_cscLabel << "\"";
	 iSetup.get<CSCSurveyRcd>().get(m_cscLabel, cscAlignments);
	 edm::ESHandle<SurveyErrors> surveyErrors;
	 iSetup.get<CSCSurveyErrorRcd>().get(m_cscLabel, surveyErrors);

	 AlignmentErrors alignmentErrors;
	 for (std::vector<SurveyError>::const_iterator iter = surveyErrors->m_surveyErrors.begin();  iter != surveyErrors->m_surveyErrors.end();  ++iter) {
	    align::ErrorMatrix matrix6by6 = iter->matrix();  // start from 0,0
	    CLHEP::HepSymMatrix matrix3by3(3);               // start from 1,1
	    for (int i = 0;  i < 3;  i++) {
	       for (int j = 0;  j < 3;  j++) {
		  matrix3by3(i+1, j+1) = matrix6by6(i, j);
	       }
	    }
	    alignmentErrors.m_alignError.push_back(AlignTransformError(matrix3by3, iter->rawId()));
	 }
	 aligner.applyAlignments<CSCGeometry>(cscGeometry, &(*cscAlignments), &alignmentErrors,
					     align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
      }
      else {
	 edm::LogWarning("MuonGeometryIntoNtuples") << "applying (mis)alignment from CSCAlignmentRcd \"" << m_cscLabel << "\"";
	 iSetup.get<CSCAlignmentRcd>().get(m_cscLabel, cscAlignments);
	 iSetup.get<CSCAlignmentErrorRcd>().get(m_cscLabel, cscAlignmentErrors);
	 aligner.applyAlignments<CSCGeometry>(cscGeometry, &(*cscAlignments), &(*cscAlignmentErrors),
					     align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
      }
      edm::LogWarning("MuonGeometryIntoNtuples") << "done!";
   }
   else {
      edm::LogWarning("MuonGeometryIntoNtuples") << "NOT applying a CSC alignment or misalignment";
   }

   // Create an alignable geometry (I'm more familiar with this version)
   AlignableMuon alignableMuon(dtGeometry, cscGeometry);
   edm::LogWarning("MuonGeometryIntoNtuples") << "created AlignableMuon tree";

   if (m_doMisalignmentScenario) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "applying the MisalignmentScenario";
      MuonScenarioBuilder muonScenarioBuilder(&alignableMuon);
      muonScenarioBuilder.applyScenario(m_misalignmentScenario);
      edm::LogWarning("MuonGeometryIntoNtuples") << "done!";
   }
   else {
      edm::LogWarning("MuonGeometryIntoNtuples") << "NOT applying any MisalignmentScenario";
   }

   // Prepare the output
   edm::Service<TFileService> tfile;

   m_dtWheels = m_dtStations = m_dtChambers = m_dtSuperLayers = m_dtLayers = m_cscStations = m_cscChambers = m_cscLayers = NULL;

   if (m_book_dtWheels) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "booking DTWheels TTree";
      m_dtWheels      = tfile->make<TTree>("DTWheels", "DTWheels");
      addBranches(m_dtWheels);
   }
   if (m_book_dtStations) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "booking DTStations TTree";
      m_dtStations    = tfile->make<TTree>("DTStations", "DTStations");
      addBranches(m_dtStations);
   }
   if (m_book_dtChambers) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "booking DTChambers TTree";
      m_dtChambers    = tfile->make<TTree>("DTChambers", "DTChambers");
      addBranches(m_dtChambers);
   }
   if (m_book_dtSuperLayers) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "booking DTSuperLayers TTree";
      m_dtSuperLayers = tfile->make<TTree>("DTSuperLayers", "DTSuperLayers");
      addBranches(m_dtSuperLayers);
   }
   if (m_book_dtLayers) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "booking DTLayers TTree";
      m_dtLayers      = tfile->make<TTree>("DTLayers", "DTLayers");
      addBranches(m_dtLayers);
   }
   if (m_book_cscStations) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "booking CSCStations TTree";
      m_cscStations   = tfile->make<TTree>("CSCStations", "CSCStations");
      addBranches(m_cscStations);
   }
   if (m_book_cscChambers) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "booking CSCChambers TTree";
      m_cscChambers   = tfile->make<TTree>("CSCChambers", "CSCChambers");
      addBranches(m_cscChambers);
   }
   if (m_book_cscLayers) {
      edm::LogWarning("MuonGeometryIntoNtuples") << "booking CSCLayers TTree";
      m_cscLayers     = tfile->make<TTree>("CSCLayers", "CSCLayers");
      addBranches(m_cscLayers);
   }

   edm::LogWarning("MuonGeometryIntoNtuples") << "filling ntuples";

   // Write it out
   recursiveWalk(&alignableMuon);

   edm::LogWarning("MuonGeometryIntoNtuples") << "done with everything!";
}

void MuonGeometryIntoNtuples::getOriginals(Alignable *ali) {
   int id = getId(ali) * 100 + ali->alignableObjectId();
   m_originalPositions[id] = ali->globalPosition();
   m_originalRotations[id] = ali->globalRotation();

   std::vector<Alignable*> components = ali->components();
   for (std::vector<Alignable*>::const_iterator iter = components.begin();  iter != components.end();  ++iter) {
      getOriginals(*iter);
   }
}

void MuonGeometryIntoNtuples::addBranches(TTree* tree) {
   // The branches are nearly the same on all eight TTrees

   tree->Branch("rawid", &m_rawid, "rawid/I");
   tree->Branch("structa", &m_structa, "structa/B");
   tree->Branch("structb", &m_structb, "structb/B");
   tree->Branch("structc", &m_structc, "structc/B");
   tree->Branch("slayer", &m_slayer, "slayer/B");
   tree->Branch("layer", &m_layer, "layer/B");

   tree->Branch("x", &m_x, "x/F");
   tree->Branch("y", &m_y, "y/F");
   tree->Branch("z", &m_z, "z/F");

   tree->Branch("xhatx", &m_xhatx, "xhatx/F");
   tree->Branch("xhaty", &m_xhaty, "xhaty/F");
   tree->Branch("xhatz", &m_xhatz, "xhatz/F");
   tree->Branch("yhatx", &m_yhatx, "yhatx/F");
   tree->Branch("yhaty", &m_yhaty, "yhaty/F");
   tree->Branch("yhatz", &m_yhatz, "yhatz/F");
   tree->Branch("zhatx", &m_zhatx, "zhatx/F");
   tree->Branch("zhaty", &m_zhaty, "zhaty/F");
   tree->Branch("zhatz", &m_zhatz, "zhatz/F");

   tree->Branch("dx", &m_dx, "dx/F");
   tree->Branch("dy", &m_dy, "dy/F");
   tree->Branch("dz", &m_dz, "dz/F");

   tree->Branch("dxx", &m_dxx, "dxx/F");
   tree->Branch("dxy", &m_dxy, "dxy/F");
   tree->Branch("dxz", &m_dxz, "dxz/F");
   tree->Branch("dyx", &m_dyx, "dyx/F");
   tree->Branch("dyy", &m_dyy, "dyy/F");
   tree->Branch("dyz", &m_dyz, "dyz/F");
   tree->Branch("dzx", &m_dzx, "dzx/F");
   tree->Branch("dzy", &m_dzy, "dzy/F");
   tree->Branch("dzz", &m_dzz, "dzz/F");

   tree->Branch("parent", &m_parent, "parent/I");
   tree->Branch("numChildren", &m_numChildren, "numChildren/I");
   tree->Branch("children", &m_children, "children[numChildren]/I");
}

int MuonGeometryIntoNtuples::getId(Alignable *ali) {
   if (ali == NULL) return 0;

   int rawid = ali->geomDetId().rawId();
   while (rawid == 0) {
      std::vector<Alignable*> components = ali->components();
      if (components.size() == 0) {
	 return rawid;
      }
      else {
	 ali = components[0];
	 rawid = ali->geomDetId().rawId();
      }
   }
   return rawid;
}

void MuonGeometryIntoNtuples::recursiveWalk(Alignable *ali) {
   edm::LogInfo("MuonGeometryIntoNtuples") << "filling ntuple for " << m_alignableObjectId.typeToName(ali->alignableObjectId()) << " " << getId(ali);

   // Depth first, so that you can skip the largest structures without
   // skipping all of their components
   std::vector<Alignable*> components = ali->components();
   for (std::vector<Alignable*>::const_iterator iter = components.begin();  iter != components.end();  ++iter) {
      recursiveWalk(*iter);
   }

   // Get a pointer to whatever ntuple we're filling (they're all so similar)
   TTree *tree;
   if (ali->alignableObjectId() == align::AlignableDTWheel) tree = m_dtWheels;
   else if (ali->alignableObjectId() == align::AlignableDTStation) tree = m_dtStations;
   else if (ali->alignableObjectId() == align::AlignableDTChamber) tree = m_dtChambers;
   else if (ali->alignableObjectId() == align::AlignableDTSuperLayer  ||
	    (ali->alignableObjectId() == align::AlignableDet  &&  ali->geomDetId().subdetId() == MuonSubdetId::DT)) tree = m_dtSuperLayers;
   else if (ali->alignableObjectId() == align::AlignableDTLayer  ||
	    (ali->alignableObjectId() == align::AlignableDetUnit  &&  ali->geomDetId().subdetId() == MuonSubdetId::DT)) tree = m_dtLayers;
   else if (ali->alignableObjectId() == align::AlignableCSCStation) tree = m_cscStations;
   else if (ali->alignableObjectId() == align::AlignableCSCChamber) tree = m_cscChambers;
   else if (ali->alignableObjectId() == align::AlignableCSCLayer  ||
	    ((ali->alignableObjectId() == align::AlignableDet  ||
	      ali->alignableObjectId() == align::AlignableDetUnit)  &&  ali->geomDetId().subdetId() == MuonSubdetId::CSC)) tree = m_cscLayers;
   else if (ali->alignableObjectId() == align::AlignableDTBarrel  ||
	    ali->alignableObjectId() == align::AlignableCSCEndcap  ||
	    ali->alignableObjectId() == align::AlignableMuon) {
      // skip these
      return;
   }
   else {
      throw cms::Exception("BadAssociation") << "Alignable in AlignableMuon has an unidentified type: " << ali->alignableObjectId() << std::endl;
   }

   if (tree == NULL) return;  // maybe I've chosen not to book this one

   // get a surface and start filling the ntuple entries
   AlignableSurface surface = ali->surface();

   m_rawid = getId(ali);

   if (DetId(m_rawid).subdetId() == MuonSubdetId::DT) {
      if (tree == m_dtLayers) {
	 DTLayerId id(m_rawid);
	 m_structa = id.wheel();
	 m_structb = id.station();
	 m_structc = id.sector();
	 m_slayer = id.superLayer();
	 m_layer = id.layer();
      }
      else if (tree == m_dtSuperLayers) {
	 DTSuperLayerId id(m_rawid);
	 m_structa = id.wheel();
	 m_structb = id.station();
	 m_structc = id.sector();
	 m_slayer = id.superLayer();
	 m_layer = 127;
      }
      else {
	 DTChamberId id(m_rawid);
	 m_structa = id.wheel();
	 m_structb = id.station();
	 m_structc = id.sector();
	 m_slayer = 127;
	 m_layer = 127;
      }
   }
   else if (DetId(m_rawid).subdetId() == MuonSubdetId::CSC) {
      CSCDetId id(m_rawid);
      m_structa = (id.endcap() == 1? 1: -1)*id.station();
      m_structb = id.ring();
      m_structc = id.chamber();
      m_slayer = 127;
      m_layer = (id.layer() == 0? 127: id.layer());
   }
   else {
      throw cms::Exception("BadAssociation") << "This alignable is neither a DT nor a CSC: " << m_rawid << std::endl;
   }

   GlobalPoint position = surface.toGlobal(LocalPoint(0., 0., 0.));
   m_x = position.x();
   m_y = position.y();
   m_z = position.z();

   GlobalVector xhat = surface.toGlobal(LocalVector(1., 0., 0.));
   GlobalVector yhat = surface.toGlobal(LocalVector(0., 1., 0.));
   GlobalVector zhat = surface.toGlobal(LocalVector(0., 0., 1.));
   m_xhatx = xhat.x();
   m_xhaty = xhat.y();
   m_xhatz = xhat.z();
   m_yhatx = yhat.x();
   m_yhaty = yhat.y();
   m_yhatz = yhat.z();
   m_zhatx = zhat.x();
   m_zhaty = zhat.y();
   m_zhatz = zhat.z();

   // get original position/orientation
   int id = m_rawid * 100 + ali->alignableObjectId();
   GlobalPoint originalPosition = m_originalPositions[id];
   align::RotationType originalRotation = m_originalRotations[id];

   // subtract off the ideal position (left to right)
   GlobalVector displacement = ali->globalPosition() - originalPosition;
   m_dx = displacement.x();
   m_dy = displacement.y();
   m_dz = displacement.z();
   
   // rotate off the ideal orientation (right to left: matrix multiplication)
   align::RotationType rotation = originalRotation.multiplyInverse(ali->globalRotation());
   m_dxx = rotation.xx();
   m_dxy = rotation.xy();
   m_dxz = rotation.xz();
   m_dyx = rotation.yx();
   m_dyy = rotation.yy();
   m_dyz = rotation.yz();
   m_dzx = rotation.zx();
   m_dzy = rotation.zy();
   m_dzz = rotation.zz();

   m_parent = getId(ali->mother());
   m_numChildren = 0;
   for (std::vector<Alignable*>::const_iterator iter = components.begin();  iter != components.end();  ++iter) {
      m_children[m_numChildren] = getId(*iter);
      m_numChildren++;

      if (m_numChildren > kMaxChildren) {
	 throw cms::Exception("BadAssociation") << "More than " << (m_numChildren-1) << " descend from a single alignable." << std::endl;
      }
   }

   // Pad it for safety...
   for (int i = m_numChildren;  i < kMaxChildren;  i++) {
      m_children[i] = 0;
   }

   // Fill 'er up
   tree->Fill();
   return;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonGeometryIntoNtuples::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGeometryIntoNtuples);
