
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "Fireworks/Eve/interface/EveService.h"

#include "Fireworks/Geometry/interface/DisplayGeomRecord.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"

#include <boost/algorithm/string/case_conv.hpp>

#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TEveGeoNode.h"
#include "TEveTrans.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TVector.h"
#include "TGLScenePad.h"
#include "TGLRnrCtx.h"
#include "TEvePointSet.h"
#include "TRandom.h"
#include "TEveUtil.h"

#include "TEveQuadSet.h"
#include "TEveStraightLineSet.h"
#include "TEveRGBAPalette.h"
#include "TSystem.h"
#include "TStyle.h"
// class decleration
//

class DisplayGeom : public edm::one::EDAnalyzer<> {
public:
  explicit DisplayGeom(const edm::ParameterSet&);
  ~DisplayGeom() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  TEveGeoTopNode* make_node(const TString& path, Int_t vis_level, Bool_t global_cs);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void endJob() override;

  edm::Service<EveService> m_eve;

  TEveElement* m_geomList;

  int m_level;

  typedef std::vector<std::string> vstring;
  vstring m_nodes;

  int m_MF_component;
  std::vector<double> m_MF_plane_d0;
  std::vector<double> m_MF_plane_d1;
  std::vector<double> m_MF_plane_d2;
  int m_MF_plane_N1;
  int m_MF_plane_N2;
  int m_MF_plane_draw_dir;
  bool m_MF_isPickable;

  edm::ESWatcher<DisplayGeomRecord> m_geomWatcher;

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_magFieldToken;
  const edm::ESGetToken<TGeoManager, DisplayGeomRecord> m_displayGeomToken;

  void remakeGeometry(const DisplayGeomRecord& dgRec);
};

DEFINE_FWK_MODULE(DisplayGeom);

DisplayGeom::DisplayGeom(const edm::ParameterSet& iConfig)
    : m_eve(),
      m_geomList(nullptr),
      m_level(iConfig.getUntrackedParameter<int>("level")),
      m_nodes(iConfig.getUntrackedParameter<vstring>("nodes")),
      m_MF_component(0),
      m_geomWatcher(this, &DisplayGeom::remakeGeometry),
      m_displayGeomToken(esConsumes()) {
  std::string component = iConfig.getUntrackedParameter<std::string>("MF_component");
  boost::algorithm::to_upper(component);

  if (component == "NONE") {
    m_MF_component = -1;
  } else if (component == "ABSBZ") {
    m_MF_component = 1;
  } else if (component == "ABSBR") {
    m_MF_component = 2;
  } else if (component == "ABSBPHI") {
    m_MF_component = 3;
  } else if (component == "BR") {
    m_MF_component = 4;
  } else if (component == "BPHI") {
    m_MF_component = 5;
  } else {  // Anything else -> |B|
    m_MF_component = 0;
  }

  if (m_MF_component != -1) {
    m_magFieldToken = esConsumes();

    m_MF_plane_d0 = iConfig.getUntrackedParameter<std::vector<double> >("MF_plane_d0");
    m_MF_plane_d1 = iConfig.getUntrackedParameter<std::vector<double> >("MF_plane_d1");
    m_MF_plane_d2 = iConfig.getUntrackedParameter<std::vector<double> >("MF_plane_d2");

    m_MF_plane_N1 = iConfig.getUntrackedParameter<int>("MF_plane_N");
    m_MF_plane_N2 = iConfig.getUntrackedParameter<int>("MF_plane_N2");
    if (m_MF_plane_N2 < 0)
      m_MF_plane_N2 = m_MF_plane_N1;

    m_MF_plane_draw_dir = iConfig.getUntrackedParameter<int>("MF_plane_draw_dir");
    m_MF_isPickable = iConfig.getUntrackedParameter<bool>("MF_pickable");
  }
}

DisplayGeom::~DisplayGeom() {}

//==============================================================================
// Protected helpers
//==============================================================================

TEveGeoTopNode* DisplayGeom::make_node(const TString& path, Int_t vis_level, Bool_t global_cs) {
  if (!gGeoManager->cd(path)) {
    Warning("make_node", "Path '%s' not found.", path.Data());
    return nullptr;
  }

  TEveGeoTopNode* tn = new TEveGeoTopNode(gGeoManager, gGeoManager->GetCurrentNode());
  tn->SetVisLevel(vis_level);
  if (global_cs) {
    tn->RefMainTrans().SetFrom(*gGeoManager->GetCurrentMatrix());
  }
  m_geomList->AddElement(tn);
  gEve->AddToListTree(tn, true);
  return tn;
}

//==============================================================================
// member functions
//==============================================================================

void DisplayGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (m_eve) {
    // Remake geometry if it has changed.
    m_geomWatcher.check(iSetup);

    if (m_MF_component != -1) {
      MagneticField const& field = iSetup.getData(m_magFieldToken);

      gStyle->SetPalette(1, nullptr);

      int minval = 0;
      int maxval = 4000;
      if (m_MF_component == 1) {  //AbsBZ
        minval = 0, maxval = 4000;
      } else if (m_MF_component == 2) {  //AbsBR
        minval = 0, maxval = 4000;
      } else if (m_MF_component == 3) {  //AbsBphi
        minval = 0, maxval = 1000;
      } else if (m_MF_component == 4) {  //BR
        minval = -4000, maxval = 4000;
      } else if (m_MF_component == 5) {  //Bphi
        minval = -200, maxval = 200;
      }

      TEveRGBAPalette* pal = new TEveRGBAPalette(minval, maxval);

      TEveStraightLineSet* ls = nullptr;
      if (m_MF_plane_draw_dir) {
        ls = new TEveStraightLineSet("MF_line_direction");
        ls->SetPickable(false);
        ls->SetLineColor(kGreen);
        ls->SetMarkerColor(kGreen);
        ls->SetMarkerStyle(1);
      }

      TEveQuadSet* q = new TEveQuadSet("MF_quad_values");
      q->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);
      q->SetOwnIds(kTRUE);
      q->SetAlwaysSecSelect(true);
      q->SetPickable(m_MF_isPickable);
      q->SetPalette(pal);

      TEveVectorD v0(m_MF_plane_d0[0], m_MF_plane_d0[1], m_MF_plane_d0[2]);
      TEveVectorD v01(m_MF_plane_d1[0], m_MF_plane_d1[1], m_MF_plane_d1[2]);
      TEveVectorD v02(m_MF_plane_d2[0], m_MF_plane_d2[1], m_MF_plane_d2[2]);

      TEveVectorD b01 = (v01 - v0);
      TEveVectorD b02 = (v02 - v0);
      TEveVectorD b03 = b01.Cross(b02);

      TEveTrans trans;
      trans.SetBaseVec(1, b01.fX, b01.fY, b01.fZ);
      trans.SetBaseVec(2, b02.fX, b02.fY, b02.fZ);
      trans.SetBaseVec(3, b03.fX, b03.fY, b03.fZ);
      trans.SetPos(v0.Arr());
      trans.OrtoNorm3();
      q->SetTransMatrix(trans.Array());

      double w_step = b01.Mag() / m_MF_plane_N1;
      double h_step = b02.Mag() / m_MF_plane_N2;

      q->SetDefWidth(w_step);
      q->SetDefHeight(h_step);
      TEveVectorD d1;
      trans.GetBaseVec(1).GetXYZ(d1);
      d1 *= w_step;
      TEveVectorD d2;
      trans.GetBaseVec(2).GetXYZ(d2);
      d2 *= h_step;

      //d1.Print();
      d2.Dump();
      double line_step_size = TMath::Min(w_step, h_step);

      for (int i = 0; i < m_MF_plane_N1; i++) {
        for (int j = 0; j < m_MF_plane_N2; j++) {
          TEveVectorD p = d1 * Double_t(i) + d2 * Double_t(j) + v0;
          GlobalPoint pos(p.fX, p.fY, p.fZ);
          GlobalVector b = field.inTesla(pos) * 1000.;  // in mT
          float value = 0.;
          if (m_MF_component == 0) {  //BMOD
            value = b.mag();
          } else if (m_MF_component == 1) {  //BZ
            value = fabs(b.z());
          } else if (m_MF_component == 2) {  //ABSBR
            value = fabs(b.x() * cos(pos.phi()) + b.y() * sin(pos.phi()));
          } else if (m_MF_component == 3) {  //ABSBPHI
            value = fabs(-b.x() * sin(pos.phi()) + b.y() * cos(pos.phi()));
          } else if (m_MF_component == 2) {  //BR
            value = b.x() * cos(pos.phi()) + b.y() * sin(pos.phi());
          } else if (m_MF_component == 5) {  //BPHI
            value = -b.x() * sin(pos.phi()) + b.y() * cos(pos.phi());
          }

          q->AddQuad(w_step * i, h_step * j);
          q->QuadValue(value);
          if (m_MF_isPickable)
            q->QuadId(new TNamed(Form("Mag (%f, %f, %f) val = %f", b.x(), b.y(), b.z(), b.mag()), "Dong!"));

          if (ls) {
            if (b.mag() > 1e-6) {
              b.unit();
              b *= line_step_size;
              ls->AddLine(p.fX, p.fY, p.fZ, p.fX + b.x(), p.fY + b.y(), p.fZ + b.z());
            } else {
              ls->AddLine(p.fX, p.fY, p.fZ, p.fX + b.x(), p.fY + b.y(), p.fZ + b.z());
            }

            ls->AddMarker(ls->GetLinePlex().Size() - 1, 0);
            ls->AddMarker(i * m_MF_plane_N1 + j, 0);
          }
        }
      }

      TEveScene* eps = gEve->SpawnNewScene("MF Map");
      gEve->GetDefaultViewer()->AddScene(eps);
      eps->GetGLScene()->SetStyle(TGLRnrCtx::kFill);
      eps->AddElement(q);
      if (ls)
        m_eve->AddElement(ls);
    } else {
      // 	 // Add a test obj
      // 	 if (!gRandom)
      // 	   gRandom = new TRandom(0);
      // 	 TRandom& r= *gRandom;

      // 	 Float_t s = 100;

      // 	 TEvePointSet* ps = new TEvePointSet();
      // 	 ps->SetOwnIds(kTRUE);
      // 	 for(Int_t i = 0; i< 100; i++)
      // 	   {
      // 	     ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      // 	     ps->SetPointId(new TNamed(Form("Point %d", i), ""));
      // 	   }

      // 	 ps->SetMarkerColor(TMath::Nint(r.Uniform(2, 9)));
      // 	 ps->SetMarkerSize(r.Uniform(1, 2));
      // 	 ps->SetMarkerStyle(4);
      // 	 m_eve->AddElement(ps);
    }
  }
  m_eve->getManager()->FullRedraw3D(true, true);
}

// ------------ method called once each job just before starting event loop  ------------
void DisplayGeom::beginJob() {
  if (m_eve) {
    m_geomList = new TEveElementList("Display Geom");
    m_eve->AddGlobalElement(m_geomList);
    //      m_eve->getManager()->GetGlobalScene()->GetGLScene()->SetStyle(TGLRnrCtx::kWireFrame);
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void DisplayGeom::endJob() {}

//------------------------------------------------------------------------------
void DisplayGeom::remakeGeometry(const DisplayGeomRecord& dgRec) {
  m_geomList->DestroyElements();

  TEveGeoManagerHolder _tgeo(const_cast<TGeoManager*>(&dgRec.get(m_displayGeomToken)));

  for (std::string& aNode : m_nodes) {
    make_node(aNode, m_level, kTRUE);
  }
}

void DisplayGeom::fillDescriptions(edm::ConfigurationDescriptions& conf) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<vstring>("nodes", vstring{"tracker:Tracker_1", "muonBase:MUON_1", "caloBase:CALO_1"})
      ->setComment("List of nodes to visualize");
  ;
  desc.addUntracked<int>("level", 4)->setComment("Levels into the geometry hierarchy visualized at startup");
  desc.addUntracked<std::string>("MF_component", "None")
      ->setComment("Component of the MF to show: 'None', 'B', 'AbsBZ', 'AbsBR', 'AbsBphi', 'BR', 'Bphi'");
  desc.addUntracked<std::vector<double> >("MF_plane_d0", std::vector<double>{0., -900., -2400.})
      ->setComment("1st corner of MF map");
  desc.addUntracked<std::vector<double> >("MF_plane_d1", std::vector<double>{0., -900., 2400.})
      ->setComment("2nd corner of MF map");
  desc.addUntracked<std::vector<double> >("MF_plane_d2", std::vector<double>{0., 900., -2400.})
      ->setComment("3rd corner of MF map");
  desc.addUntracked<int>("MF_plane_N", 200)->setComment("Number of bins for the MF map in 1st coord");
  desc.addUntracked<int>("MF_plane_N2", -1)->setComment("Number of bins for the MF map in 2nd coord");
  desc.addUntracked<int>("MF_plane_draw_dir", true)->setComment("Draw MF direction arrows (slow)");
  desc.addUntracked<bool>("MF_pickable", false)->setComment("MF values are pickable (slow)");

  conf.add("DisplayGeom", desc);
}
