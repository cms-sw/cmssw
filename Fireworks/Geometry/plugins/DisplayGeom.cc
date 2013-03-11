
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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

#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TEveGeoNode.h"
#include "TEveTrans.h"
#include "TEveScene.h"
#include "TVector.h"
#include "TGLScenePad.h"
#include "TGLRnrCtx.h"
#include "TEvePointSet.h"
#include "TRandom.h"
#include "TEveUtil.h"

#include "TEveQuadSet.h"
#include "TEveStraightLineSet.h"
#include "TEveRGBAPAlette.h"
#include "TSystem.h"
#include "TStyle.h"
// class decleration
//
       
class DisplayGeom : public edm::EDAnalyzer {

public:
   explicit DisplayGeom(const edm::ParameterSet&);
   ~DisplayGeom();

protected:
   TEveGeoTopNode* make_node(const TString& path, Int_t vis_level, Bool_t global_cs);

  
private:
   virtual void beginJob() ;
   virtual void analyze(const edm::Event&, const edm::EventSetup&);

   virtual void endJob() ;

   edm::Service<EveService>  m_eve;

   TEveElement   *m_geomList;

   int m_level;

   bool m_MF;
   std::vector<double> m_MF_plane_d0;
   std::vector<double> m_MF_plane_d1;
   std::vector<double> m_MF_plane_d2;
   int m_MF_plane_N;
   int m_MF_plane_draw_dir;

   edm::ESWatcher<DisplayGeomRecord> m_geomWatcher;

   void remakeGeometry(const DisplayGeomRecord& dgRec);

};

DEFINE_FWK_MODULE(DisplayGeom);

DisplayGeom::DisplayGeom(const edm::ParameterSet& iConfig):
   m_eve(),
   m_geomList(0),
   m_geomWatcher(this, &DisplayGeom::remakeGeometry)
{
   m_level =  iConfig.getUntrackedParameter<int>( "level", 2);

   m_MF    =  iConfig.getUntrackedParameter<int>( "MF", false);
   m_MF_plane_d0 = iConfig.getUntrackedParameter< std::vector<double> >("MF_plane_d0",  std::vector<double>(3, 0.0));
   m_MF_plane_d1 = iConfig.getParameter< std::vector<double> >("MF_plane_d1");
   m_MF_plane_d2 = iConfig.getParameter< std::vector<double> >("MF_plane_d2");

   printf("%f %f %f \n ", m_MF_plane_d1[0],  m_MF_plane_d1[1], m_MF_plane_d1[2]);
   printf("%f %f %f \n ", m_MF_plane_d2[0],  m_MF_plane_d2[1], m_MF_plane_d2[2]);
   m_MF_plane_N  = iConfig.getUntrackedParameter<UInt_t>( "MF_plane_N", 100);
   m_MF_plane_draw_dir = iConfig.getUntrackedParameter<int>( "MF_plane_draw_dir", true);
}


DisplayGeom::~DisplayGeom()
{
}


//==============================================================================
// Protected helpers
//==============================================================================

TEveGeoTopNode* DisplayGeom::make_node(const TString& path, Int_t vis_level, Bool_t global_cs)
{
   if (! gGeoManager->cd(path))
   {
      Warning("make_node", "Path '%s' not found.", path.Data());
      return 0;
   }

   TEveGeoTopNode* tn = new TEveGeoTopNode(gGeoManager, gGeoManager->GetCurrentNode());
   tn->SetVisLevel(vis_level);
   if (global_cs)
   {
      tn->RefMainTrans().SetFrom(*gGeoManager->GetCurrentMatrix());
   }
   m_geomList->AddElement(tn);
   gEve->AddToListTree(tn, true);
   return tn;
}


//==============================================================================
// member functions
//==============================================================================


void
DisplayGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if (m_eve)
   {
      // Remake geometry if it has changed.
     m_geomWatcher.check(iSetup);

     if (m_MF)
       {
          edm::ESHandle<MagneticField> field;
         iSetup.get<IdealMagneticFieldRecord>().get( field );

	 gStyle->SetPalette(1, 0);
	 TEveRGBAPalette* pal = new TEveRGBAPalette(0, 400);

	 TEveStraightLineSet* ls = new TEveStraightLineSet("MF_line_direction");
         ls->SetPickable(false);
	 ls->SetLineColor(kGreen);
	 ls->SetMarkerColor(kGreen);
	 ls->SetMarkerStyle(1);

	 TEveQuadSet* q = new TEveQuadSet("MF_quad_values");
         q->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);
	 q->SetOwnIds(kTRUE);
	 q->SetAlwaysSecSelect(1);
	 q->SetPickable(1);
	 q->SetDefWidth(8);
	 q->SetDefHeight(8);
	 q->SetPalette(pal);


	 TEveVectorD v0(m_MF_plane_d0[0], m_MF_plane_d0[1], m_MF_plane_d0[2]);
	 TEveVectorD v01(m_MF_plane_d1[0], m_MF_plane_d1[1], m_MF_plane_d1[2]);
	 TEveVectorD v02(m_MF_plane_d2[0], m_MF_plane_d2[1], m_MF_plane_d2[2]);

	 TEveVectorD b01 = (v01 -v0);
	 TEveVectorD b02 = (v02 -v0); 
	 TEveVectorD b03 = b01.Cross(b02);

	 TEveTrans trans;
	 trans.SetBaseVec(1, b01.fX, b01.fY, b01.fZ);
	 trans.SetBaseVec(2, b02.fX, b02.fY, b02.fZ);
	 trans.SetBaseVec(3, b03.fX, b03.fY, b03.fZ);
	 trans.SetPos(v0.Arr());
	 trans.OrtoNorm3();
	 q->SetTransMatrix(trans.Array());

	 double w_step = b01.Mag()/m_MF_plane_N;
	 double h_step = b02.Mag()/m_MF_plane_N;

	 TEveVectorD d1; trans.GetBaseVec(1).GetXYZ(d1); d1 *= w_step;
	 TEveVectorD d2; trans.GetBaseVec(2).GetXYZ(d2); d2 *= h_step;

	 //d1.Print();
	 d2.Dump();
         double line_step_size = TMath::Min(w_step, h_step);

	 for(int i = 0; i < m_MF_plane_N; i++)
	   {
	     for(int j=0; j <m_MF_plane_N; j++)
	       {
		 TEveVectorD  p = d1*Double_t(i) + d2*Double_t(j) + v0;
                 GlobalVector b = field->inTesla( GlobalPoint(p.fX, p.fY, p.fZ));

                 q->AddQuad(w_step*i, h_step*j);
		 q->QuadValue(b.mag()*100);
		 q->QuadId(new TNamed(Form("Mag (%f, %f, %f) val = %f", b.x(), b.y(), b.z(), b.mag() ), "Dong!"));

		 if (b.mag() > 1e-6) {
		   b.unit(); b *= line_step_size;
		   ls->AddLine(p.fX, p.fY, p.fZ, p.fX + b.x(),p.fY + b.y(), p.fZ + b.z());
		 }
		 else {
		   ls->AddLine(p.fX, p.fY, p.fZ, p.fX + b.x(),p.fY + b.y(), p.fZ + b.z());
		 }

		 ls->AddMarker(ls->GetLinePlex().Size()-1, 0);	
		 ls->AddMarker(i*m_MF_plane_N + j, 0);		   			    
	       }
	   } 

	 m_eve->AddElement(q);
      	 ls->SetRnrSelf(m_MF_plane_draw_dir);
	 m_eve->AddElement(ls);
       }
     else
       {
	 // Add a test obj
	 if (!gRandom)
	   gRandom = new TRandom(0);
	 TRandom& r= *gRandom;

	 Float_t s = 100;

	 TEvePointSet* ps = new TEvePointSet();
	 ps->SetOwnIds(kTRUE);
	 for(Int_t i = 0; i< 100; i++)
	   {
	     ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
	     ps->SetPointId(new TNamed(Form("Point %d", i), ""));
	   }

	 ps->SetMarkerColor(TMath::Nint(r.Uniform(2, 9)));
	 ps->SetMarkerSize(r.Uniform(1, 2));
	 ps->SetMarkerStyle(4);
	 m_eve->AddElement(ps);
       }
   }
}

// ------------ method called once each job just before starting event loop  ------------
void 
DisplayGeom::beginJob()
{ 
   if (m_eve)
   {
      m_geomList = new TEveElementList("Display Geom");
      m_eve->AddGlobalElement(m_geomList);
      m_eve->getManager()->GetGlobalScene()->GetGLScene()->SetStyle(TGLRnrCtx::kWireFrame);
   }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DisplayGeom::endJob() {
}

//------------------------------------------------------------------------------
void DisplayGeom::remakeGeometry(const DisplayGeomRecord& dgRec)
{
   m_geomList->DestroyElements();

   edm::ESHandle<TGeoManager> geom;
   dgRec.get(geom);
   TEveGeoManagerHolder _tgeo(const_cast<TGeoManager*>(geom.product()));

   // To have a full one, all detectors in one top-node:
   // make_node("/cms:World_1/cms:CMSE_1", 4, kTRUE);

   if (m_MF)
   {
      make_node("/cms:World_1", m_level, kTRUE);
   }
   else
   {
      make_node("/cms:World_1/cms:CMSE_1/tracker:Tracker_1", m_level, kTRUE);
      make_node("/cms:World_1/cms:CMSE_1/caloBase:CALO_1",   m_level, kTRUE);
      make_node("/cms:World_1/cms:CMSE_1/muonBase:MUON_1",   m_level, kTRUE);
   }
}
