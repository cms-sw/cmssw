/*!
  \file JetMETRenderPlugin.cc
  \\
  \\ Code shamelessly borrowed from L1T's L1TRenderPlugin.cc code,
  \\ which was shapelessly borrowed from J. Temple's HcalRenderPlugin.cc
  \\ which was shamelessly borrowed from S. Dutta's SiStripRenderPlugin.cc
  \\ code, G. Della Ricca and B. Gobbo's EBRenderPlugin.cc, and other existing
  \\ subdetector plugins
  \\ preDraw and postDraw methods now check whether histogram was a TH1
  \\ or TH2, and call a private method appropriate for the histogram type
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TBox.h"
#include "TLine.h"
#include "TLegend.h"
#include "TPRegexp.h"

#include <cassert>
#include <string>

#define REMATCH(pat, str) (TPRegexp(pat).MatchB(str))

class JetMETRenderPlugin : public DQMRenderPlugin
{
  TH2F* dummybox;
  TBox* b_box_w; // (masked) -> not set : no
  TBox* b_box_r; // error
  TBox* b_box_y; // (warning)
  TBox* b_box_g; // good
  TBox* b_box_b; // (waiting)
  int l1t_pcol[60];
  float l1t_rgb[60][3];

public:
  virtual void initialise (int, char **)
  {
    // same as RenderPlugin default for now (no special action taken)

    dummybox = new  TH2F("dummyJetMET","",22,-0.5,21.5,18,-0.5,17.5);

    for(int i=0; i<22; i++)
      {
        for(int j=0; j<18; j++)
          {
            dummybox->Fill(i,j,0.1);
          }
      }

    for( int i=0; i<60; i++ ){

      if ( i < 15 ){
        l1t_rgb[i][0] = 1.00;
        l1t_rgb[i][1] = 1.00;
        l1t_rgb[i][2] = 1.00;
      }
      else if ( i < 30 ){
        l1t_rgb[i][0] = 0.50;
        l1t_rgb[i][1] = 0.80;
        l1t_rgb[i][2] = 1.00;
      }
      else if ( i < 40 ){
        l1t_rgb[i][0] = 1.00;
        l1t_rgb[i][1] = 1.00;
        l1t_rgb[i][2] = 1.00;
      }
      else if ( i < 57 ){
        l1t_rgb[i][0] = 0.80+0.01*(i-40);
        l1t_rgb[i][1] = 0.00+0.03*(i-40);
        l1t_rgb[i][2] = 0.00;
      }
      else if ( i < 59 ){
        l1t_rgb[i][0] = 0.80+0.01*(i-40);
        l1t_rgb[i][1] = 0.00+0.03*(i-40)+0.15+0.10*(i-17-40);
        l1t_rgb[i][2] = 0.00;
      }
      else if ( i == 59 ){
        l1t_rgb[i][0] = 0.00;
        l1t_rgb[i][1] = 0.80;
        l1t_rgb[i][2] = 0.00;
      }

      l1t_pcol[i] = TColor::GetColor(l1t_rgb[i][0], l1t_rgb[i][1], l1t_rgb[i][2]);
    }

    b_box_w = new TBox();
    b_box_r = new TBox();
    b_box_y = new TBox();
    b_box_g = new TBox();
    b_box_b = new TBox();

    b_box_g->SetFillColor(l1t_pcol[59]);
    b_box_y->SetFillColor(l1t_pcol[58]);
    b_box_r->SetFillColor(l1t_pcol[40]);
    b_box_w->SetFillColor(0);
    b_box_b->SetFillColor(l1t_pcol[22]);

  }

  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
  {
    // determine whether core object is an JetMET object
    if (o.name.find( "JetMET/" ) != std::string::npos )
      return true;
    return false;
  }

  virtual void preDraw (TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &)
  {
    c->cd();

    // object is TH2 histogram
    if( dynamic_cast<TH2F*>( o.object ) )
      {
        preDrawTH2F( c, o );
      }
    // object is TH1 histogram
    else if( dynamic_cast<TH1F*>( o.object ) )
      {
        preDrawTH1F( c, o );
      }
  }

  virtual void postDraw (TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &)
  {
    // object is TH2 histogram
    if( dynamic_cast<TH2F*>( o.object ) )
      {
        postDrawTH2F( c, o );
      }
    // object is TH1 histogram
    else if( dynamic_cast<TH1F*>( o.object ) )
      {
        postDrawTH1F( c, o );
      }
  }

private:
  void preDrawTH1F ( TCanvas *, const VisDQMObject &o )
  {
    // Do we want to do anything special yet with TH1F histograms?

    TH1F* obj = dynamic_cast<TH1F*>( o.object );
    assert (obj); // checks that object indeed exists

    gStyle->SetOptStat(111111);

    // - top
    if(REMATCH("hltpath", o.name))
      {
	obj->GetXaxis()->SetTitle("HLT Path");
	return;
      }
    else if(REMATCH("lumisec", o.name))
      {
        obj->GetXaxis()->SetTitle("lumi section");
        return;
      }
    // - MET
    else if( !(REMATCH("D0", o.name)          ||
	       REMATCH("Dz", o.name)          ||
	       REMATCH("Dxy", o.name)         ||
	       REMATCH("Phi", o.name)         ||
	       REMATCH("Eta", o.name)         ||
	       REMATCH("_LS", o.name)         ||
	       REMATCH("Flag", o.name)        ||
	       REMATCH("Chi2", o.name)        ||
	       REMATCH("Flag", o.name)        ||
	       REMATCH("Frac", o.name)        ||
	       //REMATCH("_logx", o.name)       ||
	       REMATCH("NHits", o.name)       ||
	       REMATCH("NLayers", o.name)     //||
	       //REMATCH("CorrFactor", o.name)  //||
	       //REMATCH("Correction", o.name)
	       )
	     )
      {
	if (obj->GetMaximum()>0.) gPad->SetLogy(1);
      }
    else obj->SetMinimum(0.);

    if( o.name.find( "physdec" )  != std::string::npos)
      {
	obj->GetXaxis()->SetBinLabel(1,"All Events");
	obj->GetXaxis()->SetBinLabel(2,"HLT_PhysicsDeclared");
	gPad->SetLogy(0);
        obj->GetXaxis()->CenterLabels();

      }
    if( o.name.find( "CaloEmEtInEE" )  != std::string::npos)
      {
	obj->GetXaxis()->SetTitle("EM Et [GeV]");
      }

    if( REMATCH("_Hi",o.name))
      {
	std::string title = obj->GetTitle();
	std::string::iterator it = title.end();
	title.replace(it-2,it,"(Pass Hi Pt Jet Trigger)");
	obj->SetTitle(title.c_str());
      }
    if( REMATCH("_Lo",o.name))
      {
	std::string title = obj->GetTitle();
	std::string::iterator it = title.end();
	title.replace(it-2,it,"(Pass Low Pt Jet Trigger)");
	obj->SetTitle(title.c_str());
      }

    return;
  }

  void preDrawTH2F ( TCanvas *, const VisDQMObject &o )
  {
    TH2F* obj = dynamic_cast<TH2F*>( o.object );
    assert( obj );

    gStyle->SetOptStat(10);

    //put in preDrawTH2F

    // ReportSummaryMap
    if( o.name.find( "reportSummaryMap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );

	obj->SetMaximum(1.0+1e-15);
	obj->SetMinimum(-2-1e-15);
	gStyle->SetPalette(60, l1t_pcol);

	obj->GetXaxis()->SetBinLabel(1,"CaloTower");
	obj->GetXaxis()->SetBinLabel(2,"MET");
	obj->GetXaxis()->SetBinLabel(3,"Jet");
	obj->GetXaxis()->SetLabelSize(0.1);

        obj->SetOption("col");
        obj->SetTitle("JetMET Report Summary Map");

        obj->GetXaxis()->CenterLabels();
        obj->GetYaxis()->CenterLabels();

        return;
      }

    gStyle->SetCanvasBorderMode( 0 );
    gStyle->SetPadBorderMode( 0 );
    gStyle->SetPadBorderSize( 0 );

    // Use same labeling format as SiStripRenderPlugin.cc
    TAxis* xa = obj->GetXaxis();
    TAxis* ya = obj->GetYaxis();

    xa->SetTitleOffset(0.7);
    xa->SetTitleSize(0.05);
    xa->SetLabelSize(0.04);

    ya->SetTitleOffset(0.7);
    ya->SetTitleSize(0.05);
    ya->SetLabelSize(0.04);

    // Now the important stuff -- set 2D hist drawing option to "colz"
    gStyle->SetPalette(1);
    obj->SetOption("colz");

    gStyle->SetStatX(0.82);
    gStyle->SetStatY(0.86);
    gStyle->SetStatW(0.30);
    gStyle->SetStatH(0.15);

    // CaloTower Stuff
    if( o.name.find( "CT_METPhivsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
        obj->SetTitle("CT-ieta vs METPhi");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("METphi (rad)");
        return;
      }
    else if( o.name.find( "CT_METvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT-ieta vs MET");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("MET (GeV)");
        return;
      }
    else if( o.name.find( "CT_MExvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT-ieta vs MEx");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("MEx (GeV)");
        return;
      }
    else if( o.name.find( "CT_MEyvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT-ieta vs MEy");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("MEy (GeV)");
        return;
      }
    else if( o.name.find( "CT_Maxetvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT-ieta vs MaxEt");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("MaxEt (GeV)");
        return;
      }
    else if( o.name.find( "CT_Minetvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT-ieta vs MinEt");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("MinEt (GeV)");
        return;
      }
    else if( o.name.find( "CT_Occ_ieta_iphi" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Occupancy: ieta vs iphi");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("iphi");
        return;
      }
    else if( o.name.find( "CT_Occvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Occupancy vs ieta");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("Occupancy");
        return;
      }
    else if( o.name.find( "CT_SETvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Sum Et vs ieta");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("SET (GeV)");
        return;
      }
    // EM-ET
    else if( o.name.find( "CT_emEt_ieta_iphi" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT EM Et: ieta vs iphi");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("iphi");
        return;
      }
    else if( o.name.find( "CT_emEtvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT EM Et vs ieta");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("EM Et (GeV)");
        return;
      }
    // Total-ET
    else if( o.name.find( "CT_et_ieta_iphi" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Et: ieta vs iphi");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("iphi");
        return;
      }
    else if( o.name.find( "CT_etvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Et vs ieta");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("Et (GeV)");
        return;
      }
    // Had-ET
    else if( o.name.find( "CT_hadEt_ieta_iphi" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Had Et: ieta vs iphi");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("iphi");
        return;
      }
    else if( o.name.find( "CT_hadEtvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Had Et vs ieta");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("Had Et (GeV)");
        return;
      }
    // Outer-ET
    else if( o.name.find( "CT_outerEt_ieta_iphi" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Outer Et: ieta vs iphi");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("iphi");
        return;
      }
    else if( o.name.find( "CT_outerEtvsieta" ) != std::string::npos )
      {
        gPad->SetGrid(1,1);
	if (obj->GetMaximum()>0.) gPad->SetLogz(1);
        obj->SetTitle("CT Outer Et vs ieta");
        obj->GetXaxis()->SetTitle("ieta");
        obj->GetYaxis()->SetTitle("Outer ET (GeV)");
        return;
      }

  }

  void postDrawTH1F( TCanvas *, const VisDQMObject & )
  {

  }

  void postDrawTH2F( TCanvas *, const VisDQMObject &o )
  {

    TH2F* obj = dynamic_cast<TH2F*>( o.object );
    assert( obj );

    TBox* b_box = new TBox();
    TLine* l_line = new TLine();
    TText* t_text = new TText();

    // ReportSummaryMap
    if( o.name.find( "reportSummaryMap" )  != std::string::npos)
      {
        t_text->SetTextAlign(21);

	t_text->DrawText(0.5,4.3, "Barrel");
	t_text->DrawText(0.5,3.3, "Endcap");
	t_text->DrawText(0.5,2.3, "Forward");

	t_text->DrawText(1.5,4.3, "CaloMET");
	t_text->DrawText(1.5,3.3, "CaloMETNoHF");
	t_text->DrawText(1.5,2.3, "TcMET");
	t_text->DrawText(1.5,1.3, "PFMET");
	t_text->DrawText(1.5,0.3, "MuCorrMET");

	t_text->DrawText(2.5,4.3, "CaloJet Barrel");
	t_text->DrawText(2.5,3.3, "CaloJet Endcap");
	t_text->DrawText(2.5,2.3, "CaloJet HF");
	t_text->DrawText(2.5,1.3, "JPT");
	t_text->DrawText(2.5,0.3, "PFJet");

	// Fill the emptry area by gray
	b_box->SetFillColor(17);
	b_box->DrawBox(0,0,1,2);

	// Vertical lines
	l_line->SetLineWidth(2);
	l_line->DrawLine(1,0,1,5);
	l_line->DrawLine(2,0,2,5);

	// Horizontal lines
	l_line->DrawLine(0,4,3,4);
	l_line->DrawLine(0,3,3,3);
	l_line->DrawLine(0,2,3,2);
	l_line->DrawLine(1,1,3,1);

	TLegend* leg = new TLegend(0.10, 0.11, 0.35, 0.38);
	leg->AddEntry(b_box_g,"Good",   "f");
	leg->AddEntry(b_box_r,"Bad",    "f");
	leg->AddEntry(b_box_w,"Not Set","f");
	leg->Draw();

	return;

      }

  }
};

static JetMETRenderPlugin instance;
