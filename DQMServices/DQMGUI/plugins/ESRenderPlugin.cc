#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TH1F.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TGaxis.h"
#include "TColor.h"
#include "TROOT.h"
#include "TText.h"
#include "TGraph.h"
#include "TLine.h"
#include "TProfile.h"
#include "TProfile2D.h"

#include <math.h>
#include <cassert>

class ESRenderPlugin : public DQMRenderPlugin {

  public:

  static const int ixES[346];
  static const int iyES[346];
  static const int lsES[54];
  static const int lwES[54];

  int colorbar1[10];
  int colorbar2[8];

  virtual void initialise( int, char ** ) {
    float rgb[10][3] = {{0.87, 0.00, 0.00}, {0.91, 0.27, 0.00},
                        {0.95, 0.54, 0.00}, {1.00, 0.81, 0.00},
                        {0.56, 0.91, 0.00}, {0.12, 1.00, 0.00},
                        {0.06, 0.60, 0.50}, {0.00, 0.20, 1.00},
                        {0.00, 0.10, 0.94}, {0.00, 0.00, 0.87}};


    for (int i=0; i<10; ++i) {
      colorbar1[i] = TColor::GetColor(rgb[i][0], rgb[i][1], rgb[i][2]);
    }

    for (int i=0; i<8; ++i) {
      colorbar2[i] = i+1;
    }
    colorbar2[0] = 0;
    colorbar2[7] = 800;
  }

  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo &i );

  virtual void preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &i, VisDQMRenderInfo&  r);

  virtual void postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &i );

private:

  void preDrawTH1( TCanvas *c, const VisDQMObject &o );
  void preDrawTH2F( TCanvas *c, const VisDQMObject &o );
  void postDrawTH2F( TCanvas *c, const VisDQMObject &o );
  void preDrawTProfile( TCanvas *c, const VisDQMObject &o );

  void drawBorders( int plane, float sx, float sy );

  double NEntries;

};

bool ESRenderPlugin::applies( const VisDQMObject &o, const VisDQMImgInfo & ) {

  if ( o.name.find( "EcalPreshower" ) != std::string::npos ) {
    if ( o.name.find( "ESOccupancyTask" ) != std::string::npos )
      return true;
    if ( o.name.find( "ESRawDataTask" ) != std::string::npos )
      return true;
    if ( o.name.find( "ESIntegrityTask" ) != std::string::npos )
      return true;
    if ( o.name.find( "ESIntegrityClient" ) != std::string::npos )
      return true;
    if ( o.name.find( "EventInfo" ) != std::string::npos )
      return true;
  }

  return false;
}

void ESRenderPlugin::preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo & ) {

   c->cd();

   gStyle->Reset("Default");

   gStyle->SetCanvasColor(10);
   gStyle->SetPadColor(10);
   gStyle->SetFillColor(10);
   gStyle->SetFrameFillColor(10);
   gStyle->SetStatColor(10);
   gStyle->SetTitleFillColor(10);
   gStyle->SetOptTitle(kTRUE);
   gStyle->SetTitleBorderSize(0);
   gStyle->SetOptStat(kFALSE);
   gStyle->SetStatBorderSize(1);
   gStyle->SetOptFit(kFALSE);

   if ( dynamic_cast<TProfile*>( o.object ) )
     preDrawTProfile( c, o );
   else if ( dynamic_cast<TH1F*>( o.object ) || dynamic_cast<TH1D*>( o.object ))
     preDrawTH1( c, o );
   else if ( dynamic_cast<TH2F*>( o.object ) )
     preDrawTH2F( c, o );

}

void ESRenderPlugin::postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo & ) {

  c->cd();

  if ( dynamic_cast<TH2F*>( o.object ) )
    postDrawTH2F( c, o );

}

void ESRenderPlugin::preDrawTProfile(TCanvas *, const VisDQMObject &o) {

  TProfile* obj = dynamic_cast<TProfile*>( o.object );
  assert( obj );

  gStyle->SetPaintTextFormat();

  gStyle->SetOptStat(kFALSE);
  obj->SetStats(kFALSE);
  gPad->SetLogy(kFALSE);
  obj->SetMinimum(0.0);

  std::string name = o.name.substr(o.name.rfind("/")+1);
  if ( o.name.find("Trending") != std::string::npos ) {
    obj->SetLineColor(4);
  }

}

void ESRenderPlugin::preDrawTH1(TCanvas *, const VisDQMObject &o) {

  TH1* obj = dynamic_cast<TH1*>( o.object );
  assert( obj );

  std::string name = o.name.substr(o.name.rfind("/")+1);

  if ( o.name.find("ESRawDataTask") != std::string::npos ) {
    obj->SetFillColor(kRed);
  }

  if ( name.find( "Gain used for data taking" ) != std::string::npos ) {
    obj->GetXaxis()->SetBinLabel(1,"LG");
    obj->GetXaxis()->SetBinLabel(2,"HG");
    obj->GetXaxis()->SetLabelSize(0.1);
    obj->SetLineColor(6);
    obj->SetLineWidth(2);
  }

  if ( name.find( "FEDs used for data taking" ) != std::string::npos ) {
    obj->SetFillColor(kGreen);
  }

  if ( name.find( "Z 1 P 1" ) != std::string::npos ) {
    name.erase( name.find( "Z 1 P 1" ) , 7);
    name.insert( 2, "+F" );
    obj->SetTitle( name.c_str() );
  } else if ( name.find( "Z -1 P 1" ) != std::string::npos ) {
    name.erase( name.find( "Z -1 P 1" ) , 8);
    name.insert( 2, "-F" );
    obj->SetTitle( name.c_str() );
  } else if ( name.find( "Z 1 P 2" ) != std::string::npos ) {
    name.erase( name.find( "Z 1 P 2" ) , 7);
    name.insert( 2, "+R" );
    obj->SetTitle( name.c_str() );
  } else if ( name.find( "Z -1 P 2" ) != std::string::npos ) {
    name.erase( name.find( "Z -1 P 2" ) , 8);
    name.insert( 2, "-R" );
    obj->SetTitle( name.c_str() );
  }

  if ( name.find( "Num of RecHits" ) != std::string::npos ) {
    gPad->SetLogy(kTRUE);
    obj->SetLineColor(4);
    obj->SetLineWidth(2);
  }

  if ( name.find( "Num of Good RecHits" ) != std::string::npos ) {
    gPad->SetLogy(kTRUE);
    obj->SetLineColor(2);
    obj->SetLineWidth(2);
  }

  if ( name.find( "Event Energy" ) != std::string::npos ) {
    gPad->SetLogy(kTRUE);
    obj->SetLineColor(3);
    obj->SetLineWidth(2);
  }

  if ( name.find( "RecHit Energy" ) != std::string::npos ) {
    gPad->SetLogy(kTRUE);
    obj->GetXaxis()->SetNdivisions(5, kFALSE);
    obj->SetLineColor(6);
    obj->SetLineWidth(2);
  }

}

void ESRenderPlugin::preDrawTH2F( TCanvas *, const VisDQMObject &o ) {

   TH2F* obj = dynamic_cast<TH2F*>( o.object );

   assert( obj );

   std::string name = o.name.substr(o.name.rfind("/")+1);

   gStyle->SetPaintTextFormat();

   gStyle->SetOptStat(kFALSE);
   obj->SetStats(kFALSE);
   gPad->SetLogy(kFALSE);

   gStyle->SetPalette(1);
   obj->SetOption("colz");
   gPad->SetRightMargin(0.15);
   gStyle->SetPaintTextFormat("+g");

   if (name.find( "OptoRX" ) != std::string::npos||name.find( "KChip" ) != std::string::npos||name.find( "Fiber Bad Status" ) != std::string::npos || name.find( "Fiber Off" ) != std::string::npos) {
     gStyle->SetPalette(10,colorbar1);
   }

   if ( name.find( "Z 1 P 1" ) != std::string::npos ) {
      name.erase( name.find( "Z 1 P 1" ) , 7);
      name.insert( 2, "+F" );
      obj->SetTitle( name.c_str() );
   } else if ( name.find( "Z -1 P 1" ) != std::string::npos ) {
      name.erase( name.find( "Z -1 P 1" ) , 8);
      name.insert( 2, "-F" );
      obj->SetTitle( name.c_str() );
   } else if ( name.find( "Z 1 P 2" ) != std::string::npos ) {
      name.erase( name.find( "Z 1 P 2" ) , 7);
      name.insert( 2, "+R" );
      obj->SetTitle( name.c_str() );
   } else if ( name.find( "Z -1 P 2" ) != std::string::npos ) {
      name.erase( name.find( "Z -1 P 2" ) , 8);
      name.insert( 2, "-R" );
      obj->SetTitle( name.c_str() );
   }

   if ( name.find( "Integrity Summary" ) != std::string::npos ) {
      gStyle->SetPalette(8,colorbar2);
      obj->SetMinimum(0.5);
      obj->SetMaximum(8.5);
      return;
   }

   if ( name.find( "Fiber Bad Status" ) != std::string::npos ) {
      obj->GetYaxis()->SetTitle("");
      NEntries = obj->GetBinContent(56,36);
      if (NEntries != 0) {
	obj->SetBinContent(56,36,0.);
	obj->Scale(1/NEntries);
	obj->SetMaximum(1);
      }
      return;
   }

   if ( name.find( "Fiber Off" ) != std::string::npos ) {
      NEntries = obj->GetBinContent(56,36);
      if (NEntries != 0) {
	obj->SetBinContent(56,36,0.);
	obj->Scale(1/NEntries);
	obj->SetMaximum(1);
      }
      return;
   }

   if ( name.find( "Event Dropped" ) != std::string::npos ) {
      NEntries = obj->GetBinContent(56,36);
      if (NEntries != 0) {
	obj->SetBinContent(56,36,0.);
	obj->Scale(1/NEntries);
	obj->SetMaximum(1);
      }
      return;
   }

   if ( name.find( "DCC" ) != std::string::npos || name.find( "OptoRX" ) != std::string::npos) {
      NEntries = obj->GetBinContent(56,3);
      if (NEntries != 0) {
	obj->SetBinContent(56, 3, 0.);
	obj->Scale(1/NEntries);
	obj->SetMaximum(1);
      }
      return;
   }

   if ( name.find( "RecHit 2D Occupancy" ) != std::string::npos ) {
     gStyle->SetPalette(1);
     NEntries = obj->GetBinContent(40,40);
     if (NEntries != 0) { // consider changing to < 1.
       obj->SetBinContent(40,40,0.);
       obj->Scale(1/NEntries);
       obj->SetMaximum(32);
       obj->GetZaxis()->SetNdivisions(8, kFALSE);
     }
     return;
   }

   if ( name.find( "Digi 2D Occupancy" ) != std::string::npos ) {
     gStyle->SetPalette(1);
     NEntries = obj->GetBinContent(40,40);
     if (NEntries != 0) {
       obj->SetBinContent(40,40,0.);
       obj->Scale(1/NEntries);
     }
     return;
   }

   if ( name.find( "Energy Density" ) != std::string::npos || name.find( "Occupancy with" ) != std::string::npos) {
     gStyle->SetPalette(1);
     NEntries = obj->GetBinContent(40,40);
     if (NEntries != 0) {
       obj->SetBinContent(40,40,0.);
       obj->Scale(1/NEntries);
     }
     return;
   }

   if ( name.find( "reportSummaryMap" ) != std::string::npos ) {
     dqm::utils::reportSummaryMapPalette(obj);
     obj->SetTitle("EcalPreshower Report Summary Map");
     return;
   }

}

void ESRenderPlugin::postDrawTH2F( TCanvas *, const VisDQMObject &o ) {

   TH2F* obj = dynamic_cast<TH2F*>( o.object );
   assert( obj );

   std::string name = o.name.substr(o.name.rfind("/")+1);

   if ( name.find( "Z 1 P 1" ) != std::string::npos ) {
     drawBorders( 1, 0.5, 0.5 );
   }

   if ( name.find( "Z -1 P 1" ) != std::string::npos ) {
     drawBorders( 2, 0.5, 0.5 );
   }

   if ( name.find( "Z 1 P 2" ) != std::string::npos ) {
     drawBorders( 3, 0.5, 0.5 );
   }

   if ( name.find( "Z -1 P 2" ) != std::string::npos ) {
     drawBorders( 4, 0.5, 0.5 );
   }

   if ( name.find( "reportSummaryMap" ) != std::string::npos ) {
     TText t;
     t.SetTextAlign(22);
     t.DrawText(21,21,"ES+R");
     t.DrawText(61,21,"ES-R");
     t.DrawText(21,61,"ES+F");
     t.DrawText(61,61,"ES-F");

     drawBorders( 1, 0.5, 40.5 );
     drawBorders( 2, 40.5, 40.5 );
     drawBorders( 3, 0.5, 0.5 );
     drawBorders( 4, 40.5, 0.5 );
   }

}

// Draw ES borders (Ming's copyright, the idea is borrowed from Giuseppe ;-))
void ESRenderPlugin::drawBorders( int plane, float sx, float sy ) {

   TLine l;

   switch (plane) {

      case 1:	//ES+F
	 for ( int i=0; i<346; i=i+2) {
	    if (i<54*2) {
	       l.SetLineStyle(lsES[i/2]);
	       l.SetLineWidth(lwES[i/2]);
	    } else {
	       l.SetLineStyle(3);
	       l.SetLineWidth(2);
	    }
	    l.DrawLine(ixES[i]+sx, iyES[i]+sy, ixES[i+1]+sx, iyES[i+1]+sy);
	 }
	 break;

      case 2:   //ES-F
	 for ( int i=0; i<346; i=i+2) {
	    if (i<54*2) {
	       l.SetLineStyle(lsES[i/2]);
	       l.SetLineWidth(lwES[i/2]);
	    } else {
	       l.SetLineStyle(3);
	       l.SetLineWidth(2);
	    }
	    l.DrawLine(40-ixES[i]+sx, iyES[i]+sy, 40-ixES[i+1]+sx, iyES[i+1]+sy);
	 }
	 break;

      case 3:    //ES+R
	 for ( int i=0; i<346; i=i+2) {
	    if (i<54*2) {
	       l.SetLineStyle(lsES[i/2]);
	       l.SetLineWidth(lwES[i/2]);
	    } else {
	       l.SetLineStyle(3);
	       l.SetLineWidth(2);
	    }
	    l.DrawLine(40-iyES[i]+sx, ixES[i]+sy, 40-iyES[i+1]+sx, ixES[i+1]+sy);
	 }
	 break;

      case 4:    //ES-R
	 for ( int i=0; i<346; i=i+2) {
	    if (i<54*2) {
	       l.SetLineStyle(lsES[i/2]);
	       l.SetLineWidth(lwES[i/2]);
	    } else {
	       l.SetLineStyle(3);
	       l.SetLineWidth(2);
	    }
	    l.DrawLine(iyES[i]+sx, ixES[i]+sy, iyES[i+1]+sx, ixES[i+1]+sy);
	 }
	 break;

      default:
	 break;

   }
}

const int ESRenderPlugin::ixES[346] = {
   1, 13,  5,  5,  5,  7,  7,  7,  7,  9,  9,  9, 11, 11, 13, 13, 13, 15, 15, 15,
   15, 15, 15, 19, 19, 19, 21, 21, 21, 23, 23, 23, 25, 25, 25, 27, 27, 27, 27, 29,
   29, 29, 29, 31, 31, 31, 31, 31, 31, 33, 33, 33, 35, 35, 39, 27, 35, 35, 35, 33,
   33, 33, 33, 31, 31, 31, 29, 29, 27, 27, 27, 25, 25, 25, 25, 25, 25, 21, 21, 21,
   19, 19, 19, 17, 17, 17, 15, 15, 15, 13, 13, 13, 13, 11, 11, 11, 11,  9,  9,  9,
   9,  9,  9,  7,  7,  7,  5,  5,

   1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,  6,
   6,  6,  6,  7,  7,  7,  7,  9,  9,  9,  9, 10, 10, 10, 10, 13, 13, 13, 13, 15,
   15, 15, 15, 25, 25, 25, 25, 27, 27, 27, 27, 30, 30, 30, 30, 31, 31, 31, 31, 33,
   33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38,
   38, 38, 38, 39, 39, 39, 39, 38, 38, 38, 38, 37, 37, 37, 37, 36, 36, 36, 36, 35,
   35, 35, 35, 34, 34, 34, 34, 33, 33, 33, 33, 31, 31, 31, 31, 30, 30, 30, 30, 27,
   27, 27, 27, 25, 25, 25, 25, 15, 15, 15, 15, 13, 13, 13, 13, 10, 10, 10, 10,  9,
   9,  9,  9,  7,  7,  7,  7,  6,  6,  6,  6,  5,  5,  5,  5,  4,  4,  4,  4,  3,
   3,  3,  3,  2,  2,  2,  2,  1,  1,  1,

   13, 13, 13, 14, 14, 14, 14, 15, 15, 16, 16, 16, 16, 18, 18, 18, 18, 22, 22, 22,
   22, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 26, 26, 26,
   26, 25, 25, 24, 24, 24, 24, 22, 22, 22, 22, 18, 18, 18, 18, 16, 16, 16, 16, 15,
   15, 15, 15, 14, 14, 14, 14, 13
};

const int  ESRenderPlugin::iyES[346] = {
   20, 20, 20, 30, 30, 30, 30, 36, 32, 32, 32, 20, 20, 38, 39, 26, 26, 26, 26, 24,
   40, 30, 30, 30, 40, 27, 40, 35, 35, 35, 35, 26, 24, 40, 26, 26, 26, 33, 33, 33,
   38, 24, 24, 24, 20, 24, 24, 28, 28, 28, 28, 36, 20, 33, 20, 20, 20, 10, 10, 10,
   10,  4,  8,  8,  8, 20, 20,  2,  1, 14, 14, 14, 14, 16,  0, 10, 10, 10, 13,  0,
   0,  5,  5,  5,  5, 14, 16,  0, 14, 14, 14,  7,  7,  7,  2, 16, 16, 16, 20, 16,
   16, 12, 12, 12, 12,  4,  7, 20,

   20, 26, 26, 26, 26, 28, 28, 28, 28, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33,
   33, 34, 34, 34, 34, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39,
   39, 40, 40, 40, 40, 39, 39, 39, 39, 38, 38, 38, 38, 37, 37, 37, 37, 36, 36, 36,
   36, 34, 34, 34, 34, 33, 33, 33, 33, 32, 32, 32, 32, 31, 31, 31, 31, 28, 28, 28,
   28, 26, 26, 26, 26, 14, 14, 14, 14, 12, 12, 12, 12,  9,  9,  9,  9,  8,  8,  8,
   8,  7,  7,  7,  7,  6,  6,  6,  6,  4,  4,  4,  4,  3,  3,  3,  3,  2,  2,  2,
   2,  1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,
   3,  4,  4,  4,  4,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,
   9, 12, 12, 12, 12, 14, 14, 14, 14, 20,

   18, 22, 22, 22, 22, 24, 24, 24, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 26,
   26, 26, 26, 25, 25, 25, 25, 24, 24, 24, 24, 22, 22, 22, 22, 18, 18, 18, 18, 16,
   16, 16, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15,
   15, 16, 16, 16, 16, 18, 18, 18
};

const int  ESRenderPlugin::lsES[54] = { // line style
   1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2,
   1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1,
   2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2
};

const int  ESRenderPlugin::lwES[54] = { // line width
   2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1,
   2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2,
   1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1
};

static ESRenderPlugin instance;
