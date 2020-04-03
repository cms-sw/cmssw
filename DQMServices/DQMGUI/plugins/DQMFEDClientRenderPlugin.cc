/*!
  \file DQMFEDCLientRenderPlugin.cc
  \\
  \\ Code shamelessly borrowed from J. Temple's HcalRenderPlugin.cc code,
  \\ which was shamelessly borrowed from S. Dutta's SiStripRenderPlugin.cc
  \\ code, G. Della Ricca and B. Gobbo's EBRenderPlugin.cc, and other existing
  \\ subdetector plugins
  \\ preDraw and postDraw methods now check whether histogram was a TH1
  \\ or TH2, and call a private method appropriate for the histogram type
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include <cassert>

class DQMFEDClientRenderPlugin : public DQMRenderPlugin
{
public:
  // These functions may be different that parent version
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
    {
      // determine whether core object is an FED Client object
      if(o.name.find( "FED/" ) != std::string::npos  ||
         o.name.find( "FEDTest/" ) != std::string::npos )
        return true;

      return false;
    }

  virtual void preDraw (TCanvas * c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &)
    {
      c->cd();
      gPad->SetLogy(0);
      if ( o.name.find( "Fatal" )  != std::string::npos)
         gPad->SetLogy(1);

      // object is TH2 histogram
      if ( dynamic_cast<TH2F*>( o.object ) )
      {
         gPad->SetLogy(0);
         preDrawTH2F( c, o );
      }
    }

private:
  void preDrawTH2F ( TCanvas *, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      //put in preDrawTH2F
      if( o.name.find( "reportSummaryMap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        dqm::utils::reportSummaryMapPalette(obj);
        obj->SetOption("colz");
        obj->SetTitle("FED Report Summary Map");

        obj->GetXaxis()->SetNdivisions(1,true);
        obj->GetYaxis()->SetNdivisions(7,true);
        obj->GetXaxis()->CenterLabels();
        obj->GetYaxis()->CenterLabels();

        gPad->SetGrid(1,1);

        return;
      }

      gStyle->SetCanvasBorderMode( 0 );
      gStyle->SetPadBorderMode( 0 );
      gStyle->SetPadBorderSize( 0 );
    }
};

static DQMFEDClientRenderPlugin instance;
