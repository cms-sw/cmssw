/**
 * \class L1TRenderPlugin
 *
 *
 * Description: render plugin for L1 Trigger DQM histograms.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Lorenzo Agostino
 *      Initial version - based on code from HcalRenderPlugin
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *      New render plugin for report summary map
 *
 *
 * $Date: 2012/06/13 11:06:06 $
 * $Revision: 1.35 $
 *
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

#include "QualityTestStatusRenderPlugin.h"

#define REMATCH(pat, str) (TPRegexp(pat).MatchB(str))

class L1TRenderPlugin : public DQMRenderPlugin
{
  TH2F* dummybox;
  TBox* b_box_w;
  TBox* b_box_r;
  TBox* b_box_y;
  TBox* b_box_g;
  TBox* b_box_b;
  int l1t_pcol[60];
  float l1t_rgb[60][3];

public:
  virtual void initialise (int, char **)
    {

      dummybox = new  TH2F("dummyL1T","",22,-0.5,21.5,18,-0.5,17.5);

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

  l1t_pcol[i] = 1901+i;

  TColor* color = gROOT->GetColor( 1901+i );
  if( ! color ) color = new TColor( 1901+i, 0, 0, 0, "" );
  color->SetRGB( l1t_rgb[i][0], l1t_rgb[i][1], l1t_rgb[i][2] );
      }

      b_box_w = new TBox();
      b_box_r = new TBox();
      b_box_y = new TBox();
      b_box_g = new TBox();
      b_box_b = new TBox();

      b_box_g->SetFillColor(1960);
      b_box_y->SetFillColor(1959);
      b_box_r->SetFillColor(1941);
      b_box_w->SetFillColor(0);
      b_box_b->SetFillColor(1923);


    }

  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
    {
      // determine whether core object is an L1T object
      if (o.name.find( "L1T/" ) != std::string::npos )
        // Stage 2 trigger will have new render plugin
        if (o.name.find( "L1TStage2" ) == std::string::npos && 
            o.name.find( "reportSummaryMap" ) == std::string::npos ) 
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

      // GCT section
      if (o.name.find("L1TGCT") != std::string::npos) {

        // General style and stats
        //gStyle->SetPalette(1);
        //obj->SetOption("colz");
        gPad->SetGrid(1,1);
        gStyle->SetOptStat(11);

        // Axis labels

        // HF Ring Counts
        if (o.name.find("TowerCountNegEta") != std::string::npos ||
            o.name.find("TowerCountPosEta") != std::string::npos) {
          obj->GetXaxis()->SetTitle("HF Ring Tower Count");
          return;
        }        

        // HF Ring Sums
        if (o.name.find("ETSumNegEta") != std::string::npos ||
            o.name.find("ETSumPosEta") != std::string::npos) {
          obj->GetXaxis()->SetTitle("HF Ring E_{T}");
          return;
        } 

        // HF Ring ratios
        if( (o.name.find("HFRingRatioPosEta")!= std::string::npos ) ){
          obj->GetXaxis()->SetTitle("HF #eta + RING1 E_{T}/RING2 E_{T}");
          return;
        } 

        // HF Ring ratios
        if( (o.name.find("HFRingRatioNegEta")!= std::string::npos) ){
          obj->GetXaxis()->SetTitle("HF #eta - RING1 E_{T}/RING2 E_{T}");
          return;
        } 

        // Eta 1D
        if (o.name.find("Eta") != std::string::npos) {
          obj->GetXaxis()->SetTitle("#eta");
          obj->SetMinimum(0);
          return;
        }

        // Phi 1D
        if (o.name.find("Phi") != std::string::npos) {
          obj->GetXaxis()->SetTitle("#phi");
          obj->SetMinimum(0);
          return;
        }

        // Jet and electron ET
        if (o.name.find("Rank") != std::string::npos) {
          obj->GetXaxis()->SetTitle("E_{T}");
    gPad->SetLogy(1);
          return;
        }

        // Energy sums overflow
        if (o.name.find("Of") != std::string::npos) {
          obj->GetXaxis()->SetTitle("Overflow Bit");
    int bincheck = obj->GetNbinsX();
    if( bincheck==2 ){
      obj->GetXaxis()->SetBinLabel(1,"Off");
      obj->GetXaxis()->SetBinLabel(2,"On");
    }
          obj->GetXaxis()->SetNdivisions(2);
          return;
        }

        // Energy sums
        if (o.name.find("EtMiss") != std::string::npos) {
          obj->GetXaxis()->SetTitle("MET");
    gPad->SetLogy(1);
          return;
        }

        // Energy sums
        if (o.name.find("HtMiss") != std::string::npos) {
          obj->GetXaxis()->SetTitle("MHT");
    gPad->SetLogy(1);
          return;
        }

        // Energy sums
        if (o.name.find("EtTotal") != std::string::npos) {
          obj->GetXaxis()->SetTitle("Sum E_{T}");
    gPad->SetLogy(1);
          return;
        }

        // Energy sums
        if (o.name.find("EtHad") != std::string::npos) {
          obj->GetXaxis()->SetTitle("H_{T}");
    gPad->SetLogy(1);
          return;
        }

        return;      
      }


      // rate histograms
      if ( (o.name.find("rate_algobit") != std::string::npos ||
      o.name.find("rate_ttbit") != std::string::npos ||
      o.name.find("Rate_AlgoBit") != std::string::npos ||
      o.name.find("Rate_TechBit") != std::string::npos ||
      o.name.find("Rate_Ratio") != std::string::npos ||
      o.name.find("Integral_TechBit") != std::string::npos ||
      o.name.find("Integral_AlgoBit") != std::string::npos ||
      o.name.find("Physics_Trigger_Rate") != std::string::npos ||
      o.name.find("Random_Trigger_Rate") != std::string::npos ||
      o.name.find("Lost_Physics_Trigger_Rate") != std::string::npos ||
      o.name.find("Deadtime_Percent") != std::string::npos ||
      o.name.find("instTrigRate") != std::string::npos ||
      o.name.find("instEventRate") != std::string::npos ||
      o.name.find("Rate_Ratio") != std::string::npos ||
      o.name.find("Number_of_Triggers") != std::string::npos ||
      o.name.find("Physics_Triggers") != std::string::npos ||
      o.name.find("Random_Triggers") != std::string::npos ||
      o.name.find("Lost_Final_Trigger") != std::string::npos ||
      o.name.find("DeadTime") != std::string::npos ||
      o.name.find("Number_Resets") != std::string::npos ||
      o.name.find("Orbit_Number") != std::string::npos ||
      o.name.find("Number_of_Events") != std::string::npos ||
      o.name.find("totAlgoRate") != std::string::npos ||
      o.name.find("totTtRate") != std::string::npos ||
      o.name.find("Instant_Lumi") != std::string::npos ||      
      o.name.find("Instant_Lumi_Err") != std::string::npos ||
      o.name.find("Instant_Lumi_Qlty") != std::string::npos ||
      o.name.find("Instant_Et_Lumi") != std::string::npos ||
      o.name.find("Instant_Et_Lumi_Err") != std::string::npos ||
      o.name.find("Instant_Et_Lumi_Qlty") != std::string::npos ||
      o.name.find("Num_Orbits") != std::string::npos ||
      o.name.find("Start_Orbit") != std::string::npos 
      ) )
  {
    gStyle->SetOptStat(11);
    obj->GetXaxis()->SetTitle("Luminosity Segment Number");
    obj->GetYaxis()->SetTitle("Rate (Hz)");
    int nbins = obj->GetNbinsX();
    int maxRange = nbins;
    for ( int i = nbins; i > 0; --i )
      {
        if ( obj->GetBinContent(i) != 0 )
    {
      maxRange = i;
      break;
    }
      }
    int minRange = 0;
    for ( int i = 0; i <= nbins; ++i )
      {
        if ( obj->GetBinContent(i) != 0 )
    {
      minRange = i;
      break;
    }
      }
    minRange = ( minRange>0 ) ? minRange-1 : 0;
    maxRange = ( nbins>maxRange ) ? maxRange+1 : nbins;
    
    obj->GetXaxis()->SetRange(minRange, maxRange);
    
    if ( (o.name.find("Integral_TechBit") != std::string::npos ||
    o.name.find("Integral_AlgoBit") != std::string::npos ||
    o.name.find("Rate_Ratio") != std::string::npos ||
    o.name.find("Deadtime_Percent") != std::string::npos ||
    o.name.find("Physics_Triggers") != std::string::npos ||
    o.name.find("Random_Triggers") != std::string::npos ||
    o.name.find("Orbit_Number") != std::string::npos ||
    o.name.find("Number_of_Events") != std::string::npos ||
    o.name.find("Lost_Final_Trigger") != std::string::npos ||
    o.name.find("DeadTime") != std::string::npos ||
    o.name.find("Number_Resets") != std::string::npos ||
    o.name.find("Instant_Lumi") != std::string::npos ||      
    o.name.find("Instant_Lumi_Err") != std::string::npos ||
    o.name.find("Instant_Lumi_Qlty") != std::string::npos ||
    o.name.find("Instant_Et_Lumi") != std::string::npos ||
    o.name.find("Instant_Et_Lumi_Err") != std::string::npos ||
    o.name.find("Instant_Et_Lumi_Qlty") != std::string::npos ||
    o.name.find("Num_Orbits") != std::string::npos ||
    o.name.find("Start_Orbit") != std::string::npos 
    ) )
      {
        obj->GetYaxis()->SetTitle("");
      }
    
    else if ( (o.name.find("instTrigRate") != std::string::npos ||
         o.name.find("instEventRate") != std::string::npos ||
         o.name.find("Number_of_Triggers") != std::string::npos
         ) )
      {
        obj->GetXaxis()->SetTitle("Time (sec)");
      }
  }

      /// DTTF section
      if ( o.name.find("dttf_") != std::string::npos ) {
  // dqm::utils::reportSummaryMapPalette(obj);
  // obj->SetOption("colz");
  obj->GetYaxis()->SetRangeUser(0,
              obj->GetBinContent(obj->GetMaximumBin() ) * 1.1 );


  if ( o.name.find("bx") != std::string::npos ) {
    
    obj->GetXaxis()->SetNdivisions(3);
    return;
    
  } else if ( o.name.find("charge" ) != std::string::npos ) {

    obj->GetXaxis()->SetNdivisions(2);
    return;

  } else if ( o.name.find("se" ) != std::string::npos ) {

    if ( ( o.name.find("etaFine" ) != std::string::npos ) ||
         ( o.name.find("nTracks" ) != std::string::npos ) ) {
      obj->GetXaxis()->SetNdivisions(2);
    }
    return;

  } else if ( ( o.name.find("etaFine_fraction_wh") != std::string::npos )
        || ( o.name.find("nTracks_wh" ) != std::string::npos ) ) {

    obj->GetXaxis()->CenterLabels();
    obj->GetXaxis()->SetNdivisions(12);
    return;

  }

  return;

      }

    }

  void preDrawTH2F ( TCanvas *, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*> (o.object);

      // checks that object indeed exists
      assert(obj);

      // specific rendering of L1T reportSummaryMap, using
      // dqm::QualityTestStatusRenderPlugin::reportSummaryMapPalette(obj)

      if (o.name.find("reportSummaryMap") != std::string::npos) {

          obj->SetStats(kFALSE);
          dqm::QualityTestStatusRenderPlugin::reportSummaryMapPalette(obj);

          obj->GetXaxis()->SetLabelSize(0.1);

          obj->GetXaxis()->CenterLabels();
          obj->GetYaxis()->CenterLabels();

          return;
      }

      // pre-draw rendering of other L1T TH2F histograms

      gStyle->SetCanvasBorderMode( 0 );
      gStyle->SetPadBorderMode( 0 );
      gStyle->SetPadBorderSize( 0 );

      // I don't think we want to set stats to 0 for Hcal
      //gStyle->SetOptStat( 0 );
      //obj->SetStats( kFALSE );

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

      //gStyle->SetOptStat(0);

/*      if(
        o.name.find( "RctEmIsoEmEtEtaPhi" ) != std::string::npos ||
        o.name.find( "RctEmIsoEmOccEtaPhi" ) != std::string::npos ||
        o.name.find( "RctEmNonIsoEmEtEtaPhi" ) != std::string::npos ||
        o.name.find( "RctEmNonIsoEmOccEtaPhi" ) != std::string::npos ||
        o.name.find( "RctRegionsEtEtaPhi" ) != std::string::npos ||
        o.name.find( "RctRegionsOccEtaPhi" ) != std::string::npos
         ) */
      if(
        ( o.name.find( "Rct" ) != std::string::npos &&
    (o.name.find( "Jet" ) != std::string::npos ||
           o.name.find( "IsoEm" ) != std::string::npos )) &&
        o.name.find( "EtaPhi" ) != std::string::npos 
        )
      {
        gPad->SetGrid(1,1);
        gStyle->SetOptStat(11);
        obj->GetXaxis()->SetTitle("GCT eta");
        obj->GetYaxis()->SetTitle("GCT phi");
        return;
      }

      // GCT section
      if (o.name.find("L1TGCT") != std::string::npos) {

        // General style and stats
        gStyle->SetPalette(1);
        obj->SetOption("colz");
        gPad->SetGrid(1,1);
        gStyle->SetOptStat(11);

        // Axis labels

        // Energy sums MET and MHT correlations
        if (o.name.find("EtMissHtMissPhiCorr") != std::string::npos) {
          obj->GetXaxis()->SetTitle("MET #phi");
          obj->GetYaxis()->SetTitle("MHT #phi");
          return;
        }

        // Eta phi 2D plots
        if (o.name.find("EtaPhi") != std::string::npos) {
          obj->GetXaxis()->SetTitle("#eta");
          obj->GetYaxis()->SetTitle("#phi");
          return;
        }

        // BX plots
        if (o.name.find("Bx") != std::string::npos) {
          obj->GetXaxis()->SetTitle("BX");
          obj->GetXaxis()->SetNdivisions(5);
          obj->GetYaxis()->SetTitle("E_{T}");
    int bincheck = obj->GetNbinsX();
    if( bincheck==5 ){
      obj->GetXaxis()->SetBinLabel(1,"-2");
      obj->GetXaxis()->SetBinLabel(2,"-1");
      obj->GetXaxis()->SetBinLabel(3,"0");
      obj->GetXaxis()->SetBinLabel(4,"+1");
      obj->GetXaxis()->SetBinLabel(5,"+2");
    }
          return;
        }

       // Energy sums MET and MHT correlations
        if (o.name.find("EtMissHtMissCorr") != std::string::npos) {
          obj->GetXaxis()->SetTitle("MET");
          obj->GetYaxis()->SetTitle("MHT");
          return;
        }

       // Energy sums Sum ET and HT correlations
        if (o.name.find("EtTotalEtHadCorr") != std::string::npos) {
          obj->GetXaxis()->SetTitle("Sum E_{T}");
          obj->GetYaxis()->SetTitle("H_{T}");
          return;
        }


        // HF Ring correlations
        if (o.name.find("TowerCountCorr") != std::string::npos ||
            o.name.find("HFRing1Corr") != std::string::npos ||
            o.name.find("HFRing2Corr") != std::string::npos) {
          obj->GetXaxis()->SetTitle("HF #eta +");
          obj->GetYaxis()->SetTitle("HF #eta -");
          return;
        } 

        return;      
      }

      /// DTTF section
      if ( o.name.find("dttf") != std::string::npos ) {

        gPad->SetGrid(1,1);
  gStyle->SetOptStat(0);

   if ( o.name.find("bx" ) != std::string::npos ) {

     obj->GetYaxis()->CenterLabels();
    obj->GetYaxis()->SetNdivisions(3, true);

    if ( o.name.find("wh") != std::string::npos ) {
      obj->GetXaxis()->SetNdivisions(12, true);
      obj->GetXaxis()->CenterLabels();
    }

    return;

  } else if ( o.name.find("gmt") != std::string::npos ) {

    // gStyle->SetOptStat(110010);
    obj->GetYaxis()->SetNdivisions(12);
    obj->GetYaxis()->CenterLabels();
    return;

  } else if ( o.name.find("phi_vs_eta") != std::string::npos ) {

    // if ( o.name.find("wh") != std::string::npos ) {
    //   gStyle->SetOptStat(110010);
    // }
    obj->GetYaxis()->SetNdivisions(12, false);
    return;

    //obj->GetXaxis()->SetNdivisions(8, false);

  } else if ( o.name.find("highQual" ) != std::string::npos ) {

    obj->GetYaxis()->SetNdivisions(12, true);
     obj->GetZaxis()->SetRangeUser(0, 1);
    obj->GetYaxis()->CenterLabels();
    return;

  } else if ( o.name.find("occupancy" ) != std::string::npos ) {

    obj->GetYaxis()->SetNdivisions(12, true);
    obj->GetYaxis()->CenterLabels();
    if ( o.name.find("tracks_occupancy_summary" ) != std::string::npos )
      obj->GetZaxis()->SetRangeUser(0, 0.03);
    return;

  } else if  ( o.name.find("quality" ) != std::string::npos ) {

      obj->GetYaxis()->SetNdivisions(8, true);
     obj->GetXaxis()->SetNdivisions(12, false);
     obj->GetXaxis()->CenterLabels();
    obj->GetYaxis()->CenterLabels();
    return;

  }

  return;

      }


      else if(REMATCH("BX_Correlation_*", o.name)) {
        TAxis* yBX = obj->GetYaxis();
        yBX->SetTitleOffset(1.1);
        return;
      }


      if(o.name.find("CSCTF_Chamber_Occupancies") != std::string::npos)
      {
        gStyle->SetOptStat(11);
        return;
      }
      if(o.name.find("CSCTF_occupancies") != std::string::npos)
      {
        gStyle->SetOptStat(11);
        return;
      }
      if(o.name.find("GMT_etaphi") != std::string::npos)
      {
        gStyle->SetOptStat(11);
        return;
      }
      if(o.name.find("BX_diffvslumi") != std::string::npos)
  {
    obj->GetXaxis()->SetTitle("Luminosity Segment Number");
    obj->GetYaxis()->SetTitle("#Delta bx");
    //obj->GetXaxis()->SetNdivisions(6,true);
    obj->GetYaxis()->SetNdivisions(9,true);
    obj->GetYaxis()->CenterLabels();
    //gPad->SetGrid(1,1);

    int nxbins = obj->GetNbinsX();
    int nybins = obj->GetNbinsY();
    int maxRange = nxbins;
    bool ynonempty = false;
    for ( int i = nxbins; i > 0; --i ){
      for ( int j = nybins; j > 0; --j ){
        if ( obj->GetBinContent(i,j) != 0 ){
    ynonempty = true;
    break;
        }
      }
      if(ynonempty){
        maxRange = i;
        break;    
      }
    }
    int minRange = 0;
    ynonempty = false;
    for ( int i = 0; i <= nxbins; ++i ){
      for ( int j = 0; j <= nybins; ++j ){
        if ( obj->GetBinContent(i,j) != 0 ){
    ynonempty = true;
    break;
        }
      }
      if(ynonempty){
        minRange = i;
        break;
      }
    }
    minRange = ( minRange>0 ) ? minRange-1 : 0;
    maxRange = ( nxbins>maxRange ) ? maxRange+1 : nxbins;
    
    obj->GetXaxis()->SetRange(minRange, maxRange);
    
    return;
  }


    }

  void postDrawTH1F(TCanvas*, const VisDQMObject &) {

        // use DQM default rendering

    }

  void postDrawTH2F(TCanvas*, const VisDQMObject& dqmObj) {

        TH2F* obj = dynamic_cast<TH2F*> (dqmObj.object);

        // checks that object indeed exists
        assert(obj);

        if (dqmObj.name.find("reportSummaryMap") != std::string::npos) {

            TLine* l_line = new TLine();
            TText* t_text = new TText();

            t_text->DrawText(2.25, 14.3, "Mu");
            t_text->DrawText(2.25, 13.3, "NoIsoEG");
            t_text->DrawText(2.25, 12.3, "IsoEG");
            t_text->DrawText(2.25, 11.3, "CenJet");
            t_text->DrawText(2.25, 10.3, "ForJet");
            t_text->DrawText(2.25, 9.3, "TauJet");
            t_text->DrawText(2.25, 8.3, "ETT");
            t_text->DrawText(2.25, 7.3, "ETM");
            t_text->DrawText(2.25, 6.3, "HTT");
            t_text->DrawText(2.25, 5.3, "HTM");
            t_text->DrawText(2.25, 4.3, "HfBitCounts");
            t_text->DrawText(2.25, 3.3, "HfRingEtSums");
            t_text->DrawText(2.25, 2.3, "GtExternal");
            t_text->DrawText(2.25, 1.3, "TechTrig");

            t_text->DrawText(1.25, 11.3, "GT");
            t_text->DrawText(1.25, 10.3, "GMT");
            t_text->DrawText(1.25, 9.3, "RPC");
            t_text->DrawText(1.25, 8.3, "CSC TF");
            t_text->DrawText(1.25, 7.3, "CSC TPG");
            t_text->DrawText(1.25, 6.3, "DT TF");
            t_text->DrawText(1.25, 5.3, "DT TPG");
            t_text->DrawText(1.25, 4.3, "Stage1Layer2");
            t_text->DrawText(1.25, 3.3, "RCT");
            t_text->DrawText(1.25, 2.3, "HCAL TPG");
            t_text->DrawText(1.25, 1.3, "ECAL TPG");

            l_line->SetLineWidth(2);

            // vertical line

            l_line->DrawLine(2, 1, 2, 15);

            // horizontal lines

            l_line->DrawLine(1, 1, 3, 1);
            l_line->DrawLine(1, 2, 3, 2);
            l_line->DrawLine(1, 3, 3, 3);
            l_line->DrawLine(1, 4, 3, 4);
            l_line->DrawLine(1, 5, 3, 5);
            l_line->DrawLine(1, 6, 3, 6);
            l_line->DrawLine(1, 7, 3, 7);
            l_line->DrawLine(1, 8, 3, 8);
            l_line->DrawLine(1, 9, 3, 9);
            l_line->DrawLine(1, 10, 3, 10);
            l_line->DrawLine(1, 11, 3, 11);
            l_line->DrawLine(1, 12, 3, 12);
            l_line->DrawLine(2, 13, 3, 13);
            l_line->DrawLine(2, 14, 3, 14);

            return;
        }

        // post-draw rendering of other L1T TH2F histograms

        TBox* b_box = new TBox();
        TLine* l_line = new TLine();

        if (dqmObj.name.find("CSCTF_Chamber_Occupancies") != std::string::npos) {

            b_box->SetFillColor(1);
            b_box->SetFillStyle(3013);

            l_line->SetLineWidth(1);

            int Num = 6;
            for (int i = 0; i < Num; i++) {
                double x1s = double(0.25 + i * 0.1 * 9);
                double x1e = double(0.85 + i * 0.1 * 9);

                double y1s = 3.5;
                double y1e = 4.5;
                double y2s = -5.5;
                double y2e = -4.5;

                // Draw boxes
                b_box->DrawBox(x1s, y1s, x1e, y1e);
                b_box->DrawBox(x1s, y2s, x1e, y2e);

                // Draw horizontal boundary lines
                l_line->DrawLine(x1s, y1s, x1e, y1s);
                l_line->DrawLine(x1s, y2e, x1e, y2e);

                // Draw vertical boundary lines
                l_line->DrawLine(x1s, y1s, x1s, y1e);
                l_line->DrawLine(x1s, y2s, x1s, y2e);

                l_line->DrawLine(x1e, y1s, x1e, y1e);
                l_line->DrawLine(x1e, y2s, x1e, y2e);
            }

            return;
        }

        if ((dqmObj.name.find("Rct") != std::string::npos ||
                dqmObj.name.find("Jet") != std::string::npos ||
                dqmObj.name.find("IsoEm") != std::string::npos) &&
                dqmObj.name.find("EtaPhi") != std::string::npos) {

            dummybox->Draw("box,same");

            if (dqmObj.name.find("IsoEm") != std::string::npos
                    || dqmObj.name.find("CenJet") != std::string::npos
                    || dqmObj.name.find("TauJet") != std::string::npos) {
                l_line->SetLineWidth(1);
                l_line->DrawLine(3.5, -0.5, 3.5, 17.5);
                l_line->DrawLine(17.5, -0.5, 17.5, 17.5);

                b_box->SetFillColor(1);
                b_box->SetFillStyle(3013);

                b_box->DrawBox(-0.5, -0.5, 3.5, 17.5);
                b_box->DrawBox(17.5, -0.5, 21.5, 17.5);
            }

            if (dqmObj.name.find("ForJet") != std::string::npos) {
                l_line->SetLineWidth(1);
                l_line->DrawLine(3.5, -0.5, 3.5, 17.5);
                l_line->DrawLine(17.5, -0.5, 17.5, 17.5);

                b_box->SetFillColor(1);
                b_box->SetFillStyle(3013);
                b_box->DrawBox(3.5, -0.5, 17.5, 17.5);
            }

            return;
        }

    }
};

static L1TRenderPlugin instance;
