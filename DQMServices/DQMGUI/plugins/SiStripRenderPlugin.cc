/*!
  \file SiStripRenderPlugin
  \brief Display Plugin for SiStrip DQM Histograms
  \author S. Dutta
  \version $Revision: 1.45 $
  \date $Date: 2011/11/16 17:35:55 $
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TProfile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TLine.h"
#include "TMath.h"
#include <cassert>

class SiStripRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo & )
    {
      if ((o.name.find( "SiStrip/" ) == std::string::npos) &&
	  (o.name.find( "Tracking/" ) == std::string::npos))
         return false;

      if( o.name.find( "/EventInfo/" ) != std::string::npos )
        return true;

      if( o.name.find( "/MechanicalView/" ) != std::string::npos )
        return true;

      if( o.name.find( "/ReadoutView/" ) != std::string::npos )
        return true;

      if( o.name.find( "/ControlView/" ) != std::string::npos )
        return true;

      if( o.name.find( "/TrackParameters/" ) != std::string::npos )
        return true;

      if( o.name.find( "/MessageLog/" ) != std::string::npos )
        return true;

      if( o.name.find( "/BaselineValidator/" ) != std::string::npos )
        return true;

      return false;
    }

  virtual void preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo & )
    {
      c->cd();

      if( (dynamic_cast<TProfile*>( o.object ) || dynamic_cast<TProfile2D*>(o.object) || dynamic_cast<TH1F*>(o.object) || dynamic_cast<TH2F*>(o.object) ) && 
	  (o.name.find( "fedErrorsVsIdVsLumi" )!=std::string::npos || o.name.find("LumiBlock")!= std::string::npos || o.name.find("RocTrend")!= std::string::npos))
      {
        TProfile2D*  obj = dynamic_cast<TProfile2D*>(o.object);
        int min_x = (int) obj->FindFirstBinAbove(0.001);
        int max_x = (int) obj->FindLastBinAbove(0.001)+1;     

        if( o.name.find("fedErrorsVsIdVsLumi")!=std::string::npos){
          obj->GetXaxis()->SetRange(min_x, max_x);
	} 
      }

      if( dynamic_cast<TH2F*>( o.object ) )
      {
        preDrawTH2F( c, o );
      }
      else if( dynamic_cast<TH1F*>( o.object ) )
      {
        preDrawTH1F( c, o );
      }
      else if( dynamic_cast<TProfile2D*>( o.object ) )
      {
        preDrawTProfile2D( c, o );
      }
      else if( dynamic_cast<TProfile*>( o.object ) )
      {
        preDrawTProfile( c, o );
      }
    }

  virtual void postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo & )
    {
      c->cd();

      if( dynamic_cast<TH1F*>( o.object ) )
      {
        postDrawTH1F( c, o );
      }
      if( dynamic_cast<TH2F*>( o.object ) )
      {
        postDrawTH2F( c, o );
      }
      if( dynamic_cast<TProfile*>( o.object ) )
      {
        postDrawTProfile( c, o );
      }
    }

private:
  void preDrawTH2F( TCanvas *c, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      // This applies to all
      gStyle->SetCanvasBorderMode( 0 );
      gStyle->SetPadBorderMode( 0 );
      gStyle->SetPadBorderSize( 0 );
      //    (data->pad)->SetLogy( 0 );;
      //  gStyle->SetOptStat( 0 );

      TAxis* xa = obj->GetXaxis();
      TAxis* ya = obj->GetYaxis();

      xa->SetTitleOffset(0.7);
      xa->SetTitleSize(0.05);
      xa->SetLabelSize(0.04);

      ya->SetTitleOffset(0.7);
      ya->SetTitleSize(0.05);
      ya->SetLabelSize(0.04);

      if( o.name.find( "PedsEvolution" ) != std::string::npos)
      {
        gStyle->SetOptStat( 1111 );
        obj->SetStats( kTRUE );
        obj->SetOption( "lego2" );
        return;
      }
      if( o.name.find( "CMDistribution " )  != std::string::npos)
      {
        obj->GetXaxis()->LabelsOption("d");
        obj->SetOption( "lego2" );
        return;
      }
      if( o.name.find( "CMSlopeDistribution " )  != std::string::npos)
      {
        obj->GetXaxis()->LabelsOption("d");
        obj->SetOption( "lego2" );
        return;
      }
      if( o.name.find( "PedestalDistribution " )  != std::string::npos)
      {
        obj->GetXaxis()->LabelsOption("d");
        obj->SetOption( "lego" );
        return;
      }
      if( o.name.find( "reportSummaryMap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        dqm::utils::reportSummaryMapPalette(obj);
        obj->SetOption("colztext");
        return;
      }
      if( o.name.find( "detFractionReportMap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        dqm::utils::reportSummaryMapPalette(obj);
	/*
        if ( o.name.find( "_has" )  != std::string::npos)
        {
          gStyle->SetPalette(60);
        }
	*/
        obj->SetOption("colztext");
        return;
      }
      if( o.name.find( "sToNReportMap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        dqm::utils::reportSummaryMapPalette(obj);
        obj->SetOption("colztext");
        return;
      }
      if( o.name.find( "SummaryOfCabling" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        obj->SetOption("text");
        return;
      }

      if( o.name.find( "DataPresentInLastLS" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        dqm::utils::reportSummaryMapPalette(obj);
        obj->SetOption("colztext");
        return;
      }

	  if( o.name.find( "StripClusVsPixClus" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }

	  if( o.name.find( "ClusterWidths_vs_Amplitudes" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
				c->SetLogz(1);
        obj->SetOption("colz");
	return;
      }


	 if( o.name.find( "TrackEtaPhi" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
				c->SetLogz(1);
        obj->SetOption("colz");
	return;
      }

	 if( o.name.find( "Foldingmap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }
	
	 if( o.name.find( "ControlView" )  != std::string::npos) 
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
        return;
      }

	
	 if( o.name.find( "ControlView" )  != std::string::npos) 
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
        return;
      }
	
	 if( o.name.find( "PhiVsEta" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }

	 if( o.name.find( "PtVsEta" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
        return;
      }

      if( o.name.find( "SeedsVsClusters" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }

      if( o.name.find( "toppingSourceVS" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }

      if( o.name.find( "TracksVs" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }

      if( o.name.find( "DeltaBx_vs_ApvCycle" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }

      if( o.name.find( "ADCvsAPVs" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }
      if( o.name.find( "FedIdVsApvId" )  != std::string::npos)
	{
	  obj->SetStats( kFALSE );
	  gStyle->SetPalette(1,0);
	  obj->SetOption("colz");
	return;
	}
      if( o.name.find( "FEDErrorsVsId" )  != std::string::npos)
	{
	  gPad->SetGrid();
          gPad->SetLeftMargin(0.2);
	  obj->SetStats( kFALSE );
	  gStyle->SetPalette(1,0);
	  obj->SetOption("colz");
	  obj->GetYaxis()->SetTitle("");

	return;
	}
			if( o.name.find( "DataPresentInLS" )  != std::string::npos)
	{
	  obj->SetStats( kFALSE );
	  dqm::utils::reportSummaryMapPalette(obj);
	  obj->SetOption("colz");
	return;
	}
      if( o.name.find( "ErrorsVsModules" )  != std::string::npos)
	{
	  gStyle->SetPalette(1,0);
	  gStyle->SetOptStat(10);
	  obj->SetOption("colz");

	  xa->SetTitleOffset(1.5);
	  ya->SetTitleOffset(1.5);

	  gPad->SetLeftMargin(0.2);
	  gPad->SetBottomMargin( 0.2 );

	  return;
	}

      return;
    }

  void preDrawTH1F( TCanvas *c, const VisDQMObject &o )
    {
      TH1F* obj = dynamic_cast<TH1F*>( o.object );
      assert( obj );

      // This applies to all
      gStyle->SetOptStat(1110);

      //  if ( obj->GetMaximum(1.e5) > 0. ) {
      //    gPad->SetLogy(1);
      //  } else {
      //   gPad->SetLogy(0);
      //  }
      TAxis* xa = obj->GetXaxis();
      TAxis* ya = obj->GetYaxis();

      xa->SetTitleOffset(0.7);
      xa->SetTitleSize(0.05);
      xa->SetLabelSize(0.04);

      ya->SetTitleOffset(0.7);
      ya->SetTitleSize(0.04);
      ya->SetLabelSize(0.04);


       if( o.name.find( "Ratio" )  != std::string::npos)
	{
	      obj->SetOption("e");
	}

      if( o.name.find( "Summary_MeanNumberOfDigis" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
  //        obj->SetMaximum(5.0); // COSMICS
  obj->SetMaximum(20.0); // COLLISIONS run179828 (2011highPU)
  //obj->SetMaximum(120.0); // HEAVY IONS
        obj->SetMinimum(-0.1);
	//do not return, because Summary_MeanNumberOfDigis__TOB string (below) is found here as well
	//return;
      }
      if( o.name.find( "Summary_MeanNumberOfDigis__TOB" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
  obj->SetMaximum(10.0); // COLLISIONS run179828 (2011highPU)
  //obj->SetMaximum(5.0); // COSMICS
  //obj->SetMaximum(70.0); // HEAVY IONS
        obj->SetMinimum(-0.1);
        return;
      }
      if( o.name.find( "Summary_MeanNumberOfClusters" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
	//obj->SetMaximum(1.0);
	obj->SetMaximum(4.0);
        obj->SetMinimum(-0.001);
        //return;
      }
      if( o.name.find( "Summary_MeanNumberOfClusters__TOB" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
	//obj->SetMaximum(1.0);
	obj->SetMaximum(1.5);
        obj->SetMinimum(-0.001);
        return;
      }

      if( o.name.find( "Summary_MeanClusterWidth" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        obj->SetMaximum(10.0);
        obj->SetMinimum(-1.0);
        return;
      }

      if ( o.name.find( "GoodTrackPhi_" ) != std::string::npos or
	   o.name.find( "GoodTrackEta_" ) != std::string::npos or
	   o.name.find( "SeedPhi_" )  != std::string::npos
	   ) {

	size_t nbins        = obj->GetNbinsX();
	double entries      = obj->GetEntries();
	double meanbinentry = entries/double(nbins);
	double ymax = obj->GetMaximum();
	obj->SetMinimum(ymax-meanbinentry);

      }

      std::string name = o.name.substr(o.name.rfind("/")+1);

      if( name.find( "NumberOfTracks_" ) != std::string::npos or
          name.find( "Chi2oNDF_" ) != std::string::npos or
          name.find( "TrackPt_" ) != std::string::npos or
          name.find( "TrackP_" ) != std::string::npos or
          name.find( "NumberOfSeeds_") != std::string::npos or
          name.find( "SeedPt_") != std::string::npos
          ) {
        if (obj->GetEntries() > 10.0) c->SetLogy(1);
        c->SetGridy();
      }

      if ( name.find( "Summary_ClusterCharge_OffTrack__" )!= std::string::npos or
           (name.find( "Track" )!= std::string::npos and
            name.find( "Err" )!= std::string::npos) or
            name.find( "NumberOfRecHitsLostPerTrack_") != std::string::npos or
            name.find( "ClusterMultiplicityRegions") != std::string::npos
           ) {
        if (obj->GetEntries() > 10.0) c->SetLogy(1);
      }

    }

  void preDrawTProfile2D( TCanvas *, const VisDQMObject &o )
    {
      TProfile2D* obj = dynamic_cast<TProfile2D*>( o.object );
      assert( obj );

      // This applies to all
      gStyle->SetCanvasBorderMode( 0 );
      gStyle->SetPadBorderMode( 0 );
      gStyle->SetPadBorderSize( 0 );
      //    (data->pad)->SetLogy( 0 );;
      //  gStyle->SetOptStat( 0 );

      TAxis* xa = obj->GetXaxis();
      TAxis* ya = obj->GetYaxis();

      xa->SetTitleOffset(0.7);
      xa->SetTitleSize(0.05);
      xa->SetLabelSize(0.04);

      ya->SetTitleOffset(0.7);
      ya->SetTitleSize(0.05);
      ya->SetLabelSize(0.04);

      if( o.name.find( "TkHMap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
        if (o.name.find( "TkHMap_FractionOfBadChannels" )  != std::string::npos) {
          obj->SetMinimum(0.0001);
          obj->SetMaximum(1.0);
        }
        return;
      }

       if( o.name.find( "PhiVsEta" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
        return;
      }
        
         if( o.name.find( "PtVsEta" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
        return;
      }

      if( o.name.find( "fedErrorsVsIdVsLumi" )  != std::string::npos){
		obj->SetStats( kFALSE );
        	gStyle->SetPalette(1,0);
		obj->SetOption("colz");
		return;
	}

	  if( o.name.find( "StripClusVsBXandOrbit" ) != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
	return;
      }

	if( o.name.find( "NumberOfClusterPerLayerTrendVar" ) != std::string::npos)
	{
		if (((TString)obj->GetTitle()).Contains( "TIB" )) {
			obj->GetYaxis()->SetRangeUser(1,4);
		} else if (((TString)obj->GetTitle()).Contains( "TOB" ) ) {
			obj->GetYaxis()->SetRangeUser(1,6);
		} else if (((TString)obj->GetTitle()).Contains( "TEC" ) ) {
			obj->GetYaxis()->SetRangeUser(1,9);
		} else if (((TString)obj->GetTitle()).Contains( "TID" )) {
			obj->GetYaxis()->SetRangeUser(1,3);
		}
		obj->SetStats( kFALSE );
		gStyle->SetPalette(1,0);
		obj->SetOption("colz");
	}

	if( o.name.find("NumberOfClusterPerRingVsTrendVar") != std::string::npos){
		obj->SetStats( kFALSE );
		gStyle->SetPalette(1,0);
		obj->SetOption("colz");
		if (((TString)obj->GetTitle()).Contains( "TID" )) {
			obj->GetYaxis()->SetRangeUser(1,3);
		}
		if (((TString)obj->GetTitle()).Contains( "TEC" )) {
			if (((TString)obj->GetTitle()).Contains( "wheel__1" ) or ((TString)obj->GetTitle()).Contains( "wheel__2" ) or ((TString)obj->GetTitle()).Contains( "wheel__3" )) obj->GetYaxis()->SetRangeUser(1,7);
			if (((TString)obj->GetTitle()).Contains( "wheel__4" ) or ((TString)obj->GetTitle()).Contains( "wheel__5" ) or ((TString)obj->GetTitle()).Contains( "wheel__6" )) obj->GetYaxis()->SetRangeUser(2,7);
			if (((TString)obj->GetTitle()).Contains( "wheel__7" ) or ((TString)obj->GetTitle()).Contains( "wheel__8" ) ) obj->GetYaxis()->SetRangeUser(3,7);
			if (((TString)obj->GetTitle()).Contains( "wheel__9" ) ) obj->GetYaxis()->SetRangeUser(4,7);
		}
	}

      return;
    }
  void preDrawTProfile( TCanvas *c, const VisDQMObject &o )
    {
      TProfile* obj = dynamic_cast<TProfile*>( o.object );
      assert( obj );

      // This applies to all
      gStyle->SetCanvasBorderMode( 0 );
      gStyle->SetPadBorderMode( 0 );
      gStyle->SetPadBorderSize( 0 );
      //    (data->pad)->SetLogy( 0 );;
      //  gStyle->SetOptStat( 0 );

      TAxis* xa = obj->GetXaxis();
      TAxis* ya = obj->GetYaxis();

      xa->SetTitleOffset(0.7);
      xa->SetTitleSize(0.05);
      xa->SetLabelSize(0.04);

      ya->SetTitleOffset(0.7);
      ya->SetTitleSize(0.05);
      ya->SetLabelSize(0.04);

      obj->SetStats( kFALSE );
      obj->SetOption("e");

      if( o.name.find( "TkHMap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        gStyle->SetPalette(1,0);
        obj->SetOption("colz");
        if (o.name.find( "TkHMap_FractionOfBadChannels" )  != std::string::npos) {
          obj->SetMinimum(0.0001);
          obj->SetMaximum(1.0);
        }
      return;
    }

      if ( o.name.find( "DistanceOfClosestApproachToBSVsPhi") != std::string::npos ) {
	obj->SetMinimum(-0.02);
	obj->SetMaximum( 0.02);
      }

      if ( o.name.find( "zPointOfClosestApproachVsPhi") != std::string::npos ) {
	obj->SetMinimum(-5.);
	obj->SetMaximum( 5.);
      }

      if ( o.name.find( "TECLayersPerTrackVs") != std::string::npos or
	   o.name.find( "TECRecHitsPerTrackVs") != std::string::npos
	   ) {
	obj->SetMinimum(-0.5);
	obj->SetMaximum(12.);
      }
      if ( o.name.find( "TIBLayersPerTrackVs") != std::string::npos or
	   o.name.find( "TIBRecHitsPerTrackVs") != std::string::npos
	   ) {
	obj->SetMinimum(-0.5);
	obj->SetMaximum(6.);
      }
      if ( o.name.find( "TIDLayersPerTrackVs") != std::string::npos or
	   o.name.find( "TIDRecHitsPerTrackVs") != std::string::npos
	   ) {
	obj->SetMinimum(-0.5);
	obj->SetMaximum(6.);
      }
      if ( o.name.find( "TOBLayersPerTrackVs") != std::string::npos or
	   o.name.find( "TOBRecHitsPerTrackVs") != std::string::npos
	   ) {
	obj->SetMinimum(-0.5);
	obj->SetMaximum(10.);
      }
      if ( o.name.find( "NumberOfLayersPerTrackVs") != std::string::npos or
	   o.name.find( "NumberOfRecHitsPerTrackVs") != std::string::npos or
	   o.name.find( "NumberOfFoundRecHitsPerTrackVs") != std::string::npos
	   ) {
	obj->SetMinimum(-0.5);
	obj->SetMaximum(20.);
      }

      if ( o.name.find( "PixEndcapLayersPerTrackVs" ) != std::string::npos or
	   o.name.find( "PixEndcapRecHitsPerTrackVs") != std::string::npos
	   ) {
	obj->SetMinimum(-0.5);
	obj->SetMaximum(3.);
      }

      if ( o.name.find( "PixBarrelLayersPerTrackVs"  ) != std::string::npos or
	   o.name.find( "PixBarrelRecHitsPerTrackVs" ) != std::string::npos
	   ) {
	obj->SetMinimum(-0.5);
	obj->SetMaximum(4.);
      }

      if ( o.name.find( "ProbVs" ) != std::string::npos or
	   o.name.find( "Chi2oNDFVs" )  != std::string::npos
	   ){
	obj->SetMinimum(0.);
	obj->SetMaximum(1.5);
      }

      if( o.name.find( "TotalNumberOfDigiProfile__" ) != std::string::npos )
        {
         float TIBLimit1 = 45000.0;
         float TOBLimit1 = 53000.0;
         float TIDLimit1 = 9500.0;
         float TECLimit1 = 48500.0;
         obj->SetMinimum(1);
         float ymax = obj->GetMaximum()*1.2;
	
          if (obj->GetEntries() > 10.0) c->SetLogy(1);
          c->SetGridy();

          if (o.name.find( "TotalNumberOfDigiProfile__TIB" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          obj->GetXaxis()->SetRange(0, max_x);
            obj->SetMaximum(TMath::Max(ymax, TIBLimit1*50 ));
          } else if (o.name.find( "TotalNumberOfDigiProfile__TOB" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          obj->GetXaxis()->SetRange(0, max_x);
            obj->SetMaximum(TMath::Max(ymax, TOBLimit1*50 ));
          } else if (o.name.find( "TotalNumberOfDigiProfile__TEC" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          obj->GetXaxis()->SetRange(0, max_x);
            obj->SetMaximum(TMath::Max(ymax, TECLimit1*50 ));
          }  else if (o.name.find( "TotalNumberOfDigiProfile__TID" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          obj->GetXaxis()->SetRange(0, max_x);
            obj->SetMaximum(TMath::Max(ymax, TIDLimit1*50 ));
          }
        }

      if( o.name.find( "TotalNumberOfClusterProfile__" ) != std::string::npos )
	{
	  float TIBLimit2 = 10000.0;
	  float TOBLimit2 = 10000.0;
	  float TIDLimit2 = 2500.0;
	  float TECLimit2 = 13000.0;
	  obj->SetMinimum(1);
	  float ymax = obj->GetMaximum()*1.2;

          if (obj->GetEntries() > 10.0) c->SetLogy(1);
          c->SetGridy();

	  if (o.name.find( "TotalNumberOfClusterProfile__TIB" ) != std::string::npos) {
	    obj->SetMaximum(TMath::Max(ymax, TIBLimit2*20 ));
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          obj->GetXaxis()->SetRange(0, max_x);
	  } else if (o.name.find( "TotalNumberOfClusterProfile__TOB" ) != std::string::npos) {
	    obj->SetMaximum(TMath::Max(ymax, TOBLimit2*20 ));
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          obj->GetXaxis()->SetRange(0, max_x);
	  } else if (o.name.find( "TotalNumberOfClusterProfile__TEC" ) != std::string::npos) {
	    obj->SetMaximum(TMath::Max(ymax, TECLimit2*20 ));
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          obj->GetXaxis()->SetRange(0, max_x);
	  }  else if (o.name.find( "TotalNumberOfClusterProfile__TID" ) != std::string::npos) {
	    obj->SetMaximum(TMath::Max(ymax, TIDLimit2*20 ));
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          obj->GetXaxis()->SetRange(0, max_x);
	  }
	}

      return;
    }

  void postDrawTH1F( TCanvas *, const VisDQMObject &o )
  {
    TH1F* obj = dynamic_cast<TH1F*>( o.object );
    assert( obj );

    TLine tl2;
    tl2.SetLineColor(922); // 15?
    tl2.SetLineWidth(2);
    tl2.SetLineStyle(7);

    TLine tl3;
    tl3.SetLineColor(922); // 15?
    tl3.SetLineWidth(1);
    tl3.SetLineStyle(7);

    TText tt;
    tt.SetTextSize(0.12);
    
    TText tt2;
    tt2.SetTextSize(0.04);
    tt2.SetTextColor(15);
    
    if (o.flags == 0) return;
    else
      {
        if (o.flags & DQMNet::DQM_PROP_REPORT_ERROR)
	  {
	    tt.SetTextColor(2);
	    tt.DrawTextNDC(0.5, 0.5, "Error");
	  }
        else if (o.flags & DQMNet::DQM_PROP_REPORT_WARN)
	  {
	    tt.SetTextColor(5);
	    tt.DrawTextNDC(0.5, 0.5, "Warning");
	  }
        else if (o.flags & DQMNet::DQM_PROP_REPORT_OTHER)
	  {
	    tt.SetTextColor(1);
	    tt.DrawTextNDC(0.5, 0.5, "Other ");
	  }
      }
      
      if( o.name.find( "/ReadoutView/FE/VsId/" ) != std::string::npos || o.name.find( "/ReadoutView/FED/VsId/" ) != std::string::npos || o.name.find( "/ReadoutView/Fiber/VsId/" ) != std::string::npos || o.name.find( "/ReadoutView/DataPresent" ) != std::string::npos )
    {
      float plot_ymax = 1.049*(obj->GetMaximum());
      if ( plot_ymax == 0 ) plot_ymax = 1.049;
      tl3.DrawLine(134.0, 0., 134.0, plot_ymax);
      tl2.DrawLine(164.0, 0., 164.0, plot_ymax);
      tl2.DrawLine(260.0, 0., 260.0, plot_ymax);
      tl2.DrawLine(356.0, 0., 356.0, plot_ymax);

      tt2.DrawTextNDC(0.18, 0.91, "TIB/D");
      tt2.DrawTextNDC(0.38, 0.91, "TEC-");
      tt2.DrawTextNDC(0.55, 0.91, "TEC+");
      tt2.DrawTextNDC(0.72, 0.91, "TOB");
    }

  }

  void postDrawTH2F( TCanvas *c, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      std::string name = o.name.substr(o.name.rfind("/")+1);

      TLine tl1;
      tl1.SetLineColor(2);
      tl1.SetLineWidth(3);
      float mask1_xmin = 26.5;
      float mask1_xmax = 29.5;
      float mask1_ymin = 166.5;
      float mask1_ymax = 236.5;

      float mask2_xmin = 37.5;
      float mask2_xmax = 39.5;
      float mask2_ymin = 387.5;
      float mask2_ymax = 458.5;

      TLine tl2;
      tl2.SetLineColor(922); // 15?
      tl2.SetLineWidth(2);
      tl2.SetLineStyle(7);
      
      TLine tl3;
      tl3.SetLineColor(922); // 15?
      tl3.SetLineWidth(1);
      tl3.SetLineStyle(7);

      TText tt;
      tt.SetTextSize(0.12);
      
      TText tt2;
      tt2.SetTextSize(0.04);
      tt2.SetTextColor(15);
      if (o.flags != 0)
	{
	  if (o.flags & DQMNet::DQM_PROP_REPORT_ERROR)
	    {
	      tt.SetTextColor(2);
	      tt.DrawTextNDC(0.5, 0.5, "Error");
	    }
	  else
	    if (o.flags & DQMNet::DQM_PROP_REPORT_WARN)
	      {
		tt.SetTextColor(5);
		tt.DrawTextNDC(0.5, 0.5, "Warning");
	      }
	  else
	    if (o.flags & DQMNet::DQM_PROP_REPORT_OTHER)
	      {
		tt.SetTextColor(1);
		tt.DrawTextNDC(0.5, 0.5, "Other ");
	      }
	}

      if( name.find( "reportSummaryMap" ) != std::string::npos )
      {
        c->SetGridx();
        c->SetGridy();
        return;
      }
      if( name.find( "detFractionReportMap" ) != std::string::npos )
      {
        c->SetGridx();
        c->SetGridy();
        return;
      }
      if( name.find( "sToNReportMap" ) != std::string::npos )
      {
        c->SetGridx();
        c->SetGridy();
        return;
      }
      if( name.find( "SummaryOfCabling" ) != std::string::npos )
      {
        c->SetGridx();
        c->SetGridy();
        return;
      }
      if( name.find( "DeltaBx_vs_ApvCycle" )  != std::string::npos)
      {
        tl1.DrawLine(mask1_xmin, mask1_ymin, mask1_xmin, mask1_ymax);
        tl1.DrawLine(mask1_xmax, mask1_ymin+2, mask1_xmax, mask1_ymax+2);
        for (int i = 0; i<3; i++){
          tl1.DrawLine(mask1_xmin+i, mask1_ymin+i, mask1_xmin+1+i, mask1_ymin+i);
          tl1.DrawLine(mask1_xmin+1+i, mask1_ymin+i, mask1_xmin+1+i, mask1_ymin+1+i);
          tl1.DrawLine(mask1_xmin+i, mask1_ymax-1+i, mask1_xmin+i, mask1_ymax+i);
          tl1.DrawLine(mask1_xmin+i, mask1_ymax+i, mask1_xmin+1+i, mask1_ymax+i);
        }
        tl1.DrawLine(mask2_xmin, mask2_ymin, mask2_xmin, mask2_ymax);
        tl1.DrawLine(mask2_xmax, mask2_ymin+1, mask2_xmax, mask2_ymax+1);
        for (int i = 0; i<2; i++){
          tl1.DrawLine(mask2_xmin+i, mask2_ymin+i, mask2_xmin+1+i, mask2_ymin+i);
          tl1.DrawLine(mask2_xmin+1+i, mask2_ymin+i, mask2_xmin+1+i, mask2_ymin+1+i);
          tl1.DrawLine(mask2_xmin+i, mask2_ymax-1+i, mask2_xmin+i, mask2_ymax+i);
          tl1.DrawLine(mask2_xmin+i, mask2_ymax+i, mask2_xmin+1+i, mask2_ymax+i);
        }
        return;
      }
      if( o.name.find( "FEDErrorsVsId" )  != std::string::npos || o.name.find( "ApvIdVsFedId" )  != std::string::npos )
	    {
        float err_ymin = obj->GetYaxis()->GetXmin();
        float err_ymax = obj->GetYaxis()->GetXmax();
        tl3.DrawLine(134.0, err_ymin, 134.0, err_ymax);
        tl2.DrawLine(164.0, err_ymin, 164.0, err_ymax);
        tl2.DrawLine(260.0, err_ymin, 260.0, err_ymax);
        tl2.DrawLine(356.0, err_ymin, 356.0, err_ymax);

        tt2.DrawTextNDC(0.27, 0.92, "TIB/D");
        tt2.DrawTextNDC(0.43, 0.92, "TEC-");
        tt2.DrawTextNDC(0.58, 0.92, "TEC+");
        tt2.DrawTextNDC(0.77, 0.92, "TOB");
      }
    }

  void postDrawTProfile( TCanvas *, const VisDQMObject &o )
  {
    TProfile* obj = dynamic_cast<TProfile*>( o.object );
    assert( obj );

    std::string name = o.name.substr(o.name.rfind("/")+1);

    TLine tl1;
    tl1.SetLineColor(3);
    tl1.SetLineWidth(3);

    TLine tl2;
    tl2.SetLineColor(4);
    tl2.SetLineWidth(3);
    
    TLine tl3;
    tl3.SetLineColor(922); // 15?
    tl3.SetLineWidth(2);
    tl3.SetLineStyle(7);

    TLine tl4;
    tl4.SetLineColor(922); // 15?
    tl4.SetLineWidth(1);
    tl4.SetLineStyle(7);

    TText tt;
    tt.SetTextSize(0.04);
    tt.SetTextColor(15);

    float xmin = 0.0;
//    float xmax = obj->GetXaxis()->GetXmax();
//    float ymax = obj->GetMaximum()*1.2;

    // FOR PP
    /*
    float TIBLimit1 = 10000.0;
    float TOBLimit1 = 11000.0;
    float TIDLimit1 = 2000.0;
    float TECLimit1 = 10000.0;

    float TIBLimit2 = 2000.0;
    float TOBLimit2 = 2000.0;
    float TIDLimit2 = 600.0;
    float TECLimit2 = 2400.0;
    */
    float TIBLimit1 = 45000.0;
    float TOBLimit1 = 53000.0;
    float TIDLimit1 = 9500.0;
    float TECLimit1 = 48500.0;

    float TIBLimit2 = 10000.0;
    float TOBLimit2 = 10000.0;
    float TIDLimit2 = 2500.0;
    float TECLimit2 = 13000.0;
    /*
    //FOR HI
    float TIBLimit1 = 70000.0;
    float TOBLimit1 = 70000.0;
    float TIDLimit1 = 15000.0;
    float TECLimit1 = 70000.0;

    float TIBLimit2 = 15000.0;
    float TOBLimit2 = 15000.0;
    float TIDLimit2 = 4000.0;
    float TECLimit2 = 20000.0;
    */

    if( name.find( "TotalNumberOfDigiProfile__" ) != std::string::npos )
      {
        if (name.find( "TotalNumberOfDigiProfile__TIB" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)*10;
           tl1.DrawLine(xmin, TIBLimit1,     max_x, TIBLimit1);
           tl2.DrawLine(xmin, TIBLimit1*0.5, max_x, TIBLimit1*0.5);
           tl2.DrawLine(xmin, TIBLimit1*2.0, max_x, TIBLimit1*2.0);
//          obj->SetMinimum(TIBLimit1*0.1);
          //axis range set in PreDraw function to enable zooming in GUI
//          obj->SetMinimum(1);
//          obj->SetMaximum(TMath::Max(ymax, TIBLimit1*50));
        } else if (name.find( "TotalNumberOfDigiProfile__TOB" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)*10;
           tl1.DrawLine(xmin, TOBLimit1,     max_x, TOBLimit1);
           tl2.DrawLine(xmin, TOBLimit1*0.5, max_x, TOBLimit1*0.5);
           tl2.DrawLine(xmin, TOBLimit1*2.0, max_x, TOBLimit1*2.0);
//          obj->SetMinimum(TOBLimit1*0.1);
//          obj->SetMinimum(1);
//	  obj->SetMaximum(TMath::Max(ymax, TOBLimit1*50));
        } else if (name.find( "TotalNumberOfDigiProfile__TEC" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)*10;
          tl1.DrawLine(xmin, TECLimit1,     max_x, TECLimit1);
          tl2.DrawLine(xmin, TECLimit1*0.5, max_x, TECLimit1*0.5);
          tl2.DrawLine(xmin, TECLimit1*2.0, max_x, TECLimit1*2.0);
//          obj->SetMinimum(TECLimit1*0.1);
//          obj->SetMinimum(1);
//          obj->SetMaximum(TMath::Max(ymax, TECLimit1*50));
        } else if (name.find( "TotalNumberOfDigiProfile__TID" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)*10;
           tl1.DrawLine(xmin, TIDLimit1,     max_x, TIDLimit1);
           tl2.DrawLine(xmin, TIDLimit1*0.5, max_x, TIDLimit1*0.5);
           tl2.DrawLine(xmin, TIDLimit1*2.0, max_x, TIDLimit1*2.0);
//          obj->SetMinimum(TIDLimit1*0.1);
//          obj->SetMinimum(1);
//          obj->SetMaximum(TMath::Max(ymax, TIDLimit1*50));
	}
        return;
      }
    if( name.find( "TotalNumberOfClusterProfile__" ) != std::string::npos )
      {
        if (name.find( "TotalNumberOfClusterProfile__TIB" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          tl1.DrawLine(xmin, TIBLimit2,     max_x, TIBLimit2);
          tl2.DrawLine(xmin, TIBLimit2*0.5, max_x, TIBLimit2*0.5);
          tl2.DrawLine(xmin, TIBLimit2*2.0, max_x, TIBLimit2*2.0);
	  //axis range set in PreDraw function to enable zooming in GUI
          //obj->SetMinimum(1);
	  //obj->SetMaximum(TMath::Max(ymax, TIBLimit2*20));
        } else if (name.find( "TotalNumberOfClusterProfile__TOB" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          tl1.DrawLine(xmin, TOBLimit2,     max_x, TOBLimit2);
          tl2.DrawLine(xmin, TOBLimit2*0.5, max_x, TOBLimit2*0.5);
          tl2.DrawLine(xmin, TOBLimit2*2.0, max_x, TOBLimit2*2.0);
          //obj->SetMinimum(1);
          //obj->SetMaximum(TMath::Max(ymax, TOBLimit2*20));
        } else if (name.find( "TotalNumberOfClusterProfile__TEC" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          tl1.DrawLine(xmin, TECLimit2,     max_x, TECLimit2);
          tl2.DrawLine(xmin, TECLimit2*0.5, max_x, TECLimit2*0.5);
          tl2.DrawLine(xmin, TECLimit2*2.0, max_x, TECLimit2*2.0);
          //obj->SetMinimum(1);
          //obj->SetMaximum(TMath::Max(ymax, TECLimit2*20));
        }  else if (name.find( "TotalNumberOfClusterProfile__TID" ) != std::string::npos) {
            int max_x = (int) obj->FindLastBinAbove(0.001)+1;
          tl1.DrawLine(xmin, TIDLimit2,     max_x, TIDLimit2);
          tl2.DrawLine(xmin, TIDLimit2*0.5, max_x, TIDLimit2*0.5);
          tl2.DrawLine(xmin, TIDLimit2*2.0, max_x, TIDLimit2*2.0);
          //obj->SetMinimum(1);
          //obj->SetMaximum(TMath::Max(ymax, TIDLimit2*20));
        }
        return;
      }
    
    if( o.name.find( "/ReadoutView/FedEventSize" ) != std::string::npos )
    {
      float plot_ymax = 1.049*(obj->GetMaximum()+2*sqrt(obj->GetMaximum()));
      if ( plot_ymax == 0 ) plot_ymax = 1.049;
      tl4.DrawLine(134.0, -0.05, 134.0, plot_ymax);
      tl3.DrawLine(164.0, -0.05, 164.0, plot_ymax);
      tl3.DrawLine(260.0, -0.05, 260.0, plot_ymax);
      tl3.DrawLine(356.0, -0.05, 356.0, plot_ymax);

      tt.DrawTextNDC(0.18, 0.91, "TIB/D");
      tt.DrawTextNDC(0.38, 0.91, "TEC-");
      tt.DrawTextNDC(0.55, 0.91, "TEC+");
      tt.DrawTextNDC(0.72, 0.91, "TOB");
    }

    if( o.name.find( "/MechanicalView/NumberOfDigisinFED_v_FEDID" ) != std::string::npos || o.name.find( "/MechanicalView/NumberOfClustersinFED_v_FEDID" ) != std::string::npos )
    {
      float plot_ymax = 1.049*(obj->GetMaximum()+2*sqrt(obj->GetMaximum()));
      if ( plot_ymax == 0 ) plot_ymax = 1.049;
      tl4.DrawLine(134.0, -0.05, 134.0, plot_ymax);
      tl3.DrawLine(164.0, -0.05, 164.0, plot_ymax);
      tl3.DrawLine(260.0, -0.05, 260.0, plot_ymax);
      tl3.DrawLine(356.0, -0.05, 356.0, plot_ymax);

      tt.DrawTextNDC(0.18, 0.91, "TIB/D");
      tt.DrawTextNDC(0.38, 0.91, "TEC-");
      tt.DrawTextNDC(0.55, 0.91, "TEC+");
      tt.DrawTextNDC(0.72, 0.91, "TOB");
    }
  }

};

static SiStripRenderPlugin instance;
