#include <fstream>
#include <math.h>
#include <algorithm>
#include <cassert>

#include <TMath.h>

#include "Riostream.h"
#include "TSystem.h"
#include "TDatime.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TColor.h"
#include "TF1.h"
#include "TPad.h"
#include "TMarker.h"
#include "TFrame.h"
#include "TGResourcePool.h"
#include "TGCanvas.h"
#include "TGraphErrors.h"
#include "TProfile.h"
#include "TText.h"
#include "TImage.h"
#include "TAttImage.h"

#include "../../interface/MEGeom.h"
#include "../../interface/MEChannel.h"

#include "MusEcalGUI.hh"
#include "MEVarVector.hh"
#include "MECorrector2Var.hh"
#include "MEEBDisplay.hh"
#include "MEEEDisplay.hh"
#include "MEPlotWindow.hh"
#include "MERunPanel.hh"
#include "MEChanPanel.hh"
#include "MELeafPanel.hh"
#include "MEMultiVarPanel.hh"
#include "METwoVarPanel.hh"
#include "MERun.hh"
#include "MERunManager.hh"
#include "MEIntervals.hh"

ClassImp(MusEcalGUI)

MusEcalGUI::MusEcalGUI( const TGWindow *p, UInt_t w, UInt_t h,
			int type, int color )
:  TGMainFrame( p, w, h ), MECanvasHolder(), MusEcal( type, color )
{
  // init
  _isGUI       = true;
  _debug       = false;
  _write       = false;
  _ihtype      = iMAP;
  _icateg      = 0;
  _ihist       = ME::iAPD_MEAN;
  _drawIntervals = true;
  _historyType = iHistoryVsTime;
  _fRunPanel   = 0;
  _fChanPanel  = 0;
  _fLeafPanel  = 0;
  _fMultiVarPanel  = 0;
  _fTwoVarPanel  = 0;
  _psdir = TString(getenv("MEPSDIR"))+"/";
  
  
  // layouts and menus
  cout << "Setup main window" << endl;
  setupMainWindow();
  cout << "done." << endl;

  // create the run panel
  cout << "Create Run panel" << endl;
  createRunPanel();
  cout << "done." << endl;

  // configure and book histograms
  cout << "Configure histograms" << endl;
  histConfig();
  cout << "done." << endl;

  cout << "Book histograms" << endl;
  bookHistograms();
  cout << "done." << endl;
  
  // fill histograms
  cout << "Fill histograms" << endl;
  fillHistograms();
  cout << "done." << endl;

  // default drawings

  cout << "Default drawings" << endl;  

  //  drawHist(1);
  cout << "Default history" << endl;  
  //historyPlot(true);
  //  leafPlot(1);
  cout << "done." << endl;
}

// get vectors -- version with std::vectors
bool
MusEcalGUI::getTimeVector( vector< ME::Time >& time )
{
  if( _debug ) cout <<"getTimeVector: 0"<< endl;
  if( _leaf==0 ) return false;
  if( _debug ) cout <<"getTimeVector: 1"<< endl;
  time.clear();
  MEVarVector* apdVector_ = curMgr()->apdVector(_leaf);
  apdVector_->getTime( time );
  if( _debug ) cout <<"getTimeVector: 2 "<< time.size()<<endl;
  return true;
}

bool
MusEcalGUI::getHistoryVector( vector< ME::Time >& time, 
			      vector< float    >& val,
			      vector< bool     >& flag,
			      vector< float    >& eval,
			      bool& b_, 
			      float& miny, float& maxy )
{
  if( _leaf==0 ) 
    {
      cout << "Please select a channel first " << endl;
      return false;
    }
  int ig_;
  TString varName_;
  TString str_, str0_;
  if( _type==ME::iLaser )
    {
      ig_ = MusEcal::iGVar[_var];
      varName_ = MusEcal::historyVarName[_var];
    }
  else if( _type==ME::iTestPulse )
    {
      ig_ = MusEcal::iGTPVar[_var];
      varName_ = MusEcal::historyTPVarName[_var];
    }
  else 
    return false;
  MEChannel* leaf_ = _leaf;
  if( ig_<_ig )
    {
      leaf_ = _leaf->getAncestor( ig_ );
    }

  //  cout << "History plot for var=" << varName_
  //       << " and channel " << leaf_->oneLine() << endl;

  MEChannel* pnleaf_=_leaf;
  MEChannel* mtqleaf_=_leaf;
  if(_leaf->ig()>ME::iLMModule){
    while( pnleaf_->ig() != ME::iLMModule){
      pnleaf_=pnleaf_->m();
    }
  }
  if(_leaf->ig()>ME::iLMRegion){
    while( mtqleaf_->ig() != ME::iLMRegion){
      mtqleaf_=mtqleaf_->m();
    }
  }
 
  val.clear();
  eval.clear();
  flag.clear();
  vector< float > rms;
  vector< float > nevt;
  vector< bool  > flag_;
  b_   = true;
  miny =  999999; 
  maxy = -999999;

  getTimeVector( time );
  MEVarVector* apdVector_ = curMgr()->apdVector(leaf_);
  MEVarVector* pnaVector_ = curMgr()->pnVector(pnleaf_,0);
  MEVarVector* pnbVector_ = curMgr()->pnVector(pnleaf_,1);
  MEVarVector* mtqVector_ = curMgr()->mtqVector(mtqleaf_);  
  MEVarVector* midVector_ = midVector(leaf_);
  MEVarVector* nlsVector_ = nlsVector(leaf_);

  if( _type==ME::iLaser )
    {
      TString str0_("APD-"), str1_("PN-"), str2_("MTQ-");
      
      if( _var==MusEcal::iCLS )
	{ 
	  // JM
	  nlsVector_->getValAndFlag( ME::iCLS_MEAN, time, val, flag);
	  nlsVector_->getValAndFlag( ME::iCLS_RMS, time, rms, flag_ );
	  nlsVector_->getValAndFlag( ME::iCLS_NEVT, time, nevt, flag_ );


	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      if( _var==MusEcal::iCLSN )
	{ 
	  vector< float > cls, clsrms, norm;
	  // JM

	  nlsVector_->getValAndFlag( ME::iCLS_MEAN, time, cls, flag_);
	  nlsVector_->getValAndFlag( ME::iCLS_RMS,  time, clsrms, flag_ );
	  nlsVector_->getValAndFlag( ME::iCLS_NEVT, time, nevt, flag_ );
	  nlsVector_->getValAndFlag( ME::iCLS_NORM, time, norm, flag );

	  for( unsigned int icls=0; icls<time.size(); icls++ )
	    {
	      cout<<icls<<" "<<cls[icls]<<" "<<clsrms[icls]<<" "<<norm[icls]<<" "<<flag[icls]<< endl;
	      val.push_back(cls[icls]*norm[icls]);
	      rms.push_back(clsrms[icls]*norm[icls]);
	    }

	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iNLS )
	{ 	  
	  // JM
	  nlsVector_->getValAndFlag( ME::iNLS_MEAN, time, val, flag);
	  nlsVector_->getValAndFlag( ME::iNLS_RMS, time, rms, flag_ );
	  nlsVector_->getValAndFlag( ME::iNLS_NEVT, time, nevt, flag_ );

	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	} 
      else if( _var==MusEcal::iNLSN )
	{ 	   
	  vector< float > nls, nlsrms, norm;
	  // JM
	  nlsVector_->getValAndFlag( ME::iNLS_MEAN, time, nls, flag_);
	  nlsVector_->getValAndFlag( ME::iNLS_RMS, time, nlsrms, flag_ );
	  nlsVector_->getValAndFlag( ME::iNLS_NEVT, time, nevt, flag_ );
	  nlsVector_->getValAndFlag( ME::iNLS_NORM, time, norm, flag );

	  for( unsigned int inls=0; inls<time.size(); inls++ )
	    {
	      val.push_back(nls[inls]*norm[inls]);
	      rms.push_back(nlsrms[inls]*norm[inls]);
	    }
	
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      
      
      else if( _var==MusEcal::iCLSNORM )
	{ 	  
	  // JM
	  nlsVector_->getValAndFlag( ME::iCLS_NORM, time, val, flag);
	  
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      
      else if( _var==MusEcal::iNLSNORM )
	{ 	  
	  // JM
	  nlsVector_->getValAndFlag( ME::iNLS_NORM, time, val, flag);
	  
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	} 
      else if( _var==MusEcal::iMID )
	{ 
	  // JM
	  midVector_->getValAndFlag( ME::iMID_MEAN, time, val, flag);
	  midVector_->getValAndFlag( ME::iMID_RMS, time, rms, flag_ );
	  midVector_->getValAndFlag( ME::iMID_NEVT, time, nevt, flag_ );


	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iMIDA )
	{ 
	  // JM
	  midVector_->getValAndFlag( ME::iMIDA_MEAN, time, val, flag);
	  midVector_->getValAndFlag( ME::iMIDA_RMS, time, rms, flag_ );
	  midVector_->getValAndFlag( ME::iMIDA_NEVT, time, nevt, flag_ );


	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}  else if( _var==MusEcal::iMIDB )
	{ 
	  // JM
	  midVector_->getValAndFlag( ME::iMIDB_MEAN, time, val, flag);
	  midVector_->getValAndFlag( ME::iMIDB_RMS, time, rms, flag_ );
	  midVector_->getValAndFlag( ME::iMIDB_NEVT, time, nevt, flag_ );


	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPD )
	{
	  apdVector_->getValAndFlag( ME::iAPD_RMS,        time, rms, flag_ );
	  apdVector_->getValAndFlag( ME::iAPD_MEAN,       time, val, flag );
	  apdVector_->getValAndFlag( ME::iAPD_NEVT,       time, nevt, flag_ ); 
	  str_=str0_+ME::APDPrimVar[ME::iAPD_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDTime )
	{
	  apdVector_->getValAndFlag( ME::iAPD_TIME_RMS,   time, rms, flag_ );
	  apdVector_->getValAndFlag( ME::iAPD_TIME_MEAN,  time, val, flag );
	  apdVector_->getValAndFlag( ME::iAPD_TIME_NEVT,  time, nevt, flag_ ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_TIME_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDNevt )
	{
	  apdVector_->getValAndFlag( ME::iAPD_NEVT,  time, val, flag );
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_NEVT]; 
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	} 
      else if( _var==MusEcal::iAPDTimeNevt )
	{
	  apdVector_->getValAndFlag( ME::iAPD_TIME_NEVT,  time, val, flag );
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_TIME_NEVT];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDoPNA )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNA_MEAN, time, val, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNA_RMS,  time, rms, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNA_NEVT,  time, nevt, flag_ ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PNA_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDoPNB )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNB_MEAN, time, val, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNB_RMS,  time, rms, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNB_NEVT, time, nevt, flag_ ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PNB_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDoPN )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PN_MEAN, time, val, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PN_RMS,  time, rms, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PN_NEVT, time, nevt, flag_ ); 
	  
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }

	}
      else if( _var==MusEcal::iAPDoPNACOR )
	{
	  //      remove temporarely
	  //      apdVector_->getValAndFlag( ME::iAPD_OVER_PNACOR_MEAN, time, val, flag );
	  // 	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNACOR_RMS,  time, rms, flag );
	  // 	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNACOR_NEVT,  time, nevt, flag_ ); 
	  
	  // JM
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNATMPCOR_MEAN, time, val, flag);
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNATMPCOR_RMS, time, rms, flag_ );
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNATMPCOR_NEVT, time, nevt, flag_ );
	  
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PNA_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDoPNBCOR )
	{
	  // apdVector_->getValAndFlag( ME::iAPD_OVER_PNBCOR_MEAN, time, val, flag );
// 	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNBCOR_RMS,  time, rms, flag );
// 	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNBCOR_NEVT, time, nevt, flag_ ); 
	  // JM
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNBTMPCOR_MEAN, time, val, flag);
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNBTMPCOR_RMS, time, rms, flag_ );
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNBTMPCOR_NEVT, time, nevt, flag_ );
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PNB_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDoPNCOR )
	{
	 //  apdVector_->getValAndFlag( ME::iAPD_OVER_PNCOR_MEAN, time, val, flag );
// 	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNCOR_RMS,  time, rms, flag );
// 	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNCOR_NEVT, time, nevt, flag_ ); 
	  
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNTMPCOR_MEAN, time, val, flag);
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNTMPCOR_RMS, time, rms, flag_ );
	  midVector_->getValAndFlag( ME::iAPD_OVER_PNTMPCOR_NEVT, time, nevt, flag_ );

	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }

	}
     //  else if( _var==MusEcal::iAPDABFIToPNACOR )
// 	{
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNACOR_MEAN, time, val, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNACOR_RMS,  time, rms, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNACOR_NEVT,  time, nevt, flag_ ); 
// 	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
// 	  if( hist_nbin(str_)!=0 ) 
// 	    {
// 	      b_ = false;
// 	      miny = hist_min(str_);
// 	      maxy = hist_max(str_);
// 	    }
// 	}
//       else if( _var==MusEcal::iAPDABFIToPNBCOR )
// 	{
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNBCOR_MEAN, time, val, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNBCOR_RMS,  time, rms, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNBCOR_NEVT, time, nevt, flag_ ); 
// 	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
// 	  if( hist_nbin(str_)!=0 ) 
// 	    {
// 	      b_ = false;
// 	      miny = hist_min(str_);
// 	      maxy = hist_max(str_);
// 	    }
// 	}
//       else if( _var==MusEcal::iAPDABFIToPNCOR )
// 	{
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNCOR_MEAN, time, val, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNCOR_RMS,  time, rms, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIT_OVER_PNCOR_NEVT, time, nevt, flag_ ); 
	  
// 	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
// 	  if( hist_nbin(str_)!=0 ) 
// 	    {
// 	      b_ = false;
// 	      miny = hist_min(str_);
// 	      maxy = hist_max(str_);
// 	    }

// 	}else if( _var==MusEcal::iAPDABFIXoPNACOR )
// 	{
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNACOR_MEAN, time, val, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNACOR_RMS,  time, rms, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNACOR_NEVT,  time, nevt, flag_ ); 
// 	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
// 	  if( hist_nbin(str_)!=0 ) 
// 	    {
// 	      b_ = false;
// 	      miny = hist_min(str_);
// 	      maxy = hist_max(str_);
// 	    }
// 	}
//       else if( _var==MusEcal::iAPDABFIXoPNBCOR )
// 	{
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNBCOR_MEAN, time, val, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNBCOR_RMS,  time, rms, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNBCOR_NEVT, time, nevt, flag_ ); 
// 	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
// 	  if( hist_nbin(str_)!=0 ) 
// 	    {
// 	      b_ = false;
// 	      miny = hist_min(str_);
// 	      maxy = hist_max(str_);
// 	    }
// 	}
//       else if( _var==MusEcal::iAPDABFIXoPNCOR )
// 	{
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNCOR_MEAN, time, val, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNCOR_RMS,  time, rms, flag );
// 	  apdVector_->getValAndFlag( ME::iAPDABFIX_OVER_PNCOR_NEVT, time, nevt, flag_ ); 
	  
// 	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PN_MEAN];
// 	  if( hist_nbin(str_)!=0 ) 
// 	    {
// 	      b_ = false;
// 	      miny = hist_min(str_);
// 	      maxy = hist_max(str_);
// 	    }

// 	}
      else if( _var==MusEcal::iAPDoPNANevt )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNA_NEVT,  time, val, flag ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PNA_NEVT];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDoPNBNevt )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_PNB_NEVT,  time, val, flag ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_PNB_NEVT];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}

      else if( _var==MusEcal::iAPDoAPDA )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_APDA_MEAN, time, val, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_APDA_RMS,  time, rms, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_APDA_NEVT, time, nevt, flag_ ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_APDA_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;

	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDoAPDB )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_APDB_MEAN, time, val, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_APDB_RMS,  time, rms, flag );
	  apdVector_->getValAndFlag( ME::iAPD_OVER_APDB_NEVT, time, nevt, flag_ ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_APDB_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;

	      miny = hist_min(str_);
	      maxy = hist_max(str_);

	    }
	}
      else if( _var==MusEcal::iAPDoAPDANevt )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_APDA_NEVT, time, val, flag ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_APDA_NEVT];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;

	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iAPDoAPDBNevt )
	{
	  apdVector_->getValAndFlag( ME::iAPD_OVER_APDB_NEVT, time, val, flag ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_OVER_APDB_NEVT];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;

	      miny = hist_min(str_);
	      maxy = hist_max(str_);

	    }
	}
      else if( _var==MusEcal::iShapeCorAPD )
	{
	  apdVector_->getValAndFlag( ME::iAPD_SHAPE_COR, time, val, flag ); 
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_SHAPE_COR];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;

	      miny = hist_min(str_);
	      maxy = hist_max(str_);

	    }
	}
      else if( _var==MusEcal::iShapeCorRatio )
	{
	  vector< float > scapd, scpna, scpnb;
	  vector< bool > flagapd, flagpna, flagpnb;

	  apdVector_->getValAndFlag( ME::iAPD_SHAPE_COR, time, scapd, flagapd ); 
	  pnaVector_->getValAndFlag( ME::iPN_SHAPE_COR,       time, scpna, flagpna );
	  pnbVector_->getValAndFlag( ME::iPN_SHAPE_COR,       time, scpnb, flagpnb );

	  for( unsigned int ii=0; ii<time.size(); ii++ )
	    {
	      float denom=0.; int icc=0;
	      if( scpnb[ii]>0. && flagpnb[ii] ){
		denom+=scpnb[ii]; 
		icc++;
	      }
	      if( scpna[ii]>0. && flagpna[ii] ){
		denom+=scpna[ii];
		icc++;
	      }
	      if( denom==0. || icc==0 ){
		denom=1.0;
		flag.push_back(false);
	      }else{
		denom/=float(icc);
		flag.push_back(flagapd[ii]);
	      }
	      val.push_back(scapd[ii]/denom);
	      
	    }
	  
	  TString str_=str0_+ME::APDPrimVar[ME::iAPD_SHAPE_COR];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	      
	    }
	}
      else if( _var==MusEcal::iPNA )
	{
	  pnaVector_->getValAndFlag( ME::iPN_MEAN, time, val, flag );
	  pnaVector_->getValAndFlag( ME::iPN_RMS,  time, rms, flag );
	  pnaVector_->getValAndFlag( ME::iPN_NEVT, time, nevt, flag_ ); 
	  TString str_=str1_+ME::PNPrimVar[ME::iPN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;

	      miny = hist_min(str_);
	      maxy = hist_max(str_);

	    }
	}
      else if( _var==MusEcal::iPNB )
	{
	  pnbVector_->getValAndFlag( ME::iPN_MEAN, time, val, flag );
	  pnbVector_->getValAndFlag( ME::iPN_RMS,  time, rms, flag );
	  pnbVector_->getValAndFlag( ME::iPN_NEVT, time, nevt, flag_ ); 
	  TString str_=str1_+ME::PNPrimVar[ME::iPN_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	      
	    }

	}
      else if( _var==MusEcal::iPNANevt )
	{
	  pnaVector_->getValAndFlag( ME::iPN_NEVT, time, val, flag );  
	  TString str_=str1_+ME::PNPrimVar[ME::iPN_NEVT];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	      
	    }
	}
      else if( _var==MusEcal::iPNBNevt )
	{
	  pnbVector_->getValAndFlag( ME::iPN_NEVT, time, val, flag );
	  TString str_=str1_+ME::PNPrimVar[ME::iPN_NEVT];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	      
	    }

	}else if( _var==MusEcal::iPNARMS )
	{
	  
	  vector< float > val1_, rms1_;
	  vector< bool > flag1_;

	  pnaVector_->getValAndFlag( ME::iPN_RMS, time, rms1_, flag1_ ); 
	  pnaVector_->getValAndFlag( ME::iPN_NEVT, time, nevt, flag1_ ); 
	  pnaVector_->getValAndFlag( ME::iPN_MEAN, time, val1_, flag );  
	  
	  double findmean=0.0;
	  int cmean=0;
	  for( unsigned int iab=0; iab<time.size(); iab++ )
	    {
	      double ratio=0.0;
	      if(val1_[iab]>0.0) ratio=rms1_[iab]/val1_[iab];
	      val.push_back( ratio );
	      cout<< "ratio "<<ratio<<" "<<iab<<" "<<val1_[iab]<<" "<<rms1_[iab]<<endl;
	      findmean+=ratio;
	      cmean++;
	    }
	  if (cmean!=0) findmean/=double(cmean);
	  
	  cout<<"mean : "<< findmean<<" "<<cmean<< " "<< time.size()<<" "<<val.size()<<" "<<endl;
	  b_ = false;
	  miny = 0.0;
	  maxy = findmean*2.0;
	  
	}
      else if( _var==MusEcal::iPNBRMS )
	{ 
	  vector< float > val1_, rms1_;
	  vector< bool > flag1_;

	  pnbVector_->getValAndFlag( ME::iPN_RMS, time, rms1_, flag1_ );
	  pnaVector_->getValAndFlag( ME::iPN_NEVT, time, nevt, flag1_ ); 
	  pnbVector_->getValAndFlag( ME::iPN_MEAN, time, val1_, flag );  
	  
	  double findmean=0.0;
	  int cmean=0;
	  for( unsigned int iab=0; iab<time.size(); iab++ )
	    {
	      double ratio=0.0;
	      if(val1_[iab]>0.0)ratio=rms1_[iab]/val1_[iab];
	      val.push_back( ratio );
	      findmean+=ratio;
	      cmean++;
	    }
	  if (cmean!=0) findmean/=double(cmean);
	  
	  b_ = false;
	  miny = 0.0;
	  maxy = findmean*2.0;

	}
      else if( _var==MusEcal::iPNBoPNA )
	{
	  pnaVector_->getValAndFlag( ME::iPNA_OVER_PNB_MEAN, time, val, flag );
	  TString str_=str1_+ME::PNPrimVar[ME::iPNA_OVER_PNB_MEAN];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	      
	    }
	}
      else if( _var==MusEcal::iShapeCorPNA )
	{
	  pnaVector_->getValAndFlag( ME::iPN_SHAPE_COR,       time, val, flag );
	  str_=str0_+ME::PNPrimVar[ME::iPN_SHAPE_COR];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iShapeCorPNB )
	{
	  pnbVector_->getValAndFlag( ME::iPN_SHAPE_COR,       time, val, flag );
	  str_=str0_+ME::PNPrimVar[ME::iPN_SHAPE_COR];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      // else if( _var==MusEcal::iAlphaBeta )
// 	{
// 	  vector< float > alpha_, beta_;
// 	  vector< bool > flaga_, flagb_;
// 	  apdVector_->getValAndFlag( ME::iAPD_ALPHA, time, alpha_, flaga_ );
// 	  apdVector_->getValAndFlag( ME::iAPD_BETA,  time, beta_, flagb_ );
// 	  double findmean=0.0;
// 	  int cmean=0;
// 	  for( unsigned int iab=0; iab<time.size(); iab++ )
// 	    {
// 	      val.push_back( alpha_[iab]*beta_[iab] );
// 	      findmean+=alpha_[iab]*beta_[iab];
// 	      cmean++;
// 	      flag.push_back( flaga_[iab]&&flagb_[iab] );
// 	    }
// 	  if (cmean!=0) findmean/=double(cmean);
	  
// 	  b_ = false;
// 	  miny = 0.0;
// 	  maxy = findmean*1.2;
// 	}
      else if( _var==MusEcal::iMTQTrise )
	{
	  
	  mtqVector_->getValAndFlag( ME::iMTQ_RISE, time, val, flag ); 
	  TString str_=str2_+ME::MTQPrimVar[ME::iMTQ_RISE];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iMTQAmpl )
	{
	  mtqVector_->getValAndFlag( ME::iMTQ_AMPL, time, val, flag ); 
	  TString str_=str2_+ME::MTQPrimVar[ME::iMTQ_AMPL];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    }
	}
      else if( _var==MusEcal::iMTQFwhm )
	{ 

	  mtqVector_->getValAndFlag( ME::iMTQ_FWHM, time, val, flag ); 
	  TString str_=str2_+ME::MTQPrimVar[ME::iMTQ_FWHM];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    } 
	}
      else if( _var==MusEcal::iMTQFw10 )
	{
	  mtqVector_->getValAndFlag( ME::iMTQ_FW10, time, val, flag ); 
	  TString str_=str2_+ME::MTQPrimVar[ME::iMTQ_FW10];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    } 
	}
      else if( _var==MusEcal::iMTQFw05 )
	{ 
	  mtqVector_->getValAndFlag( ME::iMTQ_FW05, time, val, flag ); 
	  TString str_=str2_+ME::MTQPrimVar[ME::iMTQ_FW05];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    } 
	}
      else if( _var==MusEcal::iMTQTime )
	{
	  mtqVector_->getValAndFlag( ME::iMTQ_TIME, time, val, flag ); 
	  TString str_=str2_+ME::MTQPrimVar[ME::iMTQ_TIME];
	  if( hist_nbin(str_)!=0 ) 
	    {
	      b_ = false;
	      miny = hist_min(str_);
	      maxy = hist_max(str_);
	    } 
	}
    }
  else if( _type==ME::iTestPulse )
    {
      TString str0_("TPAPD-"), str1_("TPPN-");
      if( _var==MusEcal::iTPAPD_0 )
	{
  
	  cout << varName_ << ": not implemented yet" << endl;
	  return false;
	}
      else if( _var==MusEcal::iTPAPD_1 )
	{
	  apdVector_->getValAndFlag( ME::iTPAPD_RMS,        time, rms, flag_ );
 	  apdVector_->getValAndFlag( ME::iTPAPD_MEAN,       time, val, flag );
 	  apdVector_->getValAndFlag( ME::iTPAPD_NEVT,       time, nevt, flag_ ); 
 	  str_=str0_+ME::APDPrimVar[ME::iTPAPD_MEAN];
 	  
	  if( hist_nbin(str_)!=0 ) 
 	    {
 	      b_ = false;
 	      miny = hist_min(str_);
 	      maxy = hist_max(str_);
 	    }
	} 
      else if( _var==MusEcal::iTPAPD_2 )
	{
	  cout << varName_ << ": not implemented yet" << endl;
	  return false;
	}
      if( _var==MusEcal::iTPPNA_0 )
	{
	  cout << varName_ << ": not implemented yet" << endl;
	  return false;
	}
      else if( _var==MusEcal::iTPPNA_1 )
	{ 
	  pnaVector_->getValAndFlag( ME::iTPPN_RMS,        time, rms, flag_ );
 	  pnaVector_->getValAndFlag( ME::iTPPN_MEAN,       time, val, flag );
 	  apdVector_->getValAndFlag( ME::iTPAPD_NEVT,       time, nevt, flag_ ); 
 	  str_=str0_+ME::PNPrimVar[ME::iTPPN_MEAN];
 	  
	  if( hist_nbin(str_)!=0 ) 
 	    {
 	      b_ = false;
 	      miny = hist_min(str_);
 	      maxy = hist_max(str_);
 	    }
	}
      if( _var==MusEcal::iTPPNB_0 )
	{
	  cout << varName_ << ": not implemented yet" << endl;
	  return false;
	}
      else if( _var==MusEcal::iTPPNB_1 )
	{ 

	  pnbVector_->getValAndFlag( ME::iTPPN_RMS,        time, rms, flag_ );
 	  pnbVector_->getValAndFlag( ME::iTPPN_MEAN,       time, val, flag );
 	  apdVector_->getValAndFlag( ME::iTPAPD_NEVT,       time, nevt, flag ); 
 	  str_=str0_+ME::PNPrimVar[ME::iTPPN_MEAN];
 	  
	  if( hist_nbin(str_)!=0 ) 
 	    {
 	      b_ = false;
 	      miny = hist_min(str_);
 	      maxy = hist_max(str_);
 	    }
	}
    }
  
  assert(  val.size()==time.size() );
  assert( flag.size()==time.size() );
  if( eval.size()==0 && rms.size()!=0 )
    {
      assert( rms.size()==time.size() );
      for( unsigned int itime=0; itime<time.size(); itime++ )
	{
	  float eval_ = 100000.;
	  if( nevt.size()==time.size() )
	    {
	      float nevt_ = nevt[itime];
	      if( nevt_>0 ) eval_ = rms[itime]/sqrt( nevt_ );
	    } 
	  else
	    {
	      eval_ = rms[itime];
	    }
	  eval.push_back( eval_ );
	}
    }
  return true;
}

// get vectors -- version with C-style vectors
bool
MusEcalGUI::getHistoryVector( unsigned int& nrun, 
			       float* x, float* y, float* ex, float* ey,
			       bool* ok, 
			       float& normy, float& miny, float& maxy )
{
  vector< ME::Time > time;
  vector< float > val;
  vector< float > eval;
  vector< bool > flag;
  miny =  999999; 
  maxy = -999999;
  bool b_(true);
  bool ok_ = getHistoryVector( time, val, flag, eval, b_, miny, maxy );
  if( !ok_ ) return false;

  nrun = time.size();
  if( nrun<2 ) 
    {
      cout << "Not enough runs for an history plot, nrun=" << nrun << endl;
      return false;
    }

  normy=0; 
  int ngood_=0;

  for( unsigned int irun=0; irun<nrun; irun++ )
    {
      float dt_ = ME::timeDiff( time[irun], _time, ME::iHour );
      float val_ = val[irun];
      bool flag_ = flag[irun];
      float eval_=0.;
      if( eval.size()==time.size() ) eval_=eval[irun];
      x[irun] = dt_;
      y[irun] = val_;
      ex[irun] = 0;
      ey[irun] = eval_;
      ok[irun] = flag_;
      if( flag_ )
	{
	  ngood_++;
	  normy+=val_;
	}
      if(b_)
	{
	  if( val_>maxy ) maxy=val_;
	  if( val_<miny ) miny=val_;
	}
    }
  if( b_ ) maxy*=1.2;
  if( ngood_>0 ) normy/=ngood_;
  return true;
}

void
MusEcalGUI::historyPlot( int opt )
{
  TString WName_="History";
  MEPlotWindow* win_ = getWindow( WName_, opt, 800, 450 );
  if( win_==0 ) return;
  _curPad = win_->getPad();
  _curPad->cd();
  if(_debug) cout<< "historyPlot: 0 "<< endl;
  vector< ME::Time > time;
  if( !getTimeVector( time ) ) return;
  unsigned int nrun = time.size();
  if(_debug) cout<< "historyPlot: getTimeVector OK "<<  time.size()<< endl;

  float x[nrun];
  float y[nrun];
  float ex[nrun];
  float ey[nrun];
  bool ok[nrun];
  float miny;
  float maxy;
  float normy;						
  if( !getHistoryVector( nrun, x, y, ex, ey, ok, 
			  normy, miny, maxy ) ) return;
  if(_debug) cout<< "historyPlot: getHistoryVector OK "<<normy<<" "<<miny<<" "<< maxy<< endl;

  
  
  float minx = x[0];
  float maxx = x[nrun-1];
  float rangex = maxx-minx; 
  minx -= 0.05*rangex;
  maxx += 0.05*rangex;

  if( _normalize )
    {
      //assert( normy!=0 );
      if( normy==0 ){
	cout<< " Impossible to normalize, no valid data"<< endl ;
	normy=1.;
      }
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  y[ii]  /= normy;
	  ey[ii] /= normy;
	}
      miny   /= normy;
      maxy   /= normy;      
    }

  TString titleX;
  TString titleY;
  TString titleM;
  TString titleW;
  titleW = ME::type[_type];
  if( _type==ME::iLaser )
    {
      titleW+="."; titleW+=ME::color[_color];
      titleW+="."; titleW+=MusEcal::historyVarName[_var];
      titleY = MusEcal::historyVarName[_var];
    }
  else if( _type==ME::iTestPulse )
    {
      titleW+="."; titleW+=MusEcal::historyTPVarName[_var];
      titleY = MusEcal::historyTPVarName[_var];
    }
  titleW+="."; titleW+=_leaf->oneWord();
  titleW+="."; 
  MERun* run_ = _runMgr.begin()->second->curRun();
  //  assert( run_!=0 );
  if( run_==0 )
    {
      cout << "Warning -- no run available ";
      return;
    }
  titleW+=run_->rundir();

  if( _normalize ) titleW+=".normalized";
  titleM = titleW;
  titleM.ReplaceAll("."," ");
  titleW+=".history";

  if( _historyType==iHistoryVsTime )
    {
      titleW+=".vs.time";

      TString GraphOpt( "PSame" );
      titleX = "time (in hours, relative to current run)";
      
      //  float xcur; 
      float markerSize = 0;
      int markerStyle   = 1;
      int markerColor   = kBlack;
      int lineWidth     = 1;
      int lineColor     = kBlue;
      
      int ncol = gStyle->GetNumberOfColors();
      
      vector<TMarker*> markers;
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  float f = float(ii)/float(nrun);
	  int jj = (int) (f*ncol);      
	  if( jj<1 ) jj=1;
	  int icol = gStyle->GetColorPalette( jj );
	  int mstyle;
	  if( ok[ii] )
	    {
	      mstyle = 20;
	    }
	  else
	    {
	      mstyle = 24;
	    }
	  
	  TMarker* marker = new TMarker( x[ii], y[ii], mstyle );
	  
	  marker->SetMarkerSize( 0.8 );
	  marker->SetMarkerColor( icol );
	  markers.push_back( marker );
	}
      


      TH1* hdum  = (TH1*) gROOT->FindObject( "href" );
      if( hdum!=0 ) delete hdum;
      TH1* _href = new TH1F( "href", "href", 100,minx, maxx );
      
      setHistoStyle( _href );
      
      cout << " historyPlot: drawIntervals: "<<  _drawIntervals<< endl;
      if( _drawIntervals )
	{
	  MERun* run_ = _runMgr.begin()->second->curRun();
	  ME::Time t0_=run_->time();
	  MEIntervals* intervals_ = intervals( _leaf );
	  if(_debug) cout << " multiVarPlot : draw intervals " << endl;
	  drawIntervals( intervals_, miny, maxy, t0_ );
	}
      
      
      _href->SetMinimum( miny );
      _href->SetMaximum( maxy );
      _href->SetStats( kFALSE );
      _href->SetTitle(titleM);
      _href->GetXaxis()->SetTitle(titleX);
      _href->GetYaxis()->SetTitle(titleY);
      _href->Draw();
      drawHistoryGraph( nrun, 
			x, y, ex, ey, ok,
			markerStyle, markerSize, markerColor,
			lineWidth, lineColor, GraphOpt.Data() );
      win_->setCurHist( _href );
      _curPad->SetCrosshair(1);
      _curPad->SetGridx(0);
      _curPad->SetGridy(1);
      
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  
	  // JM: don't draw bad runs
	  //if(markers[ii]->GetMarkerStyle()==20){
	  markers[ii]->Draw();
	    //}
	  
	}  
    }
  else if( _historyType==iHistoryProjection )
    {
      titleW+=".projection";

      bool automaticScale(true);
      
      float _miny=miny;
      float _maxy=maxy;
      if( automaticScale )
	{
	  float a=0;
	  float a2=0;
	  int n = 0;
	  for( unsigned ii=0; ii<nrun; ii++ )
	    {
	      if( !ok[ii] ) continue;
	      a  += y[ii];
	      a2 += y[ii]*y[ii];
	      n++;
	    }
	  a /= n;
	  a2 /= n;
      
	  float rms = a2-a*a;
	  if( rms>0 )
	    {
	      rms = sqrt(rms);
	      _miny = a - 10*rms;
	      _maxy = a + 10*rms;
	    }
	}

      TString hname = MusEcal::historyVarName[ _var ];
      hname += ", projection";
      TH1* _href =  new TH1F( hname, hname, 200, _miny, _maxy );             
      setHistoStyle( _href );
      _href->SetTitle(titleM);
      _href->SetStats( kTRUE );      
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  if( !ok[ii] ) continue;
	  _href->Fill( y[ii] );
	}
      _href->Draw();

//       l = new TLine( xcur, _href->GetMinimum(), 
// 		     xcur, _href->GetMaximum() );
//       l->SetLineStyle(2);
//       l->SetLineColor(kRed);
//       l->Draw();

      win_->setCurHist( _href );
      _curPad->SetCrosshair(0);
      _curPad->SetGridx(0);
      _curPad->SetGridy(1);
    }

  _curPad->Modified();
  _curPad->Update();

  win_->setPrintName( _psdir+titleW+".ps" );
  if(_write) win_->write();
}

void 
MusEcalGUI::multiVarPlot( int opt )
{
  if( _leaf==0 ) return;
  TString WName_="MultiVar";

  MEPlotWindow* win_ = getWindow( WName_, opt, 700, 700 );
  if( win_==0 ) return;
  _curPad = win_->getPad();
  _curPad->cd();

  vector< ME::Time > time;
  if( !getTimeVector( time ) ) return;
  unsigned int nrun = time.size();
  float x[nrun];
  float ex[nrun];
  for( unsigned int irun=0; irun<nrun; irun++ )
    {
      float dt_ = ME::timeDiff( time[irun], _time, ME::iHour );
      x[irun] = dt_;
      ex[irun] = 0;
    }
  float minx = x[0];
  float maxx = x[nrun-1];
  float rangex = maxx-minx; 
  minx -= 0.05*rangex;
  maxx += 0.05*rangex;

  unsigned nplot = 0;
  unsigned nvar(0);
  if( _type==ME::iLaser) nvar=MusEcal::iSizeLV;
  else nvar=MusEcal::iSizeTPV;
  vector<   int   > varZoom(  nvar );
  vector< TString > varName(  nvar );
  vector<  float  > varNorm(  nvar );
  vector<    int  > varColor( nvar );
  for( unsigned var=0; var<nvar; var++ )
    {
      if( _type==ME::iLaser )
	{ 
	  varZoom[var]  = MusEcal::historyVarZoom[_type][var];
	  varName[var]  = MusEcal::historyVarName[var];
	  varColor[var] = MusEcal::historyVarColor[var];
	}
      else if( _type==ME::iTestPulse )
	{
	  varZoom[var]  = MusEcal::historyTPVarZoom[var];
	  varName[var]  = MusEcal::historyTPVarName[var];
	  varColor[var] = MusEcal::historyTPVarColor[var];
	}
      else
	{
	  abort();
	}
      if( varZoom[var]!=MusEcal::iZero ) nplot ++;      
    }
  float  y[nplot][nrun];
  float ey[nplot][nrun];
  bool   ok[nplot][nrun];
  unsigned iplot = 0;
  unsigned plottedVar[nplot];
  int curVar_ = _var;
  for( unsigned var=0; var<nvar; var++ )
    {
      if( varZoom[var]==MusEcal::iZero ) continue;
      
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  y[iplot][ii]  = 0;
	  ey[iplot][ii] = 0;
	  ok[iplot][ii] = true;
	}
	
      float norm(0); 
      float miny(0); 
      float maxy(0); 
      
      setVar( var );
      assert( getHistoryVector( nrun, x, y[iplot], ex, ey[iplot], ok[iplot],
				norm, miny, maxy ) );
      
      if( _type==ME::iLaser && (var==MusEcal::iCLS || var==MusEcal::iNLS) ) norm = 1;
      
      float zr = MusEcal::zoomRange[ varZoom[var] ];
      
      float yy(0);
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  yy = y[iplot][ii];

	  yy = (yy-norm)/norm/zr;
	  if( yy>1  ) yy = 1; 
	  if( yy<-1 ) yy = -1;

	  yy += 2*(nplot-iplot)-1;

	  y[iplot][ii] = yy;
	  ey[iplot][ii] = 0; // fixme!
	  plottedVar[iplot]   = var;

	}
      iplot++;
    }
  setVar( curVar_ );
  float miny=0;
  float maxy=2*nplot;

  //
  // OK, all is set, now plotting
  //
  TString titleX;
  TString titleY;
  TString titleM;
  TString titleW;

  titleW = ME::type[_type];
  if( _type==ME::iLaser )
    {
      titleW+="."; titleW+=ME::color[_color];
    }
  titleW+="."; titleW+=_leaf->oneWord();
  titleW+="."; 
  MERun* run_ = _runMgr.begin()->second->curRun();
  assert( run_!=0 );
  titleW+=run_->rundir();
  titleM = titleW;
  titleM.ReplaceAll("."," ");
  titleW+=".multiVar";

  titleX = "time (in hours, relative to current run)";
      
  TString hname( "_multivar" );
  TH1* hdum  = (TH1*) gROOT->FindObject( hname );
  if( hdum ) delete hdum;
  hdum = new TH1F( hname, hname, 100, minx, maxx );

  setHistoStyle( hdum );

  hdum->SetMinimum( miny );
  hdum->SetMaximum( maxy );
  hdum->SetStats( kFALSE );

  TAxis* ax = hdum->GetXaxis();
  TAxis* ay = hdum->GetYaxis();
  ax->SetTitle(titleX);
  ay->SetLabelFont(0);
  ay->SetLabelSize(0);
  ay->SetTitleFont(0);
  ay->SetTitleSize(0);
  ay->SetNdivisions( 2*nplot, true );
  hdum->SetTitle(titleM);
  hdum->Draw();
  
  _curPad->SetCrosshair(1);
  _curPad->SetGridx(0);
  _curPad->SetGridy(0);

  if( _drawIntervals )
    {
      MERun* run_ = _runMgr.begin()->second->curRun();
      ME::Time t0_=run_->time();
      MEIntervals* intervals_ = intervals (_leaf );
      if(_debug) cout << " multiVarPlot : draw intervals " << endl;
      drawIntervals( intervals_, miny, maxy, t0_ );
    }
      
  for( unsigned iplot=0; iplot<nplot; iplot++ )
    {
      TLine* line = new TLine( minx, 2*(nplot-iplot )-1, maxx, 2*(nplot-iplot )-1 );
      line->SetLineWidth(1);
      line->SetLineColor(kBlack);
      line->Draw("Same");

      int icol =  varColor[ plottedVar[iplot] ];
      float markerSize = 0.3;
      int markerStyle = 1;
      int markerColor = icol;
      int lineWidth = 2;
      int lineColor = icol;
      drawHistoryGraph( nrun, 
			x, y[iplot], ex, ey[iplot], ok[iplot],
			markerStyle, markerSize, markerColor,
			lineWidth, lineColor );

      //      TText* text = new TText( first_run - 0.75*( last_run-minx ), 
      //			       2*(nplot-iplot)-1+0.5,  varName[ plottedVar[iplot] ]  );
      TText* text = new TText( minx + 0.05*( maxx - minx ), 
			       2*(nplot-iplot)-1+0.5,  varName[ plottedVar[iplot] ]  );
      text->SetTextAlign( 11 ); // 10*left-ajusted + bottom-ajusted
      text->SetTextFont( 132 ); 
      text->SetTextSizePixels( 20 ); 
      text->Draw("Same");

      //      TText* text2 = new TText( maxx - 0.5*(maxx - last_run ), 
      //				2*(nplot-iplot)-1+0.5,  zoomName[ varZoom[ plottedVar[iplot] ] ]  );
      TText* text2 = new TText( maxx - 0.05*(maxx - minx ), 
				2*(nplot-iplot)-1+0.5, MusEcal::zoomName[ varZoom[ plottedVar[iplot] ] ]  );
      text2->SetTextAlign( 31 ); // 10*right-ajusted + bottom-ajusted
      text2->SetTextFont( 132 ); 
      text2->SetTextSizePixels( 18 ); 
      text2->Draw("Same");
      {
	TLine* line = new TLine( minx, 2*(nplot-iplot ), maxx, 2*(nplot-iplot ) );
	line->SetLineWidth(2);
	line->SetLineColor(kBlack);
	line->Draw("Same");
      }
    }
  TLine* line = new TLine( minx, 0, maxx, 0 );
  line->SetLineWidth(2);
  line->SetLineColor(kBlack);
  line->Draw("Same");
  
  gPad->Modified();
  gPad->Update();

  win_->setPrintName( _psdir+titleW+".ps" );
  if(_write) win_->write();
}

void 
MusEcalGUI::leafPlot( int opt )
{

  if( _leaf==0 ) return;
  MEChannel* curLeaf_ = _leaf;
  //  MEChannel* mother_  = _leaf->m();
  MEChannel* mother_  = _leaf;
  if( mother_==0 ) return;
  int iG = mother_->ig();
  bool doZoom = true;
  if( iG<ME::iLMRegion )
    {
      cout << "Already at Laser Monitoring Region level" << endl;
      doZoom = false;
    }
  else if( _type==ME::iTestPulse )
    {
      if( iG>=MusEcal::iGTPVar[_var] ) doZoom=false;
    }
  else
    {
      if( iG>=MusEcal::iGVar[_var] )   doZoom=false;
    }

  TString WName_="Leaf";
  if( !doZoom )
    {
      if( _window.count(WName_)!=0 ) delete _window[WName_];
      historyPlot(2);
      return;
    }
  MEPlotWindow* win_ = getWindow( WName_, opt, 800, 800 );
  if( win_==0 ) return;
  _curPad = win_->getPad();
  _curPad->cd();

  //  unsigned int nrun=2000;
  
  vector< ME::Time > time;
  if( !getTimeVector( time ) ) return;
  unsigned int nrun = time.size();

  float x[nrun];
  float y[nrun];
  float ex[nrun];
  float ey[nrun];
  bool ok[nrun];
  float miny;
  float maxy;
  float normy;						
  if( !getHistoryVector( nrun, x, y, ex, ey, ok, 
			  normy, miny, maxy ) )
    return;
  
  unsigned nl = 0;
  unsigned nc  = 0;
  unsigned nplot = mother_->n(); 
  if( nplot<=5 )
    {
      nc = 1;
      nl = nplot;
    }
  else if( nplot<=10 )
    {
      nc = 2;
      nl = (nplot+1)/2;
    }
  else if( nplot==25 )
    {
      nc = 5;
      nl = 5;
    }
  else
    {
      cout << "--- Number of plots not supported : " << nplot << endl;
      return;
    }

  TString titleX;
  TString titleY;
  TString titleM;
  TString titleW;
  //titleX = "time (in hours from first run)";

  TString varName;
  int     varColor;
  titleW = ME::type[_type];
  if( _type==ME::iLaser )
    { 
      varName  = MusEcal::historyVarName[_var];
      varColor = MusEcal::historyVarColor[_var];
      titleW+="."; titleW+=ME::color[_color]; 
    }
  else if( _type==ME::iTestPulse )
    {
      varName  = MusEcal::historyTPVarName[_var];
      varColor = MusEcal::historyTPVarColor[_var];
    }
  else
    {
      abort();
    }
  titleW+="."; titleW+=varName;
  titleY += varName;
  titleY += ", ";
  if( _zoom==MusEcal::iZero ) 
    {
      _zoom = MusEcal::iThirtyPercent;
    }      
  titleY += MusEcal::zoomName[_zoom ];

  titleW+="."; titleW+=mother_->oneWord();
  titleW+="."; 
  MERun* run_ = _runMgr.begin()->second->curRun();
  assert( run_!=0 ); 
  titleW+= run_->rundir();
  titleM = titleW;
  titleM.ReplaceAll("."," ");
  titleW+=".leaf";
  win_->setPrintName( _psdir+titleW+".ps" );

  int tx100_ = ((int)(100*(x[nrun-1]-x[0])));
  int hrs_  = tx100_/100;
  int min_  = ((tx100_%100)*60)/100;
	       
  titleM += " ("; 
  titleM += hrs_; titleM += " hrs ";
  titleM += min_; titleM += " min)";

  _curPad->SetCrosshair(1);
  _curPad->SetGridx(0);
  _curPad->SetGridy(0);
  float xxmin=0;
  float xxmax=(float)nc;
  float yymin=0;
  float yymax=(float)(2*nl);

  TString hname( "leaf_" );
  TH1* hdum  = (TH1*) gROOT->FindObject( hname );
  if( hdum ) delete hdum;

  TH1* _href = new TH1F( hname, hname, 100, xxmin, xxmax );
  setHistoStyle( _href );
  _href->SetMinimum( yymin );
  _href->SetMaximum( yymax );
  _href->SetStats( kFALSE );
  win_->setCurHist( _href );

  TAxis* ax = _href->GetXaxis();
  TAxis* ay = _href->GetYaxis();
  ax->SetTitle(titleX);
  ax->SetLabelFont(0);
  ax->SetLabelSize(0);
  ax->SetTitleFont(0);
  ax->SetTitleSize(0);
  ax->SetNdivisions(nc, true );
  ay->SetTitle(titleY);
  ay->SetLabelFont(0);
  ay->SetLabelSize(0);
  ay->SetNdivisions(2*nl, true );
  _href->SetTitle(titleM);
  _href->Draw();

  gPad->Modified();
  gPad->Update();

  float zr = MusEcal::zoomRange[ _zoom ];
  for( unsigned iplot=0; iplot<nplot; iplot++ )
    {
      int ic=iplot/nl;
      int il=iplot%nl;

      //
      // get the vectors
      //
      MEChannel* leaf_ = mother_->d(iplot);
      TString oneWord_;
      oneWord_+=leaf_->id();

      _leaf=leaf_;
      //      cout << leaf_->oneLine() << endl;
      if( !getHistoryVector( nrun, x, y, ex, ey, ok, 
			      normy, miny, maxy ) ) continue;
      unsigned int nrun_=1000;
      assert( nrun<nrun_ );
      float x_[nrun_];
      float y_[nrun_];
      float ex_[nrun_];
      float ey_[nrun_];
      bool ok_[nrun_];
      float norm_(0); 
      float miny_(1000000); 
      float maxy_(0);
      int n_(0);
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  ex_[ii] = 0.;
	  ey_[ii] = 0.;
	  x_[ii] = ( x[ii]-x[0] )/( x[nrun-1]-x[0] );
	  y_[ii] =  0.;
	  ok_[ii] = ok[ii];
	  if( ok_[ii] )
	    {
	      if( y[ii]<miny_ ) miny_=y[ii]; 
	      if( y[ii]>maxy_ ) maxy_=y[ii]; 
	      norm_+=y[ii];
	      n_++;
	    }
	}
      if( n_>0 ) norm_/=n_;
      nrun_ = nrun;
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  float yy = y[ii];

	  yy = (yy-norm_)/norm_/zr;
	  if( yy>1  ) yy = 1; 
	  if( yy<-1 ) yy = -1;
	  yy += (2*il+1);

	  x_[ii] += ic;	  
	  y_[ii] = yy;

	}
 
      TLine* baseline = new TLine( ic, 2*il+1, ic+1, 2*il+1 );
      baseline->SetLineWidth(1);
      baseline->SetLineColor(kBlack);
      baseline->Draw("Same");

      int icol =  varColor;
      float markerSize = 0.3;
      int markerStyle = 1;
      int markerColor = icol;
      int lineWidth = 2;
      int lineColor = icol;

      drawHistoryGraph( nrun_, 
			x_, y_, ex_, ey_, ok_,
			markerStyle, markerSize, markerColor,
			lineWidth, lineColor );

      TText* text = new TText( ic    +  0.97,  // 95% of 1
			       2*il  +  1.7,   // 85% of 2 
			       oneWord_ );
      text->SetTextAlign( 31 );  // 10*right-ajusted + bottom-ajusted
      text->SetTextFont( 132 ); 
      int txtsize =  (18*4)/nl;  // proportional to the number of line, 18 when nl=4
      text->SetTextSizePixels( txtsize ); 
      text->Draw("Same");

      TLine* line[4];
      line[0] = new TLine( ic, 2*il, ic+1, 2*il );
      line[1] = new TLine( ic+1, 2*il, ic+1, 2*(il+1) );
      line[2] = new TLine( ic+1, 2*(il+1), ic, 2*(il+1) );
      line[3] = new TLine( ic, 2*(il+1), ic, 2*il );
      for( int kk=0; kk<4; kk++ )
	{
	  line[kk]->SetLineWidth(2);
	  line[kk]->SetLineColor(kBlack);
	  line[kk]->Draw("Same");
	}
      gPad->Modified();
      gPad->Update();
    }

  _leaf = curLeaf_;
  
  gPad->Modified();
  gPad->Update();

  if(_write) win_->write();
}



void 
MusEcalGUI::correlation2Var( int var0, int var1, int zoom0, int zoom1, int fitDegree, int opt)
{
  
  if(_debug) cout << "Entering correlation2Var " << endl;
  
  if( _leaf==0 ) return;
  TString WName_="Correlation2Var";
  
  MEPlotWindow* win_ = getWindow( WName_, opt, 700, 700 );
  if( win_==0 ) return;
  _curPad = win_->getPad();
  _curPad->cd();
  
  if(_debug) cout << "Before getTime " << endl;
  vector< ME::Time > time;
  if( !getTimeVector( time ) ) return;
  unsigned int nrun = time.size();
  float x[nrun];
  float ex[nrun];
  for( unsigned int irun=0; irun<nrun; irun++ )
    {
      float dt_ = ME::timeDiff( time[irun], _time, ME::iHour );
      x[irun] = dt_;
      ex[irun] = 0;
    }
  if(_debug) cout << "Before setVar " << endl;
 
  int varIndex[ 2 ] = { var0, var1 };
  int varZoom[ 2 ]  = { zoom0, zoom1 };
  
  float x_[2][nrun];
  float y_[2][nrun];
  float ey_[2][nrun];
  bool   ok_[2][nrun];
  float norm_[2];
  float min_[2];
  float max_[2];
  vector<bool> keep(nrun,true);
  for( unsigned var=0; var<2; var++ )
    {
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  y_[var][ii] = 0.;
	  ey_[var][ii] = 0.;
	  ok_[var][ii] = true;
	}
      norm_[var] = 0;
      min_[var] = 0;
      max_[var] = 0;
      
      //_leaf->getValFlagAndNorm(varIndex[var],time,y_[var],ok_[var],norm_[var]);
      
      setVar(varIndex[var]);
      assert( getHistoryVector( nrun ,x_[var], y_[var], ex, ey_[var], ok_[var],
				norm_[var], min_[var], max_[var] ) );
      
      if(_debug) cout << "After getHistoryVector " << endl;
      //if( varIndex[var]==iNLS || varIndex[var]==iCorNLS ) norm_[var]=1;
      //  if( varIndex[var]==_corVar0 ) norm_[var]=_corX0;
      // if( varIndex[var]==_corVar1 ) norm_[var]=_corY0;
      
      double zr = zoomRange[ varZoom[var] ];
      
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  bool   oo = ok_[var][ii];
	  double yy = (y_[var][ii]-norm_[var])/norm_[var]/zr;
	  if( !oo ) keep[ii] = false;
	  if( yy<-1 || yy>1 ) keep[ii] = false;
	  y_[var][ii] = yy;
	}
    }
  double minx = -1;
  double maxx = +1;
  
  double miny = -1;
  double maxy = +1;
  
  // Ok, now plotting
  
  if(_debug) cout << "now plotting" << endl;
  TString titleX;
  TString titleY;
  TString titleM;
  
  titleM += ME::type[_type];
  titleM += ", ";
  if(_type==ME::iLaser) titleM+= ME::color[_color];

  titleM += ME::type[_type];
  titleM += ", ";
  if(_type==ME::iLaser) titleM+= ME::color[_color];
  //  titleM += ", Runs ";
  //titleM += first_run;
  //titleM += " to ";
  //titleM += last_run;
  titleM += "   ";
  
  titleM += "  ";
  titleM += _leaf->oneLine();
  if(_debug) cout << titleM << endl;

  TString axisTitle[ 2 ];
  TString varName[ 2 ];

  for( unsigned var=0; var<2; var++ )
    {
      if( varZoom[var]<0 )
	varZoom[var] = historyVarZoom[_type][varIndex[var]];
      varName[var] = historyVarName[varIndex[var]];
      axisTitle[var] += varName[var];
      axisTitle[var] += ", ";
      axisTitle[var] += zoomName[ varZoom[var] ];
    }
  
  TString hname( "_href" );
  TH1* hdum  = (TH1*) gROOT->FindObject( hname );
  if( hdum )  delete hdum;
  
  TH2* href = new TH2F( hname, hname, 100,minx, maxx, 100, miny, maxy );
  href->SetMinimum( 1 );
  href->SetMaximum( nrun );
  href->Fill(10*maxx,10*maxy,nrun); // fixme !!!

  if(_debug) cout << "After Fill" << endl;
  setHistoStyle( href );

  href->SetStats( kFALSE );

  _curPad->cd();

  TAxis* ax = href->GetXaxis();
  TAxis* ay = href->GetYaxis();
  ax->SetTitle( axisTitle[0] );
  ax->SetNdivisions( 2, true );
  ay->SetTitle( axisTitle[1] );
  ay->SetNdivisions( 2, true );
  href->SetTitle(titleM);
  href->Draw("COLZ");
  
  _curPad->SetCrosshair(0);
  _curPad->SetGridx(1);
  _curPad->SetGridy(1);

  int ncol = gStyle->GetNumberOfColors();
  if( _debug ) cout << "Number of palette colors "  << ncol << endl;
  
  MERun* run_ = _runMgr.begin()->second->curRun();
  if( run_==0 )
    {
      cout << "Warning -- no run available ";
      return;
    }

  vector<double> xgood;
  vector<double> ygood;
  vector<TMarker*> markers;
  for( unsigned ii=0; ii<nrun; ii++ )
    {
      double f = double(ii)/double(nrun);
      int jj = (int) (f*ncol);      
      if( jj<1 ) jj=1;
      int icol = gStyle->GetColorPalette( jj );
      if( _debug ) cout << "ii/f/jj/icol" << ii << "/" << f << "/" << jj << "/" << icol << endl;
      int mstyle;
 
      bool isCurrent =  time[ii]==run_->time();
      if( keep[ii] )
	{
	  mstyle = 20;
	  if( isCurrent ) mstyle=29;
	}
      else
	{
	  mstyle = 24;
	  if( isCurrent ) mstyle=30;
	}

      TMarker* marker = new TMarker( y_[0][ii], y_[1][ii], mstyle );

      marker->SetMarkerSize( 1 );
      if( isCurrent ) marker->SetMarkerSize( 2 );
      marker->SetMarkerColor( icol );
      markers.push_back( marker );
      if( keep[ii] )
	{
	  xgood.push_back( y_[0][ii] );
	  ygood.push_back( y_[1][ii] );
	}
    }
  
  unsigned NN = xgood.size();
  double XX[NN]; double YY[NN]; 
  for( unsigned jj=0; jj<NN; jj++ ) { XX[jj]=xgood[jj]; YY[jj]=ygood[jj]; }
  TGraph* gr = new TGraph( NN, XX, YY );
  TString dumstr = "pol"; dumstr += fitDegree;
  TF1* f1 = new TF1("f1",dumstr.Data(),minx,maxx);
  gr->Fit("f1","N0");
  f1->SetLineWidth(2);
  f1->SetLineColor(kRed);
  f1->Draw("Same");
  if(fitDegree<=2){
    float slope=f1->GetParameter(1);
    float ratio;
    if(zoomRange[varZoom[0]]*norm_[0]!=0){
      ratio= zoomRange[varZoom[1]]*norm_[1]/(zoomRange[varZoom[0]]*norm_[0]);
      cout << "Real Slope = "<<ratio*slope<< endl;
      if(fitDegree==2) cout << "Real 2nd order = "<<ratio*ratio*f1->GetParameter(2)<< endl;
    }
  }
  
  if(_debug) cout << "After Draw" << endl;
  for( unsigned ii=0; ii<nrun; ii++ )
    {
      markers[ii]->Draw();
    }  
  
  if(_debug) cout << "OK update pad " << endl;
  gPad->Modified();
  gPad->Update();
 

}

void 
MusEcalGUI::intervalPlot( MEIntervals* intervals_ )
{
  //
  // plot diff vector and corresponding intervals for a given MATACQ variable
  //
  
  if(_debug) cout << "Entering intervalPlot " << endl;

  if( _type>=ME::iSizeC ) return;

  METimeInterval* topInterval = intervals_->topInterval();

  bool twoVar = intervals_->useTwoVar();

  vector< ME::Time >& diff_key_ = intervals_->key();
  vector< float >&   diff_val_ = intervals_->diff();
  vector< float >&   val0_     = intervals_->val(0);
  vector< float >&   val1_     = intervals_->val(1);
  
  // first get the keys in interval
  vector< ME::Time> key_;
  
  MEChannel* sideLeaf = _leaf->getAncestor( ME::iLMRegion );
  MEVarVector* mtqVector_ = curMgr()->mtqVector(sideLeaf); 
  mtqVector_->getTime(key_, topInterval);

  vector< ME::Time>& timestamp_ = key_;

  unsigned int t0_ = timestamp_[0];
  
  double a(0), b(0);

  unsigned nl=2;
  if( twoVar ) nl=3;

  float diffmin = -0.08; float diffmax = +0.08;
  if( twoVar ) diffmin = -0.01; diffmax = +0.09;
  a = 2/(diffmax-diffmin);
  b = -(diffmin+diffmax)/(diffmax-diffmin);
  unsigned int ndiff = diff_key_.size();
  float xdiff[ndiff];
  float ydiff[ndiff];
  for( unsigned ii=0; ii<ndiff; ii++ )
    {
      xdiff[ii] = ( diff_key_[ii] - t0_ )/3600.;
      
      ydiff[ii] = a*diff_val_[ii]+b;
      if( ydiff[ii]<-1 ) ydiff[ii]=-1;
      if( ydiff[ii]>+1 ) ydiff[ii]=+1;
      ydiff[ii] += 2*(nl-1) + 1;
    }
  unsigned int n = ndiff;
  float y[2][n];

  float xmin;
  float xmax;
  xmin=xdiff[0];
  xmax=xdiff[n-1];
  float rangex = xmax-xmin;
  xmin -= 0.05*rangex;
  xmax += 0.05*rangex;

  for( unsigned jj=0; jj<2; jj++ )
    {
      if( !twoVar && jj==1 ) break;

      float ymin = intervals_->min( jj );
      float ymax = intervals_->max( jj );
      float rangey = ymax-ymin;
      ymin -= 0.01*rangey;
      ymax += 0.01*rangey;
      assert( ymax>ymin );
      a = 2/(ymax-ymin);
      b = -(ymin+ymax)/(ymax-ymin);
      for( unsigned ii=0; ii<n; ii++ )
	{
	  float curval = (jj==0) ? val0_[ii] : val1_[ii];
	  y[jj][ii] = a*curval+b;  // between -1 and +1
	  if( y[jj][ii]<-1 ) y[jj][ii]=-1;
	  if( y[jj][ii]>+1 ) y[jj][ii]=+1;
	  y[jj][ii] += 2*jj+1;
	  //	  ok[ii] = good_[ii];
	}
    }
  //
  //  Ok, now plotting
  //
  
  TString titleX;
  TString titleY;
  TString titleM;
  
  timestamp_       = key_;
  titleX = "time (in hours, from first run)";
  
  if(_type==ME::iLaser) titleM+= ME::color[_color];
  titleM += ME::type[_type];
  titleM += ", ";
  titleM += _leaf->oneLine( ME::iLMRegion );
  if(_debug) cout << titleM << endl;

  TString varName[2];
  int    varColor[2];

  varColor[0] = historyVarColor[ intervals_->var(0) ];
  varName[0] = historyVarName[ intervals_->var(0) ];
  if( twoVar )
    {
      varColor[1] = historyVarColor[ intervals_->var(1) ];
      varName[1] = historyVarName[ intervals_->var(1) ];
    }

  titleY += varName[0];
  if( twoVar )
    {
      titleY += "-";
      titleY += varName[1];
    }
  titleY += ", ";

  float yymin = 0;
  float yymax = 2*nl;
  TString hname = "diff";
  TH1* hdum  = (TH1*) gROOT->FindObject( hname );
  if( hdum ) delete hdum;
  TH1* h_ = new TH1F( hname, hname, 100, xmin, xmax );
  h_->SetMinimum( yymin );
  h_->SetMaximum( yymax );
  setHistoStyle( h_ );
  h_->SetStats( kFALSE );
  
  TAxis* ax = h_->GetXaxis();
  TAxis* ay = h_->GetYaxis();
  ax->SetTitle(titleX);
  ay->SetTitle(titleY);
  ay->SetLabelFont(0);
  ay->SetLabelSize(0); 
  ay->SetNdivisions(2*nl, true );

  h_->SetTitle(titleM);
  h_->Draw();
  if( _debug) cout << "href " << h_ << endl;
  if( _debug ) h_->Print();

  int icol(1), markerStyle(1), markerColor(0), lineWidth(2), lineColor(1);
  float markerSize(0);

  // first the diff plot
  for( unsigned iline=0; iline<=nl; iline++ )
    {
      TLine* separator = new TLine( xmin, 2*iline, xmax, 2*iline );

      if( _debug) cout << " diff plot Line: " << xmin<<" "<<xmax<<" " <<2*iline<< endl;
      separator->SetLineWidth(2);
      separator->SetLineColor(kBlack);
      separator->Draw("Same");
    }
  
  icol =  kBlue;
  lineColor = icol;
  markerColor = icol;
  drawHistoryGraph( ndiff, 
		    xdiff, ydiff, 0, 0, 0,
		    markerStyle, markerSize, markerColor,
		    lineWidth, lineColor );

  for( unsigned jj=0; jj<2; jj++ )
    {
      if( !twoVar && jj==1 ) break;
      
      icol =  varColor[jj];
      lineColor = icol;
      markerColor = icol;
      drawHistoryGraph( ndiff, 
			xdiff, y[jj], 0, 0, 0,
			markerStyle, markerSize, markerColor,
			lineWidth, lineColor );
    }
  
  // Now draw the intervals
  
  // loop on the levels

  double pos = -2*diffmin/(diffmax-diffmin );
  TLine* baseline = new TLine( xmin, 2*(nl-1)+pos, xmax, 2*(nl-1)+pos );
  baseline->SetLineWidth(1);
  baseline->SetLineColor(kBlack);
  baseline->Draw("Same");
  for( unsigned ilevel=MEIntervals::nlevel; ilevel!=0; ilevel-- )
    {
      double threshold = MEIntervals::threshold[ilevel-1];
      int    lineColor = MEIntervals::lineColor[ilevel-1];
      int    lineStyle = MEIntervals::lineStyle[ilevel-1];
      int    lineWidth = MEIntervals::lineWidth[ilevel-1];

      for( int jj=0; jj<2; jj++ )
	{
	  if( twoVar && jj==1 ) break;
	  int eps = 1-2*jj;
	  double pos = 2*( eps*threshold-diffmin )/(diffmax-diffmin );
	  TLine* line = new TLine( xmin, 2*(nl-1)+pos, 
				   xmax, 2*(nl-1)+pos );
	  if( _debug) cout << " intervals Line: " << xmin<<" "<<xmax<<" " <<2*(nl-1)+pos<< endl;
	  line->SetLineColor( lineColor );
	  line->SetLineStyle( lineStyle );
	  line->SetLineWidth( lineWidth );
	  line->Draw("Same");
	}
    }
  drawIntervals( intervals_, 0, 2*nl, t0_ );

  _curPad->SetGridx(0);
  _curPad->SetGridy(0);
  
  gPad->Modified();
  gPad->Update();
  
}


void
MusEcalGUI::drawIntervals( MEIntervals* intervals_, double ymin, double ymax,  unsigned int t0_ )
{

  if(_debug) cout<< " Entering MusEcalGUI::drawIntervals" << endl;

  METimeInterval* topInterval = intervals_->topInterval();

  // loop on the levels
  METimeInterval* ki(0);

  //  int nlevelmax = MEIntervals::nlevel;
  //  int nlevelmax = 2;
  int ilevel = 1;

  //  for( unsigned ilevel=nlevelmax; ilevel!=0; ilevel-- )
  {
    //      double threshold = MEIntervals::threshold[ilevel-1];
    int    lineColor = MEIntervals::lineColor[ilevel-1];
    int    lineStyle = MEIntervals::lineStyle[ilevel-1];
    int    lineWidth = MEIntervals::lineWidth[ilevel-1];
    
    if(_debug)
      {
	cout << "Intervals at level " << ilevel << endl;
	topInterval->print( ilevel );
      }
      int interval_(0);
      for( ki=topInterval->first(ilevel); ki!=0; ki=ki->next() )
	{
	  interval_++;
	  ME::Time key_    = ki->firstTime();
	  ME::Time timestamp_    = key_;
	  timestamp_ = key_;
	  unsigned fk = ( timestamp_ - t0_ )/3600;
	  if( _debug )
	    {
	      cout << "--> interval "  << interval_ << " fk=" << fk << endl; 
	      cout << ki->inBrackets();
	      cout << " next=" << ki->next();
	      cout << " previous=" << ki->previous();
	      cout << " firstIn=" << ki->firstIn();
	      cout << " lastIn=" << ki->lastIn();
	      cout << endl;
	    }
	  TLine* line = new TLine( fk, ymin, fk, ymax );
	  line->SetLineColor( lineColor );
	  line->SetLineStyle( lineStyle );
	  line->SetLineWidth( lineWidth );
	  line->Draw("Same");
	}
  }
  for( unsigned ii=0; ii<2; ii++ )
    {
      int    lineColor = kBlack;
      int    lineStyle = kSolid;
      int    lineWidth = 1;
      ME::Time key_;
      if( ii==0 ) 
	key_ = topInterval->firstTime();
      else        
	key_ = topInterval->lastTime();
      ME::Time timestamp_    = key_;
      timestamp_ = key_;

      unsigned fk = ( timestamp_ - t0_ )/3600;
      
      TLine* line = new TLine( fk, ymin, fk, ymax );
      line->SetLineColor( lineColor );
      line->SetLineStyle( lineStyle );
      line->SetLineWidth( lineWidth );
      line->Draw("Same");
    } 
}

// void
// MusEcalGUI::buildMtqIntervals( int color , int opt)
// {
//   MusEcal::buildMtqIntervals( color , true );

//   // create a new window
//   TString windowName =  "MATACQ Intervals";
//   TString str1 = "LASER Validity Intervals -- from analysis of ";
//   str1 += historyVarName[_mtqVar0];
//   if( _mtqVar1>0 )
//     {
//       str1 += ", ";
//       str1 += historyVarName[_mtqVar1];
//       str1 += ") plane";      
//     }
//   else
//     {
//       str1 += " variations";
//     }

//   TString WName_="MtqIntervals";
//   MEPlotWindow* win_ = getWindow( WName_, opt, 800, 450 );
//   if( win_==0 ) return;
//   _curPad = win_->getPad();
//   _curPad->cd();

//   MEIntervals* intervals_    = mtqIntervals( _color , true );
//   intervalPlot( intervals_ );

//   // return to the main pad
//   _curPad = fPad;
//   _curPad->cd();
  
// }




void 
MusEcalGUI::welcome()
{
  TString str_ = "Welcome";
  if( _window.count( str_ )!=0 ) return;
  _window[str_] = 
    new MEPlotWindow( 
		     gClient->GetRoot(), this, 
		     str_, 
		     fCanvas->GetWw(), fCanvas->GetWh() ,
		     "MusEcal - Monitoring and Useful Survey of Ecal",
		     "Visit our Twiki page at https://twiki.cern.ch/twiki/bin/view/CMS/EcalLaserMonitoring",
		     "MusEcal Version 2.0",
		     "J. Malcles & G. Hamel de Monchenault, CEA-Saclay"
		      );
  _window[str_]->ShowWelcome( true );
}

MusEcalGUI::~MusEcalGUI() 
{
  ClearWelcome();
}

void 
MusEcalGUI::exitProg() 
{
  cout << "Exiting MusEcalGUI. Take Care." << endl;
  gApplication->Terminate(0);
}

void 
MusEcalGUI::setPxAndPy( int px, int py ) 
{ 
  if( _debug ) cout << "MusEcalGUI: Entering setPxAndPy " << endl;

  MECanvasHolder::setPxAndPy( px, py );
}

void
MusEcalGUI::windowClicked( MEPlotWindow* win )
{
  float x, y;
  win->getCurXY( x, y );
  int ix = static_cast<int>(x > 0 ? x + 0.5 : x - 0.5);
  int iy = static_cast<int>(x > 0 ? y + 0.5 : y - 0.5);
  if( win->name()==TString("History") )
    {
      float dt_ = x;
      ME::Time t_ = ME::time( dt_, _time, ME::iHour ); 
      setTime( t_ );
      return;
    }
  if( win->name()==TString("Leaf") )
    {
      
      int nl = (int)((win->curHist()->GetMaximum()+0.00001)/2);
      int ic = (int) x;
      int il = (int) y; il/=2;
      cout << "nl/ic/il " << nl << "/" << x << "/" << y << endl;
      int iplot = nl*ic + il;
      cout << "iplot= " << iplot << endl;
      _leaf = _leaf->d(iplot);
      _leaf->print( cout );
      leafPlot();
      return;
    }
  if( win->name().Contains("_sel") )
    {
      int lmr_;
      if( win->name().Contains("LMR_") )
	{
	  int ieta = ix;
	  int iphi = iy;
	  lmr_ = MEEBGeom::lmr( ieta, iphi );
	  if( lmr_<1 || lmr_>72 ) return;
	

	}
      else 
	{
	  int iX = ix/5;
	  int iY = iy/5;
	  if( win->name().Contains("LMM_") )
	    {
	      int lm = MEEBGeom::lm_channel( iX, iY );
	      _leaf = _leaf->getAncestor( ME::iSector )
		->getDescendant( ME::iLMModule, lm );
	    }
	  else if( win->name().Contains("SC_") )
	    {
	      int sc = MEEBGeom::tt_channel( iX, iY );
	      _leaf = _leaf->getAncestor( ME::iSector )
		->getDescendant( ME::iSuperCrystal, sc );
	    }
	  else if( win->name().Contains("C_")
		   || win->name()==TString("EBLocal") )
	    {
	      _ig = ME::iCrystal;
	      int cr = MEEBGeom::crystal_channel( ix, iy );
	      _leaf = _leaf->getAncestor( ME::iSector )
		->getDescendant( ME::iCrystal, cr );
	    }
	  cout << _leaf->oneWord() << endl;
	  lmr_ = _leaf->getAncestor( ME::iLMRegion )->id();
	  cout << "lmr_/_lmr " << lmr_ << "/" << _lmr << endl;
	}
      if( lmr_!=_lmr ) 
	{
	  setChannel( runMgr( lmr_ )->tree()->getFirstDescendant( ME::iCrystal ) );	 
	}
      else
	{

	  fillEBLocalHistograms();
	  drawHist();
	  historyPlot();
	}
      delete win;
      return;
    }
}

void
MusEcalGUI::HandleFileMenu( Int_t id )
{
  if( id==0 )
    {
      exitProg();
      return;
    }  
  if( id==1 )
    {
      dumpVector( ME::iAPD_MEAN );
      return;
    }
  if( id==3 )
    {
      createRunPanel();
      return;
    }
  if( id==4 )
    {
      welcome();
      return;
    }
}

void
MusEcalGUI::HandleHistMenu( Int_t id )
{
  if( id==-1000 )
    {
      // produce animated gif files
      drawAPDAnim(1);
      return;
    }
  int kk = id/1000000;
  if( kk>0 )
    {
      int ihtype_=id%1000000; 
      if( _ihtype!=ihtype_ )
	{
	  _ihtype=ihtype_;
	  drawAPDHist();
	}      
      return;
    }  
  if( _type==ME::iLaser )
    {
      // APD? PN? MTQ?
      int icateg = id/1000;
      id = id%1000;
      _icateg = icateg;

      // color?
      int icol = id/100;
      assert( icol==_color );
      id = id%100;

      // histogram?
      _ihist = id;
      
      drawHist(1);
    }
  else if( _type==ME::iTestPulse )
    {
      // APD? PN?
      int icateg = id/1000;
      id = id%1000;
      _icateg = icateg;

      _ihist = id;
      drawHist(1);
    }
}

void
MusEcalGUI::HandleHistoryMenu( Int_t id )
{
  if( id<0 )
    {
      if( id==-1000 )
	{
	  createLeafPanel();
	  return;
	}
      if( id==-2000 )
	{
	  createMultiVarPanel();
	  return;
	}
      if( id==-3000 )
	{
	  createTwoVarPanel();
	  return;
	}
      if( id==-101 )
	{
	  cout << "History Plots: Projection" << endl;
	  _historyType = iHistoryProjection;
	} 
      else if( id==-100 )
	{
	  cout << "History Plots: versus time (hours since current Run/Sequence)" << endl;
	  _historyType = iHistoryVsTime;
	}
      else if( id==-10 )
	{
	  if( !_normalize )
	    {
	      cout << "History Plots: normalized to full range "  
		   << endl;
	      _normalize = true;
	    } 
	  else
	    {
	      cout << "History plots: unnormalized "  << endl;
	      _normalize = false;
	    }
	}
      else if( id==-11 )
	{
	  if( !_drawIntervals )
	    {
	      cout << "History plots: drawIntervals true "  
		   << endl;
	      _drawIntervals = true;
	    } 
	  else
	    {
	      cout << "History plots:  drawIntervals false"  << endl;
	      _drawIntervals = false;
	    }
	}
      historyPlot(1);
      return;
    }
  if( _type==ME::iLaser )
    {
      // color?
      int icol = id/100;
      assert( icol==_color );
      id = id%100;

      // variable?
      _var = id;

      historyPlot(1);
    }
  else if( _type==ME::iTestPulse )
    {
      // variable?
      _var = id;

      historyPlot(1);
    }
  else
    {
      cout << "Not implemented yet" << endl;
    }
}

void 
MusEcalGUI::distancePlot()
{
  //
  // plot diff vector and corresponding intervals for a given MATACQ variable
  //

  if(_debug) cout << "Entering distancePlot " << endl;

  if( _type>=ME::iSizeC ) return;

  if( _leaf==0 ) return;

  // first get the keys in interval
  vector< ME::Time> key_;
  
  MEChannel* sideLeaf = _leaf->getAncestor( ME::iLMRegion );
  MEVarVector* mtqVector_ = curMgr()->mtqVector(sideLeaf); 
  mtqVector_->getTime(key_);

  unsigned nrun = key_.size();


  // temporary
  int var0 = ME::iMTQ_FWHM;
  int var1 = ME::iMTQ_AMPL;
  //int var0 = iPNA;
  //int var1 = iPNB;
  int zoom0 = iThirtyPercent;
  int zoom1 = iThirtyPercent;

  int varIndex[ 2 ] = { var0, var1 };
  int varZoom[ 2 ]  = { zoom0, zoom1 };

  vector<float> y_[2];
  vector<bool>  ok_[2];
  double norm_[2];
  
  vector<bool> keep(nrun,true);
  for( unsigned var=0; var<2; var++ )
    {
      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  y_[var][ii] = 0.;
	  ok_[var][ii] = true;
	}
      norm_[var] = 0;

      mtqVector_->getValFlagAndNorm(varIndex[var],key_,y_[var],ok_[var],norm_[var]);

      for( unsigned ii=0; ii<nrun; ii++ )
	{
	  bool   oo = ok_[var][ii];
	  float yy = (y_[var][ii]-norm_[var])/norm_[var];
	  if( !oo ) keep[ii] = false;
	  y_[var][ii] = yy;
	}
    }
  
  vector<unsigned int> kgood;
  vector<float> xgood;
  vector<float> ygood;
  for( unsigned ii=0; ii<nrun; ii++ )
    {
      if( keep[ii] )
	{
	  kgood.push_back( key_[ii] );
	  xgood.push_back( y_[0][ii] );
	  ygood.push_back( y_[1][ii] );
	}
    }

  unsigned nl=3;
  float ymin[nl];
  ymin[0]=-1; ymin[1]=-1; ymin[2]=-0.05;
  float ymax[nl];
  ymax[0]=+1; ymax[1]=+1; ymax[2]=+0.20;

  unsigned N = xgood.size();
  float xx_[N];
  //  bool   ok_[N];
  float yy_[nl][N];
  vector<float> rho(N,0.);
  vector<float> eta(N,0.);
  vector<float> delta(N,0.);
  vector<float> phi(N,0.);
  unsigned n = 9;
  for( unsigned ii=0; ii<N; ii++ )
    {
      //      ok_[ii] = true;
      xx_[ii] = kgood[ii];
      yy_[0][ii] = xgood[ii]/zoomRange[ varZoom[0] ];
      yy_[1][ii] = ygood[ii]/zoomRange[ varZoom[1] ];
      yy_[2][ii] = 0;
      //      yy_[3][ii] = 0;
      if( ii<(n+1) ) continue;
      if( ii>=(N-(n+1)) ) continue;
      float rho_up=0;
      float rho_down=0;
      float eta_up=0;
      float eta_down=0;
      for( unsigned jj=0; jj<n; jj++ )
	{
	  rho_up   += xgood[ ii+1+jj ];
	  rho_down += xgood[ ii-1-jj ];
	  eta_up   += ygood[ ii+1+jj ];
	  eta_down += ygood[ ii-1-jj ];
	}
      rho_up   /= n; 
      rho_down /= n; 
      eta_up   /= n; 
      eta_down /= n;
      rho[ii]   = rho_up - rho_down;
      eta[ii]   = eta_up - eta_down;
      delta[ii] = sqrt( pow( rho[ii],2 ) + pow( eta[ii],2 ) );
      phi[ii]   = atan2( eta[ii], rho[ii] )/acos(-1.);
      yy_[2][ii] = delta[ii];
      //      yy_[3][ii] = phi[ii];
    }

  int il= 0;
  for( unsigned jj=0; jj<nl; jj++ )
    {
      il = 2*jj + 1;
      float a = 2/(ymax[jj]-ymin[jj]);
      float b = -(ymin[jj]+ymax[jj])/(ymax[jj]-ymin[jj]);
      for( unsigned ii=0; ii<N; ii++ )
	{
	  yy_[jj][ii] = a*yy_[jj][ii] + b;
	  if( yy_[jj][ii]>1 )  yy_[jj][ii] = 1;
	  if( yy_[jj][ii]<-1 ) yy_[jj][ii] = -1;
	  yy_[jj][ii] += il;
	}
    }

  float xmin=key_[0];
  float xmax=key_[nrun-1];
  float rangex = xmax-xmin;
  xmin -= 0.05*rangex;
  xmax += 0.05*rangex;

  //
  //  Ok, now plotting
  //

  TString titleX;
  TString titleY;
  TString titleM;

  titleX = "key";

  if(_type==ME::iLaser) titleM+= ME::color[_color];
  titleM += ME::type[_type];
  if(_debug) cout << titleM << endl;

  float yymin = 0;
  float yymax = 2*nl;
  TString hname = "dist";
  TH1* hdum  = (TH1*) gROOT->FindObject( hname );
  if( hdum ) delete hdum;
  TH1* h_ = new TH1F( hname, hname, 100, xmin, xmax );
  h_->SetMinimum( yymin );
  h_->SetMaximum( yymax );
  setHistoStyle( h_ );
  h_->SetStats( kFALSE );
  
  TAxis* ax = h_->GetXaxis();
  TAxis* ay = h_->GetYaxis();
  ax->SetTitle(titleX);
  //  ax->SetLabelFont(0);
  //  ax->SetLabelSize(0);
  //  ax->SetTitleFont(0);
  //  ax->SetTitleSize(0);
  //  ax->SetNdivisions(1, true );
  ay->SetTitle(titleY);
  ay->SetLabelFont(0);
  ay->SetLabelSize(0);
  //   ay->SetTitleFont(0);
  //   ay->SetTitleSize(0);
  ay->SetNdivisions(4, true );

  h_->SetTitle(titleM);
  h_->Draw();
  if( _debug) cout << "href " << h_ << endl;
  if( _debug ) h_->Print();

  // int il(0), icol(1);
  int markerStyle(1), markerColor(0), lineWidth(2), lineColor(1);
  int markerSize(0);
  for( unsigned jj=0; jj<nl; jj++ )
    {
      lineColor=kBlue;
      markerColor=kBlue;
      drawHistoryGraph( N, 
			xx_, yy_[jj], 0, 0, 0,
			markerStyle, markerSize, markerColor,
			lineWidth, lineColor );
    }

  _curPad->SetGridx(1);
  _curPad->SetGridy(0);

  gPad->Modified();
  gPad->Update();

}


void
MusEcalGUI::HandleChannelMenu( Int_t id )
{
  if( id==60000 )
    {
      createChanPanel();
      // one level up
      return;
    }

  if( id==50000 )
    {
      oneLevelUp();
      // one level up
      return;
    }

  if( id/10000==1 )
    {
      int icr_=id%10000;
      if( curMgr()==0 ) 
	{
	  cout << "Warning -- current manager is not present " << endl;
	  return;
	}
      setChannel( curMgr()->tree()->getDescendant( ME::iCrystal, icr_ ) );
      return;
    }

  if( id/1000==1 )
    {
      int ilmr_=id%1000;
      if( runMgr( ilmr_ )==0 ) 
	{
	  cout << "Warning -- LMR" << ilmr_ << " is not present" << endl;
	  return;
	}
      setChannel( runMgr( ilmr_ )->tree()->getFirstDescendant( ME::iCrystal ) );
      return;
    }

  if( id<0 || id>ME::iSizeG ) return;

  MEPlotWindow* win_(0);
  if( id==ME::iLMRegion )
    {      
      win_ = new MEPlotWindow( gClient->GetRoot(), this, "LMR_sel", 450, 700 );
      assert( _febgeom!=0 );
      TH2* h_  = (TH2*) _febgeom->Get("eb_lmr");
      assert( h_ !=0 );
      TPad* pad_ = (TPad*) win_->getPad();
      pad_->cd();
      TString title_="Double Click on LM Region";
      h_->SetTitle(title_);
      h_->SetStats( kFALSE );
      h_->Draw("COLZ");
      MEEBDisplay::drawEBGlobal();
      win_->setCurHist( h_ );
      pad_->Modified();
      pad_->Update();
    }
  else
    {
      _ig = id;
      TString str_ = ME::granularity[_ig];
      str_ += "_sel";
      win_ = new MEPlotWindow( gClient->GetRoot(), this, str_, 700, 450,
			       "Channel Selection");
      assert( _febgeom!=0 );
      TString hname_="eb_loc";
      TString cr_str_ = "_cr";
      if( ME::useElectronicNumbering ) cr_str_ = "_elecr";
      switch( id )
	{
	case ME::iLMModule:     hname_+="_lmmod"; break;
	case ME::iSuperCrystal: hname_+="_tt";    break;
	case ME::iCrystal:      hname_+=cr_str_;    break;
	default: return;
	}      
      TH2* h_  = (TH2*) _febgeom->Get(hname_);
      assert( h_ !=0 );
      TPad* pad_ = (TPad*) win_->getPad();
      pad_->cd();
      TString title_="Double Click to select -- ";
      title_ += _leaf->oneWord(ME::iSector);
      h_->SetTitle(title_);
      h_->SetStats( kFALSE );
      h_->Draw("COLZ");
      MEEBDisplay::drawEBLocal();
      win_->setCurHist( h_ );
      pad_->Modified();
      pad_->Update();      
    }
}

void
MusEcalGUI::drawHist( int opt )
{ 
  if( _icateg==0 )
    {
      drawAPDHist( opt );
    }
  else if( _icateg==1 )
    {
      drawPNHist( opt );
    }
  else if( _icateg==2 )
    {
      drawMTQHist( opt );
    }
}

void
MusEcalGUI::drawAPDHist( int opt )
{
  TString str_;
  TH2* h2_;
  TH1* h1_;
  TString WName_;
  TString title_;
  MEPlotWindow* win_;

  TString varName_;
  if( _type==ME::iLaser )
    {
      varName_=ME::APDPrimVar[_ihist];
      str_="APD-";
    }
  else if( _type==ME::iTestPulse )
    {
      varName_=ME::TPAPDPrimVar[_ihist];
      str_="TPAPD-";
    }
  str_+=varName_;


  if( isBarrel() )
    {
      if(_debug) cout<< "drawAPDHist: EB before 2D "<< endl;
      // the full barrel 2D
      WName_ = "EB_2D";
      win_ = getWindow( WName_, opt, 450, 700 );
      if( win_!=0 )
	{
	  _curPad =  win_->getPad();
	  _curPad->cd();
	  
	  h2_ = (TH2*) _eb_m[str_];
	  assert( h2_!=0 );
	  h2_->Draw("COLZ");
	  MEEBDisplay::drawEBGlobal();
	  _window[WName_]->setCurHist( h2_ );
	  _curPad->Modified();
	  _curPad->Update();
	  
	  title_ = h2_->GetTitle();
	  title_.ReplaceAll(" ",".");
	  title_+=".2D";
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
      if(_debug) cout<< "drawAPDHist: EB before 1D "<< endl;
      // the full barrel 1D
      WName_ = "EB_1D";
      win_ = getWindow( WName_, opt, 900, 300 );
      if( win_!=0 )
	{
	  _curPad =  win_->getPad();
	  _curPad->cd();
	  
	  TString ext_="_1D";
	  h1_ = (TH1*) _eb_m[str_+ext_];
	  assert( h1_!=0 );
	  h1_->Draw();
	  float max_ = h1_->GetMaximum();
	  float min_ = h1_->GetMinimum();
	  float x_=-0.5;
	  for( int ilmr=1; ilmr<=72; ilmr++ )
	    {	  
	      if( ilmr!=1 && (ilmr-1)%2==0 ) x_+=800/25; 
	      if( (ilmr-1)%2==1 )            x_+=900/25; 
	      if( ilmr==1 ) continue;
	      TLine* l = new TLine( x_, min_+0.01*(max_-min_), 
				    x_, max_-0.01*(max_-min_) );
	      if( (ilmr-1)%2==0 )
		{
		  l->SetLineColor(kRed);
		  l->SetLineWidth(2);
		}
	      else
		{
		  l->SetLineColor(kGreen);
		}
	      l->Draw("Same");
	      
	      if( (ilmr-1)%2==0 ) continue;
	      TText* text_ = new TText( x_, min_+0.80*(max_-min_), ME::smName( ilmr ) );
	      text_->SetTextSize(0.05);
	      text_->SetTextAngle(90);
	      text_->SetTextAlign(12);
	      text_->Draw("Same");
	    }
	  h1_->Draw("Same");
	  
	  win_->setCurHist( h1_ );
	  _curPad->Modified();
	  _curPad->Update();
	  
	  title_ = h1_->GetTitle();
	  title_.ReplaceAll(" ",".");
	  title_+=".1D";
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
    }
  else
    {

      // the full endcap 2D
      if(_debug) cout<< "drawAPDHist: EE before 2D "<< endl;
      WName_ = "EE_2D";
      win_ = getWindow( WName_, opt, 350, 600 );
      if( win_!=0 )
	{
	  _curPad =  win_->getPad();
	  _curPad->cd();
	  
	  h2_ = (TH2*) _ee_m[str_];
	  assert( h2_!=0 );
	  h2_->Draw("COLZ");
	  MEEEDisplay::drawEEGlobal();
	  
	  _window[WName_]->setCurHist( h2_ );
	  
	  _curPad->Modified();
	  _curPad->Update();
	  
	  title_ = h2_->GetTitle();
	  title_.ReplaceAll(" ",".");
	  title_+=".2D";
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
      // JM

      // the full endcap 1D
      
      if(_debug) cout<< "drawAPDHist: EE before 1D "<< endl;
      WName_ = "EE_1D";
      win_ = getWindow( WName_, opt, 900, 300 );
      if( win_!=0 )
	{
	  _curPad =  win_->getPad();
	  _curPad->cd();
	  
	  TString ext_="_1D";
	  h1_ = (TH1*) _ee_m[str_+ext_];
	  assert( h1_!=0 );
	  h1_->Draw();
	  float max_ = h1_->GetMaximum();
	  float min_ = h1_->GetMinimum();
	  float x_=-0.5;
	  for( int ilmr=73; ilmr<=92; ilmr++ )
	    {	  
	      // FIXME
	      
	      if( ilmr!=1 && (ilmr-1)%2==0 ) x_+=800/25; 
	      if( (ilmr-1)%2==1 )            x_+=900/25; 
	      if( ilmr==1 ) continue;
	      TLine* l = new TLine( x_, min_+0.01*(max_-min_), 
				    x_, max_-0.01*(max_-min_) );
	      if( (ilmr-1)%2==0 )
		{
		  l->SetLineColor(kRed);
		  l->SetLineWidth(2);
		}
	      else
		{
		  l->SetLineColor(kGreen);
		}
	      l->Draw("Same");
	      
	      if( (ilmr-1)%2==0 ) continue;
	      TText* text_ = new TText( x_, min_+0.80*(max_-min_), ME::smName( ilmr ) );
	      text_->SetTextSize(0.05);
	      text_->SetTextAngle(90);
	      text_->SetTextAlign(12);
	      text_->Draw("Same");
	    }
	  h1_->Draw("Same");
	  
	  win_->setCurHist( h1_ );
	  _curPad->Modified();
	  _curPad->Update();
	  
	  title_ = h1_->GetTitle();
	  title_.ReplaceAll(" ",".");
	  title_+=".1D";
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
    }
  

  if( isBarrel() )
    {
  
      if(_debug) cout<< "drawAPDHist: EB Local"<< endl;
      // then the current super-module
      WName_ = "EBLocal";
      win_ = getWindow( WName_, opt, 700, 450 );
      if( win_!=0 )
	{
	  _curPad = (TPad*) win_->getPad();
	  _curPad->cd();
	  TString ext_;
	  if( _ihtype==iMAP )
	    {
	      _curPad->SetGridx(0);
	      _curPad->SetGridy(0);
	      h2_ = (TH2*) _eb_loc_m[str_+ext_];
	      assert( h2_!=0 );
	      win_->setCurHist( h2_ );
	      h2_->Draw("COLZ");
	      MEEBDisplay::drawEBLocal();
	      title_ = h2_->GetTitle();
	      title_.ReplaceAll(" ",".");
	      title_+=".MAP";
	    }
	  else if( _ihtype==iHIST )
	    {
	      _curPad->SetGridx(1);
	      _curPad->SetGridy(0);
	      ext_="_HIST";
	      if( _eb_loc_m.count( str_+ext_ )==0 ) return;
	      h1_ = (TH1*) _eb_loc_m[str_+ext_];
	      if( h1_==0 ) return;
	      win_->setCurHist( h1_ );
	      h1_->Draw();
	      ext_+="_sel";
	      h1_ = (TH1*) _eb_loc_m[str_+ext_];
	      h1_->Draw("Same");
	      title_ = h1_->GetTitle();
	      title_.ReplaceAll(" ",".");
	      title_+=".HIST";
	    }
	  else if( _ihtype==iVS_CHANNEL )
	    {	  
	      _curPad->SetGridx(1);
	      _curPad->SetGridy(1);
	      ext_="_VS_CHANNEL";
	      if( _eb_loc_m.count( str_+ext_ )==0 ) return;
	      h1_ = (TH1*) _eb_loc_m[str_+ext_];
	      if( h1_==0 ) return;
	      win_->setCurHist( h1_ );
	      h1_->Draw();
	      ext_+="_sel";
	      h1_ = (TH1*) _eb_loc_m[str_+ext_];
	      h1_->Draw("Same");
	      title_ = h1_->GetTitle();
	      title_.ReplaceAll(" ",".");
	      title_+=".VS_CHANNEL";
	    }
	  _curPad->Modified();
	  _curPad->Update();
      
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
    }
  else
    {
      if(_debug) cout<< "drawAPDHist: EE Local"<< endl;
      int isect = _leaf->getAncestor( ME::iSector )->id();
      if( isect>9 ) isect-=9;
      // then the current super-module
      WName_ = "EELocal";
      win_ = getWindow( WName_, opt, 450, 450 );
      if( win_!=0 )
	{
	  _curPad = (TPad*) win_->getPad();
	  _curPad->cd();
	  TString ext_;
	  if( _ihtype==iMAP )
	    {
	      _curPad->SetGridx(0);
	      _curPad->SetGridy(0);
	      TString str__ = str_+ext_;
	      str__+="_";
	      str__+=isect;
	      h2_ = (TH2*) _ee_loc_m[str__];
	      if( h2_==0 )
		{
		  cout << str__ << endl;
		  return;
		}
	      //	      assert( h2_!=0 );
	      win_->setCurHist( h2_ );
	      h2_->Draw("COLZ");
	      MEEEDisplay::drawEELocal( isect );
	      title_ = h2_->GetTitle();
	      title_.ReplaceAll(" ",".");
	      title_+=".MAP";
	    }
	  else if( _ihtype==iHIST )
	    {
	      _curPad->SetGridx(1);
	      _curPad->SetGridy(0);
	      ext_="_HIST";
	      TString str__ = str_;
	      str__+="_";
	      str__+=isect;
	      str__+=ext_;
	      cout << "IHIST " << str__ << endl;
	      if( _ee_loc_m.count( str__ )==0 ) return;
	      h1_ = (TH1*) _ee_loc_m[str__];
	      if( h1_==0 ) return;
	      win_->setCurHist( h1_ );
	      h1_->Draw();
	      ext_+="_sel";
	      h1_ = (TH1*) _ee_loc_m[str__];
	      h1_->Draw("Same");
	      title_ = h1_->GetTitle();
	      title_.ReplaceAll(" ",".");
	      title_+=".HIST";
	    }
	  else if( _ihtype==iVS_CHANNEL )
	    {	  
	      _curPad->SetGridx(1);
	      _curPad->SetGridy(1);
	      ext_="_VS_CHANNEL";
	      TString str__ = str_;
	      str__+="_";
	      str__+=isect;
	      str__+=ext_;
	      cout << "VS_CHANNEL " << str__ << endl;
	      if( _ee_loc_m.count( str__ )==0 ) return;
	      h1_ = (TH1*) _ee_loc_m[str__];
	      if( h1_==0 ) return;
	      win_->setCurHist( h1_ );
	      h1_->Draw();
	      ext_+="_sel";
	      h1_ = (TH1*) _ee_loc_m[str__];
	      h1_->Draw("Same");
	      title_ = h1_->GetTitle();
	      title_.ReplaceAll(" ",".");
	      title_+=".VS_CHANNEL";
	    }
	  _curPad->Modified();
	  _curPad->Update();
      
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
    }
  if(_debug) cout<< "drawAPDHist: Done!!!"<< endl;
}

void
MusEcalGUI::drawPNHist( int opt )
{
  TString str_;
  TH1* h1_;
  TString WName_;
  TString title_;
  MEPlotWindow* win_;

  TString varName_;
  if( _type==ME::iLaser )
    {
      varName_=ME::PNPrimVar[_ihist];
      str_="PN-";
    }
  else if( _type==ME::iTestPulse )
    {
      varName_=ME::TPPNPrimVar[_ihist];
      str_="TPPN-";
    }
  str_+=varName_;

  if( isBarrel() )
    {
      
      // the full barrel 1D
      WName_ = "EB_PN";
      win_ = getWindow( WName_, opt, 900, 300 );
      if( win_!=0 )
	{
	  _curPad = win_->getPad();
	  _curPad->cd();
	  h1_ = (TH1*) _eb_m[str_];
	  assert( h1_!=0 );
	  float max_ = h1_->GetMaximum();
	  float min_ = h1_->GetMinimum();
	  h1_->SetMaximum(max_); // !!!
	  h1_->Draw();
	  float x_=-0.5;
	  for( int ilmr=1; ilmr<=72; ilmr++ )
	    {	  
	      if( ilmr!=1 && (ilmr-1)%2==0 ) x_+=8; 
	      if( (ilmr-1)%2==1 )             x_+=10; 
	      if( ilmr==1 ) continue;
	      TLine* l = new TLine( x_, min_+0.01*(max_-min_), 
				    x_, max_-0.01*(max_-min_) );
	      if( (ilmr-1)%2==0 )
		{
		  l->SetLineColor(kRed);
		  l->SetLineWidth(2);
		}
	      else
		{
		  l->SetLineColor(kGreen);
		}
	      l->Draw("Same");

	      if( (ilmr-1)%2==0 ) continue;
	      TText* text_ = new TText( x_, min_+0.80*(max_-min_), ME::smName( ilmr ) );
	      text_->SetTextSize(0.05);
	      text_->SetTextAngle(90);
	      text_->SetTextAlign(12);
	      text_->Draw("Same");
	    }
	  h1_->Draw("Same");
	  
	  win_->setCurHist( h1_ );
	  _curPad->Modified();
	  _curPad->Update();
	  
	  title_ = h1_->GetTitle();
	  title_.ReplaceAll(" ",".");
	  title_+=".GLOBAL";
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
  
      // then the current super-module
      WName_ = "EBLocal_PN";
      win_ = getWindow( WName_, opt, 700, 450 );
      if( win_!=0 )
	{
	  _curPad = (TPad*) win_->getPad();
	  _curPad->cd();
	  _curPad->SetGridx(1);
	  _curPad->SetGridy(1);
	  TString ext_="_LOCAL";
	  if( _eb_loc_m.count( str_+ext_ )==0 ) return;
	  h1_ = (TH1*) _eb_loc_m[str_+ext_];
	  if( h1_==0 ) return;
	  win_->setCurHist( h1_ );
	  h1_->Draw();
	  title_ = h1_->GetTitle();
	  title_.ReplaceAll(" ",".");
	  title_+=".LOCAL";
	  _curPad->Modified();
	  _curPad->Update();
	  
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
    }else{

      // JM
      // the full endcap 1D
      
      WName_ = "EE_PN";
      win_ = getWindow( WName_, opt, 900, 300 );
      if( win_!=0 )
	{
	  _curPad = win_->getPad();
	  _curPad->cd();
	  h1_ = (TH1*) _ee_m[str_];
	  assert( h1_!=0 );
	  float max_ = h1_->GetMaximum();
	  float min_ = h1_->GetMinimum();
	  h1_->SetMaximum(max_); // !!!
	  h1_->Draw();
	  float x_=-0.5;
	  float xprev=-0.5;
	  for( int ilmr=73; ilmr<=92; ilmr++ )
	    {	  
	      std::vector<int> vecmod=MEEEGeom::lmmodFromLmr( ilmr );
	      int ntoadd=vecmod.size()*2;
	      x_+=ntoadd;
	      
	      TLine* l = new TLine( x_, min_+0.01*(max_-min_), 
				    x_, max_-0.01*(max_-min_) );
	      
	      float xtxt;
	      if(ilmr==80 || ilmr==90 ) {
		l->SetLineColor(kGreen);
		l->SetLineWidth(1);
		xtxt=x_;
	      }else{
		l->SetLineColor(kRed);
		l->SetLineWidth(2);
		xtxt=xprev+(x_-xprev)/2.0;
	      }
	      
	      l->Draw("Same");
	      
	      TText* text_ = new TText( xtxt, min_+0.80*(max_-min_), ME::smName( ilmr ) );
	      text_->SetTextSize(0.05);
	      text_->SetTextAngle(90);
	      text_->SetTextAlign(12);
	      if( ilmr!=81 && ilmr!=91 ) text_->Draw("Same");
	      xprev= x_;
	    }
	  h1_->Draw("Same");
	  
	  win_->setCurHist( h1_ );
	  _curPad->Modified();
	  _curPad->Update();
	  
	  title_ = h1_->GetTitle();
	  title_.ReplaceAll(" ",".");
	  title_+=".GLOBAL";
	  win_->setPrintName( _psdir+title_+".ps" );
	  if(_write) win_->write();
	}
      
      // FIXME 
      // then the current super-module
      //  WName_ = "EELocal_PN";
      //       win_ = getWindow( WName_, opt, 700, 450 );
      //       if( win_!=0 )
      // 	{
      // 	  _curPad = (TPad*) win_->getPad();
      // 	  _curPad->cd();
      // 	  _curPad->SetGridx(1);
      // 	  _curPad->SetGridy(1);
      // 	  TString ext_="_LOCAL";
      // 	  if( _ee_loc_m.count( str_+ext_ )==0 ) return;
      // 	  h1_ = (TH1*) _ee_loc_m[str_+ext_];
      // 	  if( h1_==0 ) return;
      // 	  win_->setCurHist( h1_ );
      // 	  h1_->Draw();
      // 	  title_ = h1_->GetTitle();
      // 	  title_.ReplaceAll(" ",".");
      // 	  title_+=".LOCAL";
      // 	  _curPad->Modified();
      // 	  _curPad->Update();
      
      // 	  win_->setPrintName( _psdir+title_+".ps" );
      // 	  win_->write();
      // 	}
      
    }
      
}

void
MusEcalGUI::drawMTQHist( int opt )
{
  
//   TString str_;
//   TH1* h1_;
//   TString WName_;
//   TString title_;
//   MEPlotWindow* win_;

//   TString varName_;
//   varName_=ME::PNPrimVar[_ihist];
//   str_="MTQ-";
  
//   str_+=varName_;

//   // the full barrel 1D
//   WName_ = "EB_MTQ";
//   win_ = getWindow( WName_, opt, 900, 300 );
//   if( win_!=0 )
//     {
//       _curPad = win_->getPad();
//       _curPad->cd();
//       h1_ = (TH1*) _eb_m[str_];
//       assert( h1_!=0 );
//       float max_ = h1_->GetMaximum();
//       float min_ = h1_->GetMinimum();
//       h1_->SetMaximum(max_); // !!!
//       h1_->Draw();
//       float x_=-0.5;
//       for( int ilmr=1; ilmr<=72; ilmr++ )
// 	{	  
// 	  if( ilmr!=1 && (ilmr-1)%2==0 ) x_+=8; 
// 	  if( (ilmr-1)%2==1 )             x_+=10; 
// 	  if( ilmr==1 ) continue;
// 	  TLine* l = new TLine( x_, min_+0.01*(max_-min_), 
// 				x_, max_-0.01*(max_-min_) );
// 	  if( (ilmr-1)%2==0 )
// 	    {
// 	      l->SetLineColor(kRed);
// 	      l->SetLineWidth(2);
// 	    }
// 	  else
// 	    {
// 	      l->SetLineColor(kGreen);
// 	    }
// 	  l->Draw("Same");

// 	  if( (ilmr-1)%2==0 ) continue;
// 	  TText* text_ = new TText( x_, min_+0.80*(max_-min_), ME::smName( ilmr ) );
// 	  text_->SetTextSize(0.05);
// 	  text_->SetTextAngle(90);
// 	  text_->SetTextAlign(12);
// 	  text_->Draw("Same");
// 	}
//       h1_->Draw("Same");
        
//       win_->setCurHist( h1_ );
//       _curPad->Modified();
//       _curPad->Update();
      
//       title_ = h1_->GetTitle();
//       title_.ReplaceAll(" ",".");
//       title_+=".GLOBAL";
//       win_->setPrintName( _psdir+title_+".ps" );
//       win_->write();
//     }
  
//   // then the current super-module
//   WName_ = "EBLocal_PN";
//   win_ = getWindow( WName_, opt, 700, 450 );
//   if( win_!=0 )
//     {
//       _curPad = (TPad*) win_->getPad();
//       _curPad->cd();
//       _curPad->SetGridx(1);
//       _curPad->SetGridy(1);
//       TString ext_="_LOCAL";
//       if( _eb_loc_m.count( str_+ext_ )==0 ) return;
//       h1_ = (TH1*) _eb_loc_m[str_+ext_];
//       if( h1_==0 ) return;
//       win_->setCurHist( h1_ );
//       h1_->Draw();
//       title_ = h1_->GetTitle();
//       title_.ReplaceAll(" ",".");
//       title_+=".LOCAL";
//       _curPad->Modified();
//       _curPad->Update();

//       win_->setPrintName( _psdir+title_+".ps" );
//       win_->write();
//     }

  cout << "drawMTQHist not implemented yet "<< endl;

}

// FIXME! this is a test
void
MusEcalGUI::drawAPDAnim( int opt )
{
  TString str_;
  TH2* h2_;
  //  TH1* h1_;
  TString WName_;
  TString title_;
  MEPlotWindow* win_;

  // FIXME!!!!
  TString fname_(getenv("MESTORE"));
  fname_ += "/EBGlobalHist.root";
  FILE *test;
  //  cout << fname_ << endl;
  test = fopen( fname_, "r" );
  if( test==0 )
    {
      cout << "Histogam file not present: ";
      cout << "please run bin/createEBGlobalHistFile first";
      cout << endl;
      return;
    }
  fclose( test );
  TFile* file = TFile::Open(fname_);
  assert( file!=0 );

  TString varName_;
  if( _type==ME::iLaser )
    {
      varName_=ME::APDPrimVar[_ihist];
      str_="APD-";
    }
  else if( _type==ME::iTestPulse )
    {
      varName_=ME::TPAPDPrimVar[_ihist];
      str_="TPAPD-";
    }
  str_+=varName_;

  // the full barrel 2D
  WName_ = "EB_ANIM";
  win_ = getWindow( WName_, opt, 450, 700 );
  if( win_!=0 )
    {
      _curPad =  win_->getPad();
      _curPad->cd();

      int ii=0;
      while(1)
	{
	  ii++;
	  TString hname_ = str_;
	  hname_ += ";";
	  hname_ += ii;
	  h2_ = (TH2*) file->Get(hname_);
	  if( h2_==0 ) break;
	  h2_->Draw("COLZ");
	  MEEBDisplay::drawEBGlobal();
	  _window[WName_]->setCurHist( h2_ );
	  _curPad->Modified();
	  _curPad->Update();

	  title_ = str_;
	  //	  title_ += ".gif+50";	  
	  //	  title_ = h2_->GetTitle();
	  //	  title_.ReplaceAll(" ",".");
	  //	  title_+=".ANIM";
	  win_->setPrintName( _psdir+title_+".gif+50" );
	  if(_write) win_->write();
	  //	  string dum;
	  //	  cin >> dum;
	} 
    }

  file->Close();
  setTime( _time );

//   // the full barrel 1D
//   WName_ = "EB_1D";
//   win_ = getWindow( WName_, opt, 900, 300 );
//   if( win_!=0 )
//     {
//       _curPad =  win_->getPad();
//       _curPad->cd();

//       TString ext_="_1D";
//       h1_ = (TH1*) _eb_m[str_+ext_];
//       assert( h1_!=0 );
//       h1_->Draw();
//       float max_ = h1_->GetMaximum();
//       float min_ = h1_->GetMinimum();
//       float x_=-0.5;
//       for( int ilmr=1; ilmr<=72; ilmr++ )
// 	{	  
// 	  if( ilmr!=1 && (ilmr-1)%2==0 ) x_+=800/25; 
// 	  if( (ilmr-1)%2==1 )             x_+=900/25; 
// 	  if( ilmr==1 ) continue;
// 	  TLine* l = new TLine( x_, min_+0.01*(max_-min_), 
// 				x_, max_-0.01*(max_-min_) );
// 	  if( (ilmr-1)%2==0 )
// 	    {
// 	      l->SetLineColor(kRed);
// 	      l->SetLineWidth(2);
// 	    }
// 	  else
// 	    {
// 	      l->SetLineColor(kGreen);
// 	    }
// 	  l->Draw("Same");

// 	  if( (ilmr-1)%2==0 ) continue;
// 	  TText* text_ = new TText( x_, min_+0.80*(max_-min_), ME::smName( ilmr ) );
// 	  text_->SetTextSize(0.05);
// 	  text_->SetTextAngle(90);
// 	  text_->SetTextAlign(12);
// 	  text_->Draw("Same");
// 	}
//       h1_->Draw("Same");

//       win_->setCurHist( h1_ );
//       _curPad->Modified();
//       _curPad->Update();

//       title_ = h1_->GetTitle();
//       title_.ReplaceAll(" ",".");
//       title_+=".1D";
//       win_->setPrintName( _psdir+title_ );
//       win_->write();
//     }

//   // then the current super-module
//   WName_ = "EBLocal";
//   win_ = getWindow( WName_, opt, 700, 450 );
//   if( win_!=0 )
//     {
//       _curPad = (TPad*) win_->getPad();
//       _curPad->cd();
//       TString ext_;
//       if( _ihtype==iMAP )
// 	{
// 	  _curPad->SetGridx(0);
// 	  _curPad->SetGridy(0);
// 	  h2_ = (TH2*) _eb_loc_m[str_+ext_];
// 	  assert( h2_!=0 );
// 	  win_->setCurHist( h2_ );
// 	  h2_->Draw("COLZ");
// 	  MEEBDisplay::drawEBLocal();
// 	  title_ = h2_->GetTitle();
// 	  title_.ReplaceAll(" ",".");
// 	  title_+=".MAP";
// 	}
//       else if( _ihtype==iHIST )
// 	{
// 	  _curPad->SetGridx(1);
// 	  _curPad->SetGridy(0);
// 	  ext_="_HIST";
// 	  if( _eb_loc_m.count( str_+ext_ )==0 ) return;
// 	  h1_ = (TH1*) _eb_loc_m[str_+ext_];
// 	  if( h1_==0 ) return;
// 	  win_->setCurHist( h1_ );
// 	  h1_->Draw();
// 	  ext_+="_sel";
// 	  h1_ = (TH1*) _eb_loc_m[str_+ext_];
// 	  h1_->Draw("Same");
// 	  title_ = h1_->GetTitle();
// 	  title_.ReplaceAll(" ",".");
// 	  title_+=".HIST";
// 	}
//       else if( _ihtype==iVS_CHANNEL )
// 	{	  
// 	  _curPad->SetGridx(1);
// 	  _curPad->SetGridy(1);
// 	  ext_="_VS_CHANNEL";
// 	  if( _eb_loc_m.count( str_+ext_ )==0 ) return;
// 	  h1_ = (TH1*) _eb_loc_m[str_+ext_];
// 	  if( h1_==0 ) return;
// 	  win_->setCurHist( h1_ );
// 	  h1_->Draw();
// 	  ext_+="_sel";
// 	  h1_ = (TH1*) _eb_loc_m[str_+ext_];
// 	  h1_->Draw("Same");
// 	  title_ = h1_->GetTitle();
// 	  title_.ReplaceAll(" ",".");
// 	  title_+=".VS_CHANNEL";
// 	}
//       _curPad->Modified();
//       _curPad->Update();

//       win_->setPrintName( _psdir+title_ );
//       win_->write();
//     }
}

void
MusEcalGUI::createRunPanel()
{
  delete _fRunPanel;
  _fRunPanel = new MERunPanel(gClient->GetRoot(), this, 600, 300);
}

void
MusEcalGUI::createChanPanel( bool ifexists )
{
  if( ifexists && _fChanPanel==0 ) return;
  delete _fChanPanel;
  _fChanPanel = new MEChanPanel(gClient->GetRoot(), this, 400, 150);
}

void
MusEcalGUI::createLeafPanel()
{
  delete _fLeafPanel;
  _fLeafPanel = new MELeafPanel(gClient->GetRoot(), this, 400, 150);
}

void
MusEcalGUI::createMultiVarPanel()
{
  delete _fMultiVarPanel;
  _fMultiVarPanel = new MEMultiVarPanel(gClient->GetRoot(), this, 600, 300);
}

void 
MusEcalGUI::createTwoVarPanel( )
{
  // first delete it if it already exists
  delete _fTwoVarPanel;

  // then create
  _fTwoVarPanel = new METwoVarPanel(gClient->GetRoot(), this, 600, 300);

}

void
MusEcalGUI::refresh()
{
  MusEcal::refresh();
  fillHistograms();
  drawAPDHist();
  drawPNHist();
  historyPlot();
}

void 
MusEcalGUI::setType( int type, int color )
{
  MusEcal::setType( type, color );
  refresh();
}

void 
MusEcalGUI::setLMRegion( int lmr )
{
  MusEcal::setLMRegion( lmr );
  setLMRMenu();
}

void 
MusEcalGUI::setTime( ME::Time time )
{
  MusEcal::setTime( time );
  refresh();
}

void 
MusEcalGUI::setChannel( MEChannel* leaf )
{
  MusEcal::setChannel( leaf );


  if( isBarrel() )
    {
      if(_debug) cout << "GHM !!!! call to fillEBLocalHistogram !!! " << endl;
      fillEBLocalHistograms();
    }
  else
    {
     if(_debug)  cout << "GHM !!!! call to fillEELocalHistogram !!! " << endl;
      fillEELocalHistograms();
    }

  if(_debug) cout << "....Done !!! " << endl;

  if(_debug) cout << "drawAPDHist " << endl;
  drawAPDHist();
  if(_debug) cout << "....Done !!! " << endl;
  // JM: added drawPNhist here
  if(_debug) cout << "drawPNHist " << endl;
  drawPNHist();
  if(_debug) cout << "....Done !!! " << endl;
  if(_debug) cout << "historyPlot " << endl;
  historyPlot();
  if(_debug) cout << "....Done !!! " << endl;
}

void
MusEcalGUI::drawHistoryGraph( unsigned int n, 
			      float* x, float* y, float* ex, float* ey, bool* ok,
			      int markerStyle, float markerSize, int markerColor,
			      int lineWidth, int lineColor, const char* graphOpt )
{
  int nn=0;
  float xx[n];
  float yy[n];
  float exx[n];
  float eyy[n];
  float ok_(true);

  for( unsigned int ii=0; ii<n; ii++  )
    {
      if( ok!=0 )
	{
	  ok_ = ok[ii]; 
	}
      else
	{
	  ok_ = true;
	}
      if( ok_ )
	{
	  xx[nn]  = x[ii];
	  yy[nn]  = y[ii];
	  if( ex!=0 )
	    {
	      exx[nn] = ex[ii];
	      eyy[nn] = ey[ii];
	    }
	  else
	    {
	      exx[nn] = 0;
	      eyy[nn] = 0;
	    }
	  nn++;
	}
      if( !ok_ || ii==n-1 )
	{
	  if( nn!=0 )
	    {
	      TGraph* gr = new TGraphErrors( nn, xx, yy, exx, eyy );
	      gr->SetMarkerStyle( markerStyle );
	      gr->SetMarkerSize( markerSize );
	      gr->SetMarkerColor( markerColor );
	      gr->SetLineWidth( lineWidth );
	      gr->SetLineColor( lineColor );
	      gr->Draw(graphOpt);
	    }
	  nn = 0;
	}
    }
}

void
MusEcalGUI::setLMRMenu()
{
  TGMenuEntry* lmrentry_ = f_Channel_Menu->GetEntry("LMR");
  if( lmrentry_!=0 )
    {
      // cleanup the LMR menu
      for( unsigned int ii=0; ii<f_tree_menu.size(); ii++ )
	{    
	  delete f_tree_menu[ii];
	}
      f_tree_menu.clear();
      f_Channel_Menu->DeleteEntry( lmrentry_ );
      //      delete lmrentry_;
    }
  TGPopupMenu* menu_ = new TGPopupMenu(gClient->GetRoot());
  f_Channel_Menu->AddPopup("LMR",menu_);

  MEChannel* tree_ = curMgr()->tree();
  for( unsigned int ii=0; ii<tree_->n(); ii++ )
    {
      MEChannel* lmm_ = tree_->d(ii);
      TGPopupMenu* menu1_ = new TGPopupMenu(gClient->GetRoot());
      f_tree_menu.push_back(menu1_);
      menu_->AddPopup(lmm_->oneLine(),menu1_);
      for( unsigned int jj=0; jj<lmm_->n(); jj++ )
 	{
 	  MEChannel* sc_ = lmm_->d(jj);
	  TGPopupMenu* menu2_ = new TGPopupMenu(gClient->GetRoot());
	  f_tree_menu.push_back(menu2_);
	  menu2_->Connect("Activated(Int_t)", "MusEcalGUI", this,
			  "HandleChannelMenu(Int_t)");
	  menu1_->AddPopup(sc_->oneWord(),menu2_);
	  for( unsigned int kk=0; kk<sc_->n(); kk++ )
	    {
	      MEChannel* c_ = sc_->d(kk);
	      menu2_->AddEntry(c_->oneWord(),10000+c_->id());
	    }  
 	}  
    } 
}

void
MusEcalGUI::setupMainWindow()
{
  
  UInt_t h = GetHeight();
  UInt_t w = GetWidth();
  const UInt_t LimSmallScreen = 800;
  const UInt_t hlow           = 75;
  const UInt_t wminus         = 180;
  UInt_t hless = h - hlow;

  fL2 = new TGLayoutHints(kLHintsCenterX | kLHintsExpandX | kLHintsExpandY,5,5,5,5);
  if (h>LimSmallScreen) {
    fL1 = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,5,5,5,0);
    fL5 = new TGLayoutHints(kLHintsCenterX | kLHintsExpandX,5,5,5,5);
    fLb = new TGLayoutHints(kLHintsLeft,5,5,5,5);
  }
  else {
    fL1 = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,0,0,0,0);
    fL5 = new TGLayoutHints(kLHintsCenterX | kLHintsExpandX,0,0,0,0);
    fLb = new TGLayoutHints(kLHintsLeft,0,0,0,0);
  }
  fL8 = new TGLayoutHints(kLHintsTop | kLHintsCenterX,5,5,5,5);

  fHFrame1 = new TGHorizontalFrame(this,w,hless);
  fVFrame  = new TGVerticalFrame(fHFrame1,wminus-10,hless);
  fHFrame1->AddFrame(fVFrame,fLb);
  
  fMenuDock = new TGDockableFrame(this);
  AddFrame(fMenuDock, new TGLayoutHints(kLHintsExpandX, 0, 0, 1, 0));
  fMenuDock->SetWindowName("MusEcal Menu");

  fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsExpandX);
  fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);

  //
  // File menu
  //
  f_File_Menu = new TGPopupMenu( gClient->GetRoot() );

  f_File_Menu->AddEntry("D&ump Vector to Ascii File", 1);
  f_File_Menu->AddSeparator();
 
  f_File_Menu->AddEntry("R&un Panel", 3);
  f_File_Menu->AddSeparator();

  f_File_Menu->AddEntry("W&elcome", 4);
  f_File_Menu->AddSeparator();

  f_File_Menu->AddEntry("E&xit", 0);
  f_File_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
		       "HandleFileMenu(Int_t)");
  //
  // Histogram menu
  //
  f_Hist_Menu = new TGPopupMenu(gClient->GetRoot());
  f_Hist_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
		       "HandleHistMenu(Int_t)");
  f_Laser_Menu = new TGPopupMenu(gClient->GetRoot());
  f_Laser_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
			"HandleHistMenu(Int_t)");
  f_APD_Menu = new TGPopupMenu(gClient->GetRoot());
  f_APD_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
		       "HandleHistMenu(Int_t)");
  f_PN_Menu = new TGPopupMenu(gClient->GetRoot());
  f_PN_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
		       "HandleHistMenu(Int_t)");
  f_MTQ_Menu = new TGPopupMenu(gClient->GetRoot());
  f_MTQ_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
		       "HandleHistMenu(Int_t)");
  for( int icol=0; icol<ME::iSizeC; icol++ )
    {
      f_APD_Hist_Menu[icol][0] = new TGPopupMenu( gClient->GetRoot() );
      f_APD_Hist_Menu[icol][1] = new TGPopupMenu( gClient->GetRoot() );

      f_APD_Hist_Menu[icol][0]->Connect("Activated(Int_t)", "MusEcalGUI", this,
				     "HandleHistMenu(Int_t)");
      int halfmenu=int(float(ME::iSizeAPD)/2.);

      // too many variables for one menu test new popup:
      for( int ii=0; ii<halfmenu; ii++ )
	{
	  int jj = 100*icol+ii; 
	  f_APD_Hist_Menu[icol][0]->AddEntry( ME::APDPrimVar[ii], jj );
	  if( _type!=ME::iLaser || _color!=icol )f_APD_Hist_Menu[icol][0]->DisableEntry(jj); 
	} 
      for( int ii=halfmenu; ii<ME::iSizeAPD; ii++ )
	{
	  int jj = 100*icol+ii; 
	  f_APD_Hist_Menu[icol][1]->AddEntry( ME::APDPrimVar[ii], jj );
	  if( _type!=ME::iLaser || _color!=icol )f_APD_Hist_Menu[icol][1]->DisableEntry(jj); 
	}
      
      f_APD_Hist_Menu[icol][0]->AddPopup("more...",f_APD_Hist_Menu[icol][1]);
      f_APD_Menu->AddPopup(ME::color[icol],f_APD_Hist_Menu[icol][0]);

      f_PN_Hist_Menu[icol] = new TGPopupMenu( gClient->GetRoot() );
      f_PN_Hist_Menu[icol]->Connect("Activated(Int_t)", "MusEcalGUI", this,
				    "HandleHistMenu(Int_t)");
      for( int ii=0; ii<ME::iSizePN; ii++ )
	{
	  int jj = 1000+100*icol+ii;
	  f_PN_Hist_Menu[icol]->AddEntry( ME::PNPrimVar[ii], jj );
	  if( _type!=ME::iLaser || _color!=icol )f_PN_Hist_Menu[icol]->DisableEntry(jj); 
	}
      f_PN_Menu->AddPopup(ME::color[icol],f_PN_Hist_Menu[icol]);

      f_MTQ_Hist_Menu[icol] = new TGPopupMenu( gClient->GetRoot() );
      f_MTQ_Hist_Menu[icol]->Connect("Activated(Int_t)", "MusEcalGUI", this,
				     "HandleHistMenu(Int_t)");
      for( int ii=0; ii<ME::iSizeMTQ; ii++ )
	{
	  int jj = 2000+100*icol+ii;
	  f_MTQ_Hist_Menu[icol]->AddEntry( ME::MTQPrimVar[ii], jj );
	  if( _type!=ME::iLaser || _color!=icol )f_MTQ_Hist_Menu[icol]->DisableEntry(jj); 
	}
      f_MTQ_Menu->AddPopup(ME::color[icol],f_MTQ_Hist_Menu[icol]);
    }
  f_Laser_Menu->AddPopup("APD",f_APD_Menu);
  f_Laser_Menu->AddPopup("PN", f_PN_Menu);
  f_Laser_Menu->AddPopup("MTQ",f_MTQ_Menu);
  f_Hist_Menu->AddPopup("Laser",f_Laser_Menu);
  f_TP_Menu = new TGPopupMenu(gClient->GetRoot());
  f_TP_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
		     "HandleHistMenu(Int_t)");
  f_TPAPD_Hist_Menu = new TGPopupMenu( gClient->GetRoot() );
  f_TPAPD_Hist_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
			     "HandleHistMenu(Int_t)");
  for( int ii=0; ii<ME::iSizeTPAPD; ii++ )
    {
      int jj = ii; 
      f_TPAPD_Hist_Menu->AddEntry( ME::TPAPDPrimVar[ii], jj );
      if( _type!=ME::iTestPulse )f_TPAPD_Hist_Menu->DisableEntry(jj); 
    }
  f_TPPN_Hist_Menu = new TGPopupMenu( gClient->GetRoot() );
  f_TPPN_Hist_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
			    "HandleHistMenu(Int_t)");
  for( int ii=0; ii<ME::iSizeTPPN; ii++ )
    {
      int jj = 1000+ii;
      f_TPPN_Hist_Menu->AddEntry( ME::TPPNPrimVar[ii], jj );
      if( _type!=ME::iTestPulse )f_TPPN_Hist_Menu->DisableEntry(jj); 
    }
  f_TP_Menu->AddPopup("APD",f_TPAPD_Hist_Menu);
  f_TP_Menu->AddPopup("PN", f_TPPN_Hist_Menu);
  f_Hist_Menu->AddPopup("Test Pulse",f_TP_Menu);

  f_Hist_Menu->AddSeparator();
  f_Hist_Menu->AddEntry( "Histogram", 1000000+iHIST );
  f_Hist_Menu->AddEntry( "vs Channel-ID", 1000000+iVS_CHANNEL );
  f_Hist_Menu->AddEntry( "Map", 1000000+iMAP );

  f_Hist_Menu->AddSeparator();
  f_Hist_Menu->AddEntry( "Animation", -1000 );
  //
  // Channel menu
  //
  f_Channel_Menu = new TGPopupMenu(gClient->GetRoot());
  f_Channel_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
			  "HandleChannelMenu(Int_t)");
  f_Channel_Menu->AddEntry("Laser Monitoring Region", ME::iLMRegion     );
  f_Channel_Menu->AddEntry("Laser Monitoring Module", ME::iLMModule     );
  f_Channel_Menu->AddEntry("Super Crystal",           ME::iSuperCrystal );
  f_Channel_Menu->AddEntry("Crystal",                 ME::iCrystal      );

  f_Channel_Menu->AddSeparator();
  f_Channel_Menu->AddEntry("One Level Up",50000);

  f_Channel_Menu->AddSeparator();

  f_Channel_Menu->AddEntry("Channel panel",60000);
  
  f_Channel_Menu->AddSeparator();
  
  map< TString, TGPopupMenu* > _menu;
  TString str_ = "ECAL";
  _menu[str_] = new TGPopupMenu(gClient->GetRoot());
  for( int ireg=0; ireg<ME::iSizeE; ireg++ )
    {
      TString str1_ = str_+"_";
      str1_+=ME::region[ireg];
      _menu[str1_] = new TGPopupMenu(gClient->GetRoot());
      _menu[str_]->AddPopup(ME::region[ireg],_menu[str1_]);
      MEChannel* tree_ = ME::regTree(ireg);
      for( unsigned int ii=0; ii<tree_->n(); ii++ )
	{
	  MEChannel* sect_ = tree_->d(ii);
	  int isect_ = sect_->id();
	  TString str2_ = str1_+"_";
	  str2_+=ii;
	  _menu[str2_] = new TGPopupMenu(gClient->GetRoot());
	  _menu[str2_]->Connect("Activated(Int_t)", "MusEcalGUI", this,
				"HandleChannelMenu(Int_t)");
	  _menu[str1_]->AddPopup(ME::smName(ireg,isect_),_menu[str2_]);
	  for( unsigned int jj=0; jj<sect_->n(); jj++ )
	    {
	      MEChannel* lmr_ = sect_->d(jj);
	      int ilmr_=lmr_->id();
	      TString str3_ = lmr_->oneLine();
	      _menu[str2_]->AddEntry( str3_, 1000+ilmr_); 
	    }  
	}
    }
  f_Channel_Menu->AddPopup(str_,_menu[str_]);
  setLMRMenu();
  //
  // History menu
  //
  f_History_Menu = new TGPopupMenu(gClient->GetRoot());
  f_History_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
			  "HandleHistoryMenu(Int_t)");
  f_History_L_Menu = new TGPopupMenu(gClient->GetRoot());
  f_History_L_Menu->Connect("Activated(Int_t)", "MusEcalGUI", this,
			    "HandleHistoryMenu(Int_t)");
  
  for( int icol=0; icol<ME::iSizeC; icol++ ){
    for( int isplit=0; isplit<2; isplit++ ){
      f_History_LV_Menu[icol][isplit] = new TGPopupMenu( gClient->GetRoot() );
      
      if(isplit==0) f_History_LV_Menu[icol][isplit]->Connect("Activated(Int_t)", 
							     "MusEcalGUI", this,
							     "HandleHistoryMenu(Int_t)");
     
      
      int iimin, iimax;
      if(isplit==0){
	iimin=0; iimax=int(double(MusEcal::iSizeLV)/2.);
      }else{
	iimin=int(double(MusEcal::iSizeLV)/2.);
	iimax=MusEcal::iSizeLV;
      }
      for( int ii=iimin; ii<iimax; ii++ )
	{
	  int jj = 100*icol+ii; 
	  f_History_LV_Menu[icol][isplit]->AddEntry( MusEcal::historyVarName[ii], jj );
	  if( _type!=ME::iLaser || _color!=icol )
	    f_History_LV_Menu[icol][isplit]->DisableEntry(jj); 
	}      
    }
    f_History_LV_Menu[icol][0]->AddPopup("more...",f_History_LV_Menu[icol][1]);
    f_History_L_Menu->AddPopup(ME::color[icol],f_History_LV_Menu[icol][0] );
  } 
  f_History_Menu->AddPopup("Laser",f_History_L_Menu );
  
  f_History_TPV_Menu = new TGPopupMenu( gClient->GetRoot() );
  f_History_TPV_Menu->Connect("Activated(Int_t)", 
			      "MusEcalGUI", this,
			      "HandleHistoryMenu(Int_t)");
  for( int ii=0; ii<MusEcal::iSizeTPV; ii++ )
    {
      int jj = ii; 
      f_History_TPV_Menu->AddEntry( MusEcal::historyTPVarName[ii], jj );
      if( _type!=ME::iTestPulse )
	f_History_TPV_Menu->DisableEntry(jj); 
    }      
  f_History_Menu->AddPopup("TestPulse",f_History_TPV_Menu );

  f_History_Menu->AddSeparator();
  f_History_Menu->AddEntry("Normalized (toggle)", -10 );
  f_History_Menu->AddSeparator();
  f_History_Menu->AddEntry("Draw Intervals (toggle)",    -11 );
  f_History_Menu->AddEntry("vs Time (hrs)", -100 );
  f_History_Menu->AddEntry("Projection",    -101 );
  f_History_Menu->AddSeparator();
  f_History_Menu->AddEntry("Correlation",-3000 );
  f_History_Menu->AddEntry("Multi-Var",-2000 );
  f_History_Menu->AddEntry("Leaf",-1000 );

  fMenuDock->EnableUndock(kTRUE);
  fMenuDock->EnableHide(kTRUE);

  fMenuBar = new TGMenuBar(fMenuDock, 1, 1, kHorizontalFrame);
  fMenuBar->AddPopup("&MusEcal", f_File_Menu, fMenuBarItemLayout);
  fMenuBar->AddPopup("&Histograms", f_Hist_Menu, fMenuBarItemLayout);
  fMenuBar->AddPopup("&Channels", f_Channel_Menu, fMenuBarItemLayout);
  fMenuBar->AddPopup("&Histories", f_History_Menu, fMenuBarItemLayout);

  fMenuDock->AddFrame(fMenuBar, fMenuBarLayout);

  fEcanvas = new MEClickableCanvas("Ecanvas",fHFrame1,w-wminus,hless,this);  
  
  fHFrame1->AddFrame(fEcanvas,fL1);
  fHFrame2 = new TGHorizontalFrame(this,w,65);

  AddFrame(fHFrame1,fL2);
  AddFrame(fHFrame2, fL5);

  TString windowname = "MusEcal";
  SetWindowName(windowname);

  MapSubwindows();
  Resize(GetDefaultSize());

  SetWMPosition(2,2);
  MapWindow();

  SetCanvas( fEcanvas->GetCanvas(),
	     "MusEcal - Main Window",
	     "",
	     "",
	     ""
	     );

  //  welcome();
  ShowWelcome( true );

  cout << "... and now enjoy the GUI!" << endl;
}

MEPlotWindow*
MusEcalGUI::getWindow( TString WName_, int opt, int w, int h )
{
  if( opt!=0 )
    {
      if( opt==2 && _window.count(WName_)!=0 ) 
	{
	  delete _window[WName_];
	  _window[WName_]=0;
	}
      if( _window.count(WName_)==0 || _window[WName_]==0 ) 
	{
	  _window[WName_] = new MEPlotWindow( gClient->GetRoot(), 
					      this, WName_, w, h );
	}
    }
  if(_window.count(WName_)!=0 ) 
    {
      return _window[WName_];
    } 
  return 0;
}
