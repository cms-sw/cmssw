//
//  The implementation of the PixelResolutinHistograms.cc class.  Please
//  look at PixelResolutionHistograms.h header file for the interface.
//
//------------------------------------------------------------------------------

// The switch, undefined in CMSSW release, and defined by standalone compilation script:


#ifdef SI_PIXEL_TEMPLATE_STANDALONE
//
//--- Stand-alone: Include a the header file from the local directory, as well as
//    dummy implementations of SimpleHistogramGenerator, LogInfo, LogError and LogDebug...
//
#include "PixelResolutionHistograms.h"
//
class TH1F;
class TH2F;
class SimpleHistogramGenerator {
public:
  SimpleHistogramGenerator(TH1F * hist) : hist_(hist) {};
private:
  TH1F * hist_;         // we don't own it 
};
#define LOGDEBUG std::cout
#define LOGERROR std::cout
#define LOGINFO  std::cout
//
#else
//--- We're inside a CMSSW release: Include the real thing.
//
#include "FastSimulation/TrackingRecHitProducer/interface/PixelResolutionHistograms.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/Utilities/interface/SimpleHistogramGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#define LOGDEBUG LogDebug("")
#define LOGERROR edm::LogError("Error")
#define LOGINFO  edm::LogInfo("Info")
//
#endif




// Generic C stuff
#include <cmath>
#include <iostream>
#include <string>

// ROOT
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>

// Global definitions 
const float cmtomicron = 10000.0;


//------------------------------------------------------------------------------
//  Constructor: Books the FastSim Histograms, given the input parameters
//  which are provided as arguments. These variables are then const inside
//  the class. (That is, once we make the histograms, we can't change the
//  definition of the binning.)
//------------------------------------------------------------------------------
PixelResolutionHistograms::
PixelResolutionHistograms( std::string filename,   // ROOT file for histograms
			   std::string rootdir,    // Subdirectory in the file, "" if none
			   std::string descTitle,  // Descriptive title	     
			   unsigned int detType,    // Where we are... (&&& do we need this?)
			   double 	cotbetaBinWidth,
			   double 	cotbetaLowEdge,
			   int	cotbetaBins,
			   double	cotalphaBinWidth,
			   double	cotalphaLowEdge,
			   int	cotalphaBins) //,
			   //int	qbinWidth,
			   //int	qbins )
  : weOwnHistograms_(true),                          // we'll be making some histos
    detType_          ( detType ),
    cotbetaBinWidth_  ( cotbetaBinWidth ),
    cotbetaLowEdge_   ( cotbetaLowEdge ), 
    cotbetaBins_      ( cotbetaBins ),
    cotalphaBinWidth_ ( cotalphaBinWidth  ),
    cotalphaLowEdge_  ( cotalphaLowEdge ),
    cotalphaBins_     ( cotalphaBins ),
    qbinWidth_        ( 1 ),
    qbins_            ( 4 ),
    binningHisto_(nullptr),
    resMultiPixelXHist_(), resSinglePixelXHist_(),  // all to nullptr
    resMultiPixelYHist_(), resSinglePixelYHist_(),  // all to nullptr
    qbinHist_(),                                    // all to nullptr
    file_(nullptr),
    status_(0),
    resMultiPixelXGen_(), resSinglePixelXGen_(), 
    resMultiPixelYGen_(), resSinglePixelYGen_(),
    qbinGen_()
{

  file_ = std::make_unique<TFile>( filename.c_str(), "RECREATE" );
  //Resolution binning
  // const double 	cotbetaBinWidth = 1.0;
  // const double 	cotbetaLowEdge	= -11.5 ;
  // const int	cotbetaBins	= 23;
  // const double	cotalphaBinWidth	= 0.08 ;
  // const double	cotalphaLowEdge	= -0.36 ;
  // const int	cotalphaBins	= 9;
  // const int	qbinWidth	= 1;
  // const int	qbins		= 4;

  // Dummy 2D histogram to store binning:
  binningHisto_ = new TH2F( "ResHistoBinning", descTitle.c_str(),
			    cotbetaBins,  cotbetaLowEdge,  cotbetaLowEdge  + cotbetaBins*cotbetaBinWidth,
			    cotalphaBins, cotalphaLowEdge, cotalphaLowEdge + cotalphaBins*cotalphaBinWidth );

  // Store detType in the underflow bin
  binningHisto_->SetBinContent(0, 0, detType_);
  
  // All other histograms:
  Char_t histo[200];
  Char_t title[200];
  //
  //--- Histograms for clusters with multiple pixels hit in a given direction.
  //
  for( int ii=0; ii<cotbetaBins_; ii++ ) {
    for( int jj=0; jj<cotalphaBins_; jj++ ) {
      for( int kk=0; kk<qbins_; kk++ ) {
	//
	sprintf( histo, "hx%d1%02d%d%d", detType_, ii+1, jj+1, kk+1 );  //information of bits of histogram names
	//--- First bit 1/0 barrel/forward, second 1/0 multi/single, cotbeta, cotalpha, qbins
	sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f qbin %d npixel>1 X",
		 cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
		 cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_,
		 kk+1 );
	//
	resMultiPixelXHist_ [ ii ][ jj ][ kk ] = new TH1F(histo, title, 1000, -0.05, 0.05);

	sprintf( histo, "hy%d1%02d%d%d", detType_, ii+1, jj+1, kk+1 );
	sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f qbin %d npixel>1 Y",
		 cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
		 cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_,
		 kk+1 );
	//
	resMultiPixelYHist_ [ ii ][ jj ][ kk ] = new TH1F(histo, title, 1000, -0.05, 0.05);
      }
    }
  }


  //
  //--- Histograms for clusters where only a single pixel was hit in a given direction.
  //
  for( int ii=0; ii<cotbetaBins_; ii++) {
    for( int jj=0; jj<cotalphaBins_; jj++) {

      sprintf( histo, "hx%d0%02d%d", detType_, ii+1, jj+1 );  //information of bits of histogram names
      //first bit 1/0 barrel/forward, second 1/0 multi/single, cotbeta, cotalpha
      sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f npixel=1 X",
		 cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
		 cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_ );
      //
      resSinglePixelXHist_ [ ii ][ jj ] = new TH1F(histo, title, 1000, -0.05, 0.05);

      sprintf( histo, "hy%d0%02d%d", detType_, ii+1, jj+1 );
      sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f npixel=1 Y",
		 cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
	       cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_ );
      //
      resSinglePixelYHist_ [ ii ][ jj ] = new TH1F(histo, title, 1000, -0.05, 0.05);

      sprintf( histo, "hqbin%d%02d%d", detType_, ii+1, jj+1 );
      sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f qbin",
	       cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
	       cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_ );
      //
      qbinHist_ [ ii ][ jj ] = new TH1F(histo, title, 4, -0.49, 3.51);
      
    }
  }

}



//------------------------------------------------------------------------------
//  Another constructor: load the histograms from one file.
//     filename = full path to filename
//     rootdir  = ROOT directory inside the file
//
//  The other parameters are the same (needed later) and must correspond
//  to the histograms we are loading from the file.
//------------------------------------------------------------------------------
PixelResolutionHistograms::
PixelResolutionHistograms( std::string filename, 
			   std::string rootdir,
			   int   detType,
			   bool  ignore_multi,
			   bool  ignore_single, 
			   bool  ignore_qBin )
  : weOwnHistograms_(false),      // resolution histograms are owned by the ROOT file
    detType_          (-1),
    cotbetaBinWidth_  (0),
    cotbetaLowEdge_   (0), 
    cotbetaBins_      (0),
    cotalphaBinWidth_ (0),
    cotalphaLowEdge_  (0),
    cotalphaBins_     (0),
    qbinWidth_        (1),
    qbins_            (4),
    binningHisto_(nullptr),
    resMultiPixelXHist_(), resSinglePixelXHist_(),  // all to nullptr
    resMultiPixelYHist_(), resSinglePixelYHist_(),  // all to nullptr
    qbinHist_(),                                    // all to nullptr
    file_(nullptr),
    status_(0),
    resMultiPixelXGen_(), resSinglePixelXGen_(), 
    resMultiPixelYGen_(), resSinglePixelYGen_(),
    qbinGen_()
{
  Char_t histo[200];       // the name of the histogram
  Char_t title[200];       // histo title, for debugging and sanity checking (compare inside file)
  TH1F * tmphist = nullptr;  // cache for histo pointer

  //--- Open the file for reading.
  file_ = std::make_unique<TFile>( filename.c_str()  ,"READ");
  if ( !file_ ) {
    status_ = 1;
    LOGERROR << "PixelResolutionHistograms:: Error, file " << filename << " not found.";
    return;          // PixelTemplateSmearerBase will throw an exception upon our return.
  }

  //--- The dummy 2D histogram with the binning of cot\beta and cot\alpha:
  binningHisto_ = (TH2F*) file_->Get( Form( "%s%s" , rootdir.c_str(), "ResHistoBinning" ) );
  if ( !binningHisto_ ) {
    status_ = 11;
    LOGERROR << "PixelResolutionHistograms:: Error, binning histogrram ResHistoBinning not found.";
    return;          // PixelTemplateSmearerBase will throw an exception upon our return.
  }

  if ( detType == -1 ) {
    //--- Fish out detType from the underflow bin:
    detType_ = binningHisto_->GetBinContent(0, 0);
  }
  else {
    detType_ = detType;     // constructor's argument overrides what's in ResHistoBinning histogram.
  }

  //--- Now we fill the binning variables:
  cotbetaAxis_      = binningHisto_->GetXaxis();
  cotbetaBinWidth_  = binningHisto_->GetXaxis()->GetBinWidth(1);  // assume all same width
  cotbetaLowEdge_   = binningHisto_->GetXaxis()->GetXmin();       // low edge of the first bin
  cotbetaBins_      = binningHisto_->GetXaxis()->GetNbins();
  cotalphaAxis_     = binningHisto_->GetYaxis();
  cotalphaBinWidth_ = binningHisto_->GetYaxis()->GetBinWidth(1);  // assume all same width;
  cotalphaLowEdge_  = binningHisto_->GetYaxis()->GetXmin();       // low edge of the first bin;
  cotalphaBins_     = binningHisto_->GetYaxis()->GetNbins();

  //--- We want the following information to show up in *every* log file!
  // LOGINFO << std::endl 
  std::cout << std::endl
	  << "Loading pixel resolution file = " << filename << std::endl 
	  << " cotBeta[" << cotbetaLowEdge_ <<","<< cotbetaBinWidth_ <<","<< cotbetaBins_ << "]" << std::endl 
	  << " cotAlpha[" << cotalphaLowEdge_ <<","<< cotalphaBinWidth_ <<","<< cotalphaBins_ << "]" 
	  << std::endl;
  

  if ( !ignore_multi ) {
    //
    //--- Histograms for clusters with multiple pixels hit in a given direction.
    //
    for( int ii=0; ii<cotbetaBins_; ii++ ) {
      for( int jj=0; jj<cotalphaBins_; jj++ ) {
	for( int kk=0; kk<qbins_; kk++ ) {
	  //
	  sprintf( histo, "hx%d1%02d%d%d", detType_, ii+1, jj+1, kk+1 );  //information of bits of histogram names
	  //--- First bit 1/0 barrel/forward, second 1/0 multi/single, cotbeta, cotalpha, qbins
	  sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f qbin %d npixel>1 X",
		   cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
		   cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_,
		   kk+1 );
	  //
	  tmphist = (TH1F*) file_->Get( Form( "%s%s" , rootdir.c_str(), histo ) );
	  if ( !tmphist ) {
	    status_ = 2;
	    LOGERROR << "Failed to find histogram=" << std::string( histo );
	    return;
	  }
	  LOGDEBUG << "Found histo " << std::string(histo)
		   << " with title = " << std::string( tmphist->GetTitle() ) << std::endl;
	  if ( tmphist->GetEntries() < 5 ) {
	    LOGINFO << "Histogram " << std::string(histo) << " has only " << tmphist->GetEntries()
		    << " entries. Trouble ahead." << std::endl;
	  }
	  resMultiPixelXHist_ [ ii ][ jj ][ kk ] = tmphist;
	  resMultiPixelXGen_  [ ii ][ jj ][ kk ] = new SimpleHistogramGenerator( tmphist );
	  
	  
	  sprintf( histo, "hy%d1%02d%d%d", detType_, ii+1, jj+1, kk+1 );
	  sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f qbin %d npixel>1 Y",
		   cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
		   cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_,
		   kk+1 );
	  //
	  tmphist = (TH1F*) file_->Get( Form( "%s%s" , rootdir.c_str(), histo ) );
	  if ( !tmphist ) {
	    status_ = 3;
	    LOGERROR << "Failed to find histogram=" << std::string( histo );
	    return;
	  }
	  LOGDEBUG << "Found histo " << std::string(histo)
		   << " with title = " << std::string( tmphist->GetTitle() ) << std::endl;
	  if ( tmphist->GetEntries() < 5 ) {
	    LOGINFO << "Histogram " << std::string(histo) << " has only " << tmphist->GetEntries()
		    << " entries. Trouble ahead." << std::endl;
	  }
	  resMultiPixelYHist_ [ ii ][ jj ][ kk ] = tmphist;
	  resMultiPixelYGen_  [ ii ][ jj ][ kk ] = new SimpleHistogramGenerator( tmphist );
	}
      }
    }
    //
  } // if (not ignore multi)
  

  //
  //--- Histograms for clusters where only a single pixel was hit in a given direction.
  //
  for( int ii=0; ii<cotbetaBins_; ii++) {
    for( int jj=0; jj<cotalphaBins_; jj++) {

      //--- Single pixel, along X.
      //
      sprintf( histo, "hx%d0%02d%d", detType_, ii+1, jj+1 );  //information of bits of histogram names
      //--- First bit 1/0 barrel/forward, second 1/0 multi/single, cotbeta, cotalpha
      sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f npixel=1 X",
	       cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
	       cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_ );
      //
      tmphist = (TH1F*) file_->Get( Form( "%s%s" , rootdir.c_str(), histo ) );
      if ( !tmphist ) {
	if ( !ignore_single ) {
	  LOGERROR << "Failed to find histogram=" << std::string( histo );
	  status_ = 4;
	  return;
	}
      }
      else {
	LOGDEBUG << "Found histo " << std::string(histo)
		 << " with title = " << std::string( tmphist->GetTitle() ) << std::endl;
	LOGDEBUG << "Found histo with title = " << std::string( tmphist->GetTitle() ) << std::endl;
	if ( tmphist->GetEntries() < 5 ) {
	  LOGINFO << "Histogram " << std::string(histo) << " has only " << tmphist->GetEntries()
		  << " entries. Trouble ahead." << std::endl;
	}
	resSinglePixelXHist_ [ ii ][ jj ] = tmphist;
	resSinglePixelXGen_  [ ii ][ jj ] = new SimpleHistogramGenerator( tmphist );
      }


      //--- Single pixel, along Y.
      //
      sprintf( histo, "hy%d0%02d%d", detType_, ii+1, jj+1 );
      sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f npixel=1 Y",
	       cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
	       cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_ );
      //
      tmphist = (TH1F*) file_->Get( Form( "%s%s" , rootdir.c_str(), histo ) );
      if ( !tmphist ) {
	if ( !ignore_single ) {
	  LOGERROR << "Failed to find histogram=" << std::string( histo );
	  status_ = 5;
	  return;
	}
      }
      else {
	LOGDEBUG << "Found histo " << std::string(histo)
		 << " with title = " << std::string( tmphist->GetTitle() ) << std::endl;
	if ( tmphist->GetEntries() < 5 ) {
	  LOGINFO << "Histogram " << std::string(histo) << " has only " << tmphist->GetEntries()
		  << " entries. Trouble ahead." << std::endl;
	}
	resSinglePixelYHist_ [ ii ][ jj ] = tmphist;
	resSinglePixelYGen_  [ ii ][ jj ] = new SimpleHistogramGenerator( tmphist );
      }


      //--- qBin distribution, for this (cotbeta, cotalpha) bin.
      //
      sprintf( histo, "hqbin%d%02d%d", detType_, ii+1, jj+1 );
      sprintf( title, "cotbeta %.1f-%.1f cotalpha %.2f-%.2f qbin",
	       cotbetaLowEdge_ + ii*cotbetaBinWidth_ , cotbetaLowEdge_ + (ii+1)*cotbetaBinWidth_,
	       cotalphaLowEdge_ +jj*cotalphaBinWidth_, cotalphaLowEdge_ +(jj+1)*cotalphaBinWidth_ );
      //
      tmphist = (TH1F*) file_->Get( Form( "%s%s" , rootdir.c_str(), histo ) );
      if ( !tmphist ) {
	if ( !ignore_qBin ) {
	  LOGERROR << "Failed to find histogram=" << std::string( histo );
	  status_ = 6;
	  return;
	}
      }
      else {
	LOGDEBUG << "Found histo " << std::string(histo)
		 << " with title = " << std::string( tmphist->GetTitle() ) << std::endl;
	if ( tmphist->GetEntries() < 5 ) {
	  LOGINFO << "Histogram " << std::string(histo) << " has only " << tmphist->GetEntries()
		  << " entries. Trouble ahead." << std::endl;
	}
	qbinHist_ [ ii ][ jj ] = tmphist;
	qbinGen_  [ ii ][ jj ] = new SimpleHistogramGenerator( tmphist );
      }


    }
  }
}




//------------------------------------------------------------------------------
//  Destructor.  Use file_ pointer to tell whether we loaded the histograms
//  from a file (and do not own them), or we built them ourselves and thus need
//  to delete them.
//------------------------------------------------------------------------------
PixelResolutionHistograms::~PixelResolutionHistograms()
{
  //--- Delete histograms, but only if we own them. If 
  //--- they came from a file, let them be.
  //
  if ( ! weOwnHistograms_ ) {
    //--- Read the histograms from the TFile, the file will take care of them.
    file_->Close();
    /// delete file_ ;   // no need to delete if unique_ptr<>
    /// file_ = 0;
  }
  else {
    //--- We made the histograms, so first write them inthe output ROOT file and close it.
    LOGINFO
      << "PixelResHistoStore: Writing the histograms to the output file. " // << filename 
      << std::endl;
    file_->Write();
    file_->Close();

    // ROOT file has the ownership, and once the file is closed,
    // all of these guys are deleted.  So, we don't need to do anything.
  } // else



  //--- Delete FastSim generators. (It's safe to delete a nullptr.)
  for( int ii=0; ii<cotbetaBins_; ii++ ) {
    for( int jj=0; jj<cotalphaBins_; jj++ ) {
      for( int kk=0; kk<qbins_; kk++ ) {
	delete resMultiPixelXGen_ [ ii ][ jj ][ kk ];
	delete resMultiPixelYGen_ [ ii ][ jj ][ kk ];
      }
    }
  }
  for( int ii=0; ii<cotbetaBins_; ii++ ) {
    for( int jj=0; jj<cotalphaBins_; jj++ ) {
      delete resSinglePixelXGen_ [ ii ][ jj ];
      delete resSinglePixelYGen_ [ ii ][ jj ]; 
      delete qbinGen_ [ ii ][ jj ];
    }
  }

}


  



//------------------------------------------------------------------------------
//  Fills the appropriate FastSim histograms.
//  Returns 0 if the relevant histogram(s) were found and filled, 1 if not.
//------------------------------------------------------------------------------
int
PixelResolutionHistograms::
Fill( double dx, double dy, double cotalpha, double cotbeta, 
      int qbin, int nxpix, int nypix ) 
{
  int icotalpha, icotbeta, iqbin ;
  icotalpha = (int)floor( (cotalpha - cotalphaLowEdge_)/cotalphaBinWidth_ ) ;
  icotbeta  = (int)floor( (cotbeta - cotbetaLowEdge_) /cotbetaBinWidth_ ) ;
  iqbin = qbin > 2 ? 3 : qbin;
  if( icotalpha >= 0 && icotalpha < cotalphaBins_ && icotbeta >= 0 && icotbeta < cotbetaBins_ ) {
    qbinHist_[icotbeta][icotalpha]->Fill((double)iqbin);
    if( nxpix == 1 )
      resSinglePixelXHist_[icotbeta][icotalpha]->Fill(dx/cmtomicron);
    else
      resMultiPixelXHist_[icotbeta][icotalpha][iqbin]->Fill(dx/cmtomicron);
    if( nypix == 1 )
      resSinglePixelYHist_[icotbeta][icotalpha]->Fill(dy/cmtomicron);
    else
      resMultiPixelYHist_[icotbeta][icotalpha][iqbin]->Fill(dy/cmtomicron);
  }
  
  return 0;  
}


//------------------------------------------------------------------------------
//  Return the histogram generator for resolution in X.  A generator contains
//  both the histogram and knows how to throw a random number off it.  It is
//  called from FastSim (from PixelTemplateSmearerBase).
//  If cotalpha or cotbeta are outside of the range, return the end of the range.
//------------------------------------------------------------------------------
const SimpleHistogramGenerator * 
PixelResolutionHistograms::
getGeneratorX( double cotalpha, double cotbeta, int qbin, bool single )
{
  int icotalpha, icotbeta, iqbin ;
  icotalpha = (int)floor( (cotalpha - cotalphaLowEdge_) / cotalphaBinWidth_ ) ;
  icotbeta  = (int)floor( (cotbeta - cotbetaLowEdge_) / cotbetaBinWidth_ ) ;
  iqbin = qbin > 2 ? 3 : qbin;     // if (qbin>2) then = 3, else return qbin
  //
  //if( icotalpha >= 0 && icotalpha < cotalphaBins_ && icotbeta >= 0 && icotbeta < cotbetaBins_ ) {

  if (icotalpha < 0) 
    icotalpha = 0;
  if (icotalpha >= cotalphaBins_)
    icotalpha = cotalphaBins_ - 1;

  if (icotbeta < 0) 
    icotbeta = 0;
  if (icotbeta >= cotbetaBins_)
    icotbeta = cotbetaBins_ - 1;

  // At this point we are sure to return *some bin* from the 3D histogram

  if( single )
    return resSinglePixelXGen_[icotbeta][icotalpha];
  else
    return resMultiPixelXGen_[icotbeta][icotalpha][iqbin];

  // }
  //else
  //return nullptr;
}

//------------------------------------------------------------------------------
//  Return the histogram generator for resolution in Y.  A generator contains
//  both the histogram and knows how to throw a random number off it.  It is
//  called from FastSim (from PixelTemplateSmearerBase).
//  If cotalpha or cotbeta are outside of the range, return the end of the range.
//------------------------------------------------------------------------------
const SimpleHistogramGenerator * 
PixelResolutionHistograms::
getGeneratorY( double cotalpha, double cotbeta, int qbin, bool single )
{
  int icotalpha, icotbeta, iqbin ;
  icotalpha = (int)floor( (cotalpha - cotalphaLowEdge_) / cotalphaBinWidth_ ) ;
  icotbeta  = (int)floor( (cotbeta - cotbetaLowEdge_) / cotbetaBinWidth_ ) ;
  iqbin = qbin > 2 ? 3 : qbin;     // if (qbin>2) then = 3, else return qbin
  //
  //if( icotalpha >= 0 && icotalpha < cotalphaBins_ && icotbeta >= 0 && icotbeta < cotbetaBins_ ) {

  if (icotalpha < 0) 
    icotalpha = 0;
  if (icotalpha >= cotalphaBins_)
    icotalpha = cotalphaBins_ - 1;

  if (icotbeta < 0) 
    icotbeta = 0;
  if (icotbeta >= cotbetaBins_)
    icotbeta = cotbetaBins_ - 1;

  // At this point we are sure to return *some bin* from the 3D histogram

  if( single )
    return resSinglePixelYGen_[icotbeta][icotalpha];
  else
    return resMultiPixelYGen_[icotbeta][icotalpha][iqbin];

  //}
  //else
  //return nullptr;
}
