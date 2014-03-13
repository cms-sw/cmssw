// Include our own header first
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateReco.h"

// Geometry services
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

//#define DEBUG

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

// The template header files
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateSplit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include "boost/multi_array.hpp"

#include <iostream>

using namespace SiPixelTemplateReco;
//using namespace SiPixelTemplateSplit;
using namespace std;

const float PI = 3.141593;
const float HALFPI = PI * 0.5;
const float degsPerRad = 57.29578;

// &&& need a class const
const float micronsToCm = 1.0e-4;

const int cluster_matrix_size_x = 13;
const int cluster_matrix_size_y = 21;

//-----------------------------------------------------------------------------
//  Constructor.  All detUnit-dependent quantities will be initialized later,
//  in setTheDet().  Here we only load the templates into the template store templ_ .
//-----------------------------------------------------------------------------
PixelCPETemplateReco::PixelCPETemplateReco(edm::ParameterSet const & conf, 
					   const MagneticField * mag, const SiPixelLorentzAngle * lorentzAngle, 
					   const SiPixelTemplateDBObject * templateDBobject) 
  : PixelCPEBase(conf, mag, lorentzAngle, 0, templateDBobject, 0)
{
  //cout << endl;
  //cout << "Constructing PixelCPETemplateReco::PixelCPETemplateReco(...)................................................." << endl;
  //cout << endl;

  // &&& initialize the templates, etc.
  
  //-- Use Magnetic field at (0,0,0) to select a template ID [Morris, 6/25/08] (temporary until we implement DB access)
  
  DoCosmics_ = conf.getParameter<bool>("DoCosmics");
  //DoLorentz_ = conf.getParameter<bool>("DoLorentz"); // True when LA from alignment is used
  DoLorentz_ = conf.existsAs<bool>("DoLorentz")?conf.getParameter<bool>("DoLorentz"):false;

  LoadTemplatesFromDB_ = conf.getParameter<bool>("LoadTemplatesFromDB");

  //cout << " PixelCPETemplateReco : (int)LoadTemplatesFromDB_ = " << (int)LoadTemplatesFromDB_ << endl;
  //cout << "field_magnitude = " << field_magnitude << endl;
  
  // ggiurgiu@fnal.gov, 12/17/2008: use configuration parameter to decide between DB or text file template access
  if ( LoadTemplatesFromDB_ )
    {
      //cout << "PixelCPETemplateReco: Loading templates from database (DB) ------------------------------- " << endl;
      
      // Initialize template store to the selected ID [Morris, 6/25/08]  
      if ( !templ_.pushfile( *templateDBobject_) )
	throw cms::Exception("PixelCPETemplateReco") 
	  << "\nERROR: Templates not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version " 
	  << (*templateDBobject_).version() << "\n\n";
    }
  else 
    {
      //cout << "PixelCPETemplateReco : Loading templates 40 and 41 from ASCII files ------------------------" << endl;

      if ( !templ_.pushfile( 40 ) )
	throw cms::Exception("PixelCPETemplateReco") 
	  << "\nERROR: Templates 40 not loaded correctly from text file. Reconstruction will fail.\n\n";
    
      if ( !templ_.pushfile( 41 ) )
	throw cms::Exception("PixelCPETemplateReco") 
	  << "\nERROR: Templates 41 not loaded correctly from text file. Reconstruction will fail.\n\n";
    }
  
  speed_ = conf.getParameter<int>( "speed");
  LogDebug("PixelCPETemplateReco::PixelCPETemplateReco:") <<
    "Template speed = " << speed_ << "\n";
  
  UseClusterSplitter_ = conf.getParameter<bool>("UseClusterSplitter");
  
}

//-----------------------------------------------------------------------------
//  Clean up.
//-----------------------------------------------------------------------------
PixelCPETemplateReco::~PixelCPETemplateReco()
{
  // &&& delete template store?
}



//------------------------------------------------------------------
//  Public methods mandated by the base class.
//------------------------------------------------------------------

//------------------------------------------------------------------
//  The main call to the template code.
//------------------------------------------------------------------
LocalPoint
PixelCPETemplateReco::localPosition(const SiPixelCluster& cluster) const 
{
  bool fpix;  //  barrel(false) or forward(true)
  if ( thePart == GeomDetEnumerators::PixelBarrel )   
    fpix = false;    // no, it's not forward -- it's barrel
  else                                              
    fpix = true;     // yes, it's forward
  
 // Compute the Lorentz shifts for this detector element for the Alignment Group 
 if ( DoLorentz_ ) computeLorentzShifts();



  int ID = -9999;

  if ( LoadTemplatesFromDB_ )
    {
      ID = templateDBobject_->getTemplateID(theDet->geographicalId());
    }
  else
    {
      if ( !fpix )
	ID = 40; // barrel
      else 
	ID = 41; // endcap
    }
  
  //cout << "PixelCPETemplateReco : ID = " << ID << endl;
  

  // Make from cluster (a SiPixelCluster) a boost multi_array_2d called 
  // clust_array_2d.
  boost::multi_array<float, 2> clust_array_2d(boost::extents[cluster_matrix_size_x][cluster_matrix_size_y]);
  
  // Preparing to retrieve ADC counts from the SiPixelCluster.  In the cluster,
  // we have the following:
  //   int minPixelRow(); // Minimum pixel index in the x direction (low edge).
  //   int maxPixelRow(); // Maximum pixel index in the x direction (top edge).
  //   int minPixelCol(); // Minimum pixel index in the y direction (left edge).
  //   int maxPixelCol(); // Maximum pixel index in the y direction (right edge).
  // So the pixels from minPixelRow() will go into clust_array_2d[0][*],
  // and the pixels from minPixelCol() will go into clust_array_2d[*][0].

  int row_offset = cluster.minPixelRow();
  int col_offset = cluster.minPixelCol();
  
  // Store the coordinates of the center of the (0,0) pixel of the array that 
  // gets passed to PixelTempReco2D
  // Will add these values to the output of  PixelTempReco2D
  float tmp_x = float(row_offset) + 0.5f;
  float tmp_y = float(col_offset) + 0.5f;
  
  // Store these offsets (to be added later) in a LocalPoint after tranforming 
  // them from measurement units (pixel units) to local coordinates (cm)

  // ggiurgiu@jhu.edu 12/09/2010 : update call with trk angles needed for bow/kink corrections
  LocalPoint lp;
  
  if ( with_track_angle )
    lp = theTopol->localPosition( MeasurementPoint(tmp_x, tmp_y), loc_trk_pred_ );
  else
    {
      edm::LogError("PixelCPETemplateReco") 
	<< "@SUB = PixelCPETemplateReco::localPosition" 
	<< "Should never be here. PixelCPETemplateReco should always be called with track angles. This is a bad error !!! ";
      
      lp = theTopol->localPosition( MeasurementPoint(tmp_x, tmp_y) );
    }
    
  
  // Visualize large clusters ---------------------------------------------------------
  // From Petar: maybe this should be moved into a method in the base class?
  /*
    char cluster_matrix[100][100];
    for (int i=0; i<100; i++)
    for (int j=0; j<100; j++)
    cluster_matrix[i][j] = '.';
    
    if ( cluster.sizeX()>cluster_matrix_size_x || cluster.sizeY()>cluster_matrix_size_y )
    {		
    cout << "cluster.size()  = " << cluster.size()  << endl;
    cout << "cluster.sizeX() = " << cluster.sizeX() << endl;
    cout << "cluster.sizeY() = " << cluster.sizeY() << endl;
    
    for ( std::vector<SiPixelCluster::Pixel>::const_iterator pix = pixVec.begin(); pix != pixVec.end(); ++pix )
    {
    int i = (int)(pix->x) - row_offset;
    int j = (int)(pix->y) - col_offset;
    cluster_matrix[i][j] = '*';
    }
    
    for (int i=0; i<(int)cluster.sizeX()+2; i++)
    {
    for (int j=0; j<(int)cluster.sizeY()+2; j++)
    cout << cluster_matrix[i][j];
    cout << endl;
    }
    } // if ( cluster.sizeX()>cluster_matrix_size_x || cluster.sizeY()>cluster_matrix_size_y )
  */
  // End Visualize clusters ---------------------------------------------------------
  

  // Copy clust's pixels (calibrated in electrons) into clust_array_2d;
  for (int i=0 ; i!=cluster.size(); ++i ) 
    {
      auto pix = cluster.pixel(i);
      // *pixIter dereferences to Pixel struct, with public vars x, y, adc (all float)
      // 02/13/2008 ggiurgiu@fnal.gov: type of x, y and adc has been changed to unsigned char, unsigned short, unsigned short
      // in DataFormats/SiPixelCluster/interface/SiPixelCluster.h so the type cast to int is redundant. Leave it there, it 
      // won't hurt. 
      int irow = int(pix.x) - row_offset;   // &&& do we need +0.5 ???
      int icol = int(pix.y) - col_offset;   // &&& do we need +0.5 ???
      
      // Gavril : what do we do here if the row/column is larger than cluster_matrix_size_x/cluster_matrix_size_y = 7/21 ?
      // Ignore them for the moment...
      if ( irow<cluster_matrix_size_x && icol<cluster_matrix_size_y )
	// 02/13/2008 ggiurgiu@fnal.gov typecast pixIter->adc to float
	clust_array_2d[irow][icol] = (float)pix.adc;
    }
  
  // Make and fill the bool arrays flagging double pixels
  std::vector<bool> ydouble(cluster_matrix_size_y), xdouble(cluster_matrix_size_x);
  // x directions (shorter), rows
  for (int irow = 0; irow < cluster_matrix_size_x; ++irow)
    {
      xdouble[irow] = theRecTopol->isItBigPixelInX( irow+row_offset );
    }
      
  // y directions (longer), columns
  for (int icol = 0; icol < cluster_matrix_size_y; ++icol) 
    {
      ydouble[icol] = theRecTopol->isItBigPixelInY( icol+col_offset );
    }

  // Output:
  float nonsense = -99999.9f; // nonsense init value
  templXrec_ = templYrec_ = templSigmaX_ = templSigmaY_ = nonsense;
  // If the template recontruction fails, we want to return 1.0 for now
  templProbY_ = templProbX_ = templProbQ_ = 1.0f;
  templQbin_ = 0;
  // We have a boolean denoting whether the reco failed or not
  hasFilledProb_ = false;
	
  float templYrec1_ = nonsense;
  float templXrec1_ = nonsense;
  float templYrec2_ = nonsense;
  float templXrec2_ = nonsense;

  // ******************************************************************
  // Do it! Use cotalpha_ and cotbeta_ calculated in PixelCPEBase

 
  float locBz = (*theParam).bz;
    
  ierr =
    PixelTempReco2D( ID, cotalpha_, cotbeta_,
		     locBz, 
		     clust_array_2d, ydouble, xdouble,
		     templ_,
		     templYrec_, templSigmaY_, templProbY_,
		     templXrec_, templSigmaX_, templProbX_, 
		     templQbin_, 
		     speed_,
		     templProbQ_
		     );

  // ******************************************************************

  // Check exit status
  if unlikely( ierr != 0 ) 
    {
      LogDebug("PixelCPETemplateReco::localPosition") <<
	"reconstruction failed with error " << ierr << "\n";

      // Gavril: what do we do in this case ? For now, just return the cluster center of gravity in microns
      // In the x case, apply a rough Lorentz drift average correction
      // To do: call PixelCPEGeneric whenever PixelTempReco2D fails
      float lorentz_drift = -999.9;
      if ( thePart == GeomDetEnumerators::PixelBarrel )
	lorentz_drift = 60.0f; // in microns
      else if ( thePart == GeomDetEnumerators::PixelEndcap )
	lorentz_drift = 10.0f; // in microns
      else 
	throw cms::Exception("PixelCPETemplateReco::localPosition :") 
	  << "A non-pixel detector type in here?" << "\n";
      // ggiurgiu@jhu.edu, 21/09/2010 : trk angles needed to correct for bows/kinks
      if ( with_track_angle )
	{
	  templXrec_ = theTopol->localX( cluster.x(), loc_trk_pred_ ) - lorentz_drift * micronsToCm; // rough Lorentz drift correction
	  templYrec_ = theTopol->localY( cluster.y(), loc_trk_pred_ ); 
	}
      else
	{
	  edm::LogError("PixelCPETemplateReco") 
	    << "@SUB = PixelCPETemplateReco::localPosition" 
	    << "Should never be here. PixelCPETemplateReco should always be called with track angles. This is a bad error !!! ";
	  
	  templXrec_ = theTopol->localX( cluster.x() ) - lorentz_drift * micronsToCm; // rough Lorentz drift correction
	  templYrec_ = theTopol->localY( cluster.y() );
	}
    }
  else if unlikely( UseClusterSplitter_ && templQbin_ == 0 )
    {
      cout << " PixelCPETemplateReco : We should never be here !!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << "                 (int)UseClusterSplitter_ = " << (int)UseClusterSplitter_ << endl;

      //ierr = 
      //PixelTempSplit( ID, fpix, cotalpha_, cotbeta_, 
      //		clust_array_2d, ydouble, xdouble, 
      //		templ_, 
      //		templYrec1_, templYrec2_, templSigmaY_, templProbY_,
      //		templXrec1_, templXrec2_, templSigmaX_, templProbX_, 
      //		templQbin_ );


      float dchisq;
      float templProbQ_;
      SiPixelTemplate2D templ2D_;
      templ2D_.pushfile(ID);
      	
      ierr =
	SiPixelTemplateSplit::PixelTempSplit( ID, cotalpha_, cotbeta_,
					      clust_array_2d, 
					      ydouble, xdouble,
					      templ_,
					      templYrec1_, templYrec2_, templSigmaY_, templProbY_,
					      templXrec1_, templXrec2_, templSigmaX_, templProbX_,
					      templQbin_, 
					      templProbQ_,
					      true, 
					      dchisq, 
					      templ2D_ );
      

      if ( ierr != 0 )
	{
	  LogDebug("PixelCPETemplateReco::localPosition") <<
	    "reconstruction failed with error " << ierr << "\n";
	  
	  // Gavril: what do we do in this case ? For now, just return the cluster center of gravity in microns
	  // In the x case, apply a rough Lorentz drift average correction
	  // To do: call PixelCPEGeneric whenever PixelTempReco2D fails
	  float lorentz_drift = -999.9f;
	  if ( thePart == GeomDetEnumerators::PixelBarrel )
	    lorentz_drift = 60.0f; // in microns
	  else if ( thePart == GeomDetEnumerators::PixelEndcap )
	    lorentz_drift = 10.0f; // in microns
	  else 
	    throw cms::Exception("PixelCPETemplateReco::localPosition :") 
	      << "A non-pixel detector type in here?" << "\n";

	  // ggiurgiu@jhu.edu, 12/09/2010 : trk angles needed to correct for bows/kinks
	  if ( with_track_angle )
	    {
	      templXrec_ = theTopol->localX( cluster.x(),loc_trk_pred_ ) - lorentz_drift * micronsToCm; // rough Lorentz drift correction
	      templYrec_ = theTopol->localY( cluster.y(),loc_trk_pred_ );
	    }
	  else
	    {
	      edm::LogError("PixelCPETemplateReco") 
		<< "@SUB = PixelCPETemplateReco::localPosition" 
		<< "Should never be here. PixelCPETemplateReco should always be called with track angles. This is a bad error !!! ";
	      
	      templXrec_ = theTopol->localX( cluster.x() ) - lorentz_drift * micronsToCm; // very rough Lorentz drift correction
	      templYrec_ = theTopol->localY( cluster.y() );
	      
	    }
	}
      else
	{
	  // go from micrometer to centimeter      
	  templXrec1_ *= micronsToCm;
	  templYrec1_ *= micronsToCm;	  
	  templXrec2_ *= micronsToCm;
	  templYrec2_ *= micronsToCm;
      
	  // go back to the module coordinate system   
	  templXrec1_ += lp.x();
	  templYrec1_ += lp.y();
	  templXrec2_ += lp.x();
	  templYrec2_ += lp.y();
      
	  // calculate distance from each hit to the track and choose the 
	  // hit closest to the track
	  float distance11 = sqrt( (templXrec1_ - trk_lp_x)*(templXrec1_ - trk_lp_x) + 
				   (templYrec1_ - trk_lp_y)*(templYrec1_ - trk_lp_y) );
	  
	  float distance12 = sqrt( (templXrec1_ - trk_lp_x)*(templXrec1_ - trk_lp_x) + 
				   (templYrec2_ - trk_lp_y)*(templYrec2_ - trk_lp_y) );
	  
	  float distance21 = sqrt( (templXrec2_ - trk_lp_x)*(templXrec2_ - trk_lp_x) + 
				   (templYrec1_ - trk_lp_y)*(templYrec1_ - trk_lp_y) );
	  
	  float distance22 = sqrt( (templXrec2_ - trk_lp_x)*(templXrec2_ - trk_lp_x) + 
				   (templYrec2_ - trk_lp_y)*(templYrec2_ - trk_lp_y) );
	  
	  float min_templXrec_ = -999.9;
	  float min_templYrec_ = -999.9;
	  float distance_min = 9999999999.9;
	  if ( distance11 < distance_min )
	    {
	      distance_min = distance11;
	      min_templXrec_ = templXrec1_;
	      min_templYrec_ = templYrec1_;
	    }
	  if ( distance12 < distance_min )
	    {
	      distance_min = distance12;
	      min_templXrec_ = templXrec1_;
	      min_templYrec_ = templYrec2_;
	    }
	  if ( distance21 < distance_min )
	    {
	      distance_min = distance21;
	      min_templXrec_ = templXrec2_;
	      min_templYrec_ = templYrec1_;
	    }
	  if ( distance22 < distance_min )
	    {
	      distance_min = distance22;
	      min_templXrec_ = templXrec2_;
	      min_templYrec_ = templYrec2_;
	    }
	  	  
	  templXrec_ = min_templXrec_;
	  templYrec_ = min_templYrec_;
	}
    } // else if ( UseClusterSplitter_ && templQbin_ == 0 )

  else // apparenly this is he good one!
    {
      // go from micrometer to centimeter      
      templXrec_ *= micronsToCm;
      templYrec_ *= micronsToCm;
      
      // go back to the module coordinate system 
      templXrec_ += lp.x();
      templYrec_ += lp.y();

      // Compute the Alignment Group Corrections [template ID should already be selected from call to reco procedure]
      if ( DoLorentz_ ) {
	  // Donly if the lotentzshift has meaningfull numbers
	if( lorentzShiftInCmX_!= 0.0 ||  lorentzShiftInCmY_!= 0.0 ) {   
	  // the LA width/shift returned by templates use (+)
	  // the LA width/shift produced by PixelCPEBase for positive LA is (-)
	  // correct this by iserting (-)
	  float templateLorwidthCmX = -micronsToCm*templ_.lorxwidth();
	  float templateLorwidthCmY = -micronsToCm*templ_.lorywidth();
	  // now, correctly, we can use the difference of shifts  
	  templXrec_ += 0.5*(lorentzShiftInCmX_ - templateLorwidthCmX);
	  templYrec_ += 0.5*(lorentzShiftInCmY_ - templateLorwidthCmY);
	  //cout << "Templates: la lorentz offset = " <<(0.5*(lorentzShiftInCmX_-templateLorwidthCmX))<< endl; //dk
	} //else {cout<<" LA is 0, disable offset corrections "<<endl;} //dk
      } //else {cout<<" Do not do LA offset correction "<<endl;} //dk

    }
    
  // Save probabilities and qBin in the quantities given to us by the base class
  // (for which there are also inline getters).  &&& templProbX_ etc. should be retired...
  probabilityX_  = templProbX_;
  probabilityY_  = templProbY_;
  probabilityQ_  = templProbQ_;
  qBin_          = templQbin_;
  
  if ( ierr == 0 ) // always true here
    hasFilledProb_ = true;
  
  return LocalPoint( templXrec_, templYrec_ );      
  
}
  
//------------------------------------------------------------------
//  localError() relies on localPosition() being called FIRST!!!
//------------------------------------------------------------------
LocalError  
PixelCPETemplateReco::localError( const SiPixelCluster& cluster) const 
{
  //cout << endl;
  //cout << "Set PixelCPETemplate errors .............................................." << endl;

  //cout << "CPETemplate : " << endl;

  //--- Default is the maximum error used for edge clusters.
  const float sig12 = 1./sqrt(12.0);
  float xerr = thePitchX *sig12;
  float yerr = thePitchY *sig12;
  
  // Check if the errors were already set at the clusters splitting level
  if ( cluster.getSplitClusterErrorX() > 0.0f && cluster.getSplitClusterErrorX() < 7777.7f && 
       cluster.getSplitClusterErrorY() > 0.0f && cluster.getSplitClusterErrorY() < 7777.7f )
    {
      xerr = cluster.getSplitClusterErrorX() * micronsToCm;
      yerr = cluster.getSplitClusterErrorY() * micronsToCm;

      //cout << "Errors set at cluster splitting level : " << endl;
      //cout << "xerr = " << xerr << endl;
      //cout << "yerr = " << yerr << endl;
    }
  else
    {
      // If errors are not split at the cluster splitting level, set the errors here

      //cout  << "Errors are not split at the cluster splitting level, set the errors here : " << endl; 
       
      int maxPixelCol = cluster.maxPixelCol();
      int maxPixelRow = cluster.maxPixelRow();
      int minPixelCol = cluster.minPixelCol();
      int minPixelRow = cluster.minPixelRow();
      
      //--- Are we near either of the edges?
      bool edgex = ( theRecTopol->isItEdgePixelInX( minPixelRow ) || theRecTopol->isItEdgePixelInX( maxPixelRow ) );
      bool edgey = ( theRecTopol->isItEdgePixelInY( minPixelCol ) || theRecTopol->isItEdgePixelInY( maxPixelCol ) );
      
      if ( ierr !=0 ) 
	{
	  // If reconstruction fails the hit position is calculated from cluster center of gravity 
	  // corrected in x by average Lorentz drift. Assign huge errors.
	  //xerr = 10.0 * (float)cluster.sizeX() * xerr;
	  //yerr = 10.0 * (float)cluster.sizeX() * yerr;
	  
	  // Assign better errors based on the residuals for failed template cases
	  if ( thePart == GeomDetEnumerators::PixelBarrel )
	    {
	      xerr = 55.0f * micronsToCm;
	      yerr = 36.0f * micronsToCm;
	    }
	  else if ( thePart == GeomDetEnumerators::PixelEndcap )
	    {
	      xerr = 42.0f * micronsToCm;
	      yerr = 39.0f * micronsToCm;
	    }
	  else 
	    throw cms::Exception("PixelCPETemplateReco::localError :") << "A non-pixel detector type in here?" ;

	  //cout << "xerr = " << xerr << endl;
	  //cout << "yerr = " << yerr << endl;
	  
	  //return LocalError(xerr*xerr, 0, yerr*yerr);
	}
      else if ( edgex || edgey )
	{
	  // for edge pixels assign errors according to observed residual RMS 
	  if      ( edgex && !edgey )
	    {
	      xerr = 23.0f * micronsToCm;
	      yerr = 39.0f * micronsToCm;
	    }
	  else if ( !edgex && edgey )
	    {
	      xerr = 24.0f * micronsToCm;
	      yerr = 96.0f * micronsToCm;
	    }
	  else if ( edgex && edgey )
	    {
	      xerr = 31.0f * micronsToCm;
	      yerr = 90.0f * micronsToCm;
	    }
	  else
	    {
	      throw cms::Exception(" PixelCPETemplateReco::localError: Something wrong with pixel edge flag !!!");
	    }

	  //cout << "xerr = " << xerr << endl;
	  //cout << "yerr = " << yerr << endl;
	}
      else 
	{
	  // &&& need a class const
	  //const float micronsToCm = 1.0e-4;
	  
	  xerr = templSigmaX_ * micronsToCm;
	  yerr = templSigmaY_ * micronsToCm;
	  
	  //cout << "xerr = " << xerr << endl;
	  //cout << "yerr = " << yerr << endl;

	  // &&& should also check ierr (saved as class variable) and return
	  // &&& nonsense (another class static) if the template fit failed.
	}       
      
      if (theVerboseLevel > 9) 
	{
	  LogDebug("PixelCPETemplateReco") <<
	    " Sizex = " << cluster.sizeX() << " Sizey = " << cluster.sizeY() << " Edgex = " << edgex << " Edgey = " << edgey << 
	    " ErrX  = " << xerr            << " ErrY  = " << yerr;
	}

    } // else
  
  if ( !(xerr > 0.0f) )
    throw cms::Exception("PixelCPETemplateReco::localError") 
      << "\nERROR: Negative pixel error xerr = " << xerr << "\n\n";
  
  if ( !(yerr > 0.0f) )
    throw cms::Exception("PixelCPETemplateReco::localError") 
      << "\nERROR: Negative pixel error yerr = " << yerr << "\n\n";

  //cout << "Final errors set to: " << endl;
  //cout << "xerr = " << xerr << endl;
  //cout << "yerr = " << yerr << endl;
  //cout << "Out of PixelCPETemplateREco..........................................................................." << endl;
  //cout << endl;

  return LocalError(xerr*xerr, 0, yerr*yerr);
}

