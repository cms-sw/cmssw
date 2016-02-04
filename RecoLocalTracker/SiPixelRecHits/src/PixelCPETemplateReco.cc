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
  : PixelCPEBase(conf, mag, lorentzAngle, 0, templateDBobject)
{
  //cout << "From PixelCPETemplateReco::PixelCPETemplateReco(...)" << endl;

  // &&& initialize the templates, etc.
  
  //-- Use Magnetic field at (0,0,0) to select a template ID [Morris, 6/25/08] (temporary until we implement DB access)
  
  DoCosmics_ = conf.getParameter<bool>("DoCosmics");
  LoadTemplatesFromDB_ = conf.getParameter<bool>("LoadTemplatesFromDB");
  //cout << "(int)LoadTemplatesFromDB_ = " << (int)LoadTemplatesFromDB_ << endl;
  //cout << "field_magnitude = " << field_magnitude << endl;
  
  // ggiurgiu@fnal.gov, 12/17/2008: use configuration parameter to decide between DB or text file template access
  if ( LoadTemplatesFromDB_ )
    {
      // Initialize template store to the selected ID [Morris, 6/25/08]  
      if ( !templ_.pushfile( *templateDBobject_) )
	throw cms::Exception("PixelCPETemplateReco") 
	  << "\nERROR: Templates not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version " 
	  << (*templateDBobject_).version() << "\n\n";
    }
  else 
    {
      if ( !templ_.pushfile( templID_ ) )
	throw cms::Exception("PixelCPETemplateReco") 
	  << "\nERROR: Templates not loaded correctly from text file. Reconstruction will fail.\n\n";
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


MeasurementPoint 
PixelCPETemplateReco::measurementPosition(const SiPixelCluster& cluster, 
					  const GeomDetUnit & det) const
{
  LocalPoint lp = localPosition(cluster,det);
 
  // ggiurgiu@jhu.edu 12/09/2010 : trk angles needed to correct for bows/kinks
  if ( with_track_angle )
    return theTopol->measurementPosition(lp, Topology::LocalTrackAngles( loc_traj_param_.dxdz(), loc_traj_param_.dydz() ) );
  else
    {
      edm::LogError("PixelCPETemplateReco") 
	<< "@SUB = PixelCPETemplateReco::measurementPosition" 
	<< "Should never be here. PixelCPETemplateReco should always be called with track angles. This is a bad error !!! ";
      
      return theTopol->measurementPosition( lp ); 
    }
}


//------------------------------------------------------------------
//  Public methods mandated by the base class.
//------------------------------------------------------------------

//------------------------------------------------------------------
//  The main call to the template code.
//------------------------------------------------------------------
LocalPoint
PixelCPETemplateReco::localPosition(const SiPixelCluster& cluster, const GeomDetUnit & det) const 
{
  setTheDet( det, cluster );
  templID_ = templateDBobject_->getTemplateID(theDet->geographicalId());
  
  //int ierr;   //!< return status
  int ID = templID_; //!< take the template ID that was selected by the constructor [Morris, 6/25/2008]


  bool fpix;  //!< barrel(false) or forward(true)
  if ( thePart == GeomDetEnumerators::PixelBarrel )   
    fpix = false;    // no, it's not forward -- it's barrel
  else                                              
    fpix = true;     // yes, it's forward
  
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
  float tmp_x = float(cluster.minPixelRow()) + 0.5;
  float tmp_y = float(cluster.minPixelCol()) + 0.5;
  
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

  const std::vector<SiPixelCluster::Pixel> & pixVec = cluster.pixels();
  std::vector<SiPixelCluster::Pixel>::const_iterator 
    pixIter = pixVec.begin(), pixEnd = pixVec.end();
  
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
  for ( ; pixIter != pixEnd; ++pixIter ) 
    {
      // *pixIter dereferences to Pixel struct, with public vars x, y, adc (all float)
      // 02/13/2008 ggiurgiu@fnal.gov: type of x, y and adc has been changed to unsigned char, unsigned short, unsigned short
      // in DataFormats/SiPixelCluster/interface/SiPixelCluster.h so the type cast to int is redundant. Leave it there, it 
      // won't hurt. 
      int irow = int(pixIter->x) - row_offset;   // &&& do we need +0.5 ???
      int icol = int(pixIter->y) - col_offset;   // &&& do we need +0.5 ???
      
      // ggiurgiu@jhu.edu : what do we do here if the row/column is larger than cluster_matrix_size_x/cluster_matrix_size_y = 7/21 ?
      // Ignore them for the moment...
      if ( irow<cluster_matrix_size_x && icol<cluster_matrix_size_y )
	// 02/13/2008 ggiurgiu@fnal.gov typecast pixIter->adc to float
	clust_array_2d[irow][icol] = (float)pixIter->adc;
    }
  
  // Make and fill the bool arrays flagging double pixels
  std::vector<bool> ydouble(cluster_matrix_size_y), xdouble(cluster_matrix_size_x);
  // x directions (shorter), rows
  for (int irow = 0; irow < cluster_matrix_size_x; ++irow)
    {
      xdouble[irow] = theTopol->isItBigPixelInX( irow+row_offset );
    }
      
  // y directions (longer), columns
  for (int icol = 0; icol < cluster_matrix_size_y; ++icol) 
    {
      ydouble[icol] = theTopol->isItBigPixelInY( icol+col_offset );
    }

  // Output:
  float nonsense = -99999.9; // nonsense init value
  templXrec_ = templYrec_ = templSigmaX_ = templSigmaY_ = nonsense;
	// If the template recontruction fails, we want to return 1.0 for now
	templProbY_ = templProbX_ = templProbQ_ = 1.0;
	templQbin_ = 0;
	// We have a boolean denoting whether the reco failed or not
	hasFilledProb_ = false;
	
  float templYrec1_ = nonsense;
  float templXrec1_ = nonsense;
  float templYrec2_ = nonsense;
  float templXrec2_ = nonsense;

  // ******************************************************************
  // Do it! Use cotalpha_ and cotbeta_ calculated in PixelCPEBase

  GlobalVector bfield = magfield_->inTesla( theDet->surface().position() ); 
  
  Frame detFrame( theDet->surface().position(), theDet->surface().rotation() );
  LocalVector Bfield = detFrame.toLocal( bfield );
  float locBz = Bfield.z();
    
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
  if ( ierr != 0 ) 
    {
      LogDebug("PixelCPETemplateReco::localPosition") <<
	"reconstruction failed with error " << ierr << "\n";

      // ggiurgiu@jhu.edu: what do we do in this case ? For now, just return the cluster center of gravity in microns
      // In the x case, apply a rough Lorentz drift average correction
      // To do: call PixelCPEGeneric whenever PixelTempReco2D fails
      double lorentz_drift = -999.9;
      if ( thePart == GeomDetEnumerators::PixelBarrel )
	lorentz_drift = 60.0; // in microns
      else if ( thePart == GeomDetEnumerators::PixelEndcap )
	lorentz_drift = 10.0; // in microns
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
  else if ( UseClusterSplitter_ && templQbin_ == 0 )
    {
      ierr = 
	PixelTempSplit( ID, fpix, cotalpha_, cotbeta_, 
			clust_array_2d, ydouble, xdouble, 
			templ_, 
			templYrec1_, templYrec2_, templSigmaY_, templProbY_,
			templXrec1_, templXrec2_, templSigmaX_, templProbX_, 
			templQbin_ );

      if ( ierr != 0 )
	{
	  LogDebug("PixelCPETemplateReco::localPosition") <<
	    "reconstruction failed with error " << ierr << "\n";
	  
	  // ggiurgiu@jhu.edu: what do we do in this case ? For now, just return the cluster center of gravity in microns
	  // In the x case, apply a rough Lorentz drift average correction
	  // To do: call PixelCPEGeneric whenever PixelTempReco2D fails
	  double lorentz_drift = -999.9;
	  if ( thePart == GeomDetEnumerators::PixelBarrel )
	    lorentz_drift = 60.0; // in microns
	  else if ( thePart == GeomDetEnumerators::PixelEndcap )
	    lorentz_drift = 10.0; // in microns
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
	  
	  int index_dist = -999;
	  float min_templXrec_ = -999.9;
	  float min_templYrec_ = -999.9;
	  float distance_min = 9999999999.9;
	  if ( distance11 < distance_min )
	    {
	      distance_min = distance11;
	      min_templXrec_ = templXrec1_;
	      min_templYrec_ = templYrec1_;
	      index_dist = 1;
	    }
	  if ( distance12 < distance_min )
	    {
	      distance_min = distance12;
	      min_templXrec_ = templXrec1_;
	      min_templYrec_ = templYrec2_;
	      index_dist = 2;
	    }
	  if ( distance21 < distance_min )
	    {
	      distance_min = distance21;
	      min_templXrec_ = templXrec2_;
	      min_templYrec_ = templYrec1_;
	      index_dist = 3;
	    }
	  if ( distance22 < distance_min )
	    {
	      distance_min = distance22;
	      min_templXrec_ = templXrec2_;
	      min_templYrec_ = templYrec2_;
	      index_dist = 4;
	    }
	  	  
	  templXrec_ = min_templXrec_;
	  templYrec_ = min_templYrec_;
	}
    } // else if ( UseClusterSplitter_ && templQbin_ == 0 )
  else 
    {
      // go from micrometer to centimeter      
      templXrec_ *= micronsToCm;
      templYrec_ *= micronsToCm;
      
      // go back to the module coordinate system 
      templXrec_ += lp.x();
      templYrec_ += lp.y();
    }
    
  // Save probabilities and qBin in the quantities given to us by the base class
  // (for which there are also inline getters).  &&& templProbX_ etc. should be retired...
  probabilityX_  = templProbX_;
  probabilityY_  = templProbY_;
  probabilityQ_  = templProbQ_;
  qBin_          = templQbin_;
	if(ierr==0) hasFilledProb_ = true;
	
  LocalPoint template_lp = LocalPoint( nonsense, nonsense );
  template_lp = LocalPoint( templXrec_, templYrec_ );      
  
  return template_lp;

}
  
//------------------------------------------------------------------
//  localError() relies on localPosition() being called FIRST!!!
//------------------------------------------------------------------
LocalError  
PixelCPETemplateReco::localError( const SiPixelCluster& cluster, 
				  const GeomDetUnit& det ) const 
{
  setTheDet( det, cluster );
  
  //--- Default is the maximum error used for edge clusters.
  float xerr = thePitchX / sqrt(12.0);
  float yerr = thePitchY / sqrt(12.0);
  
  int maxPixelCol = cluster.maxPixelCol();
  int maxPixelRow = cluster.maxPixelRow();
  int minPixelCol = cluster.minPixelCol();
  int minPixelRow = cluster.minPixelRow();
  
  //--- Are we near either of the edges?
  bool edgex = ( theTopol->isItEdgePixelInX( minPixelRow ) || theTopol->isItEdgePixelInX( maxPixelRow ) );
  bool edgey = ( theTopol->isItEdgePixelInY( minPixelCol ) || theTopol->isItEdgePixelInY( maxPixelCol ) );
  
  if ( ierr !=0 ) 
    {
      // If reconstruction fails the hit position is calculated from cluster center of gravity 
      // corrected in x by average Lorentz drift. Assign huge errors.
      //xerr = 10.0 * (float)cluster.sizeX() * xerr;
      //yerr = 10.0 * (float)cluster.sizeX() * yerr;

      // Assign better errors based on the residuals for failed template cases
      if ( thePart == GeomDetEnumerators::PixelBarrel )
	{
	  xerr = 55.0 * micronsToCm;
	  yerr = 36.0 * micronsToCm;
	}
      else if ( thePart == GeomDetEnumerators::PixelEndcap )
	{
	  xerr = 42.0 * micronsToCm;
	  yerr = 39.0 * micronsToCm;
	}
      else 
	throw cms::Exception("PixelCPETemplateReco::localError :") << "A non-pixel detector type in here?" ;


      return LocalError(xerr*xerr, 0, yerr*yerr);
    }
  else if ( edgex || edgey )
    {
      // for edge pixels assign errors according to observed residual RMS 
      if      ( edgex && !edgey )
	{
	  xerr = 23.0 * micronsToCm;
	  yerr = 39.0 * micronsToCm;
	}
      else if ( !edgex && edgey )
	{
	  xerr = 24.0 * micronsToCm;
	  yerr = 96.0 * micronsToCm;
	}
      else if ( edgex && edgey )
	{
	  xerr = 31.0 * micronsToCm;
	  yerr = 90.0 * micronsToCm;
	}
      else
	{
	  throw cms::Exception(" PixelCPETemplateReco::localError: Something wrong with pixel edge flag !!!");
	}
    }
  else 
    {
      // &&& need a class const
      const float micronsToCm = 1.0e-4;

      xerr = templSigmaX_ * micronsToCm;
      yerr = templSigmaY_ * micronsToCm;
      
      // &&& should also check ierr (saved as class variable) and return
      // &&& nonsense (another class static) if the template fit failed.
    }       
  
  if (theVerboseLevel > 9) 
    {
      LogDebug("PixelCPETemplateReco") <<
	" Sizex = " << cluster.sizeX() << " Sizey = " << cluster.sizeY() << 	" Edgex = " << edgex           << " Edgey = " << edgey << 
	" ErrX  = " << xerr            << " ErrY  = " << yerr;
    }
  
  return LocalError(xerr*xerr, 0, yerr*yerr);
}

