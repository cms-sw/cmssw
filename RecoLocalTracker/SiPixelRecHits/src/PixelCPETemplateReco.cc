// Include our own header first
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateReco.h"

// Geometry services
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

//#define DEBUG

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

// The template header files
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateSplit.h"

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

//-----------------------------------------------------------------------------
//  Constructor.  All detUnit-dependent quantities will be initialized later,
//  in setTheDet().  Here we only load the templates into the template store templ_ .
//-----------------------------------------------------------------------------
PixelCPETemplateReco::PixelCPETemplateReco(edm::ParameterSet const & conf, 
					   const MagneticField *mag, const SiPixelLorentzAngle * lorentzAngle) 
  : PixelCPEBase(conf, mag, lorentzAngle)
{
  // &&& initialize the templates, etc.
  
  // Initialize template store, CMSSW simulation as thePixelTemp[0]
  //templ_.pushfile(201);

  // Initialize template store, Pixelav 125V simulation as
  // thePixelTemp[1]
  templ_.pushfile(4);

  // Initialize template store, CMSSW simulation w/ reduced difusion
  // as thePixelTemp[2]
  //templ_.pushfile(401);
  
  //cout << "About to read speed..." << endl;
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
  return theTopol->measurementPosition(lp);
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

  setTheDet( det );

  //int ierr;   //!< return status
  int ID = 4; //!< picks the third entry from the template store

  bool fpix;  //!< barrel(false) or forward(true)
  if ( thePart == GeomDetEnumerators::PixelBarrel )   
    fpix = false;    // no, it's not forward -- it's barrel
  else                                              
    fpix = true;     // yes, it's forward
  
  // Make cot(alpha) and cot(beta)... cot(x) = 1.0/tan(x);
  // No longer need these. cotalpha_and cotbeta_ are calculated in PixelCPEBase  
  //float cotalpha = 1.0/tan(alpha_);  
  //float cotbeta  = 1.0/tan(beta_);   
  
  // Make from cluster (a SiPixelCluster) a boost multi_array_2d called 
  // clust_array_2d.
  boost::multi_array<float, 2> clust_array_2d(boost::extents[7][21]);
  
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
  LocalPoint lp = theTopol->localPosition( MeasurementPoint(tmp_x, tmp_y) );
    
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
    
    if ( cluster.sizeX()>7 || cluster.sizeY()>21 )
    //if ( cluster.sizeX()>0 || cluster.sizeY()>0 )
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
    } // if ( cluster.sizeX()>7 || cluster.sizeY()>21 )
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
      
      // Gavril : what do we do here if the row/column is larger than 7/21 ?
      // Ignore them for the moment...
      if ( irow<7 && icol<21 )
	// 02/13/2008 ggiurgiu@fnal.gov typecast pixIter->adc to float
	clust_array_2d[irow][icol] = (float)pixIter->adc;
      //else
      //cout << " ----- Cluster is too large" << endl;
    }
  
  // Make and fill the bool arrays flagging double pixels
  // &&& Need to define constants for 7 and 21 somewhere!
  std::vector<bool> ydouble(21), xdouble(7);
  // x directions (shorter), rows
  for (int irow = 0; irow < 7; ++irow)
    {
      xdouble[irow] = RectangularPixelTopology::isItBigPixelInX( irow+row_offset );
    }
      
  // y directions (longer), columns
  for (int icol = 0; icol < 21; ++icol) 
    {
      ydouble[icol] = RectangularPixelTopology::isItBigPixelInY( icol+col_offset );
    }

  // Output:
  float nonsense = -99999.9; // nonsense init value
  templXrec_ = templYrec_ = templSigmaX_ = templSigmaY_ = nonsense;
  templProbY_ = templProbX_ = nonsense;
  
  float templYrec1_ = nonsense;
  float templXrec1_ = nonsense;
  float templYrec2_ = nonsense;
  float templXrec2_ = nonsense;

  // ******************************************************************
  // Do it! Use cotalpha_ and cotbeta_ calculated in PixelCPEBase

  ierr =
    PixelTempReco2D( ID, fpix, cotalpha_, cotbeta_,
		     clust_array_2d, ydouble, xdouble,
		     templ_,
		     templYrec_, templSigmaY_, templProbY_,
		     templXrec_, templSigmaX_, templProbX_, 
		     templQbin_, 
		     speed_ );
  // ******************************************************************

  // Check exit status
  if ( ierr != 0 ) 
    {
      //cout << "Error from PixelTempReco2D" << endl;

      LogDebug("PixelCPETemplateReco::localPosition") <<
	"reconstruction failed with error " << ierr << "\n";

      // Gavril: what do we do in this case ? For now, just return the cluster center of gravity in microns
      // In the x case, apply a rough Lorentz drift correction
      double lorentz_drift = 60.0; // in microns
      templXrec_ = theTopol->localX( cluster.x() ) - lorentz_drift * micronsToCm; // very rough Lorentz drift correction
      templYrec_ = theTopol->localY( cluster.y() );

      // go back to the module coordinate system 
      templXrec_ += lp.x();
      templYrec_ += lp.y();    
    }
  else if ( UseClusterSplitter_ && templQbin_ == 0 )
    {
      //cout << "PixelTempReco2D returned without error and templQbin_ == 0" << endl;

      ierr = 
	PixelTempSplit( ID, fpix, cotalpha_, cotbeta_, 
			clust_array_2d, ydouble, xdouble, 
			templ_, 
			templYrec1_, templYrec2_, templSigmaY_, templProbY_,
			templXrec1_, templXrec2_, templSigmaX_, templProbX_, 
			templQbin_ );

      if ( ierr != 0 )
	{
	  //cout << "Error from PixelTempSplit" << endl;
	  
	  LogDebug("PixelCPETemplateReco::localPosition") <<
	    "reconstruction failed with error " << ierr << "\n";
	  
	  double lorentz_drift = 60.0; // in microns
	  templXrec_ = theTopol->localX( cluster.x() ) - lorentz_drift * micronsToCm; // very rough Lorentz drift correction
	  templYrec_ = theTopol->localY( cluster.y() );
	
	  // go back to the module coordinate system 
	  templXrec_ += lp.x();
	  templYrec_ += lp.y();    
	}
      else
	{
	  //cout << "PixelTempSplit returned without error " << endl;

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
      
      	  //cout << "templXrec1_ = " << templXrec1_ << endl;
	  //cout << "templYrec1_ = " << templYrec1_ << endl;
	  //cout << "templXrec2_ = " << templXrec2_ << endl;
	  //cout << "templYrec2_ = " << templYrec2_ << endl;
	  //cout << "trk_lp_x    = " << trk_lp_x    << endl;
	  //cout << "trk_lp_y    = " << trk_lp_y    << endl;
	        
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
	  
	  //cout << "distance11 = " << distance11 << endl;
	  //cout << "distance12 = " << distance12 << endl;
	  //cout << "distance21 = " << distance21 << endl;
	  //cout << "distance22 = " << distance22 << endl;
	  
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
	  
	  //cout << "index_dist = " << index_dist << endl;
	  
	  templXrec_ = min_templXrec_;
	  templYrec_ = min_templYrec_;
	}
    } // else if ( UseClusterSplitter_ && templQbin_ == 0 )
  else 
    {
      //cout << "PixelTempReco2D returned without error and templQbin_ != 0" << endl;

      // go from micrometer to centimeter      
      templXrec_ *= micronsToCm;
      templYrec_ *= micronsToCm;
      
      // go back to the module coordinate system 
      templXrec_ += lp.x();
      templYrec_ += lp.y();
    }
  
  //cout << "templXrec_ = " << templXrec_ << endl;
  //cout << "templYrec_ = " << templYrec_ << endl;
  
  // Save probabilities and qBin in the quantities given to us by the base class
  // (for which there are also inline getters).  &&& templProbX_ etc. should be retired...
  probabilityX_ = templProbX_;
  probabilityY_ = templProbY_;
  qBin_         = templQbin_;

  
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
  setTheDet( det );
  
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
  
  // if ( edgex || edgey || ierr !=0 ) 
  if ( ierr !=0 ) 
    {
      // If reconstruction fails the hit position is calculated from cluster center of gravity 
      // corrected in x by average Lorentz drift. The template reconstruction fails when the cluster 
      // size is larger than 7 in y or larger than 21 in y. These kind of clusters must be delta rays, 
      // so they are not reliable. Assign huge errors.
      xerr = 10.0 * xerr;
      yerr = 10.0 * yerr;
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

