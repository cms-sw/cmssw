// Include our own header first
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEClusterRepair.h"

// Geometry services
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

//#define DEBUG

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"


// Commented for now (3/10/17) until we figure out how to resuscitate 2D template splitter
/// #include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateSplit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include "boost/multi_array.hpp"

#include <iostream>

using namespace SiPixelTemplateReco;
//using namespace SiPixelTemplateSplit;
using namespace std;

namespace {
   constexpr float micronsToCm = 1.0e-4;
   constexpr int cluster_matrix_size_x = 13;
   constexpr int cluster_matrix_size_y = 21;
}

//-----------------------------------------------------------------------------
//  Constructor.
//
//-----------------------------------------------------------------------------
PixelCPEClusterRepair::PixelCPEClusterRepair(edm::ParameterSet const & conf,
                                           const MagneticField * mag,
                                           const TrackerGeometry& geom,
                                           const TrackerTopology& ttopo,
                                           const SiPixelLorentzAngle * lorentzAngle,
					   const SiPixelTemplateDBObject * templateDBobject,
					   const SiPixel2DTemplateDBObject * templateDBobject2D )
: PixelCPEBase(conf, mag, geom, ttopo, lorentzAngle, nullptr, templateDBobject, nullptr,1)
{
   //cout << endl;
   //cout << "Constructing PixelCPEClusterRepair::PixelCPEClusterRepair(...)................................................." << endl;
   //cout << endl;
   
   // Configurable parameters
   //DoCosmics_ = conf.getParameter<bool>("DoCosmics"); // Not used in templates
   //LoadTemplatesFromDB_ = conf.getParameter<bool>("LoadTemplatesFromDB"); // Moved to Base
   
   //cout << " PixelCPEClusterRepair : (int)LoadTemplatesFromDB_ = " << (int)LoadTemplatesFromDB_ << endl;
   //cout << "field_magnitude = " << field_magnitude << endl;
   
   // configuration parameter to decide between DB or text file template access
   
   if ( LoadTemplatesFromDB_ )
   {
      //cout << "PixelCPEClusterRepair: Loading templates from database (DB) --------- " << endl;
      
      // Initialize template store to the selected ID [Morris, 6/25/08]
      if ( !SiPixelTemplate::pushfile( *templateDBobject_, thePixelTemp_) )
         throw cms::Exception("PixelCPEClusterRepair")
         << "\nERROR: Templates not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version "
         << (*templateDBobject_).version() << "\n\n";

      // Initialize template store to the selected ID [Morris, 6/25/08]
      if ( !SiPixelTemplate2D::pushfile( *templateDBobject2D , thePixelTemp2D_) )
         throw cms::Exception("PixelCPEClusterRepair")
         << "\nERROR: Templates not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version "
         << (*templateDBobject_).version() << "\n\n";
   }
   else
   {
      //cout << "PixelCPEClusterRepair : Loading templates for barrel and forward from ASCII files ----------" << endl;
      //--- (Archaic) Get configurable template IDs.  This code executes only if we loading pixels from ASCII
      //    files, and then they are mandatory.
      barrelTemplateID_  = conf.getParameter<int>( "barrelTemplateID" );
      forwardTemplateID_ = conf.getParameter<int>( "forwardTemplateID" );
      templateDir_       = conf.getParameter<int>( "directoryWithTemplates" );
     
      if ( !SiPixelTemplate::pushfile( barrelTemplateID_  , thePixelTemp_ , templateDir_ ) )
         throw cms::Exception("PixelCPEClusterRepair")
	 << "\nERROR: Template ID " << barrelTemplateID_ << " not loaded correctly from text file. Reconstruction will fail.\n\n";
      
      if ( !SiPixelTemplate::pushfile( forwardTemplateID_ , thePixelTemp_ , templateDir_ ) )
         throw cms::Exception("PixelCPEClusterRepair")
	 << "\nERROR: Template ID " << forwardTemplateID_ << " not loaded correctly from text file. Reconstruction will fail.\n\n";
   }
   
   speed_ = conf.getParameter<int>( "speed");
   LogDebug("PixelCPEClusterRepair::PixelCPEClusterRepair:") <<
   "Template speed = " << speed_ << "\n";
   
   UseClusterSplitter_ = conf.getParameter<bool>("UseClusterSplitter");   


   //--- Configure 3D reco.
   if ( conf.exists("MinProbY") )
     minProbY_ = conf.getParameter<double>("MinProbY");
   else
     minProbY_ = 0.001;           // probabilityY < 0.001

   if ( conf.exists("MaxSizeMismatchInY") )
     maxSizeMismatchInY_ = conf.getParameter<int>("MaxSizeMismatchInY");
   else
     maxSizeMismatchInY_ = 1;     // ( templ.clsleny() - nypix > 1)

}



//-----------------------------------------------------------------------------
//  Clean up.
//-----------------------------------------------------------------------------
PixelCPEClusterRepair::~PixelCPEClusterRepair()
{
   for(auto x : thePixelTemp_) x.destroy();
   // Note: this is not needed for Template 2D
}

PixelCPEBase::ClusterParam* PixelCPEClusterRepair::createClusterParam(const SiPixelCluster & cl) const
{
   return new ClusterParamTemplate(cl);
}



//------------------------------------------------------------------
//  Public methods mandated by the base class.
//------------------------------------------------------------------

//------------------------------------------------------------------
//  The main call to the template code.
//------------------------------------------------------------------
LocalPoint
PixelCPEClusterRepair::localPosition(DetParam const & theDetParam, ClusterParam & theClusterParamBase) const
{
   
   ClusterParamTemplate & theClusterParam = static_cast<ClusterParamTemplate &>(theClusterParamBase);
   
   if(!GeomDetEnumerators::isTrackerPixel(theDetParam.thePart))
      throw cms::Exception("PixelCPEClusterRepair::localPosition :")
      << "A non-pixel detector type in here?";
   
   int ID = -9999;
   if ( LoadTemplatesFromDB_ ) {
      int ID0 = templateDBobject_->getTemplateID(theDetParam.theDet->geographicalId()); // just to comapre
      ID = theDetParam.detTemplateId;
      if(ID0!=ID) cout<<" different id"<< ID<<" "<<ID0<<endl;
   } else { // from asci file
      if ( ! GeomDetEnumerators::isEndcap(theDetParam.thePart) )
	ID = barrelTemplateID_  ; // barrel
      else
	ID = forwardTemplateID_ ; // forward
   }
   //cout << "PixelCPEClusterRepair : ID = " << ID << endl;

   // &&& PM, note for later: PixelCPEBase calculates minInX,Y, and maxInX,Y
   //     Why can't we simply use that and save time with row_offset, col_offset
   //     and mrow = maxInX-minInX, mcol = maxInY-minInY ... Except that we
   //     also need to take into account cluster_matrix_size_x,y.

   
   //--- Preparing to retrieve ADC counts from the SiPixeltheClusterParam.theCluster
   //    The pixels from minPixelRow() will go into clust_array_2d[0][*],
   //    and the pixels from minPixelCol() will go into clust_array_2d[*][0].
   int row_offset = theClusterParam.theCluster->minPixelRow();
   int col_offset = theClusterParam.theCluster->minPixelCol();
   
   //--- Store the coordinates of the center of the (0,0) pixel of the array that
   //    gets passed to PixelTempReco2D.  Will add these values to the output of TemplReco2D
   float tmp_x = float(row_offset) + 0.5f;
   float tmp_y = float(col_offset) + 0.5f;

   
   //--- Store these offsets (to be added later) in a LocalPoint after tranforming
   //    them from measurement units (pixel units) to local coordinates (cm)
   LocalPoint lp;
   if ( theClusterParam.with_track_angle )
      //--- Update call with trk angles needed for bow/kink corrections  // Gavril
      lp = theDetParam.theTopol->localPosition( MeasurementPoint(tmp_x, tmp_y), theClusterParam.loc_trk_pred );
   else
   {
      edm::LogError("PixelCPEClusterRepair")
      << "@SUB = PixelCPEClusterRepair::localPosition"
      << "Should never be here. PixelCPEClusterRepair should always be called with track angles. This is a bad error !!! ";
      lp = theDetParam.theTopol->localPosition( MeasurementPoint(tmp_x, tmp_y) );
   }
   
   //--- Compute the size of the matrix which will be passed to TemplateReco.
   //    We'll later make  clustMatrix[ mrow ][ mcol ]
   int mrow=0, mcol=0;
   for (int i=0 ; i!=theClusterParam.theCluster->size(); ++i )
   {
      auto pix = theClusterParam.theCluster->pixel(i);
      int irow = int(pix.x);
      int icol = int(pix.y);
      mrow = std::max(mrow,irow);
      mcol = std::max(mcol,icol);
   }
   mrow -= row_offset; mrow+=1; mrow = std::min(mrow,cluster_matrix_size_x);
   mcol -= col_offset; mcol+=1; mcol = std::min(mcol,cluster_matrix_size_y);
   assert(mrow>0); assert(mcol>0);


   //--- Make and fill the bool arrays flagging double pixels
   bool xdouble[mrow], ydouble[mcol];
   // x directions (shorter), rows
   for (int irow = 0; irow < mrow; ++irow)
      xdouble[irow] = theDetParam.theRecTopol->isItBigPixelInX( irow+row_offset );
   //
   // y directions (longer), columns
   for (int icol = 0; icol < mcol; ++icol)
      ydouble[icol] = theDetParam.theRecTopol->isItBigPixelInY( icol+col_offset );

   //--- C-style matrix.  We'll need it in either case.
   float clustMatrix[mrow][mcol];
   float clustMatrix2[mrow][mcol];

   //--- Prepare struct that passes pointers to TemplateReco.  It doesn't own anything.
   SiPixelTemplateReco::ClusMatrix   clusterPayload  { &clustMatrix[0][0], xdouble, ydouble, mrow,mcol};
   SiPixelTemplateReco2D::ClusMatrix clusterPayload2d{ &clustMatrix2[0][0], xdouble, ydouble, mrow,mcol};


   //--- Copy clust's pixels (calibrated in electrons) into clustMatrix;
   memset( clustMatrix, 0, sizeof(float)*mrow*mcol );   // Wipe it clean.
   for (int i=0 ; i!=theClusterParam.theCluster->size(); ++i )
   {
      auto pix = theClusterParam.theCluster->pixel(i);
      int irow = int(pix.x) - row_offset;
      int icol = int(pix.y) - col_offset;
      // &&& Do we ever get pixels that are out of bounds ???  Need to check.
      if ( (irow<mrow) & (icol<mcol) ) clustMatrix[irow][icol] =  float(pix.adc);
   }
   // fillClustMatrix( float * clustMatrix );

   //--- Save a copy of clustMatrix into clustMatrix2
   memcpy( clustMatrix2, clustMatrix, sizeof(float)*mrow*mcol);

   //--- Set both return statuses, since we may call only one.
   theClusterParam.ierr  = 0;
   theClusterParam.ierr2 = 0;



   //--- Are we on edge?
   if ( theClusterParam.isOnEdge_ ) {
     //--- Call the Template Reco 2d with cluster repair.0
     callTempReco3D( theDetParam, theClusterParam, clusterPayload2d, ID, lp );
   }
   else {
     theClusterParam.recommended3D_ = false;
     //--- Call the vanilla Template Reco
     callTempReco2D( theDetParam, theClusterParam, clusterPayload, ID, lp );

     //--- Did we find a cluster which has bad probability and not enough charge?
     if ( theClusterParam.recommended3D_ ) {
       //--- Yes. So run Template Reco 2d with cluster repair.
       
       // //--- Once again (!) copy clust's pixels (calibrated in electrons) into 
       // //    clustMatrix.  We need to do that because the vanilla template reco
       // //    will modify the ADC counts in situ (during decapitation)
       // memset( clustMatrix, 0, sizeof(float)*mrow*mcol );   // Wipe it clean.
       // for (int i=0 ; i!=theClusterParam.theCluster->size(); ++i )
       // 	 {
       // 	   auto pix = theClusterParam.theCluster->pixel(i);
       // 	   int irow = int(pix.x) - row_offset;
       // 	   int icol = int(pix.y) - col_offset;
       // 	   // &&& Do we ever get pixels that are out of bounds ???  Need to check.
       // 	   if ( (irow<mrow) & (icol<mcol) ) clustMatrix[irow][icol] =  float(pix.adc);
       // 	 }
       // // fillClustMatrix( float * clustMatrix );
       

       //--- Call the Template Reco 2d with cluster repair
       callTempReco3D( theDetParam, theClusterParam, clusterPayload2d, ID, lp );
     }

   }


   
   return LocalPoint( theClusterParam.templXrec_, theClusterParam.templYrec_ );
}


//------------------------------------------------------------------
//  Helper function to aggregate call & handling of Template Reco
//------------------------------------------------------------------
void 
PixelCPEClusterRepair::callTempReco2D( DetParam const & theDetParam, 
				       ClusterParamTemplate & theClusterParam,
				       SiPixelTemplateReco::ClusMatrix & clusterPayload,
				       int ID, LocalPoint & lp ) const
{
   SiPixelTemplate templ(thePixelTemp_);
   
   // Output:
   float nonsense = -99999.9f; // nonsense init value
   theClusterParam.templXrec_ = theClusterParam.templYrec_ = theClusterParam.templSigmaX_ = theClusterParam.templSigmaY_ = nonsense;
   // If the template recontruction fails, we want to return 1.0 for now
   theClusterParam.templProbY_ = theClusterParam.templProbX_ = theClusterParam.templProbQ_ = 1.0f;
   theClusterParam.templQbin_ = 0;
   // We have a boolean denoting whether the reco failed or not
   theClusterParam.hasFilledProb_ = false;
   
   // ******************************************************************
   //--- Call normal TemplateReco
   //
   float locBz = theDetParam.bz;
   float locBx = theDetParam.bx;
   //
   const bool deadpix = false;
   std::vector<std::pair<int, int> > zeropix;
   int nypix =0, nxpix = 0;
   //
   theClusterParam.ierr =
   PixelTempReco2D( ID, theClusterParam.cotalpha, theClusterParam.cotbeta,
                   locBz, locBx,
                   clusterPayload,
                   templ,
                   theClusterParam.templYrec_, theClusterParam.templSigmaY_, theClusterParam.probabilityY_,
                   theClusterParam.templXrec_, theClusterParam.templSigmaX_, theClusterParam.probabilityX_,
                   theClusterParam.qBin_,
		   speed_, deadpix, zeropix,
		   theClusterParam.probabilityQ_, nypix, nxpix
                   );
   // ******************************************************************
   
   //--- Check exit status
   if unlikely( theClusterParam.ierr != 0 )
   {
      LogDebug("PixelCPEClusterRepair::localPosition") <<
      "reconstruction failed with error " << theClusterParam.ierr << "\n";
      
      // Gavril: what do we do in this case ? For now, just return the cluster center of gravity in microns
      // In the x case, apply a rough Lorentz drift average correction
      // To do: call PixelCPEGeneric whenever PixelTempReco2D fails
      float lorentz_drift = -999.9;
      if ( ! GeomDetEnumerators::isEndcap(theDetParam.thePart) )
         lorentz_drift = 60.0f; // in microns
      else
         lorentz_drift = 10.0f; // in microns
      // ggiurgiu@jhu.edu, 21/09/2010 : trk angles needed to correct for bows/kinks
      if ( theClusterParam.with_track_angle )
      {
         theClusterParam.templXrec_ = theDetParam.theTopol->localX( theClusterParam.theCluster->x(), theClusterParam.loc_trk_pred ) - lorentz_drift * micronsToCm; // rough Lorentz drift correction
         theClusterParam.templYrec_ = theDetParam.theTopol->localY( theClusterParam.theCluster->y(), theClusterParam.loc_trk_pred );
      }
      else
      {
         edm::LogError("PixelCPEClusterRepair")
         << "@SUB = PixelCPEClusterRepair::localPosition"
         << "Should never be here. PixelCPEClusterRepair should always be called with track angles. This is a bad error !!! ";
         
         theClusterParam.templXrec_ = theDetParam.theTopol->localX( theClusterParam.theCluster->x() ) - lorentz_drift * micronsToCm; // rough Lorentz drift correction
         theClusterParam.templYrec_ = theDetParam.theTopol->localY( theClusterParam.theCluster->y() );
      }
   }   
   else 
   {
      //--- Template Reco succeeded.  The probabilities are filled.
      theClusterParam.hasFilledProb_ = true;

      //--- templ.clsleny() is the expected length of the cluster along y axis.
      if ( (theClusterParam.probabilityY_ < minProbY_ ) && (templ.clsleny() - nypix > 1) ) {
	theClusterParam.recommended3D_ = true;
      }
      
      //--- Go from microns to centimeters
      theClusterParam.templXrec_ *= micronsToCm;
      theClusterParam.templYrec_ *= micronsToCm;
      
      //--- Go back to the module coordinate system
      theClusterParam.templXrec_ += lp.x();
      theClusterParam.templYrec_ += lp.y();
            
   }
   return;
}




//------------------------------------------------------------------
//  Helper function to aggregate call & handling of Template 2D fit
//------------------------------------------------------------------
void 
PixelCPEClusterRepair::callTempReco3D( DetParam const & theDetParam, 
				       ClusterParamTemplate & theClusterParam,
				       SiPixelTemplateReco2D::ClusMatrix & clusterPayload,
				       int ID, LocalPoint & lp ) const
{
   SiPixelTemplate2D templ2d(thePixelTemp2D_);
   
   // Output:
   float nonsense = -99999.9f; // nonsense init value
   theClusterParam.templXrec_ = theClusterParam.templYrec_ = theClusterParam.templSigmaX_ = theClusterParam.templSigmaY_ = nonsense;
   // If the template recontruction fails, we want to return 1.0 for now
   theClusterParam.templProbY_ = theClusterParam.templProbX_ = theClusterParam.templProbQ_ = 1.0f;
   theClusterParam.templQbin_ = 0;
   // We have a boolean denoting whether the reco failed or not
   theClusterParam.hasFilledProb_ = false;
   
   // ******************************************************************
   //--- Call 2D TemplateReco
   //
   float locBz = theDetParam.bz;
   float locBx = theDetParam.bx;

   //--- Input:
   //   edgeflagy - (input) flag to indicate the present of edges in y: 
   //           0=none (or interior gap),1=edge at small y, 2=edge at large y, 3=edge at either end
   //
   //   edgeflagx - (input) flag to indicate the present of edges in x: 
   //           0=none, 1=edge at small x, 2=edge at large x
   //
   //   These two variables are calculated in setTheClu() and stored in edgeTypeX_ and edgeTypeY_
   //
   //--- Output:
   //   deltay - (output) template y-length - cluster length [when > 0, possibly missing end]
   //   npixels - ???     &&& Ask Morris

   float deltay = 0;
   int npixels = 0;

   theClusterParam.ierr2 =
   PixelTempReco3D( ID, theClusterParam.cotalpha, theClusterParam.cotbeta,
                   locBz, locBx,
		   theClusterParam.edgeTypeY_ , theClusterParam.edgeTypeX_ ,
                   clusterPayload,
                   templ2d,
                   theClusterParam.templYrec_, theClusterParam.templSigmaY_, 
                   theClusterParam.templXrec_, theClusterParam.templSigmaX_, 
		   theClusterParam.templProbXY_,
         	   theClusterParam.probabilityQ_,
		   theClusterParam.qBin_,
		   deltay, npixels
                   );
   // ******************************************************************
   
   //--- Check exit status
   if unlikely( theClusterParam.ierr2 != 0 )
   {
      LogDebug("PixelCPEClusterRepair::localPosition") <<
      "3D reconstruction failed with error " << theClusterParam.ierr2 << "\n";
      
      // Gavril: what do we do in this case ? For now, just return the cluster center of gravity in microns
      // In the x case, apply a rough Lorentz drift average correction
      // To do: call PixelCPEGeneric whenever PixelTempReco2D fails
      float lorentz_drift = -999.9;
      if ( ! GeomDetEnumerators::isEndcap(theDetParam.thePart) )
         lorentz_drift = 60.0f; // in microns
      else
         lorentz_drift = 10.0f; // in microns
      // ggiurgiu@jhu.edu, 21/09/2010 : trk angles needed to correct for bows/kinks
      if ( theClusterParam.with_track_angle )
      {
         theClusterParam.templXrec_ = theDetParam.theTopol->localX( theClusterParam.theCluster->x(), theClusterParam.loc_trk_pred ) - lorentz_drift * micronsToCm; // rough Lorentz drift correction
         theClusterParam.templYrec_ = theDetParam.theTopol->localY( theClusterParam.theCluster->y(), theClusterParam.loc_trk_pred );
      }
      else
      {
         edm::LogError("PixelCPEClusterRepair")
         << "@SUB = PixelCPEClusterRepair::localPosition"
         << "Should never be here. PixelCPEClusterRepair should always be called with track angles. This is a bad error !!! ";
         
         theClusterParam.templXrec_ = theDetParam.theTopol->localX( theClusterParam.theCluster->x() ) - lorentz_drift * micronsToCm; // rough Lorentz drift correction
         theClusterParam.templYrec_ = theDetParam.theTopol->localY( theClusterParam.theCluster->y() );
      }
   }   
   else 
   {
      //--- Template Reco succeeded.
      theClusterParam.hasFilledProb_ = true;

      //--- Go from microns to centimeters
      theClusterParam.templXrec_ *= micronsToCm;
      theClusterParam.templYrec_ *= micronsToCm;
      
      //--- Go back to the module coordinate system
      theClusterParam.templXrec_ += lp.x();
      theClusterParam.templYrec_ += lp.y();
   }
   return;
}




//------------------------------------------------------------------
//  localError() relies on localPosition() being called FIRST!!!
//------------------------------------------------------------------
LocalError
PixelCPEClusterRepair::localError(DetParam const & theDetParam,  ClusterParam & theClusterParamBase) const
{
   
   ClusterParamTemplate & theClusterParam = static_cast<ClusterParamTemplate &>(theClusterParamBase);
   
   //cout << endl;
   //cout << "Set PixelCPETemplate errors .............................................." << endl;
   
   //cout << "CPETemplate : " << endl;
   
   //--- Default is the maximum error used for edge clusters.
   //--- (never used, in fact: let comment it out, shut up the complains of the static analyzer, and save a few CPU cycles)
   //   const float sig12 = 1./sqrt(12.0);
   //   float xerr = theDetParam.thePitchX *sig12;
   //   float yerr = theDetParam.thePitchY *sig12;
   float xerr = 0.0f, yerr = 0.0f;
   
   // Check if the errors were already set at the clusters splitting level
   if ( theClusterParam.theCluster->getSplitClusterErrorX() > 0.0f && theClusterParam.theCluster->getSplitClusterErrorX() < 7777.7f &&
       theClusterParam.theCluster->getSplitClusterErrorY() > 0.0f && theClusterParam.theCluster->getSplitClusterErrorY() < 7777.7f )
   {
      xerr = theClusterParam.theCluster->getSplitClusterErrorX() * micronsToCm;
      yerr = theClusterParam.theCluster->getSplitClusterErrorY() * micronsToCm;
      
      //cout << "Errors set at cluster splitting level : " << endl;
      //cout << "xerr = " << xerr << endl;
      //cout << "yerr = " << yerr << endl;
   }
   else
   {
      // &&& Duplicate from above! Argh!
      int maxPixelCol = theClusterParam.theCluster->maxPixelCol();
      int maxPixelRow = theClusterParam.theCluster->maxPixelRow();
      int minPixelCol = theClusterParam.theCluster->minPixelCol();
      int minPixelRow = theClusterParam.theCluster->minPixelRow();
      
      //--- Are we near either of the edges?
      bool edgex = ( theDetParam.theRecTopol->isItEdgePixelInX( minPixelRow ) || theDetParam.theRecTopol->isItEdgePixelInX( maxPixelRow ) );
      bool edgey = ( theDetParam.theRecTopol->isItEdgePixelInY( minPixelCol ) || theDetParam.theRecTopol->isItEdgePixelInY( maxPixelCol ) );
      

      //--- Check status of both template calls.
      if ( (theClusterParam.ierr !=0) || (theClusterParam.ierr2 !=0) )   
      {
         // If reconstruction fails the hit position is calculated from cluster center of gravity
         // corrected in x by average Lorentz drift. Assign huge errors.
         //xerr = 10.0 * (float)theClusterParam.theCluster->sizeX() * xerr;
         //yerr = 10.0 * (float)theClusterParam.theCluster->sizeX() * yerr;
         
         if(!GeomDetEnumerators::isTrackerPixel(theDetParam.thePart))
            throw cms::Exception("PixelCPEClusterRepair::localPosition :")
            << "A non-pixel detector type in here?";
         
         // Assign better errors based on the residuals for failed template cases
         if ( GeomDetEnumerators::isBarrel(theDetParam.thePart) )
         {
            xerr = 55.0f * micronsToCm;
            yerr = 36.0f * micronsToCm;
         }
         else
         {
            xerr = 42.0f * micronsToCm;
            yerr = 39.0f * micronsToCm;
         }
         
         //cout << "xerr = " << xerr << endl;
         //cout << "yerr = " << yerr << endl;
         
         //return LocalError(xerr*xerr, 0, yerr*yerr);
      }
      else if ( theClusterParam.edgeTypeX_ || theClusterParam.edgeTypeY_ )
      {
         // for edge pixels assign errors according to observed residual RMS
         if      ( theClusterParam.edgeTypeX_ && !theClusterParam.edgeTypeY_ )
         {
            xerr = 23.0f * micronsToCm;
            yerr = 39.0f * micronsToCm;
         }
         else if ( !theClusterParam.edgeTypeX_ && theClusterParam.edgeTypeY_ )
         {
            xerr = 24.0f * micronsToCm;
            yerr = 96.0f * micronsToCm;
         }
         else if ( theClusterParam.edgeTypeX_ && theClusterParam.edgeTypeY_ )
         {
            xerr = 31.0f * micronsToCm;
            yerr = 90.0f * micronsToCm;
         }
      }
      else 
      {
         xerr = theClusterParam.templSigmaX_ * micronsToCm;
         yerr = theClusterParam.templSigmaY_ * micronsToCm;
                  
         // &&& should also check ierr (saved as class variable) and return
         // &&& nonsense (another class static) if the template fit failed.
      }       
      
      //cout << "xerr = " << xerr << endl;
      //cout << "yerr = " << yerr << endl;

      if (theVerboseLevel > 9) 
      {
         LogDebug("PixelCPEClusterRepair") <<
         " Sizex = " << theClusterParam.theCluster->sizeX() << " Sizey = " << theClusterParam.theCluster->sizeY() << " Edgex = " << edgex << " Edgey = " << edgey << 
         " ErrX  = " << xerr            << " ErrY  = " << yerr;
      }
      
   } // else
   
   if ( !(xerr > 0.0f) )
      throw cms::Exception("PixelCPEClusterRepair::localError") 
      << "\nERROR: Negative pixel error xerr = " << xerr << "\n\n";
   
   if ( !(yerr > 0.0f) )
      throw cms::Exception("PixelCPEClusterRepair::localError") 
      << "\nERROR: Negative pixel error yerr = " << yerr << "\n\n";
   
   //cout << "Final errors set to: " << endl;
   //cout << "xerr = " << xerr << endl;
   //cout << "yerr = " << yerr << endl;
   //cout << "Out of PixelCPETemplateREco..........................................................................." << endl;
   //cout << endl;
   
   return LocalError(xerr*xerr, 0, yerr*yerr);
}

