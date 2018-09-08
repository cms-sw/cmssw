// Move geomCorrection to the concrete class. d.k. 06/06.
// Change drift direction. d.k. 06/06
// G. Giurgiu (ggiurgiu@pha.jhu.edu), 12/01/06, implemented the function:
// computeAnglesFromDetPosition(const SiPixelCluster & cl,
// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008
// Change to use Generic error & Template calibration from DB - D.Fehling 11/08


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/ProxyPixelTopology.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"

#define CORRECT_FOR_BIG_PIXELS

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"


#include <iostream>

using namespace std;

//-----------------------------------------------------------------------------
//  A constructor run for generic and templates
//
//-----------------------------------------------------------------------------
PixelCPEBase::PixelCPEBase(edm::ParameterSet const & conf,
                           const MagneticField *mag,
                           const TrackerGeometry& geom,
                           const TrackerTopology& ttopo,
                           const SiPixelLorentzAngle * lorentzAngle,
                           const SiPixelGenErrorDBObject * genErrorDBObject,
                           const SiPixelTemplateDBObject * templateDBobject,
                           const SiPixelLorentzAngle * lorentzAngleWidth,
                           int flag)
//  : useLAAlignmentOffsets_(false), useLAOffsetFromConfig_(false),
: useLAOffsetFromConfig_(false),
useLAWidthFromConfig_(false), useLAWidthFromDB_(false), theFlag_(flag),
magfield_(mag), geom_(geom), ttopo_(ttopo)
{
   
#ifdef EDM_ML_DEBUG
   nRecHitsTotal_=0;
   nRecHitsUsedEdge_=0,
#endif
   
   //--- Lorentz angle tangent per Tesla
   lorentzAngle_ = lorentzAngle;
   lorentzAngleWidth_ = lorentzAngleWidth;
   
   //-- GenError Calibration Object (different from SiPixelCPEGenericErrorParm) from DB
   genErrorDBObject_ = genErrorDBObject;
   
   //-- Template Calibration Object from DB
   if(theFlag_!=0) templateDBobject_ = templateDBobject; // flag to check if it is generic or templates
   
   // Configurables
   // For both templates & generic
   
   // Read templates and/or generic errors from DB
   LoadTemplatesFromDB_ = conf.getParameter<bool>("LoadTemplatesFromDB");
   
   //--- Algorithm's verbosity
   theVerboseLevel =
   conf.getUntrackedParameter<int>("VerboseLevel",0);
   
   //-- Switch on/off E.B
   alpha2Order = conf.getParameter<bool>("Alpha2Order");
   
   //--- A flag that could be used to change the behavior of
   //--- clusterProbability() in TSiPixelRecHit (the *transient* one).
   //--- The problem is that the transient hits are made after the CPE runs
   //--- and they don't get the access to the PSet, so we pass it via the
   //--- CPE itself...
   //
   clusterProbComputationFlag_
   = (unsigned int) conf.getParameter<int>("ClusterProbComputationFlag");
   
   // This LA related parameters are only relevant for the Generic algo
   // They still have to be used in Base since the LA computation is in Base
   
   // Use LA-width from DB.
   // If both (this and from config) are false LA-width is calcuated from LA-offset
   useLAWidthFromDB_ = conf.existsAs<bool>("useLAWidthFromDB")?
   conf.getParameter<bool>("useLAWidthFromDB"):false;
   
   // Use Alignment LA-offset in generic
   // (Experimental; leave commented out)
   //useLAAlignmentOffsets_ = conf.existsAs<bool>("useLAAlignmentOffsets")?
   //conf.getParameter<bool>("useLAAlignmentOffsets"):false;
   
   // Used only for testing
   lAOffset_ = conf.existsAs<double>("lAOffset")?  // fixed LA value
   conf.getParameter<double>("lAOffset"):0.0;
   lAWidthBPix_ = conf.existsAs<double>("lAWidthBPix")?   // fixed LA width
   conf.getParameter<double>("lAWidthBPix"):0.0;
   lAWidthFPix_ = conf.existsAs<double>("lAWidthFPix")?   // fixed LA width
   conf.getParameter<double>("lAWidthFPix"):0.0;
   
   // Use LA-offset from config, for testing only
   if (lAOffset_>0.0) 
     useLAOffsetFromConfig_ = true;
   // Use LA-width from config, split into fpix & bpix, for testing only
   if (lAWidthBPix_>0.0 || lAWidthFPix_>0.0) 
     useLAWidthFromConfig_ = true;
   
   
   // For Templates only
   // Compute the Lorentz shifts for this detector element for templates (from Alignment)
   DoLorentz_ = conf.existsAs<bool>("DoLorentz")?conf.getParameter<bool>("DoLorentz"):false;
   
   LogDebug("PixelCPEBase") <<" LA constants - "
			    << lAOffset_ << " " << lAWidthBPix_ << " " <<lAWidthFPix_ << endl; //dk
   
   fillDetParams();
}


//-----------------------------------------------------------------------------
//  Fill all variables which are constant for an event (geometry)
//-----------------------------------------------------------------------------
void PixelCPEBase::fillDetParams()
{
  // &&& PM: I have no idea what this code is doing, and what it is doing here!???
  //
   auto const & dus = geom_.detUnits();
   unsigned m_detectors = dus.size();
   for(unsigned int i=1;i<7;++i) {
      LogDebug("PixelCPEBase:: LookingForFirstStrip") << "Subdetector " << i
      << " GeomDetEnumerator " << GeomDetEnumerators::tkDetEnum[i]
      << " offset " << geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i])
      << " is it strip? " << (geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() ?
                              dus[geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip() : false);
      if(geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() &&
         dus[geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip()) {
         if(geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_detectors) m_detectors = geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
      }
   }
   LogDebug("LookingForFirstStrip") << " Chosen offset: " << m_detectors;
   
   
   m_DetParams.resize(m_detectors);
   LogDebug("PixelCPEBase::fillDetParams():") <<"caching "<<m_detectors<<" pixel detectors"<<endl;
   for (unsigned i=0; i!=m_detectors;++i) {
      auto & p=m_DetParams[i];
      p.theDet = dynamic_cast<const PixelGeomDetUnit*>(dus[i]);
      assert(p.theDet);
      assert(p.theDet->index()==int(i));
      
      p.theOrigin = p.theDet->surface().toLocal(GlobalPoint(0,0,0));
      
      //--- p.theDet->type() returns a GeomDetType, which implements subDetector()
      p.thePart = p.theDet->type().subDetector();
      
      
      //--- The location in of this DetUnit in a cyllindrical coord system (R,Z)
      //--- The call goes via BoundSurface, returned by p.theDet->surface(), but
      //--- position() is implemented in GloballyPositioned<> template
      //--- ( BoundSurface : Surface : GloballyPositioned<float> )
      //p.theDetR = p.theDet->surface().position().perp();  //Not used, AH
      //p.theDetZ = p.theDet->surface().position().z();  //Not used, AH
      //--- Define parameters for chargewidth calculation
      
      //--- bounds() is implemented in BoundSurface itself.
      p.theThickness = p.theDet->surface().bounds().thickness();
      
      // Cache the det id for templates and generic erros
      
      if(theFlag_==0) { // for generic
         if(LoadTemplatesFromDB_ ) // do only if genError requested
            p.detTemplateId = genErrorDBObject_->getGenErrorID(p.theDet->geographicalId().rawId());
      } else {          // for templates
         p.detTemplateId = templateDBobject_->getTemplateID(p.theDet->geographicalId());
      }
      
      
      auto topol = &(p.theDet->specificTopology());
      p.theTopol=topol;
      auto const proxyT = dynamic_cast<const ProxyPixelTopology*>(p.theTopol);
      if (proxyT) p.theRecTopol = dynamic_cast<const RectangularPixelTopology*>(&(proxyT->specificTopology()));
      else p.theRecTopol = dynamic_cast<const RectangularPixelTopology*>(p.theTopol);
      assert(p.theRecTopol);
      
      //--- The geometrical description of one module/plaquette
      //p.theNumOfRow = p.theRecTopol->nrows();	// rows in x //Not used, AH. PM: leave commented out.
      //p.theNumOfCol = p.theRecTopol->ncolumns();	// cols in y //Not used, AH. PM: leave commented out.
      std::pair<float,float> pitchxy = p.theRecTopol->pitch();
      p.thePitchX = pitchxy.first;	     // pitch along x
      p.thePitchY = pitchxy.second;	     // pitch along y
      
      
      LocalVector Bfield = p.theDet->surface().toLocal(magfield_->inTesla(p.theDet->surface().position()));
      p.bz = Bfield.z();
      p.bx = Bfield.x();
      
      
      //---  Compute the Lorentz shifts for this detector element
      if ( (theFlag_==0) || DoLorentz_ ) {  // do always for generic and if(DOLorentz) for templates
         p.driftDirection = driftDirection(p, Bfield );
         computeLorentzShifts(p);
      }
      
      
      LogDebug("PixelCPEBase::fillDetParams()") << "***** PIXEL LAYOUT *****"
      << " thePart = " << p.thePart
      << " theThickness = " << p.theThickness
      << " thePitchX  = " << p.thePitchX
      << " thePitchY  = " << p.thePitchY;
      //			     << " theLShiftX  = " << p.theLShiftX;
      
      
   }
}

//-----------------------------------------------------------------------------
//  One function to cache the variables common for one DetUnit.
//-----------------------------------------------------------------------------
void
PixelCPEBase::setTheClu( DetParam const & theDetParam, ClusterParam & theClusterParam ) const
{
   
   //--- Geometric Quality Information
   int minInX,minInY,maxInX,maxInY=0;
   minInX = theClusterParam.theCluster->minPixelRow();
   minInY = theClusterParam.theCluster->minPixelCol();
   maxInX = theClusterParam.theCluster->maxPixelRow();
   maxInY = theClusterParam.theCluster->maxPixelCol();

   if      ( theDetParam.theRecTopol->isItEdgePixelInX(minInX) )
     theClusterParam.edgeTypeX_ = 1;
   else if ( theDetParam.theRecTopol->isItEdgePixelInX(maxInX) )
     theClusterParam.edgeTypeX_ = 2;
   else
     theClusterParam.edgeTypeX_ = 0;
     
   if      ( theDetParam.theRecTopol->isItEdgePixelInY(minInY) )
     theClusterParam.edgeTypeY_ = 1;
   else if ( theDetParam.theRecTopol->isItEdgePixelInY(maxInY) )
     theClusterParam.edgeTypeY_ = 2;
   else
     theClusterParam.edgeTypeY_ = 0;
   
   theClusterParam.isOnEdge_ = ( theClusterParam.edgeTypeX_ || theClusterParam.edgeTypeY_ );
   
   // &&& FOR NOW UNUSED. KEEP IT IN CASE WE WANT TO USE IT IN THE FUTURE
   // Bad Pixels have their charge set to 0 in the clusterizer
   //hasBadPixels_ = false;
   //for(unsigned int i=0; i<theClusterParam.theCluster->pixelADC().size(); ++i) {
   //if(theClusterParam.theCluster->pixelADC()[i] == 0) { hasBadPixels_ = true; break;}
   //}
   
   theClusterParam.spansTwoROCs_ = theDetParam.theRecTopol->containsBigPixelInX(minInX,maxInX) |
   theDetParam.theRecTopol->containsBigPixelInY(minInY,maxInY);
   
}


//-----------------------------------------------------------------------------
//  Compute alpha_ and beta_ from the LocalTrajectoryParameters.
//  Note: should become const after both localParameters() become const.
//-----------------------------------------------------------------------------
void PixelCPEBase::
computeAnglesFromTrajectory( DetParam const & theDetParam, ClusterParam & theClusterParam,
                            const LocalTrajectoryParameters & ltp) const
{
   
   theClusterParam.cotalpha = ltp.dxdz();
   theClusterParam.cotbeta  = ltp.dydz();
   
   
   LocalPoint trk_lp = ltp.position();
   theClusterParam.trk_lp_x = trk_lp.x();
   theClusterParam.trk_lp_y = trk_lp.y();
   
   theClusterParam.with_track_angle = true;
   
   // GG: needed to correct for bows/kinks
   theClusterParam.loc_trk_pred = 
      Topology::LocalTrackPred(theClusterParam.trk_lp_x, theClusterParam.trk_lp_y, 
                               theClusterParam.cotalpha,theClusterParam.cotbeta);
   
}

//-----------------------------------------------------------------------------
//  Estimate theAlpha for barrel, based on the det position.
//  &&& Needs to be consolidated from the above.
//-----------------------------------------------------------------------------
//float
//PixelCPEBase::estimatedAlphaForBarrel(float centerx) const
//{
//  float tanalpha = theSign * (centerx-theOffsetX) * thePitchX / theDetR;
//  return PI/2.0 - atan(tanalpha);
//}




//-----------------------------------------------------------------------------
//  Compute alpha_ and beta_ from the position of this DetUnit.
//  &&& DOES NOT WORK FOR NOW. d.k. 6/06
// The angles from dets are calculated internaly in the PixelCPEInitial class
//-----------------------------------------------------------------------------
// G. Giurgiu, 12/01/06 : implement the function
void PixelCPEBase::
computeAnglesFromDetPosition(DetParam const & theDetParam, ClusterParam & theClusterParam ) const
{
   
   LocalPoint lp = theDetParam.theTopol->localPosition( MeasurementPoint(theClusterParam.theCluster->x(), theClusterParam.theCluster->y()) );
   auto gvx = lp.x()-theDetParam.theOrigin.x();
   auto gvy = lp.y()-theDetParam.theOrigin.y();
   auto gvz = -1.f/theDetParam.theOrigin.z();
   //--- Note that the normalization is not required as only the ratio used  
   
   //theClusterParam.zneg = (gvz < 0); // Not used, AH
   
   // calculate angles
   theClusterParam.cotalpha = gvx*gvz;
   theClusterParam.cotbeta  = gvy*gvz;
   
   theClusterParam.with_track_angle = false;
   
   
   /*
    // used only in dberror param...
    auto alpha = HALF_PI - std::atan(cotalpha_);
    auto beta = HALF_PI - std::atan(cotbeta_);
    if (zneg) { beta -=PI; alpha -=PI;}
    
    auto alpha_ = atan2( gv_dot_gvz, gv_dot_gvx );
    auto beta_  = atan2( gv_dot_gvz, gv_dot_gvy );
    
    assert(std::abs(std::round(alpha*10000.f)-std::round(alpha_*10000.f))<2);
    assert(std::abs(std::round(beta*10000.f)-std::round(beta_*10000.f))<2);
    */
   
}



//------------------------------------------------------------------------
PixelCPEBase::DetParam const & PixelCPEBase::detParam(const GeomDetUnit & det) const 
{
   auto i = det.index();
   //cout << "get parameters of detector " << i << endl;
   assert(i<int(m_DetParams.size()));
   //if (i>=int(m_DetParams.size())) m_DetParams.resize(i+1);  // should never happen!
   const DetParam & p = m_DetParams[i];
   return p;
}

//-----------------------------------------------------------------------------
//  Drift direction.
//  Works OK for barrel and forward.
//  The formulas used for dir_x,y,z have to be exactly the same as the ones
//  used in the digitizer (SiPixelDigitizerAlgorithm.cc).
//  &&& PM: needs to be consolidated, discuss with PS.
//-----------------------------------------------------------------------------
LocalVector
PixelCPEBase::driftDirection(DetParam & theDetParam, GlobalVector bfield ) const 
{
   Frame detFrame(theDetParam.theDet->surface().position(), theDetParam.theDet->surface().rotation());
   LocalVector Bfield = detFrame.toLocal(bfield);
   return driftDirection(theDetParam,Bfield);
}


LocalVector
PixelCPEBase::driftDirection(DetParam & theDetParam, LocalVector Bfield ) const 
{
   // Use LA from DB or from config
   float langle = 0.;
   if( !useLAOffsetFromConfig_ ) {  // get it from DB
      if(lorentzAngle_ != nullptr) {  // a real LA object
         langle = lorentzAngle_->getLorentzAngle(theDetParam.theDet->geographicalId().rawId());
         LogDebug("PixelCPEBase::driftDirection()") 
	   <<" la "<<langle<<" "<< theDetParam.theDet->geographicalId().rawId() <<endl;
      } else { // no LA, unused
	 langle = 0; // set to a fake value
	 LogDebug("PixelCPEBase::driftDirection()") <<" LA object is NULL, assume LA = 0"<<endl; //dk
      }
      LogDebug("PixelCPEBase::driftDirection()") << " Will use LA Offset from DB "<<langle<<endl;
   } else {  // from config file
      langle = lAOffset_;
      LogDebug("PixelCPEBase::driftDirection()") << " Will use LA Offset from config "<<langle<<endl;
   }
   
   // Now do the LA width stuff
   theDetParam.widthLAFractionX = 1.; // predefine to 1 (default) if things fail
   theDetParam.widthLAFractionY = 1.;
   
   // Compute the charge width, generic only
   if(theFlag_==0) {
      
      if(useLAWidthFromDB_ && (lorentzAngleWidth_ != nullptr) ) {
         // take it from a separate, special LA DB object (forWidth)
         
         auto langleWidth = lorentzAngleWidth_->getLorentzAngle(theDetParam.theDet->geographicalId().rawId());
         if(langleWidth!=0.0) theDetParam.widthLAFractionX = std::abs(langleWidth/langle);
         // leave the widthLAFractionY=1.
         //cout<<" LAWidth lorentz width "<<theDetParam.widthLAFractionX<<" "<<theDetParam.widthLAFractionY<<endl;
         
      } else if(useLAWidthFromConfig_) { // get from config
         
         double lAWidth=0;
         if( GeomDetEnumerators::isTrackerPixel(theDetParam.thePart) && GeomDetEnumerators::isBarrel(theDetParam.thePart) ) lAWidth = lAWidthBPix_; // barrel
         else lAWidth = lAWidthFPix_;
         
         if(langle!=0.0) theDetParam.widthLAFractionX = std::abs(lAWidth/langle);
         // fix the FractionY at 1
         
         //cout<<" Lorentz width from config "<<theDetParam.widthLAFractionX<<" "<<theDetParam.widthLAFractionY<<endl;
         
      } else { // get if from the offset LA (old method used until 2013)
         // do nothing
         //cout<<" Old default LA width method "<<theDetParam.widthLAFractionX<<" "<<theDetParam.widthLAFractionY<<endl;
         
      }
      
      //cout<<" Final LA fraction  "<<theDetParam.widthLAFractionX<<" "<<theDetParam.widthLAFractionY<<endl;
      
   }  // if flag
   

   float alpha2 = alpha2Order ?  langle*langle : 0; //
   
   // **********************************************************************
   // Our convention is the following:
   // +x is defined by the direction of the Lorentz drift!
   // +z is defined by the direction of E field (so electrons always go into -z!)
   // +y is defined by +x and +z, and it turns out to be always opposite to the +B field.
   // **********************************************************************
   
   float dir_x = -( langle * Bfield.y() + alpha2* Bfield.z()* Bfield.x() );
   float dir_y =  ( langle * Bfield.x() - alpha2* Bfield.z()* Bfield.y() );
   float dir_z = -( 1.f + alpha2* Bfield.z()*Bfield.z() );
   auto scale = 1.f/std::abs( dir_z );  // same as 1 + alpha2*Bfield.z()*Bfield.z()
   LocalVector  dd(dir_x*scale, dir_y*scale, -1.f );  // last is -1 !
   
   LogDebug("PixelCPEBase") << " The drift direction in local coordinate is "  << dd   ;
   
   return dd;
}

//-----------------------------------------------------------------------------
//  One-shot computation of the driftDirection and both lorentz shifts
//-----------------------------------------------------------------------------
void
PixelCPEBase::computeLorentzShifts(DetParam & theDetParam) const 
{
   // Max shift (at the other side of the sensor) in cm
   theDetParam.lorentzShiftInCmX = theDetParam.driftDirection.x()/theDetParam.driftDirection.z() * theDetParam.theThickness;  //
   theDetParam.lorentzShiftInCmY = theDetParam.driftDirection.y()/theDetParam.driftDirection.z() * theDetParam.theThickness;  //
   
   LogDebug("PixelCPEBase::computeLorentzShifts()") << " lorentzShiftsInCmX,Y = "
						    << theDetParam.lorentzShiftInCmX <<" "
						    << theDetParam.lorentzShiftInCmY <<" "
						    << theDetParam.driftDirection;
}

//-----------------------------------------------------------------------------
//! A convenience method to fill a whole SiPixelRecHitQuality word in one shot.
//! This way, we can keep the details of what is filled within the pixel
//! code and not expose the Transient SiPixelRecHit to it as well.  The name
//! of this function is chosen to match the one in SiPixelRecHit.
//-----------------------------------------------------------------------------
SiPixelRecHitQuality::QualWordType
PixelCPEBase::rawQualityWord(ClusterParam & theClusterParam) const
{
   SiPixelRecHitQuality::QualWordType qualWord(0);
   if (theClusterParam.hasFilledProb_) {
     float probabilityXY=0;
     if(theClusterParam.filled_from_2d) probabilityXY = theClusterParam.probabilityX_;
     else if ( theClusterParam.probabilityX_ !=0 && theClusterParam.probabilityY_ !=0 )
       probabilityXY = theClusterParam.probabilityX_ * theClusterParam.probabilityY_ * (1.f - std::log(theClusterParam.probabilityX_ * theClusterParam.probabilityY_) ) ;
     SiPixelRecHitQuality::thePacking.setProbabilityXY ( probabilityXY ,
                                                        qualWord );
   
     SiPixelRecHitQuality::thePacking.setProbabilityQ  ( theClusterParam.probabilityQ_ ,
                                                        qualWord );
   }
   SiPixelRecHitQuality::thePacking.setQBin          ( theClusterParam.qBin_, 
                                                      qualWord );
   
   SiPixelRecHitQuality::thePacking.setIsOnEdge      ( theClusterParam.isOnEdge_,
                                                      qualWord );
   
   SiPixelRecHitQuality::thePacking.setHasBadPixels  ( theClusterParam.hasBadPixels_,
                                                      qualWord );
   
   SiPixelRecHitQuality::thePacking.setSpansTwoROCs  ( theClusterParam.spansTwoROCs_,
                                                      qualWord );
   
   SiPixelRecHitQuality::thePacking.setHasFilledProb ( theClusterParam.hasFilledProb_,
                                                      qualWord );
   
   return qualWord;
}
