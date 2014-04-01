// Move geomCorrection to the concrete class. d.k. 06/06.
// Change drift direction. d.k. 06/06

// G. Giurgiu (ggiurgiu@pha.jhu.edu), 12/01/06, implemented the function: 
// computeAnglesFromDetPosition(const SiPixelCluster & cl, 
//			        const GeomDetUnit    & det ) const
//                                    09/09/07, replaced assert statements with throw cms::Exception 
//                                              and fix an invalid pointer check in setTheDet function 
//                                    09/21/07, implement caching of Lorentz drift direction
//                                    01/24/09, use atan2 to get the alpha and beta angles
// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008
// Change to use Generic error & Template calibration from DB - D.Fehling 11/08

#define NEW

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/ProxyPixelTopology.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"

#define CORRECT_FOR_BIG_PIXELS

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include <iostream>

using namespace std;


namespace {
  constexpr float degsPerRad = 57.29578;
  constexpr float HALF_PI = 1.57079632679489656;
  constexpr float PI = 2*HALF_PI;
}

//-----------------------------------------------------------------------------
//  A fairly boring constructor.  All quantities are DetUnit-dependent, and
//  will be initialized in setTheDet().
//-----------------------------------------------------------------------------
#ifdef NEW
PixelCPEBase::PixelCPEBase(edm::ParameterSet const & conf, const MagneticField *mag, const TrackerGeometry& geom,
			   const SiPixelLorentzAngle * lorentzAngle, 
			   const SiPixelGenErrorDBObject * genErrorDBObject, 
			   const SiPixelTemplateDBObject * templateDBobject,
			   const SiPixelLorentzAngle * lorentzAngleWidth)
#else
PixelCPEBase::PixelCPEBase(edm::ParameterSet const & conf, const MagneticField *mag, const TrackerGeometry& geom,
			   const SiPixelLorentzAngle * lorentzAngle, 
			   const SiPixelCPEGenericErrorParm * genErrorParm, 
			   const SiPixelTemplateDBObject * templateDBobject,
			   const SiPixelLorentzAngle * lorentzAngleWidth)
#endif
  : theParam(nullptr), nRecHitsTotal_(0), nRecHitsUsedEdge_(0),
    probabilityX_(0.0), probabilityY_(0.0),
    probabilityQ_(0.0), qBin_(0),
    isOnEdge_(false), hasBadPixels_(false),
    spansTwoROCs_(false), hasFilledProb_(false),
    useLAAlignmentOffsets_(false), useLAOffsetFromConfig_(false),
    useLAWidthFromConfig_(false), useLAWidthFromDB_(false),
    magfield_(mag), geom_(geom),
    loc_trk_pred_(0.0, 0.0, 0.0, 0.0)
{
  //--- Lorentz angle tangent per Tesla

  lorentzAngle_ = lorentzAngle;
  lorentzAngleWidth_ = lorentzAngleWidth;
 
  //--- Algorithm's verbosity
  theVerboseLevel = 
    conf.getUntrackedParameter<int>("VerboseLevel",0);
  
#ifdef NEW
   //-- GenError Calibration Object (different from SiPixelCPEGenericErrorParm) from DB
   genErrorDBObject_ = genErrorDBObject;
#else
  //-- Error Parametriaztion from DB for CPE Generic
  genErrorParm_ = genErrorParm;
#endif

  //-- Template Calibration Object from DB
  templateDBobject_ = templateDBobject;

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

  // For safety initilaize the parameters which are used by generic algo only to 0
  lAOffset_ = 0.0;
  lAWidthBPix_  = 0.0;
  lAWidthFPix_  = 0.0;

  LogDebug("PixelCPEBase") <<" LA constants - "
			   <<lAOffset_<<" "<<lAWidthBPix_<<" "<<lAWidthFPix_<<endl; //dk
  
  fillParams();
  
}

//-----------------------------------------------------------------------------
//  Fill all variables which are constant for an event (geometry)
//-----------------------------------------------------------------------------
void PixelCPEBase::fillParams()
{
  const unsigned m_detectors = geom_.offsetDU(GeomDetEnumerators::TIB); //first non-pixel detector unit
  auto const & dus = geom_.detUnits();
  m_Params.resize(m_detectors);
  //cout<<"caching "<<m_detectors<<" pixel detectors"<<endl;
  for (unsigned i=0; i!=m_detectors;++i) {
    auto & p=m_Params[i];
    theParam=&p;
    p.theDet = dynamic_cast<const PixelGeomDetUnit*>(dus[i]);
    assert(p.theDet); 
    assert(p.theDet->index()==int(i)); 

    p.theOrigin = p.theDet->surface().toLocal(GlobalPoint(0,0,0));
    
    //--- p.theDet->type() returns a GeomDetType, which implements subDetector()
    p.thePart = p.theDet->type().subDetector();
    
    //cout<<" in PixelCPEBase:settheDet - in det "<<thePart<<endl; //dk

    switch ( p.thePart ) {
    case GeomDetEnumerators::PixelBarrel:
      // A barrel!  A barrel!
      p.lAWidth = lAWidthBPix_;
      break;
    case GeomDetEnumerators::PixelEndcap:
      // A forward!  A forward!
      p.lAWidth = lAWidthFPix_;
      break;
     default:
       // does one need this exception?
       //cout<<" something wrong"<<endl;
       throw cms::Exception("PixelCPEBase::settheDet :")
	 << "PixelCPEBase: A non-pixel detector type in here?" ;
    }
    
    //--- The location in of this DetUnit in a cyllindrical coord system (R,Z)
    //--- The call goes via BoundSurface, returned by p.theDet->surface(), but
    //--- position() is implemented in GloballyPositioned<> template
    //--- ( BoundSurface : Surface : GloballyPositioned<float> )
    p.theDetR = p.theDet->surface().position().perp();
    p.theDetZ = p.theDet->surface().position().z();
    //--- Define parameters for chargewidth calculation
    
    //--- bounds() is implemented in BoundSurface itself.
    p.theThickness = p.theDet->surface().bounds().thickness();
    
    //--- Cache the topology.
    // ggiurgiu@jhu.edu 12/09/2010 : no longer need to dynamyc cast to RectangularPixelTopology
    //theTopol
    //= dynamic_cast<const RectangularPixelTopology*>( & (p.theDet->specificTopology()) );

    auto topol = &(p.theDet->specificTopology());
    if unlikely(topol!=p.theTopol) { // there is ONE topology!)
       p.theTopol=topol;
       auto const proxyT = dynamic_cast<const ProxyPixelTopology*>(p.theTopol);
       if (proxyT) p.theRecTopol = dynamic_cast<const RectangularPixelTopology*>(&(proxyT->specificTopology()));
       else p.theRecTopol = dynamic_cast<const RectangularPixelTopology*>(p.theTopol);
       assert(p.theRecTopol);
       
       //---- The geometrical description of one module/plaquette
       p.theNumOfRow = p.theRecTopol->nrows();	// rows in x
       p.theNumOfCol = p.theRecTopol->ncolumns();	// cols in y
       std::pair<float,float> pitchxy = p.theRecTopol->pitch();
       p.thePitchX = pitchxy.first;	     // pitch along x
       p.thePitchY = pitchxy.second;	     // pitch along y
      }
     
    p.theSign = isFlipped() ? -1 : 1;

    LocalVector Bfield = p.theDet->surface().toLocal(magfield_->inTesla(p.theDet->surface().position()));
    p.bz = Bfield.z();
    p.driftDirection = driftDirection( Bfield );
    p.widthLAFraction = widthLAFraction_;
    
    LogDebug("PixelCPEBase") << "***** PIXEL LAYOUT *****" 
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
PixelCPEBase::setTheDet( const GeomDetUnit & det, const SiPixelCluster & cluster ) const 
{

  theParam = &param(det);
    
  //--- Geometric Quality Information
  int minInX,minInY,maxInX,maxInY=0;
  minInX = cluster.minPixelRow();
  minInY = cluster.minPixelCol();
  maxInX = cluster.maxPixelRow();
  maxInY = cluster.maxPixelCol();
  
  isOnEdge_ = theParam->theRecTopol->isItEdgePixelInX(minInX) | theParam->theRecTopol->isItEdgePixelInX(maxInX) |
    theParam->theRecTopol->isItEdgePixelInY(minInY) | theParam->theRecTopol->isItEdgePixelInY(maxInY) ;
  
  // FOR NOW UNUSED. KEEP IT IN CASE WE WANT TO USE IT IN THE FUTURE  
  // Bad Pixels have their charge set to 0 in the clusterizer 
  //hasBadPixels_ = false;
  //for(unsigned int i=0; i<cluster.pixelADC().size(); ++i) {
  //if(cluster.pixelADC()[i] == 0) { hasBadPixels_ = true; break;}
  //}
  
  spansTwoROCs_ = theParam->theRecTopol->containsBigPixelInX(minInX,maxInX) |
    theParam->theRecTopol->containsBigPixelInY(minInY,maxInY);

}


//-----------------------------------------------------------------------------
//  Compute alpha_ and beta_ from the LocalTrajectoryParameters.
//  Note: should become const after both localParameters() become const.
//-----------------------------------------------------------------------------
void PixelCPEBase::
computeAnglesFromTrajectory( const SiPixelCluster & cl,
			     const LocalTrajectoryParameters & ltp) const
{
  //cout<<" in PixelCPEBase:computeAnglesFromTrajectory - "<<endl; //dk

  loc_traj_param_ = ltp;

  LocalVector localDir = ltp.momentum();
  
  
  float locx = localDir.x();
  float locy = localDir.y();
  float locz = localDir.z();
  
  /*
  // Danek's definition 
  alpha_ = acos(locx/sqrt(locx*locx+locz*locz));
  if ( isFlipped() )                    // &&& check for FPIX !!!
  alpha_ = PI - alpha_ ;
  beta_ = acos(locy/sqrt(locy*locy+locz*locz));
  */
  
  
  cotalpha_ = locx/locz;
  cotbeta_  = locy/locz;
  zneg = (locz < 0);
  
  
  LocalPoint trk_lp = ltp.position();
  trk_lp_x = trk_lp.x();
  trk_lp_y = trk_lp.y();
  
  with_track_angle = true;


  // ggiurgiu@jhu.edu 12/09/2010 : needed to correct for bows/kinks
  AlgebraicVector5 vec_trk_parameters = ltp.mixedFormatVector();
  loc_trk_pred_ = Topology::LocalTrackPred( vec_trk_parameters );
  
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
computeAnglesFromDetPosition(const SiPixelCluster & cl ) const
{
 
  
  /*
  // get cluster center of gravity (of charge)
  float xcenter = cl.x();
  float ycenter = cl.y();
  
  // get the cluster position in local coordinates (cm)

  // ggiurgiu@jhu.edu 12/09/2010 : This function is called without track info, therefore there are no track 
  // angles to provide here. Call the default localPosition (without track info)
  LocalPoint lp = theTopol->localPosition( MeasurementPoint(xcenter, ycenter) );


  // get the cluster position in global coordinates (cm)
  GlobalPoint gp = theDet->surface().toGlobal( lp );
  float gp_mod = sqrt( gp.x()*gp.x() + gp.y()*gp.y() + gp.z()*gp.z() );

  // normalize
  float gpx = gp.x()/gp_mod;
  float gpy = gp.y()/gp_mod;
  float gpz = gp.z()/gp_mod;

  // make a global vector out of the global point; this vector will point from the 
  // origin of the detector to the cluster
  GlobalVector gv(gpx, gpy, gpz);

  // make local unit vector along local X axis
  const Local3DVector lvx(1.0, 0.0, 0.0);

  // get the unit X vector in global coordinates/
  GlobalVector gvx = theDet->surface().toGlobal( lvx );

  // make local unit vector along local Y axis
  const Local3DVector lvy(0.0, 1.0, 0.0);

  // get the unit Y vector in global coordinates
  GlobalVector gvy = theDet->surface().toGlobal( lvy );
   
  // make local unit vector along local Z axis
  const Local3DVector lvz(0.0, 0.0, 1.0);

  // get the unit Z vector in global coordinates
  GlobalVector gvz = theDet->surface().toGlobal( lvz );
    
  // calculate the components of gv (the unit vector pointing to the cluster) 
  // in the local coordinate system given by the basis {gvx, gvy, gvz}
  // note that both gv and the basis {gvx, gvy, gvz} are given in global coordinates
  float gv_dot_gvx = gv.x()*gvx.x() + gv.y()*gvx.y() + gv.z()*gvx.z();
  float gv_dot_gvy = gv.x()*gvy.x() + gv.y()*gvy.y() + gv.z()*gvy.z();
  float gv_dot_gvz = gv.x()*gvz.x() + gv.y()*gvz.y() + gv.z()*gvz.z();
  */
  
  // all the above is equivalent to 
  LocalPoint lp = theParam->theTopol->localPosition( MeasurementPoint(cl.x(), cl.y()) );
  auto gvx = lp.x()-theParam->theOrigin.x();
  auto gvy = lp.y()-theParam->theOrigin.y();
  auto gvz = -1.f/theParam->theOrigin.z();
  //  normalization not required as only ratio used... 
  

  zneg = (gvz < 0);

  // calculate angles
  cotalpha_ = gvx*gvz;
  cotbeta_  = gvy*gvz;

  with_track_angle = false;


  /*
  // used only in dberror param...
  auto alpha = HALF_PI - std::atan(cotalpha_);
  auto beta = HALF_PI - std::atan(cotbeta_); 
  if (zneg) { beta -=PI; alpha -=PI;}

  auto alpha_ = atan2( gv_dot_gvz, gv_dot_gvx );
  auto beta_  = atan2( gv_dot_gvz, gv_dot_gvy );

  std::cout << "alpha/beta " << alpha_ <<','<<alpha <<' '<< beta_<<','<<beta <<','<< HALF_PI-beta << std::endl;
  assert(std::abs(std::round(alpha*10000.f)-std::round(alpha_*10000.f))<2);
  assert(std::abs(std::round(beta*10000.f)-std::round(beta_*10000.f))<2);
  */

}



//-----------------------------------------------------------------------------
// The isFlipped() is a silly way to determine which detectors are inverted.
// In the barrel for every 2nd ladder the E field direction is in the
// global r direction (points outside from the z axis), every other
// ladder has the E field inside. Something similar is in the 
// forward disks (2 sides of the blade). This has to be recognised
// because the charge sharing effect is different.
//
// The isFliped does it by looking and the relation of the local (z always
// in the E direction) to global coordinates. There is probably a much 
// better way.
//-----------------------------------------------------------------------------
bool PixelCPEBase::isFlipped() const 
{
  // Check the relative position of the local +/- z in global coordinates.
  float tmp1 = theParam->theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp2();
  float tmp2 = theParam->theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp2();
  //cout << " 1: " << tmp1 << " 2: " << tmp2 << endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}
//------------------------------------------------------------------------
PixelCPEBase::Param const & PixelCPEBase::param(const GeomDetUnit & det) const {
  auto i = det.index();
  //cout << "get parameters of detector " << i << endl;
  if (i>=int(m_Params.size())) m_Params.resize(i+1);  // should never happen!
  Param & p = m_Params[i];
  return p;
}

//-----------------------------------------------------------------------------
//  Drift direction.
//  Works OK for barrel and forward.
//  The formulas used for dir_x,y,z have to be exactly the same as the ones
//  used in the digitizer (SiPixelDigitizerAlgorithm.cc).
//  Assumption: setTheDet() has been called already.
//
//-----------------------------------------------------------------------------
LocalVector 
PixelCPEBase::driftDirection( GlobalVector bfield ) const {

  Frame detFrame(theParam->theDet->surface().position(), theParam->theDet->surface().rotation());
  LocalVector Bfield = detFrame.toLocal(bfield);
  return driftDirection(Bfield);
  
}

LocalVector 
PixelCPEBase::driftDirection( LocalVector Bfield ) const {
  const bool LocalPrint = false;
  const bool useLAWidthFromGenError = false;

  //auto langle = lorentzAngle_->getLorentzAngle(theDet->geographicalId().rawId());
  // Use LA from DB or from config 
  float langle = 0.;
  if( !useLAOffsetFromConfig_ ) {  // get it from DB
    if(lorentzAngle_ != NULL) {  // a real LA object 
      langle = lorentzAngle_->getLorentzAngle(theParam->theDet->geographicalId().rawId());
    } else { // no LA, unused 
      //cout<<" LA object is NULL, assume LA = 0"<<endl; //dk
      langle = 0; // set to a fake value
    }
    if(LocalPrint) cout<<" Will use LA Offset from DB "<<langle<<endl;
  } else {  // from config file 
    langle = lAOffset_;
    if(LocalPrint) cout<<" Will use LA Offset from config "<<langle<<endl;
  } 
 
  //We also need the LA values used for the charge width
  // I do not know where to put it best, try here!
  if(useLAWidthFromDB_ && (lorentzAngleWidth_ != NULL) ) {  // get it from DB

    float langleWidth = 0.;
    if(useLAWidthFromGenError) {
      // or from the new error object
      // for the moment this does not compile, gtemp_ is defined only in generic
      auto gtemplid = genErrorDBObject_->getGenErrorID(theParam->theDet->geographicalId().rawId());
      cout<<gtemplid<<endl;
      //auto qbin = gtempl_.qbin( gtemplid);  // inputs
      //langleWidth = -micronsToCm*gtempl_.lorxwidth();
      ////chargeWidthY = -micronsToCm*gtempl_.lorywidth();
    } else {
      // take it from LA object label=from-width
      langleWidth = lorentzAngleWidth_->getLorentzAngle(theParam->theDet->geographicalId().rawId());
    }

    if(langleWidth!=0.0) widthLAFraction_ = std::abs(langleWidth/langle);
    else widthLAFraction_ = 1.0;
    if(LocalPrint)  cout<<" Will use LA Width from DB "<<langleWidth<<" "<<widthLAFraction_<<endl;
  } else if(useLAWidthFromConfig_) { // get from config 
    if(langle!=0.0) widthLAFraction_ = std::abs(lAWidth_/langle);
    if(LocalPrint)  cout<<" Will use LA Width from config "<<lAWidth_<<endl;
  } else { // get if from the offset LA (old method used until 2013)
    widthLAFraction_ = 1.0; // use the same angle
    if(LocalPrint)  cout<<" Will use LA Width from LA Offset "<<widthLAFraction_<<endl;
  }
    
  float alpha2 = alpha2Order ?  langle*langle : 0;

  if(LocalPrint) cout<<" in PixelCPEBase:driftDirection - "<<langle<<" "<<Bfield<<endl; //dk

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
PixelCPEBase::computeLorentzShifts() const {

  //cout<<" in PixelCPEBase:computeLorentzShifts - "<<driftDirection_<<endl; //dk

  // Max shift (at the other side of the sensor) in cm 
  lorentzShiftInCmX_ = theParam->driftDirection.x()/theParam->driftDirection.z() * theParam->theThickness;  // 
  lorentzShiftInCmY_ = theParam->driftDirection.y()/theParam->driftDirection.z() * theParam->theThickness;  //
  
  //cout<<" in PixelCPEBase:computeLorentzShifts - "
  //<<lorentzShiftInCmX_<<" "
  //<<lorentzShiftInCmY_<<" "
  //<<endl; //dk
  
  LogDebug("PixelCPEBase") << " The drift direction in local coordinate is " 
			   << theParam->driftDirection    ;
}

//-----------------------------------------------------------------------------
//! A convenience method to fill a whole SiPixelRecHitQuality word in one shot.
//! This way, we can keep the details of what is filled within the pixel
//! code and not expose the Transient SiPixelRecHit to it as well.  The name
//! of this function is chosen to match the one in SiPixelRecHit.
//-----------------------------------------------------------------------------
SiPixelRecHitQuality::QualWordType 
PixelCPEBase::rawQualityWord() const
{
  SiPixelRecHitQuality::QualWordType qualWord(0);
  
  SiPixelRecHitQuality::thePacking.setProbabilityXY ( probabilityXY() ,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setProbabilityQ  ( probabilityQ_ , 
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setQBin          ( (int)qBin_, 
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setIsOnEdge      ( isOnEdge_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setHasBadPixels  ( hasBadPixels_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setSpansTwoROCs  ( spansTwoROCs_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setHasFilledProb ( hasFilledProb_,
                                                      qualWord );
  
  return qualWord;
}
