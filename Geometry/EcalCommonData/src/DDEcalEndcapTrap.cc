#include "Geometry/EcalCommonData/interface/DDEcalEndcapTrap.h"

// Implementation of DDEcalEndcapTrap class

DDEcalEndcapTrap::DDEcalEndcapTrap( const int hand,
				    const double front,
				    const double rear,
				    const double length) 
{
  //
  //  Initialise corners of supercrystal.

  // Start out with bottom surface on (x,z) plane, front face in (x,y) plane.

  double xsign;

  if (hand==2) {
   xsign = -1.;
  } else {
    xsign = 1.;
  }

  m_hand = hand;
  m_front = front;
  m_rear = rear;
  m_length = length;

  int icorner;
  icorner = 1;
  m_corners[3*icorner-3] = xsign*front;
  m_corners[3*icorner-2] = front;
  m_corners[3*icorner-1] = 0.;
  icorner = 2;
  m_corners[3*icorner-3] = xsign*front;
  m_corners[3*icorner-2] = 0.;
  m_corners[3*icorner-1] = 0.;
  icorner = 3;
  m_corners[3*icorner-3] = 0.;
  m_corners[3*icorner-2] = 0.;
  m_corners[3*icorner-1] = 0.;
  icorner = 4;
  m_corners[3*icorner-3] = 0.;
  m_corners[3*icorner-2] = front;
  m_corners[3*icorner-1] = 0.;

  icorner = 5;
  m_corners[3*icorner-3] = xsign*rear;
  m_corners[3*icorner-2] = rear;
  m_corners[3*icorner-1] = length;
  icorner = 6;
  m_corners[3*icorner-3] = xsign*rear;
  m_corners[3*icorner-2] = 0.;
  m_corners[3*icorner-1] = length;
  icorner = 7;
  m_corners[3*icorner-3] = 0.;
  m_corners[3*icorner-2] = 0.;
  m_corners[3*icorner-1] = length;
  icorner = 8;
  m_corners[3*icorner-3] = 0.;
  m_corners[3*icorner-2] = rear;
  m_corners[3*icorner-1] = length;
  
  calculateCentres();

  // Move centre of SC to (0,0,0)

  translate();

  // Rotate into standard position (face centres on z axis)

  //  this->rotate();

  calculateCentres();
}


//void DDEcalEndcapTrap::rotate() {
//  //
//  //  Rotate supercrystal to standard position
//  //
//  std::cout << "DDEcalEndcapTrap::rotate() - not yet implemented" << std::endl;
//}

void 
DDEcalEndcapTrap::rotate( const DDTranslation frontCentre, 
			  const DDTranslation rearCentre   ) 
{
  //
  //  Rotate supercrystal to bring front and rear face centres to specified points
  //
  edm::LogInfo("EcalGeom") << "DDEcalEndcapTrap::rotate(DDTranslation,DDTranslation) - not yet implemented" << std::endl;
}

void 
DDEcalEndcapTrap::rotate(const DDRotationMatrix rot) 
{
  //
  //  Rotate supercrystal by specified rotation about (0,0,0)
  //

  int icorner;
  DDTranslation cc;
  //  edm::LogInfo("EcalGeom") << "DDEcalEndcapTrap::rotate - rotation " << rot << std::endl;
  for (icorner=1; icorner<=8; icorner++) {
     cc = cornerPos(icorner);
     //     edm::LogInfo("EcalGeom") << "   Corner (orig) " << icorner << cc << std::endl;
     cc = rot*cc;
     //     edm::LogInfo("EcalGeom") << "   Corner (rot)  " << icorner << cc << std::endl;
     cornerPos(icorner,cc);
  }
  m_rotation = rot*m_rotation;
  calculateCentres();
}

void 
DDEcalEndcapTrap::translate() 
{
  //  edm::LogInfo("EcalGeom") << "DDEcalEndcapTrap::translate() not yet implemented" << std::endl;
   translate(-1.*centrePos());
}

void 
DDEcalEndcapTrap::translate(const DDTranslation trans) 
{
  //
  //  Translate supercrystal by specified amount
  //

  DDTranslation tcorner;
  for (int icorner=1; icorner<=8; icorner++) 
  {
    tcorner = cornerPos(icorner) + trans;
    cornerPos(icorner,tcorner);
  }
  calculateCentres();
  m_translation = trans + m_translation;
}

void 
DDEcalEndcapTrap::moveto( const DDTranslation frontCentre,
			  const DDTranslation rearCentre   ) 
{
  //
  //  Rotate (about X then about Y) and translate supercrystal to bring axis joining front and rear face centres parallel to line connecting specified points
  //

  //  Get azimuthal and polar angles of current axis and target axis
  double currentTheta = elevationAngle();
  double currentPhi   = polarAngle();
  double targetTheta  = elevationAngle(frontCentre-rearCentre);
  double targetPhi    = polarAngle(frontCentre-rearCentre);

  //  Rotate to correct angle (X then Y)
  // edm::LogInfo("EcalGeom") << "moveto: frontCentre " << frontCentre << std::endl;
  // edm::LogInfo("EcalGeom") << "moveto: rearCentre  " << rearCentre << std::endl;
  // edm::LogInfo("EcalGeom") << "moveto: X rotation: " << targetTheta << " " << currentTheta << " " << targetTheta-currentTheta << std::endl;
  // edm::LogInfo("EcalGeom") << "moveto: Y rotation: " << targetPhi << " " << currentPhi << " " << " " << targetPhi-currentPhi << std::endl;
  rotateX(targetTheta-currentTheta);
  rotateY(targetPhi-currentPhi);

  //  Translate SC to final position
  DDTranslation targetCentre = 0.5*(frontCentre + rearCentre);
  // edm::LogInfo("EcalGeom") << "moveto: translation " << targetCentre-centrePos() << std::endl;
  translate(targetCentre-centrePos());
}

void 
DDEcalEndcapTrap::rotateX(const double angle) 
{
  //
  //  Rotate SC through given angle about X axis
  //

   const CLHEP::HepRotation tmp ( CLHEP::Hep3Vector(1.,0.,0.), angle ) ;

   rotate( DDRotationMatrix( tmp.xx(), tmp.xy(), tmp.xz(),
			     tmp.yx(), tmp.yy(), tmp.yz(),
			     tmp.zx(), tmp.zy(), tmp.zz()  ) );
}

void 
DDEcalEndcapTrap::rotateY(const double angle) 
{
   //
   //  Rotate SC through given angle about Y axis
   //
   const CLHEP::HepRotation tmp ( CLHEP::Hep3Vector(0.,1.,0.), angle ) ;

   rotate( DDRotationMatrix( tmp.xx(), tmp.xy(), tmp.xz(),
			     tmp.yx(), tmp.yy(), tmp.yz(),
			     tmp.zx(), tmp.zy(), tmp.zz()  ) );
}

void DDEcalEndcapTrap::calculateCentres() 
{
  //
  //  Calculate crystal centre and front & rear face centres
  //

  int ixyz, icorner;

  for (ixyz = 0; ixyz <3; ixyz++) 
  {
    m_centre[ixyz] = 0;
    m_fcentre[ixyz] = 0;
    m_rcentre[ixyz] = 0;
  }

  for (icorner=1; icorner<=4; icorner++) 
  {
    for (ixyz=0; ixyz<3; ixyz++) {
       m_centre[ixyz] = m_centre[ixyz] + 0.125*m_corners[3*icorner-3+ixyz];
       m_fcentre[ixyz] = m_fcentre[ixyz] + 0.25*m_corners[3*icorner-3+ixyz];
    }
  }
  for (icorner=5; icorner<=8; icorner++) {
    for (ixyz=0; ixyz<3; ixyz++) {
       m_centre[ixyz] = m_centre[ixyz] + 0.125*m_corners[3*icorner-3+ixyz];
       m_rcentre[ixyz] = m_rcentre[ixyz] +  0.25*m_corners[3*icorner-3+ixyz];
    }
  }
}

DDTranslation 
DDEcalEndcapTrap::cornerPos(const int icorner) 
{
  //
  //  Return specified corner as a DDTranslation
  //
  return DDTranslation( m_corners[3*icorner-3],
			m_corners[3*icorner-2],
			m_corners[3*icorner-1]  );
}

void 
DDEcalEndcapTrap::cornerPos( const int           icorner, 
			     const DDTranslation cornerxyz ) 
{
  //
  //  Save position of specified corner.
  //
  for (int ixyz=0; ixyz<3; ixyz++) 
  {
     m_corners[3*icorner-3+ixyz] = ( 0==ixyz ? cornerxyz.x() :
				     ( 1==ixyz ? cornerxyz.y() :
				       cornerxyz.z() ) ) ;;
  }
}

DDTranslation 
DDEcalEndcapTrap::centrePos() 
{
  //
  //  Return SC centre as a DDTranslation
  //
   return DDTranslation(m_centre[0], m_centre[1], m_centre[2]);
}

DDTranslation 
DDEcalEndcapTrap::fcentrePos() 
{
  //
  //  Return SC front face centre as a DDTranslation
  //
   return DDTranslation( m_fcentre[0], 
			 m_fcentre[1], 
			 m_fcentre[2]  ) ;
}

DDTranslation 
DDEcalEndcapTrap::rcentrePos() 
{
  //
  //  Return SC rear face centre as a DDTranslation
  //
  return DDTranslation( m_rcentre[0], 
			m_rcentre[1], 
			m_rcentre[2]  );
}

double 
DDEcalEndcapTrap::elevationAngle( const DDTranslation trans ) 
{
  //
  //  Return elevation angle (out of x-z plane) of a given translation (seen as a vector from the origin).
  //
  double sintheta = trans.y()/trans.r();
  return asin(sintheta);
}

double 
DDEcalEndcapTrap::elevationAngle() 
{
  //
  //  Return elevation angle (out of x-z plane) of SC in current position.
  //
  DDTranslation current = fcentrePos() - rcentrePos();
  return elevationAngle(current);
}

double 
DDEcalEndcapTrap::polarAngle(const DDTranslation trans) 
{
  //
  //  Return polar angle (from x to z) of a given translation (seen as a vector from the origin).
  //
  double tanphi = trans.x()/trans.z();
  return atan(tanphi);
}

double 
DDEcalEndcapTrap::polarAngle() 
{
  //
  //  Return elevation angle (out of x-z plane) of SC in current position.
  //
  DDTranslation current = fcentrePos() - rcentrePos();
  return polarAngle(current);
}

void 
DDEcalEndcapTrap::print() 
{
  //
  //  Print SC coordinates for debugging
  //
  edm::LogInfo("EcalGeom") << "Endcap supercrystal" << std::endl;
   for (int ic=1; ic<=8; ic++) 
   {
      DDTranslation cc = cornerPos(ic);
      edm::LogInfo("EcalGeom") << "Corner " << ic << " " << cc << std::endl;
   }
   edm::LogInfo("EcalGeom") << "    Centre " << centrePos() << std::endl;
   edm::LogInfo("EcalGeom") << "   fCentre " << fcentrePos() << std::endl;
   edm::LogInfo("EcalGeom") << "   rCentre " << rcentrePos() << std::endl;
}




