#ifndef EcalAlgo_DDEcalEndcapTrap_h
#define EcalAlgo_DDEcalEndcapTrap_h

#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <CLHEP/Vector/EulerAngles.h>

// Define Endcap Supercrystal class

class DDEcalEndcapTrap {

   public:
      DDEcalEndcapTrap( const int hand,
			const double front , 
			const double rear  ,
			const double length  ) ;

      //  virtual ~DDEcalEndcapTrap();

      void rotate( const DDRotationMatrix rot ) ;
      void rotate( const DDTranslation frontCentre,
		   const DDTranslation rearCentre ) ;
      void translate( const DDTranslation trans ) ;

  //  void rotate();

      void rotateX( const double angle ) ;
      void rotateY( const double angle ) ;
      void translate();
      void moveto( const DDTranslation frontCentre,
		   const DDTranslation rearCentre  );
      double elevationAngle( const DDTranslation trans );
      double polarAngle(     const DDTranslation trans);
      double elevationAngle();
      double polarAngle();
      DDTranslation cornerPos( const int icorner );
      void cornerPos( const int           icorner ,
		      const DDTranslation cc)         ;
      DDTranslation centrePos();
      DDTranslation fcentrePos();
      DDTranslation rcentrePos();
      void calculateCorners();
      void calculateCentres();
      DDRotationMatrix rotation() {return m_rotation;}
      void print();
  

 private:
      DDEcalEndcapTrap(); // forbid default constructor

      double m_front;
      double m_rear;
      double m_length;
      int m_hand;
      DDRotationMatrix m_rotation;
      DDTranslation m_translation;

      double m_centre[4];
      double m_fcentre[4];
      double m_rcentre[4];
      double m_corners[25];

      int m_update;
};

#endif
