#ifndef EcalAlgo_DDEcalEndcapTrap_h
#define EcalAlgo_DDEcalEndcapTrap_h

#include <vector>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"

// Define Endcap Supercrystal class

class DDEcalEndcapTrap {
public:
  DDEcalEndcapTrap(const int hand, const double front, const double rear, const double length);
  DDEcalEndcapTrap() = delete;

  void rotate(const DDRotationMatrix& rot);
  void rotate(const DDTranslation& frontCentre, const DDTranslation& rearCentre);
  void translate(const DDTranslation& trans);

  void rotateX(const double angle);
  void rotateY(const double angle);
  void translate();
  void moveto(const DDTranslation& frontCentre, const DDTranslation& rearCentre);
  double elevationAngle(const DDTranslation& trans);
  double polarAngle(const DDTranslation& trans);
  double elevationAngle();
  double polarAngle();
  DDTranslation cornerPos(const int icorner);
  void cornerPos(const int icorner, const DDTranslation& cc);
  DDTranslation centrePos();
  DDTranslation fcentrePos();
  DDTranslation rcentrePos();
  void calculateCorners();
  void calculateCentres();
  DDRotationMatrix rotation() { return m_rotation; }
  void print();

private:
  DDRotationMatrix m_rotation;
  DDTranslation m_translation;

  double m_centre[4];
  double m_fcentre[4];
  double m_rcentre[4];
  double m_corners[25];
  double m_front;
  double m_rear;
  double m_length;

  int m_hand;
  int m_update;
};

#endif
