#ifndef _HIT_H_
#define _HIT_H_
#include <iostream>
#include <cmath>
using namespace std;

/**
   \brief A hit in the detector
**/
class Hit{

 private :
  char layer;
  char ladder;
  char zPos;
  char segment;
  short stripNumber;
  short stub_idx;
  int part_id;
  float part_pt;
  float part_ip;
  float part_eta;
  float part_phi0;
  float x;
  float y;
  float z;
  float X0;
  float Y0;
  float Z0;

  

 public :
  /**
     \brief Constructor
     \param l The layer of the hit
     \param lad The ladder of the hit
     \param zp The Z position of the module
     \param seg The segment of the hit in the module
     \param strip The strip touched
     \param idx The index of the stub creating this hit
     \param tp The belonging particule's ID (used to check efficiency) 
     \param pt The belonging particule's PT (used to check efficiency) 
     \param ip Distance between particule's origine and interaction point
     \param eta The belonging particule's ETA
     \param phi0 The belonging particule's PHI0
     \param p_x The X coordinate of the hit in the tracker
     \param p_y The Y coordinate of the hit in the tracker
     \param p_z The Z coordinate of the hit in the tracker
     \param p_x0 The X0 coordinate of the hit in the tracker
     \param p_y0 The Y0 coordinate of the hit in the tracker
     \param p_z0 The Z0 coordinate of the hit in the tracker
  **/
  Hit(char l, char lad, char zp, char seg, short strip, short idx, int tp, float pt, float ip, float eta, float phi0, float p_x, float p_y, float p_z, float p_x0, float p_y0, float p_z0);
  /**
     \brief The copy Constructor
  **/
  Hit(const Hit& h);
  /**
     \brief Get the layer of the hit
     \return The layer of the Hit
  **/
  char getLayer() const;
  /**
     \brief Get the ladder of the hit
     \return The ladder of the Hit
  **/
  char getLadder() const;
  /**
     \brief Get the Z position of the module of the hit
     \return The position of the module
  **/
  char getModule() const;
  /**
     \brief Get the segment of the hit
     \return The segment of the Hit
  **/
  char getSegment() const;
  /**
     \brief Get the strip position of the hit
     \return The strip of the Hit
  **/
  short getStripNumber() const;
  /**
     \brief Get the ID of the hit in the event
     \return The ID of the Hit
  **/
  short getID() const;
  /**
     \brief Get the original particule ID of the hit
     \return The particule ID of the Hit
  **/
  int getParticuleID() const;
  /**
     \brief Get the original particule PT of the hit
     \return The particule's PT of the Hit
  **/
  float getParticulePT() const;
  /**
     \brief Get the distance between particule's origine and IP
     \return The distance in 0,1 mm
  **/
  float getParticuleIP() const;
  /**
     \brief Get the eta value of the original particule
     \return The Eta value
  **/
  float getParticuleETA() const;
  /**
     \brief Get the PHI0 value of the original particule
     \return The PHI0 value
  **/
  float getParticulePHI0() const;
  /**
     \brief Get the X coordinate of the hit
     \return The distance in cm
  **/
  float getX() const;
  /**
     \brief Get the Y coordinate of the hit
     \return The distance in cm
  **/
  float getY() const;
  /**
     \brief Get the Z coordinate of the hit
     \return The distance in cm
  **/
  float getZ() const;
  /**
     \brief Get the X0 of the hit
     \return The value as a float
  **/
  float getX0() const;
  /**
     \brief Get the Y0 of the hit
     \return The value as a float
  **/
  float getY0() const;
  /**
     \brief Get the Z0 of the hit
     \return The value as a float
  **/
  float getZ0() const;


  /**
     \brief Allows to display a Hit as a string
  **/
  friend ostream& operator<<(ostream& out, const Hit& h);
  
};
#endif
