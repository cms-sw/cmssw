#ifndef _TRACK_H_
#define _TRACK_H_

#include <vector>

using namespace std;

/**
   \brief A track is defined with its curve (PT), D0, Phi0, Eta0 and Z0 parameters
**/
class Track{

 private:
  double curve;
  double d0;
  double phi0;
  double eta0;
  double z0;
  double w_xy;
  double w_rz;
  vector<int> stub_ids;

 public:
  /**
     \brief Default Constructor : all values set to 0
  **/
  Track();
  /**
     \brief Constructor
     \param c The PT of the track
     \param d The D0 of the track
     \param p The PHI0 of the track
     \param p_a The Eta0 of the track
     \param p_b The Z0 of the track
     \param w_xy The weight of the XY-retina maximum
     \param w_rz The weight of the RZ-retina maximum
  **/
  Track(double c, double d, double p, double p_a, double p_b, double Wxy=-1., double Wrz=-1.);
  /**
     \brief Copy Constructor
  **/
  Track(const Track&);

  /**
     \brief Set the PT of the track
     \param p The PT of the track
  **/
  void setCurve(double p);
  /**
     \brief Set the D0 of the track
     \param d The D0 of the track
  **/
  void setD0(double d);
  /**
     \brief Set the Phi of the track
     \param p The Phi of the track
  **/
  void setPhi0(double p);
  /**
     \brief Set the Eta of the track
     \param e The Eta of the track
  **/
  void setEta0(double e);
  /**
     \brief Set the Z0 of the track
     \param z The Z0 of the track
  **/
  void setZ0(double z);
  /**
     \brief Set the weight of the XY-retina maximum
     \param Wxy The weight of the XY-retina maximum
  **/
  void setWxy(double Wxy);
  /**
     \brief Set the weight of the RZ-retina maximum
     \param Wrz The weight of the RZ-retina maximum
  **/
  void setWrz(double Wrz);

  /**
     \brief Add a stub to the list of stubs used to create the track
     \param s The ID of the stub
  **/
  void addStubIndex(int s);
  /**
     \brief Get the list of the index of stubs used to compute the track
     \return A vector with the list of index
  **/
  vector<int> getStubs();

  /**
     \brief CLear the list of stubs used to create the track
  **/
  void clearStubList();

  /**
     \brief Get the PT of the track
     \return The PT of the track
  **/
  double getCurve();
  /**
     \brief Get the D0 of the track
     \return The D0 of the track
  **/
  double getD0();
  /**
     \brief Get the Phi of the track
     \return The Phi of the track
  **/
  double getPhi0();
  /**
     \brief Get the Eta of the track
     \return The Eta of the track
  **/
  double getEta0();
  /**
     \brief Get the Z0 of the track
     \return The Z0 of the track
  **/
  double getZ0();
  /**
     \brief Get the weight of the XY-retina maximum
     \return The weight of the XY-retina maximum
  **/
  double getWxy();
  /**
     \brief Get the weight of the RZ-retina maximum
     \return The weight of the RZ-retina maximum
  **/
  double getWrz();

};
#endif
