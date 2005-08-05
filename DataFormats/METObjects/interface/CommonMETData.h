#ifndef METOBJECTS_COMMON_MET_DATA_H
#define METOBJECTS_COMMON_MET_DATA_H

 /** \class CommonMETData
 *
 * \short Structure containing data common to all types of MET
 *
 * CommonMETData is an structure that is inherited by all types of MET
 * It holds information common to all types of MET.
 * More to be added...
 *
 * \author Michael Schmitt, The University of Florida
 *
 * \version   1st Version June 14, 2005.
 *
 ************************************************************/

const int MET_LABEL_LEN = 24;

struct CommonMETData {

  char label[MET_LABEL_LEN];

  double met;
  double metx;
  double mety;
  double metz;
  double sumet;
  double phi;

};

#endif // METOBJECTS_COMMON_MET_DATA_H
