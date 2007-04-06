// Helper function for output of Tracker Det Id info that involves next data:
//   - Subdetector: TIB/TID/TOB/TEC
//   - Layer/Wheel
//   - Module Number
//
// Author : Samvel Khalatyan (samvel at fnal dot gov)
// Created: 12/04/06
// Licence: GPL

#ifndef TRACKER_DETID_OSTREAM_H
#define TRACKER_DETID_OSTREAM_H

#include <iosfwd>

class DetId;

std::ostream &operator<< ( std::ostream &roOut, const DetId &roDETID);

#endif // TRACKER_DETID_OSTREAM_H
