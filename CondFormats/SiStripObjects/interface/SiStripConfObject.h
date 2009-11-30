#ifndef SiStripConfObject_h
#define SiStripConfObject_h

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

/**
 * Author M. De Mattia - 16/11/2009
 *
 * Simple class used to store configuration values. <br>
 * It stores a vector<string> and a vector<int> containing the name and value of the parameter. <br>
 * The put and get methods are provied to store and access the parameters. <br>
 * The printSummary and printDebug method return both the full list of parameters. <br>
 * The vectors with names and parameters are public. <br>
 * WARNING: the get method assumes that the elements in the two vectors correspond (vector<string>[i] <-> vector<int>[i]).
 * This is the case if the values are input with the put method.
 */

class SiStripConfObject
{
 public:
  SiStripConfObject() {}

  bool put( const string & name, const int & value );
  int get( const string & name );

  /// Prints the number of ranges as well as the value of singleLatency and singleMode
  void printSummary(std::stringstream & ss) const;
  /// Prints the full list of all ranges and corresponding values of latency and mode
  void printDebug(std::stringstream & ss) const;

  vector<string> names;
  vector<int> values;
};

#endif
