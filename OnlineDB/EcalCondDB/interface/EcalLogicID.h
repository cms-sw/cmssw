// $Id: EcalLogicID.h,v 1.1 2006/03/01 23:39:50 egeland Exp $

#ifndef ECALLOGICID_HH
#define ECALLOGICID_HH

#include <string>
// Defines an object to contain the DB logic ID for conditions
class EcalLogicID {
  
 public:

  // Default constructor
  EcalLogicID();

  // Initializing constructor
  EcalLogicID( std::string name,
	       int logicID,
	       int id1 = NULLID,
	       int id2 = NULLID,
	       int id3 = NULLID,
	       std::string mapsTo = "" );

  // Destructor
  virtual ~EcalLogicID();

  // Getters
  std::string getName() const;

  int getLogicID() const;

  int getID1() const;

  int getID2() const;

  int getID3() const;

  std::string getMapsTo() const;

  // Constants
  static const int NULLID = -100;

  // Operators
  friend inline bool operator<(EcalLogicID lhs, EcalLogicID rhs) { return (lhs.logicID < rhs.logicID); }
  
 private:
  std::string name;
  int logicID;
  int id1;
  int id2;
  int id3;
  std::string mapsTo;    

};

#endif
