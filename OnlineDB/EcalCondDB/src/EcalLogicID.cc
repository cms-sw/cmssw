// $Id: EcalLogicID.cc,v 1.2 2008/05/08 13:14:37 fra Exp $

#include <string>

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

//Constructors
EcalLogicID::EcalLogicID()
{
}

EcalLogicID::EcalLogicID( std::string name,
			  int logicID,
			  int id1,
			  int id2,
			  int id3,
			  std::string mapsTo )
{
  this->name = name;
  this->logicID = logicID;
  this->id1 = id1;
  this->id2 = id2;
  this->id3 = id3;
  if (mapsTo.size() == 0) {
    this->mapsTo = name;
  } else {
    this->mapsTo = mapsTo;
  }
}

// Destructor
EcalLogicID::~EcalLogicID()
{
}

// Getters
std::string EcalLogicID::getName() const
{
  return name;
}

int EcalLogicID::getLogicID() const
{
  return logicID;
}

int EcalLogicID::getID1() const
{
  return id1;
}

int EcalLogicID::getID2() const
{
  return id2;
}

int EcalLogicID::getID3() const
{
  return id3;
}

std::string EcalLogicID::getMapsTo() const
{
  return mapsTo;
}
