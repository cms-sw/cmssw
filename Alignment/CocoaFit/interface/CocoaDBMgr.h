#ifndef CocoaDBMgr_HH
#define CocoaDBMgr_HH
//-*- C++ -*-
//
// Package:    Alignment/CocoaFit
// Class:      CocoaDBMgr
// 
/*

 Description: 
*/
//


#include <stdexcept>
#include <string>
#include <iostream>
#include <map>

class Event;
class EventSetup;
class Entry;
class Alingments;
//#include "FWCore/Framework/interface/EventSetup.h"

class OpticalAlignments;
class OpticalAlignInfo;
class OpticalAlignMeasurements;
class Alignments;
class AlignmentErrors;
class AlignTransform;
class AlignTransformError;

class OpticalObject;


class CocoaDBMgr
{
private:
  CocoaDBMgr();

 public:
  static CocoaDBMgr* getInstance();
  
  ~CocoaDBMgr() { }
  
 public:
  bool DumpCocoaResults(); 

 private:
  OpticalAlignments* BuildOpticalAlignments();
  double GetEntryError( const Entry* entry );
  double GetEntryError( const Entry* entry1, const Entry* entry2 );

  OpticalAlignInfo GetOptAlignInfoFromOptO( OpticalObject* opto );

  std::pair< Alignments*,AlignmentErrors*> BuildAlignments(bool bDT);
  AlignTransform* GetAlignInfoFromOptO( OpticalObject* opto);
  AlignTransformError* GetAlignInfoErrorFromOptO( OpticalObject* opto);

 private:
  static CocoaDBMgr* instance;

};

#endif
