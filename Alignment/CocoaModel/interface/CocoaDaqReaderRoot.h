#ifndef CocoaDaqReaderRoot_h
#define CocoaDaqReaderRoot_h
#include "TFile.h"
#include "TTree.h"
#include "Alignment/CocoaDaq/interface/CocoaDaqReader.h"
class AlignmentEvent;
class Position2D;
class Position4x1D;
class Tilt1D;
class Distance;
class OpticalAlignMeasurementInfo;

class CocoaDaqReaderRoot : public CocoaDaqReader {
 public:
  CocoaDaqReaderRoot(const std::string& m_inFileName );
  ~CocoaDaqReaderRoot();
  virtual bool ReadNextEvent();
  virtual bool ReadEvent( int nev );
  virtual void BuildMeasurementsFromOptAlign( std::vector<OpticalAlignMeasurementInfo>& measList );

 public:
  int GetNEvents() const { return nev; }
 private:
  OpticalAlignMeasurementInfo GetMeasFromPosition2D( Position2D* pos2D );
  OpticalAlignMeasurementInfo GetMeasFromPosition4x1D( Position4x1D* pos4x1D );
  OpticalAlignMeasurementInfo GetMeasFromTilt1D( Tilt1D* tilt1D );
  OpticalAlignMeasurementInfo GetMeasFromDist( Distance* dist );

  private:
 AlignmentEvent *theEvent;
 TFile* theFile;
 TTree* theTree;
 int nev;
 int nextEvent;
};

#endif
