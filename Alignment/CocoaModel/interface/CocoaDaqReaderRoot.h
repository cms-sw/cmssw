#ifndef CocoaDaqReaderRoot_h
#define CocoaDaqReaderRoot_h
#include "TFile.h"
#include "TTree.h"
#include "Alignment/CocoaDaq/interface/CocoaDaqReader.h"
class CocoaDaqRootEvent;
class AliDaqPosition2D;
class AliDaqPositionCOPS;
class AliDaqTilt;
class AliDaqDistance;
class OpticalAlignMeasurementInfo;

class CocoaDaqReaderRoot : public CocoaDaqReader {
 public:
  CocoaDaqReaderRoot(const std::string& m_inFileName );
  ~CocoaDaqReaderRoot() override;
  bool ReadNextEvent() override;
  bool ReadEvent( int nev ) override;
  void BuildMeasurementsFromOptAlign( std::vector<OpticalAlignMeasurementInfo>& measList ) override;

 public:
  int GetNEvents() const { return nev; }
 private:
  OpticalAlignMeasurementInfo GetMeasFromPosition2D( AliDaqPosition2D* pos2D );
  OpticalAlignMeasurementInfo GetMeasFromPositionCOPS( AliDaqPositionCOPS* posCOPS );
  OpticalAlignMeasurementInfo GetMeasFromTilt( AliDaqTilt* tilt );
  OpticalAlignMeasurementInfo GetMeasFromDist( AliDaqDistance* dist );

  private:
 CocoaDaqRootEvent *theEvent;
 TFile* theFile;
 TTree* theTree;
 int nev;
 int nextEvent;
};

#endif
