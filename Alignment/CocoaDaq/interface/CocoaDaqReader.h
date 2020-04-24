#ifndef CocoaDaqReader_h
#define CocoaDaqReader_h
#include <string>
#include <vector>

class OpticalAlignMeasurementInfo;


class CocoaDaqReader {
 public:
  CocoaDaqReader(){ };
  static CocoaDaqReader* GetDaqReader(){
    return theDaqReader; }
  static void SetDaqReader( CocoaDaqReader* reader );

  virtual ~CocoaDaqReader(){ };

  virtual bool ReadNextEvent() = 0;
  virtual bool ReadEvent( int nev ){ return false; };
  virtual void BuildMeasurementsFromOptAlign( std::vector<OpticalAlignMeasurementInfo>& measList );

 public:
  int GetNEvents() const { return nev; }

 private:
  static CocoaDaqReader* theDaqReader;

 protected:
  int nev;
  int nextEvent;
};

#endif
