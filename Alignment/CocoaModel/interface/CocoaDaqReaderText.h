#ifndef CocoaDaqReaderText_h
#define CocoaDaqReaderText_h
#include <string>
#include <vector>

#include "Alignment/CocoaDaq/interface/CocoaDaqReader.h"
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
class OpticalAlignMeasurementInfo;

class CocoaDaqReaderText : public CocoaDaqReader {
public:
  CocoaDaqReaderText(const std::string& fileName);
  ~CocoaDaqReaderText() override;

  bool ReadNextEvent() override;
  void BuildMeasurementsFromOptAlign(std::vector<OpticalAlignMeasurementInfo>& measList) override;

public:
  int GetNEvents() const { return nev; }

protected:
  int nev;
  int nextEvent;
  ALIFileIn theFilein;
};

#endif
