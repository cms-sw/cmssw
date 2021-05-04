#ifndef CalibTracker_SiStripCommon_SiStripFedIdListReader_h
#define CalibTracker_SiStripCommon_SiStripFedIdListReader_h

#include <fstream>
#include <ostream>
#include <vector>
#include <cstdint>

class SiStripFedIdListReader;

/** */
std::ostream& operator<<(std::ostream&, const SiStripFedIdListReader&);

/**
   @class SiStripFedIdListReader
   @author R.Bainbridge
*/
class SiStripFedIdListReader {
public:
  /** */
  explicit SiStripFedIdListReader(std::string filePath);

  /** */
  explicit SiStripFedIdListReader(const SiStripFedIdListReader&);

  /** */
  SiStripFedIdListReader& operator=(const SiStripFedIdListReader&);

  /** */
  ~SiStripFedIdListReader();

  /** */
  inline const std::vector<uint16_t>& fedIds() const;

private:
  /** */
  explicit SiStripFedIdListReader() { ; }

  std::ifstream inputFile_;

  std::vector<uint16_t> fedIds_;
};

const std::vector<uint16_t>& SiStripFedIdListReader::fedIds() const { return fedIds_; }

#endif
