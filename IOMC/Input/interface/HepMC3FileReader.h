#ifndef IOMC_Input_HepMC3FileReader_h
#define IOMC_Input_HepMC3FileReader_h

/** \class HepMC3FileReader
 *
 *  Sequential reader of HepMC3 event files. The concrete format is deduced
 *  from the file itself, so the ASCII HepMC3 format, the ASCII HepMC2 format,
 *  LHEF and HEPEVT are all accepted, as well as their compressed flavours.
 *  A list of files can be given, in which case they are read one after the
 *  other as a single stream of events.
 */

#include <memory>
#include <string>
#include <vector>

namespace HepMC3 {
  class GenEvent;
  class Reader;
}  // namespace HepMC3

class HepMC3FileReader {
public:
  explicit HepMC3FileReader(std::vector<std::string> fileNames);
  ~HepMC3FileReader();

  /// read the next event of the stream, returns nullptr when the input is exhausted
  /// the returned event is owned by the reader and is valid until the next call
  const HepMC3::GenEvent* fillCurrentEventData();

  void printCurrentEvent() const;

private:
  /// open the next file of the list, returns false if there is none left
  bool openNextFile();

  const std::vector<std::string> fileNames_;
  unsigned int nextFile_;
  std::shared_ptr<HepMC3::Reader> reader_;
  std::unique_ptr<HepMC3::GenEvent> evt_;
};

#endif
