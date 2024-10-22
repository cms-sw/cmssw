#ifndef jhugon_StatisticsFile_h
#define jhugon_StatisticsFile_h

// system include files
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

// user include files

namespace csctf_analysis {
  class StatisticsFile {
  public:
    StatisticsFile();
    StatisticsFile(const std::string);
    ~StatisticsFile();

    void Create(const std::string);
    void Close() { statFileOut.close(); }

    void WriteStatistics(TrackHistogramList tfHistList, TrackHistogramList refHistList);

    ofstream statFileOut;

  private:
  };
}  // namespace csctf_analysis
#endif
