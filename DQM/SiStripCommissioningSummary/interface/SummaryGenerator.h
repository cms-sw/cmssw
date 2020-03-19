#ifndef DQM_SiStripCommon_SummaryGenerator_H
#define DQM_SiStripCommon_SummaryGenerator_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <string>
#include <vector>
#include <map>
#include <cstdint>

class TH1;

/**
   @file DQM/SiStripCommon/interface/SummaryGenerator.h
   @class SummaryGenerator
   @author M.Wingham, R.Bainbridge
   @brief : Fills summary histograms.
*/

class SummaryGenerator {
public:
  // ---------- General ----------

  SummaryGenerator(std::string name);
  virtual ~SummaryGenerator() { ; }

  // Some useful typedefs
  typedef std::pair<float, float> Data;
  typedef std::vector<Data> BinData;
  typedef std::map<std::string, BinData> HistoData;

  /** Creates instance of derived class based on view parameter. */
  static SummaryGenerator* instance(const sistrip::View&);

  // ---------- Contruct name and TObject ----------

  /** Constructs the summary histogram name. */
  static std::string name(const sistrip::RunType&,
                          const sistrip::Monitorable&,
                          const sistrip::Presentation&,
                          const sistrip::View&,
                          const std::string& directory);

  /** Creates instance of derived class based on view parameter. */
  static TH1* histogram(const sistrip::Presentation&, const uint32_t& xbins);

  // ---------- Methods to fill and update histograms ----------

  /** Fills the map that is used to generate the histogram(s). */
  void fillMap(const std::string& top_level_dir,
               const sistrip::Granularity&,
               const uint32_t& key,
               const float& value,
               const float& error = 0.);

  /** Clear the map that is used to generate the histogram(s). */
  void clearMap();

  /** Print contents of map used to generate the histogram(s). */
  void printMap();

  /** Creates simple 1D histogram of the parameter values. */
  void histo1D(TH1&);

  /** Creates a 1D histogram, with the weighted sum of the parameter
      (y-axis) binned as a function of position within the given
      logical structure, which is view-dependent (x-axis). */
  void histo2DSum(TH1&);

  /** Creates a 2D scatter histogram, with individual values of the
      parameter (y-axis) binned as a function of position within the
      given logical structure, which is view-dependent (x-axis). */
  void histo2DScatter(TH1&);

  /** Creates a profile histogram, with the mean and spread of the
      parameter (y-axis) binned as a function of position within the
      given logical structure, which is view-dependent (x-axis). */
  void profile1D(TH1&);

  // ---------- Histogram formatting ----------

  /** Some generic formatting of histogram. */
  void format(const sistrip::RunType&,
              const sistrip::Monitorable&,
              const sistrip::Presentation&,
              const sistrip::View&,
              const std::string& directory,
              const sistrip::Granularity&,
              TH1&);

  /** Optionally set axis label */
  inline void axisLabel(const std::string&);

  /** Retrieve size of map (ie, number of bins). */
  inline uint32_t nBins() const;
  inline uint32_t size() const { return nBins(); }  //@@ TEMPORARY!!!

  // ---------- Utility methods ----------

  /** Returns name of generator object. */
  inline const std::string& myName() const;

protected:
  // ---------- Protected methods ----------

  /** Fills the map used to generate the histogram. */
  virtual void fill(const std::string& top_level_dir,
                    const sistrip::Granularity&,
                    const uint32_t& key,
                    const float& value,
                    const float& error);

protected:
  // ---------- Protected member data ----------

  /** A map designed to holds a set of values. The map containing
      these values should be indexed by a key.*/
  HistoData map_;

  float entries_;

  float max_;

  float min_;

  std::string label_;

private:
  // ---------- Private member data ----------

  std::string myName_;
};

const std::string& SummaryGenerator::myName() const { return myName_; }
uint32_t SummaryGenerator::nBins() const { return map_.size(); }
void SummaryGenerator::axisLabel(const std::string& label) { label_ = label; }

#endif  // DQM_SiStripCommon_SummaryGenerator_H
