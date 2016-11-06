#ifndef SiStripApvGainFromFileBuilder_H
#define SiStripApvGainFromFileBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

#include <map>
#include <vector>
#include <stdint.h>


class SiStripDetCabling;


class SiStripApvGainFromFileBuilder : public edm::EDAnalyzer {

 public:

  //enum ExceptionType = { NotConnected, ZeroGainFromScan, NegativeGainFromScan };

  typedef std::map< uint32_t,float > Gain;

  typedef struct {
                   uint32_t  det_id;
                   uint16_t  offlineAPV_id;
                   int       onlineAPV_id;
                   int       FED_id;
                   int       FED_ch;
                   int       i2cAdd;
                   bool      is_connected;
                   bool      is_scanned;
                   float     gain_from_scan;
                   float     gain_in_db;
                 } Summary;

  /** Brief Constructor.
   */ 
  explicit SiStripApvGainFromFileBuilder( const edm::ParameterSet& iConfig);

  /** Brief Destructor performing the memory cleanup.
   */ 
  ~SiStripApvGainFromFileBuilder();

  /** Brief One dummy-event analysis to create the database record.
   */ 
  virtual void analyze(const edm::Event& , const edm::EventSetup& );


 private:
  edm::FileInPath gfp_;          /*!< File Path for the ideal geometry. */
  edm::FileInPath tfp_;          /*!< File Path for the tickmark scan with the APV gains. */
  double gainThreshold_;         /*!< Threshold for accepting the APV gain in the tickmark scan file. */
  double dummyAPVGain_;          /*!< Dummy value for the APV gain. */
  bool putDummyIntoUncabled_;    /*!< Flag for putting the dummy gain in the channels not actuall cabled. */
  bool putDummyIntoUnscanned_;   /*!< Flag for putting the dummy gain in the chennals not scanned. */
  bool putDummyIntoOffChannels_; /*!< Flag for putting the dummy gain in the channels that were off during the tickmark scan. */
  bool putDummyIntoBadChannels_; /*!< Flag for putting the dummy gain in the channels with negative gains. */
  bool outputMaps_;              /*!< Flag for dumping the internal maps on ASCII files. */
  bool outputSummary_;           /*!< Flag for dumping the summary of the exceptions during the DB filling. */


  edm::ESHandle<SiStripDetCabling> detCabling_; /*!< Description of detector cabling. */

  /** Brief Maps [det_id <--> gains] arranged per APV indexes.
   */
  std::vector<Gain*> gains_;          /*!< Mapping channels with positive heights. */
  std::vector<Gain*> negative_gains_; /*!< Mapping channels sending bad data. */
  std::vector<Gain*> null_gains_;     /*!< Mapping channels switched off during the scan. */

  /** Brief Collection of the channels entered in the DB without exceptions.
   * The channels whose APV gain has been input in the DB straight from the 
   * tickmark scan are collected in the summary vector. The summary list is
   * dumped in the SiStripApvGainSummary.txt at the end of the job. 
   */
  std::vector<Summary> summary_;    /*!< Collection of channel with no DB filling exceptions. */ 

  /** Brief Collection of the exceptions encountered when filling the DB. 
   * An exception occur for all the non-cabled channels ( no gain associated
   * in the tikmark file) and for all the channels that were off ( with zero
   * gain associated) or sending corrupted data (with negative values in the
   * tickmark file). At the end of the job the exception summary is dumped in
   * SiStripApvGainExceptionSummary.txt.
   */
  std::vector<Summary> ex_summary_; /*!< Collection of DB filling exceptions. */


  /** Brief Read the ASCII file containing the tickmark gains.
   * This method reads the ASCII files that contains the tickmark heights for 
   * every APV. The heights are first translated into gains, dividing by 640,
   * then are stored into maps to be associated to the detector ids. Maps are
   * created for every APV index. 
   * Negative and Zero heights, yielding to a non physical gain, are stored 
   * into separate maps.
   *   Negative gain: channels sending bad data at the tickmark scan. 
   *   Zero gain    : channels switched off during the tickmark scan. 
   */ 
  void read_tickmark(void);

  /** Brief Returns the mapping among channels and gain heights for the APVs.
   * This method searchs the mapping of detector Ids <-> gains provided. If the
   * mapping exists for the requested APV it is returned; if not a new empty
   * mapping is created, inserted and retruned. The methods accepts onlineIDs
   * running from 0 to 5. 
   */
  Gain* get_map(std::vector<Gain*>* maps, int onlineAPV_id);

  /** Brief Dumps the internal mapping on a ASCII files.
   * This method dumps the detector id <-> gain maps into acii files separated
   * for each APV. The basenmae of for the acii file has to be provided as a
   * input parameter.
   */ 
  void output_maps(std::vector<Gain*>* maps, const char* basename) const;

  /** Brief Dump the exceptions summary on a ASCII file.
   * This method dumps the online coordinate of the channels for which  there
   * was an exception for filling the database record. Exceptions are the non
   * cabled modules, the channels that were off during the tickmark scan, the
   * channels sending corrupted data duirng the tickmark scan. These exceptions
   * have been solved putting a dummy gain into the DB record or putting a zero
   * gain.
   */
  void output_summary() const;

  /** Brief Format the output line for the channel summary.
   */
  void format_summary (std::stringstream& line, Summary summary) const;

  /** Brief Find the gain value for a pair det_id, APV_id in the internal maps.
   */
  bool gain_from_maps(uint32_t det_id, int onlineAPV_id, float& gain);
  void gain_from_maps(uint32_t det_id, uint16_t totalAPVs, std::vector< std::pair<int,float> >& gain) const;

  /** Brief Convert online APV id into offline APV id.
   */
  int online2offline(uint16_t onlineAPV_id, uint16_t totalAPVs) const; 
};

#endif
