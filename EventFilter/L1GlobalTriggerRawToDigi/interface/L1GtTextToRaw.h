#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GtTextToRaw_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GtTextToRaw_h

/**
 * \class L1GtTextToRaw
 * 
 * 
 * Description: generate raw data from dumped text file.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna 
 * 
 *
 */

// system include files
#include <memory>
#include <string>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations

// class declaration
class L1GtTextToRaw : public edm::one::EDProducer<> {
public:
  /// constructor(s)
  explicit L1GtTextToRaw(const edm::ParameterSet&);

  /// destructor
  ~L1GtTextToRaw() override;

private:
  /// beginning of job stuff
  void beginJob() override;

  /// clean the text file, if needed
  virtual void cleanTextFile();

  /// get the size of the record
  virtual int getDataSize();

  /// loop over events
  void produce(edm::Event&, const edm::EventSetup&) override;

  /// end of job stuff
  void endJob() override;

private:
  /// file type for the text file
  std::string m_textFileType;

  /// file name for the text file
  std::string m_textFileName;

  /// raw event size (including header and trailer) in units of 8 bits
  int m_rawDataSize;

  /// FED ID for the system
  int m_daqGtFedId;

  /// the file itself
  std::ifstream m_textFile;
};

#endif  // EventFilter_L1GlobalTriggerRawToDigi_L1GtTextToRaw_h
