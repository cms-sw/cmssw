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
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <memory>
#include <string>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations

// class declaration
class L1GtTextToRaw : public edm::EDProducer
{

public:

    /// constructor(s)
    explicit L1GtTextToRaw(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GtTextToRaw();

private:

    /// beginning of job stuff
    virtual void beginJob();

    /// clean the text file, if needed
    virtual void cleanTextFile();

    /// get the size of the record
    virtual int getDataSize();

    /// loop over events
    virtual void produce(edm::Event&, const edm::EventSetup&);

    /// end of job stuff
    virtual void endJob();

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

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GtTextToRaw_h
