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

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GtTextToRaw.h"

// system include files
#include <vector>
#include <iostream>
#include <iomanip>

#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"







#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructor(s)
L1GtTextToRaw::L1GtTextToRaw(const edm::ParameterSet& pSet)
{

    m_textFileType = pSet.getUntrackedParameter<std::string>("TextFileType", "VmeSpyDump");

    LogDebug("L1GtTextToRaw")
    << "\nText file type: " << m_textFileType << "\n"
    << std::endl;

    m_textFileName = pSet.getUntrackedParameter<std::string>("TextFileName", "gtfe.dmp");

    LogDebug("L1GtTextToRaw")
    << "\nText file name: " << m_textFileName << "\n"
    << std::endl;

    // event size
    m_rawDataSize = pSet.getUntrackedParameter<int>("RawDataSize");

    LogDebug("L1GtTextToRaw")
    << "\nRaw data size (units of 8 bits): " << m_rawDataSize
    << "\n  If negative value, the size is retrieved from the trailer." << "\n"
    << std::endl;

    // FED Id for GT DAQ record
    // default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    // default value: assume the DAQ record is the last GT record 
    m_daqGtFedId = pSet.getUntrackedParameter<int>(
                       "DaqGtFedId", FEDNumbering::MAXTriggerGTPFEDID);

    LogDebug("L1GtTextToRaw")
    << "\nFED Id for DAQ GT record: "
    << m_daqGtFedId << " \n"
    << std::endl;

    // open test file
    m_textFile.open(m_textFileName.c_str(), std::ios::in);
    if( !m_textFile.good() ) {
        throw cms::Exception("NotFound")
        << "\nError: failed to open text file = " << m_textFileName << "\n"
        << std::endl;
    }

    //
    produces<FEDRawDataCollection>();

}

// destructor
L1GtTextToRaw::~L1GtTextToRaw()
{

    // empty now

}

// member functions

// beginning of job stuff
void L1GtTextToRaw::beginJob()
{

    cleanTextFile();

}

// clean the text file, if needed
void L1GtTextToRaw::cleanTextFile()
{

    LogDebug("L1GtTextToRaw")
    << "\nCleaning the text file\n"
    << std::endl;



}

// get the size of the record
int L1GtTextToRaw::getDataSize()
{

    LogDebug("L1GtTextToRaw")
    << "\nComputing raw data size with getRecordSize() method."
    << std::endl;

    int rawDataSize = 0;

    LogDebug("L1GtTextToRaw")
    << "\nComputed raw data size: " << rawDataSize
    << std::endl;


    return rawDataSize;

}


// method called to produce the data
void L1GtTextToRaw::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // get the size of the record

    int rawDataSize = 0;

    if (m_rawDataSize < 0) {
        rawDataSize = getDataSize();
    } else {
        rawDataSize = m_rawDataSize;

    }

    // define new FEDRawDataCollection
    // it contains ALL FEDs in an event
    std::auto_ptr<FEDRawDataCollection> fedRawColl(new FEDRawDataCollection);

    FEDRawData& rawData = fedRawColl->FEDData(m_daqGtFedId);
    // resize, GT raw data record has variable length,
    // depending on active boards (read in GTFE)
    rawData.resize(rawDataSize);


    LogDebug("L1GtTextToRaw")
    << "\n Size of raw data: " << rawData.size() << "\n"
    << std::endl;


    // read the text file
    // the file must have one 64 bits per line (usually in hex format)
    // events are separated by empty lines
    
    std::string lineString;

    boost::uint64_t lineInt = 0ULL;
    int sizeL = sizeof(lineInt);

    int fedBlockSize = 8; // block size in bits for FedRawData
    int maskBlock = 0xff; // fedBlockSize and maskBlock must be consistent

    int iLine = 0;

    while (std::getline(m_textFile, lineString)) {

        if (lineString.empty()) {
            break;
        }

        // convert string to int
        std::istringstream iss(lineString);

        iss >> std::hex >> lineInt;

        LogTrace("L1GtTextToRaw")
        << std::dec << std::setw(4) << std::setfill('0') << iLine << ": " 
        << std::hex << std::setw(sizeL*2) << lineInt 
        << std::dec << std::setfill(' ')
        << std::endl;

        // copy data
        for (int j = 0; j < sizeL; j++) {
            char blockContent = (lineInt >> (fedBlockSize * j)) & maskBlock;
            rawData.data()[iLine*sizeL + j] = blockContent;
        }


        ++iLine;
    }

    // put the raw data in the event
    iEvent.put(fedRawColl);
}


//
void L1GtTextToRaw::endJob()
{

    // empty now
}


// static class members
