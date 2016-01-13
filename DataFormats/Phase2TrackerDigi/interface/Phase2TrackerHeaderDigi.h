#ifndef DataFormats_Phase2TrackerDigi_Phase2TrackerHeaderDigi_H
#define DataFormats_Phase2TrackerDigi_Phase2TrackerHeaderDigi_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"

class Phase2TrackerHeaderDigi
{
    public:
        Phase2TrackerHeaderDigi() :
            dataFormatVersion_(0),
            debugMode_(0),
            readoutMode_(0),
            conditionData_(0),
            dataType_(0),
            glibStatusCode_(0),
            numberOfCBC_(0) {}

        Phase2TrackerHeaderDigi( Phase2Tracker::Phase2TrackerFEDHeader head ) :  
            dataFormatVersion_(head.getDataFormatVersion()),
            debugMode_(head.getDebugMode()),
            readoutMode_(head.getReadoutMode()),
            conditionData_(head.getConditionData()),
            dataType_(head.getDataType()),
            glibStatusCode_(head.getGlibStatusCode()),
            numberOfCBC_(head.getNumberOfCBC()) {}

        ~Phase2TrackerHeaderDigi() {}

        inline uint8_t getDataFormatVersion() const { return dataFormatVersion_; }
        inline uint8_t getDebugMode() const { return debugMode_; }
        inline uint8_t getReadoutMode() const { return readoutMode_; }
        inline uint8_t getConditionData() const { return conditionData_; }
        inline uint8_t getDataType() const { return dataType_; }
        inline uint64_t getGlibStatusCode() const { return glibStatusCode_; }
        inline uint16_t getNumberOfCBC() const { return numberOfCBC_; }

    private:
        uint8_t dataFormatVersion_; 
        uint8_t debugMode_;       
        uint8_t readoutMode_; 
        uint8_t conditionData_;      
        uint8_t dataType_;           
        uint64_t glibStatusCode_;    
        uint16_t numberOfCBC_;       
};

#endif
