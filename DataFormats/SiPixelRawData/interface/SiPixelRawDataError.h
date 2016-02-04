#ifndef DataFormats_SiPixelRawDataError_h
#define DataFormats_SiPixelRawDataError_h

//---------------------------------------------------------------------------
//!  \class SiPixelRawDataError
//!  \brief Pixel error -- collection of errors and error information
//!
//!  Class to contain and store all information about errors
//!  
//!
//!  \author Andrew York, University of Tennessee
//---------------------------------------------------------------------------

#include <string>
#include <stdint.h>

class SiPixelRawDataError {
  public:

    /// Default constructor
    SiPixelRawDataError();
    /// Constructor for 32-bit error word
    SiPixelRawDataError(uint32_t errorWord32, const int errorType, int fedId);
    /// Constructor with 64-bit error word and type included (header or trailer word)
    SiPixelRawDataError(uint64_t errorWord64, const int errorType, int fedId);
    /// Destructor
    ~SiPixelRawDataError();

    void setWord32(uint32_t errorWord32);		// function to allow user to input the error word (if 32-bit) after instantiation
    void setWord64(uint64_t errorWord64);		// function to allow user to input the error word (if 64-bit) after instantiation
    void setType(int errorType);			// function to allow user to input the error type after instantiation
    void setFedId(int fedId);			        // function to allow user to input the fedID after instantiation
    void setMessage();				 // function to create an error message based on errorType

    inline uint32_t getWord32() const {return errorWord32_;} // the 32-bit word that contains the error information
    inline uint64_t getWord64() const {return errorWord64_;}    // the 64-bit word that contains the error information
    inline int getType() const {return errorType_;} 	         // the number associated with the error type (26-31 for ROC number errors, 32-33 for calibration errors)
    inline int getFedId() const {return fedId_;} 	         // the fedId where the error occured
    inline std::string getMessage() const {return errorMessage_;}     // the error message to be displayed with the error	

  private:

    uint32_t errorWord32_;
    uint64_t errorWord64_;
    int errorType_;
    int fedId_;
    std::string errorMessage_;
    
};

// Comparison operators
inline bool operator<( const SiPixelRawDataError& one, const SiPixelRawDataError& other) {
  return one.getFedId() < other.getFedId();
}

#endif
