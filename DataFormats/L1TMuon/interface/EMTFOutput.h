// Class for EMTF DAQ readout

#ifndef __l1t_EMTF_output_h__
#define __l1t_EMTF_output_h__

#include <vector>
#include <boost/cstdint.hpp> // For uint64_t and other types. Also found in DataFormats/L1Trigger/src/classes.h

#include "EMTF/AMC13Header.h"
#include "EMTF/MTF7Header.h"
#include "EMTF/EventHeader.h"
#include "EMTF/Counters.h"
#include "EMTF/ME.h"
#include "EMTF/RPC.h"
#include "EMTF/SP.h"
#include "EMTF/EventTrailer.h"
#include "EMTF/MTF7Trailer.h"
#include "EMTF/AMC13Trailer.h"

// All comments below apply equally to classes in EMTF/ directory - AWB 28.01.16

namespace l1t {
  class EMTFOutput {
    
  public:
    explicit EMTFOutput(uint64_t dataword); // What does this do? - AWB 28.01.16
    
  // Empty constructor - should we include classes and collections? - AWB 28.01.16
  EMTFOutput() :
    hasAMC13Header(false), hasMTF7Header(false), hasEventHeader(false), hasCounters(false), numME(0), numRPC(0), 
      numSP(0), hasEventTrailer(false), hasMTF7Trailer(false), hasAMC13Trailer(false), format_errors(0), dataword(-99)
      {};
    
  /* // Fill constructor - should we included a copy constructor for classes and collections (vectors)? - AWB 28.01.16 */
  /* EMTFOutput(emtf::AMC13Header class_AMC13Header, emtf::SPCollection coll_SPCollection) : */
  /*   AMC13Header(class_AMC13Header), SPCollection(coll_SPCollection), format_errors(0), dataword(-99) */
  /*   {}; */
    
    virtual ~EMTFOutput() {};

    void set_AMC13Header(emtf::AMC13Header bits)    { AMC13Header = bits;  hasAMC13Header  = true; };
    void set_MTF7Header(emtf::MTF7Header bits)      { MTF7Header  = bits;  hasMTF7Header   = true; };
    void set_EventHeader(emtf::EventHeader bits)    { EventHeader = bits;  hasEventHeader  = true; };
    void set_Counters(emtf::Counters bits)          { Counters    = bits;  hasCounters     = true; };
    void push_ME(emtf::ME bits)                     { MECollection.push_back(bits);  numME  += 1; };
    void push_RPC(emtf::RPC bits)                   { RPCCollection.push_back(bits); numRPC += 1; };
    void push_SP(emtf::SP bits)                     { SPCollection.push_back(bits);  numSP  += 1; };
    void set_EventTrailer(emtf::EventTrailer bits)  { EventTrailer = bits; hasEventTrailer = true; };
    void set_MTF7Trailer(emtf::MTF7Trailer bits)    { MTF7Trailer = bits;  hasMTF7Trailer  = true; };
    void set_AMC13Trailer(emtf::AMC13Trailer bits)  { AMC13Trailer = bits; hasAMC13Trailer = true; };
    void add_format_error()                         { format_errors += 1; };
    void set_dataword(uint64_t bits)                { dataword = bits;               };
    
    bool HasAMC13Header()                   { return hasAMC13Header;  };
    bool HasMTF7Header()                    { return hasMTF7Header;   };
    bool HasEventHeader()                   { return hasEventHeader;  };
    bool HasCounters()                      { return hasCounters;     };
    int  NumSP()                            { return numSP;           };
    int  NumRPC()                           { return numRPC;          };
    int  NumME()                            { return numME;           };
    bool HasAMC13Trailer()                  { return hasAMC13Trailer; };
    bool HasMTF7Trailer()                   { return hasMTF7Trailer;  };
    bool HasEventTrailer()                  { return hasEventTrailer; };
    emtf::AMC13Header GetAMC13Header()      { return AMC13Header;   };
    emtf::MTF7Header GetMTF7Header()        { return MTF7Header;    };
    emtf::EventHeader GetEventHeader()      { return EventHeader;   };
    emtf::Counters GetCounters()            { return Counters;      };
    emtf::MECollection GetMECollection()    { return MECollection;  };
    emtf::RPCCollection GetRPCCollection()  { return RPCCollection; };
    emtf::SPCollection GetSPCollection()    { return SPCollection;  };
    emtf::EventTrailer GetEventTrailer()    { return EventTrailer;  };
    emtf::MTF7Trailer GetMTF7Trailer()      { return MTF7Trailer;   };
    emtf::AMC13Trailer GetAMC13Trailer()    { return AMC13Trailer;  };
    const int Format_Errors() const { return format_errors; };
    const uint64_t Dataword() const { return dataword; };
    
  private:
    bool hasAMC13Header; 
    bool hasMTF7Header; 
    bool hasEventHeader; 
    bool hasCounters; 
    int  numME; 
    int  numRPC; 
    int  numSP; 
    bool hasEventTrailer; 
    bool hasMTF7Trailer; 
    bool hasAMC13Trailer; 
    emtf::AMC13Header AMC13Header;
    emtf::MTF7Header MTF7Header;
    emtf::EventHeader EventHeader;
    emtf::Counters Counters;
    emtf::MECollection MECollection;
    emtf::RPCCollection RPCCollection;
    emtf::SPCollection SPCollection;
    emtf::EventTrailer EventTrailer;
    emtf::MTF7Trailer MTF7Trailer;
    emtf::AMC13Trailer AMC13Trailer;
    int  format_errors;
    uint64_t dataword; // Should this be more or fewer bits? - AWB 28.01.16
    
  }; // End class EMTFOutput

  // Define a vector of EMTFOutput
  typedef std::vector<EMTFOutput> EMTFOutputCollection;

} // End namespace l1t

#endif /* define __l1t_EMTF_output_h__ */
