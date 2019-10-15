// Class for EMTF DAQ readout

#ifndef __l1t_EMTF_output_h__
#define __l1t_EMTF_output_h__

#include <vector>

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
#include <cstdint>

// All comments below apply equally to classes in EMTF/ directory - AWB 28.01.16

namespace l1t {
  class EMTFDaqOut {
  public:
    explicit EMTFDaqOut(uint64_t dataword);  // What does this do? - AWB 28.01.16

    // Empty constructor - should we include classes and collections? - AWB 28.01.16
    EMTFDaqOut()
        : hasAMC13Header(false),
          hasMTF7Header(false),
          hasEventHeader(false),
          hasCounters(false),
          numME(0),
          numRPC(0),
          numSP(0),
          hasEventTrailer(false),
          hasMTF7Trailer(false),
          hasAMC13Trailer(false),
          format_errors(0),
          dataword(-99){};

    /* // Fill constructor - should we included a copy constructor for classes and collections (vectors)? - AWB 28.01.16 */
    /* EMTFDaqOut(emtf::AMC13Header class_AMC13Header, emtf::SPCollection coll_SPCollection) : */
    /*   AMC13Header(class_AMC13Header), SPCollection(coll_SPCollection), format_errors(0), dataword(-99) */
    /*   {}; */

    virtual ~EMTFDaqOut(){};

    void set_AMC13Header(emtf::AMC13Header bits) {
      AMC13Header = bits;
      hasAMC13Header = true;
    }
    void set_MTF7Header(emtf::MTF7Header bits) {
      MTF7Header = bits;
      hasMTF7Header = true;
    }
    void set_EventHeader(emtf::EventHeader bits) {
      EventHeader = bits;
      hasEventHeader = true;
    }
    void set_Counters(emtf::Counters bits) {
      Counters = bits;
      hasCounters = true;
    }
    void set_MECollection(emtf::MECollection bits) {
      MECollection = bits;
      numME = MECollection.size();
    }
    void push_ME(emtf::ME bits) {
      MECollection.push_back(bits);
      numME += 1;
    }
    void set_RPCCollection(emtf::RPCCollection bits) {
      RPCCollection = bits;
      numRPC = RPCCollection.size();
    }
    void push_RPC(emtf::RPC bits) {
      RPCCollection.push_back(bits);
      numRPC += 1;
    }
    void set_SPCollection(emtf::SPCollection bits) {
      SPCollection = bits;
      numSP = SPCollection.size();
    }
    void push_SP(emtf::SP bits) {
      SPCollection.push_back(bits);
      numSP += 1;
    }
    void set_EventTrailer(emtf::EventTrailer bits) {
      EventTrailer = bits;
      hasEventTrailer = true;
    }
    void set_MTF7Trailer(emtf::MTF7Trailer bits) {
      MTF7Trailer = bits;
      hasMTF7Trailer = true;
    }
    void set_AMC13Trailer(emtf::AMC13Trailer bits) {
      AMC13Trailer = bits;
      hasAMC13Trailer = true;
    }
    void add_format_error() { format_errors += 1; }
    void set_dataword(uint64_t bits) { dataword = bits; }

    bool HasAMC13Header() const { return hasAMC13Header; }
    bool HasMTF7Header() const { return hasMTF7Header; }
    bool HasEventHeader() const { return hasEventHeader; }
    bool HasCounters() const { return hasCounters; }
    int NumSP() const { return numSP; }
    int NumRPC() const { return numRPC; }
    int NumME() const { return numME; }
    bool HasAMC13Trailer() const { return hasAMC13Trailer; }
    bool HasMTF7Trailer() const { return hasMTF7Trailer; }
    bool HasEventTrailer() const { return hasEventTrailer; }
    emtf::AMC13Header GetAMC13Header() const { return AMC13Header; }
    emtf::MTF7Header GetMTF7Header() const { return MTF7Header; }
    emtf::EventHeader GetEventHeader() const { return EventHeader; }
    emtf::Counters GetCounters() const { return Counters; }
    emtf::MECollection GetMECollection() const { return MECollection; }
    emtf::RPCCollection GetRPCCollection() const { return RPCCollection; }
    emtf::SPCollection GetSPCollection() const { return SPCollection; }
    emtf::EventTrailer GetEventTrailer() const { return EventTrailer; }
    emtf::MTF7Trailer GetMTF7Trailer() const { return MTF7Trailer; }
    emtf::AMC13Trailer GetAMC13Trailer() const { return AMC13Trailer; }
    const emtf::AMC13Header* PtrAMC13Header() const { return &AMC13Header; }
    const emtf::MTF7Header* PtrMTF7Header() const { return &MTF7Header; }
    const emtf::EventHeader* PtrEventHeader() const { return &EventHeader; }
    const emtf::Counters* PtrCounters() const { return &Counters; }
    const emtf::MECollection* PtrMECollection() const { return &MECollection; }
    const emtf::RPCCollection* PtrRPCCollection() const { return &RPCCollection; }
    const emtf::SPCollection* PtrSPCollection() const { return &SPCollection; }
    const emtf::EventTrailer* PtrEventTrailer() const { return &EventTrailer; }
    const emtf::MTF7Trailer* PtrMTF7Trailer() const { return &MTF7Trailer; }
    const emtf::AMC13Trailer* PtrAMC13Trailer() const { return &AMC13Trailer; }
    int Format_Errors() const { return format_errors; }
    uint64_t Dataword() const { return dataword; }

  private:
    bool hasAMC13Header;
    bool hasMTF7Header;
    bool hasEventHeader;
    bool hasCounters;
    int numME;
    int numRPC;
    int numSP;
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
    int format_errors;
    uint64_t dataword;  // Should this be more or fewer bits? - AWB 28.01.16

  };  // End class EMTFDaqOut

  // Define a vector of EMTFDaqOut
  typedef std::vector<EMTFDaqOut> EMTFDaqOutCollection;

}  // End namespace l1t

#endif /* define __l1t_EMTF_output_h__ */
