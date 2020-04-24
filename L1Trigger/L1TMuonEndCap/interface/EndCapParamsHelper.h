#ifndef l1t_EndCapParamsHelper_h_
#define l1t_EndCapParamsHelper_h_

#include <cassert>
#include <vector>
#include <map>

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"

// If you want to create a new object that you can read and write, use this constructor:
//
//   l1t::EndCapParamsHelper x(new L1TPrescalesVetors());
//
// If you wish to read the table from the EventSetup, and will only read, use this:
//
//   const EndCapParamsHelper * x = EndCapParamsHelper::readFromEventSetup(...)
//   //...
//   delete x;
//
// If you wish to read the table from the EventSetup, but then be able to edit the values locally, use this:
//
//   EndCapParamsHelper * x = EndCapParamsHelper::readAndWriteFromEventSetup(...)
//   //...
///  delete x;
//
// but there's a performance penalty as a copy is made.


//
// This class does not take over responsibility for deleting the pointers it is
// initialized with.  That is responsibility of the calling code.
//

#include "TXMLEngine.h"

namespace l1t {

  class EndCapParamsHelper {
  public:
    enum {VERSION = 1};
    
    ~EndCapParamsHelper();

    //ctor if creating a new table (e.g. from XML or python file)
    EndCapParamsHelper(L1TMuonEndCapParams * w);
    //create for reading only, from the EventSetup:
    static const EndCapParamsHelper * readFromEventSetup(const L1TMuonEndCapParams * es);
    // create for reading and writing, starting from the EventSetup:
    static EndCapParamsHelper * readAndWriteFromEventSetup(const L1TMuonEndCapParams * es);

    void SetPtAssignVersion (unsigned version) {write_->PtAssignVersion_ = version;}
    void SetFirmwareVersion (unsigned version) {write_->firmwareVersion_ = version;}
    // "PhiMatchWindowSt1" arbitrarily re-mapped to Primitive conversion (PC LUT) version
    // because of rigid CondFormats naming conventions - AWB 02.06.17
    void SetPrimConvVersion (unsigned version) {write_->PhiMatchWindowSt1_ = version;}
    
    unsigned GetPtAssignVersion() const {return read_->PtAssignVersion_;}
    unsigned GetFirmwareVersion() const {return read_->firmwareVersion_;}
    unsigned GetPrimConvVersion() const {return read_->PhiMatchWindowSt1_;}

    // print all the parameters
    void print(std::ostream&) const;

    // access to underlying pointers, mainly for ESProducer:
    const L1TMuonEndCapParams *  getReadInstance() const {return read_;}
    L1TMuonEndCapParams *  getWriteInstance(){return write_; }
       
  private:
    EndCapParamsHelper(const L1TMuonEndCapParams * es);
    void useCopy();
    void check_write() { assert(write_); }
    // separating read from write allows for a high-performance read-only mode (as no copy is made):
    const L1TMuonEndCapParams * read_;  // when reading/getting, use this.
    L1TMuonEndCapParams * write_; // when writing/setting, use this.     
    bool we_own_write_;
  };

}
#endif
