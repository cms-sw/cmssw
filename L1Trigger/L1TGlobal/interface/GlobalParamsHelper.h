#ifndef GLOBALPARAMSHELPER_H__
#define GLOBALPARAMSHELPER_H__

#include <cassert>
#include "CondFormats/L1TObjects/interface/L1TGlobalParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalParametersRcd.h"

// If you want to create a new object that you can read and write, use this constructor:
//
//   l1t::GlobalParamsHelper x(new L1TPrescalesVetors());
//
// If you wish to read the table from the EventSetup, and will only read, use this:
//
//   const GlobalParamsHelper * x = GlobalParamsHelper::readFromEventSetup(...)
//   //...
//   delete x;
//
// If you wish to read the table from the EventSetup, but then be able to edit the values locally, use this:
//
//   GlobalParamsHelper * x = GlobalParamsHelper::readAndWriteFromEventSetup(...)
//   //...
///  delete x;
//
// but there's a performance penalty as a copy is made.


//
// This class does not take over responsibility for deleting the pointers it is
// initialized with.  That is responsibility of the calling code.
//


namespace l1t {

  class GlobalParamsHelper {
  public:
    enum {VERSION = 1};
    
    ~GlobalParamsHelper();

    //ctor if creating a new table (e.g. from XML or python file)
    GlobalParamsHelper(L1TGlobalParameters * w);
    //create for reading only, from the EventSetup:
    static const GlobalParamsHelper * readFromEventSetup(const L1TGlobalParameters * es);
    // create for reading and writing, starting from the EventSetup:
    static GlobalParamsHelper * readAndWriteFromEventSetup(const L1TGlobalParameters * es);

    //int bxMaskDefault() const { return read_->bxmask_default_; };
    //void setBxMaskDefault(int value) { check_write(); write_->bxmask_default_ = value; };


    /// get / set the number of bx in hardware
    inline  int totalBxInEvent() const {
        return read_->m_totalBxInEvent;
    }

    void setTotalBxInEvent(const int&);


    /// get / set the number of physics trigger algorithms
    inline unsigned int numberPhysTriggers() const {
        return read_->m_numberPhysTriggers;
    }

    void setNumberPhysTriggers(const unsigned int&);


    ///  get / set the number of L1 muons received by GT
    inline unsigned int numberL1Mu() const {
        return read_->m_numberL1Mu;
    }

    void setNumberL1Mu(const unsigned int&);

    ///  get / set the number of L1 e/gamma objects received by GT
    inline unsigned int numberL1EG() const {
        return read_->m_numberL1EG;
    }

    void setNumberL1EG(const unsigned int&);


    ///  get / set the number of L1  jets received by GT
    inline unsigned int numberL1Jet() const {
        return read_->m_numberL1Jet;
    }

    void setNumberL1Jet(const unsigned int&);


    ///  get / set the number of L1 tau  received by GT
    inline unsigned int numberL1Tau() const {
        return read_->m_numberL1Tau;
    }

    void setNumberL1Tau(const unsigned int&);



    ///   get / set the number of condition chips in GTL
    inline unsigned int numberChips() const {
        return read_->m_numberChips;
    }

    void setNumberChips(const unsigned int&);

    ///   get / set the number of pins on the GTL condition chips
    inline unsigned int pinsOnChip() const {
        return read_->m_pinsOnChip;
    }

    void setPinsOnChip(const unsigned int&);

    ///   get / set the correspondence "condition chip - GTL algorithm word"
    ///   in the hardware
    inline const std::vector<int>& orderOfChip() const {
        return read_->m_orderOfChip;
    }

    void setOrderOfChip(const std::vector<int>&);

    /// print all the L1 GT  parameters
    void print(std::ostream&) const;



    // access to underlying pointers, mainly for ESProducer:
    const L1TGlobalParameters *  getReadInstance() const {return read_;}
    L1TGlobalParameters *  getWriteInstance(){return write_; }
       
  private:
    GlobalParamsHelper(const L1TGlobalParameters * es);
    void useCopy();
    void check_write() { assert(write_); }
    // separating read from write allows for a high-performance read-only mode (as no copy is made):
    const L1TGlobalParameters * read_;  // when reading/getting, use this.
    L1TGlobalParameters * write_; // when writing/setting, use this.     
    bool we_own_write_;
  };

}
#endif
