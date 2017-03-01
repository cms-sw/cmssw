#ifndef l1t_ForestHelper_h_
#define l1t_ForestHelper_h_

#include <cassert>
#include <vector>
#include <map>

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"

// If you want to create a new object that you can read and write, use this constructor:
//
//   l1t::ForestHelper x(new L1TPrescalesVetors());
//
// If you wish to read the table from the EventSetup, and will only read, use this:
//
//   const ForestHelper * x = ForestHelper::readFromEventSetup(...)
//   //...
//   delete x;
//
// If you wish to read the table from the EventSetup, but then be able to edit the values locally, use this:
//
//   ForestHelper * x = ForestHelper::readAndWriteFromEventSetup(...)
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

  class ForestHelper {
  public:
    enum {VERSION = 1};

    typedef L1TMuonEndCapForest::DTreeNode DTreeNode;
    typedef L1TMuonEndCapForest::DTree DTree;
    typedef L1TMuonEndCapForest::DForest DForest;
    typedef L1TMuonEndCapForest::DForestColl DForestColl;
    typedef L1TMuonEndCapForest::DForestMap DForestMap;

    ~ForestHelper();

    //ctor if creating a new table (e.g. from XML or python file)
    ForestHelper(L1TMuonEndCapForest * w);
    //create for reading only, from the EventSetup:
    static const ForestHelper * readFromEventSetup(const L1TMuonEndCapForest * es);
    // create for reading and writing, starting from the EventSetup:
    static ForestHelper * readAndWriteFromEventSetup(const L1TMuonEndCapForest * es);

    void initializeFromXML(const char * dirname, const std::vector<int> & modes, int ntrees);

    double evaluate(int mode, const std::vector<double> & data) const;

    // print all the L1 GT  parameters
    void print(std::ostream&) const;

    // access to underlying pointers, mainly for ESProducer:
    const L1TMuonEndCapForest *  getReadInstance() const {return read_;}
    L1TMuonEndCapForest *  getWriteInstance(){return write_; }
       
  private:
    ForestHelper(const L1TMuonEndCapForest * es);
    void useCopy();
    void check_write() { assert(write_); }
    // separating read from write allows for a high-performance read-only mode (as no copy is made):
    const L1TMuonEndCapForest * read_;  // when reading/getting, use this.
    L1TMuonEndCapForest * write_; // when writing/setting, use this.     
    bool we_own_write_;



    void loadTreeFromXMLRecursive(TXMLEngine* xml, XMLNodePointer_t xnode, DTree & tree, unsigned index);
    double evalTreeRecursive(const std::vector<double> & data, const DTree & tree, int index) const;
  };

}
#endif
