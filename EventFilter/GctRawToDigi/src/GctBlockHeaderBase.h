#ifndef GCTBLOCKHEADERBASE_H_
#define GCTBLOCKHEADERBASE_H_

#include <vector>
#include <map>
#include <ostream>
#include <string>

/// Abstract base class for block headers.
class GctBlockHeaderBase
{
 public:
  GctBlockHeaderBase(const uint32_t data=0):d(data){};
  GctBlockHeaderBase(const unsigned char * data);
  virtual ~GctBlockHeaderBase() {};
  
  /// this is a valid block header
  bool valid() const { return ( blockLength_.find(this->id()) != blockLength_.end() ); }

  /// the raw header data
  uint32_t data() const { return d; }

  /// the block ID
  virtual unsigned int id() const = 0;

  /// number of time samples
  virtual unsigned int nSamples() const = 0;
  
  /// bunch crossing ID
  virtual unsigned int bcId() const = 0;

  /// event ID
  virtual unsigned int eventId() const = 0;

  /// fundamental block length (for 1 time sample)
  unsigned int length() const;

  /// block name
  std::string name() const;

 protected:
 
  uint32_t d; /// The header.

  static std::map<unsigned int, unsigned int> blockLength_;  // fundamental size of a block (ie for 1 readout sample)
  static std::map<unsigned int, std::string> blockName_;  // block name!
};

std::ostream& operator<<(std::ostream& os, const GctBlockHeaderBase& h);

#endif /*GCTBLOCKHEADERBASE_H_*/
