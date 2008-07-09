
#ifndef L1GCTPROCESSOR_H_
#define L1GCTPROCESSOR_H_

/*!
 * \author Jim Brooke
 * \date April 2006
 */

/*! \class L1GctProcessor
 * \brief ABC for a GCT trigger data processing unit
 * 
 * A processing unit can be a card, an FPGA, a chunk of firmware. This class
 * exists simply to enforce a common interface. 
 *
 * There should not be any L1GctProcessor* pointers!
 * 
 */


class L1GctProcessor {

 public:

  L1GctProcessor() {};
  virtual ~L1GctProcessor() {};
  
  /// clear internal trigger data buffers
  virtual void reset() = 0;
  
  /// set the input buffers
  virtual void fetchInput() = 0;
  
  /// process the data and set outputs
  virtual void process() = 0;

};

#endif
