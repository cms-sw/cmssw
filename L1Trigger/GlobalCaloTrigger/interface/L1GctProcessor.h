
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

#include <vector>

class L1GctProcessor {
public:
  L1GctProcessor() : m_verbose(false), m_bx(0), m_bxStart(0), m_numOfBx(1){};
  virtual ~L1GctProcessor(){};

  /// complete reset of processor
  inline void reset() {
    m_bxStart = 0;
    m_numOfBx = 1;
    m_bx = 0;
    resetPipelines();
    resetProcessor();
    setupObjects();
  }

  /// set the input buffers
  virtual void fetchInput() = 0;

  /// process the data and set outputs
  virtual void process() = 0;

  /// define the bunch crossing range to process
  inline void setBxRange(const int firstBx, const int numberOfBx) {
    m_bxStart = firstBx;
    m_numOfBx = numberOfBx;
    resetPipelines();
  }

  /// clear input data buffers and process a new bunch crossing
  inline void setNextBx(const int bxnum) {
    if ((bxnum - m_bxStart >= 0) && (bxnum - m_bxStart < m_numOfBx)) {
      m_bx = bxnum;
    } else {
      m_bx = 0;
    }
    resetProcessor();
    setupObjects();
  }

  /// Method to check the setup for this processor. Returns true by default.
  bool setupOk() const { return true; }

  /// control output messages
  void setVerbose() { m_verbose = true; }
  void setTerse() { m_verbose = false; }

protected:
  /// Separate reset methods for the processor itself and any data stored in pipelines
  virtual void resetProcessor() = 0;
  virtual void resetPipelines() = 0;

  /// Initialise inputs with null objects for the correct bunch crossing if required
  virtual void setupObjects() = 0;

  /// Support for multiple beam crossing operation
  inline int bxMin() const { return m_bxStart; }
  inline int bxMax() const { return (m_bxStart + m_numOfBx - 1); }
  inline int numOfBx() const { return m_numOfBx; }
  inline int bxAbs() const { return m_bx; }
  inline int bxRel() const { return (m_bx - m_bxStart); }

  template <class T>
  struct Pipeline {
    std::vector<T> contents;
    unsigned entriesPerBx;

    Pipeline() : contents(1), entriesPerBx(1) {}
    Pipeline(const unsigned size) : contents(size), entriesPerBx(size) {}

    void resize(const unsigned size) { entriesPerBx = size; }

    void reset(const unsigned nBx) {
      contents.clear();
      contents.resize(nBx * entriesPerBx);
    }

    void store(const T& thisBx, const int bxNum) { contents.at(bxNum) = thisBx; }

    void store(const std::vector<T>& thisBx, const int bxNum) {
      unsigned pos = entriesPerBx * bxNum;
      for (unsigned i = 0; i < entriesPerBx; i++) {
        contents.at(pos++) = thisBx.at(i);
      }
    }
  };

  /// Flag to control output messages
  bool m_verbose;

private:
  /// Support for multiple beam crossing operation
  int m_bx;
  ///
  int m_bxStart;
  int m_numOfBx;
};

#endif
