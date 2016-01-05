//-------------------------------------------------
//
/**  \class L1MuBMDataBuffer
 *
 *   Data Buffer:
 *
 *   The Data Buffer stores track 
 *   segment data during the            
 *   execution of the track assembler
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_DATA_BUFFER_H
#define L1MUBM_DATA_BUFFER_H

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

class L1MuBMSectorProcessor;
class L1MuBMTrackSegPhi;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMDataBuffer {

  public:

    /// container to store phi track segments 
    typedef std::vector<L1MuBMTrackSegPhi*> TSPhivector;

    /// constructor
    L1MuBMDataBuffer(const L1MuBMSectorProcessor& );

    /// destructor
    virtual ~L1MuBMDataBuffer();
    
    /// clear Data Buffer
    void reset(); 

    /// get all track segments from the buffer
    const TSPhivector& getTSphi() const { return *m_tsphi; }

    /// get phi track segment of a given station from the buffer
    const L1MuBMTrackSegPhi* getTSphi(int station, int address) const;
    
    /// add new phi track segment to the Data Buffer
    void addTSphi(int adr, const L1MuBMTrackSegPhi&);
    
    /// print all phi track segments which are in the buffer
    void printTSphi() const;
    
    /// return number of non-empty phi track segments
    int numberTSphi() const;
    
  private:

    const L1MuBMSectorProcessor& m_sp;
    TSPhivector*                 m_tsphi;
   
};
  
#endif
