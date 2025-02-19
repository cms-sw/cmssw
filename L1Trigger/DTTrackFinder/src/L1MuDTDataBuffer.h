//-------------------------------------------------
//
/**  \class L1MuDTDataBuffer
 *
 *   Data Buffer:
 *
 *   The Data Buffer stores track 
 *   segment data during the            
 *   execution of the track assembler
 *
 *
 *   $Date: 2007/02/27 11:44:00 $
 *   $Revision: 1.2 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_DATA_BUFFER_H
#define L1MUDT_DATA_BUFFER_H

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

class L1MuDTSectorProcessor;
class L1MuDTTrackSegPhi;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTDataBuffer {

  public:

    /// container to store phi track segments 
    typedef std::vector<L1MuDTTrackSegPhi*> TSPhivector;

    /// constructor
    L1MuDTDataBuffer(const L1MuDTSectorProcessor& );

    /// destructor
    virtual ~L1MuDTDataBuffer();
    
    /// clear Data Buffer
    void reset(); 

    /// get all track segments from the buffer
    const TSPhivector& getTSphi() const { return *m_tsphi; }

    /// get phi track segment of a given station from the buffer
    const L1MuDTTrackSegPhi* getTSphi(int station, int address) const;
    
    /// add new phi track segment to the Data Buffer
    void addTSphi(int adr, const L1MuDTTrackSegPhi&);
    
    /// print all phi track segments which are in the buffer
    void printTSphi() const;
    
    /// return number of non-empty phi track segments
    int numberTSphi() const;
    
  private:

    const L1MuDTSectorProcessor& m_sp;
    TSPhivector*                 m_tsphi;
   
};
  
#endif
