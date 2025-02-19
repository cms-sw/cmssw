//-------------------------------------------------
//
/**  \class L1TriggerLutFile
 *
 *   Auxiliary class to handle Look-up table files
 *
 *
 *   $Date: 2010/01/19 18:39:54 $
 *   $Revision: 1.2 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1_TRIGGER_LUT_FILE_H
#define L1_TRIGGER_LUT_FILE_H

//---------------
// C++ Headers --
//---------------

#include <string>
#include <fstream>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1TriggerLutFile {

  public:
 
    /// constructor
    L1TriggerLutFile(const std::string name = "" );
    
    /// copy constructor
    L1TriggerLutFile(const L1TriggerLutFile& ); 

    /// destructor
    virtual ~L1TriggerLutFile();
 
    /// assignment operator
    L1TriggerLutFile& operator=(const L1TriggerLutFile&);

    /// return filename
    inline const std::string& getName() const { return m_file; }
    
    /// open file
    int open();
    
    /// return status of file stream
    inline bool good() { return m_fin.good(); }

    /// return status of file stream
    inline bool bad() { return m_fin.bad(); }
    
    /// close file
    inline void close() { m_fin.close(); }

    /// read and ignore n lines from file
    void ignoreLines(int n);
    
    /// read one integer from file
    int readInteger();

    /// read one hex from file
    int readHex();

    /// read one string from file
    std::string readString();

  private:

    std::ifstream m_fin;       // input file stream
    std::string   m_file;      // file name

}; 

#endif
