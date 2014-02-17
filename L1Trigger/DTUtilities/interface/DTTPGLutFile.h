//-------------------------------------------------
//
/**  \class DTTPGLutFile
 *
 *   Auxiliary class to handle Look-up table files
 *
 *
 *   $Date: 2007/10/23 13:44:22 $
 *   $Revision: 1.1 $
 *
 *   \author   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef DTTPG_LUT_FILE_H
#define DTTPG_LUT_FILE_H

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

class DTTPGLutFile {

  public:
 
    /// constructor
    DTTPGLutFile(const std::string name = "" );
    
    /// copy constructor
    DTTPGLutFile(const DTTPGLutFile& ); 

    /// destructor
    virtual ~DTTPGLutFile();
 
    /// assignment operator
    DTTPGLutFile& operator=(const DTTPGLutFile&);

    /// return filename
    inline std::string getName() const { return m_file; }
    
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
 
	std::ifstream m_fin;	   // input file stream
	std::string   m_file;      // file name

}; 

#endif
