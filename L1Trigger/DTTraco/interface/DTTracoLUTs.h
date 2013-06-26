//-------------------------------------------------
//
/**  \class DTTracoLUTs
 *
 *   Look-up tables for traco angle and position
 *
 *
 *   $Date: 2006/07/19 10:24:02 $
 *   $Revision: 1.1 $
 *
 *   \author S. Vanini - INFN Padova
 */
//
//--------------------------------------------------
#ifndef DT_TRACO_LUTS_H
#define DT_TRACO_LUTS_H

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <string> 

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef std::vector<unsigned short int> LUT;

class DTTracoLUTs{

  public:
//    typedef static std::vector<unsigned short int> LUT;
 
    /// constructor
    DTTracoLUTs(std::string filename);

    /// destructor
    virtual ~DTTracoLUTs();

    /// reset look-up tables
    void reset();
    
    /// load look-up tables
    int load();

    /// print look-up tables
    void print() const;

    /// get radial angle from traco position and flag: 0=outer, 1=inner, 2=correl.
    unsigned short int getPhiRad(int pos, int qualflag) const;

    /// get psi angle from traco k parameter
    unsigned short int getPsi(int ang) const;

    /// return bending angle from pos and ang
    unsigned short int getBendAng(int pos, int ang, int qualflag) const;

    /// return number of entries in the LUT
    inline int size_psiLUT() const { return psi_lut.size(); }
    inline int size_phiLUT(int i) const { return phi_lut[i].size(); }

  private:

    LUT phi_lut[3];  // phi rad: elem.0: inner; elem.1: outer; elem.2: corr.
    LUT psi_lut;     // psi 
    std::string _testfile;

};

#endif
