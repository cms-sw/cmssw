//-------------------------------------------------
//
/**  \class L1MuGMTLUT
 *
 *   Description: A general-purpose Look-Up-Table Class
 *  
 *   Base class for all LUTs in the Global Muon Trigger Simulation
 *
 *
 *   INPUTS              OUTPUTS
 *
 *
 *             --------
 *             |      |
 *             |      |-- Y(1)
 *    eta(6) --|      |   1
 *      8..13  |      |
 *    phi(8) --|      |
 *      0..7   |      |-- X(1)
 *             |      |   0
 *             --------
 *
 *  vector<unsigned> inp; 
 *  inp [0] = phi
 *  inp [1] = eta
 *
 *  vector<unsigned> out = MyLUT.Lookup (lutidx, inp);
 *  X = out[0];
 *  Y = out[1];
 *
 *
 * 
*/ 
//
//   $Date: 2006/07/07 16:57:06 $
//   $Revision: 1.2 $
//
//   Author :
//   H. Sakulin            HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTLUT_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTLUT_h

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <iostream>
//#include <sstream>
#include <string>
#include <stdlib.h>
#include "stdio.h"

//----------------------
// Base Class Headers --
//----------------------
class L1MuGMTLUTConverter;

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include <L1Trigger/GlobalMuonTrigger/src/L1MuGMTLUTHelpers.h>


#include "FWCore/MessageLogger/interface/MessageLogger.h"


//              ---------------------
//              -- Class Interface --
//              ---------------------
using namespace std;

class L1MuGMTLUT {

  public:  
    typedef pair<string, unsigned> port;

    /// Init and Destruct

    /// default constructor w/o init
    L1MuGMTLUT() : m_initialized(0), m_NLUTS(0), m_UseLookupFunction(true), m_saveFlag(false) {};

    /// constructor with init
    L1MuGMTLUT(const char* name, 
	       const vector<string>&  instances, 
	       const vector<port>& in_widths, 
	       const vector<port>& out_widths, 
	       unsigned vme_addr_width=0, bool distrRAM=false) : m_initialized(0), m_NLUTS(0), m_UseLookupFunction(true), m_saveFlag(false) {
      Init (name, instances, in_widths, out_widths, vme_addr_width, distrRAM); 
      };

    L1MuGMTLUT(const char* name, 
	       const string& instances, 
	       const string& inputs, 
	       const string& outputs, 
	       unsigned vme_addr_width=0, bool distrRAM=false) : m_initialized(0), m_NLUTS(0), m_UseLookupFunction(true), m_saveFlag(false) {
      Init (name, L1MuGMTLUTHelpers::Tokenizer(" ",instances), PortDecoder(inputs), PortDecoder(outputs), 
	    vme_addr_width, distrRAM); 
      };

    /// destructor
    virtual ~L1MuGMTLUT();

    /// main lookup function (unsigned -> unsigned)
    /// all lookup functions go through this one
    inline unsigned LookupPacked (int idx, unsigned ) const;

    /// additional lookup function (vector -> unisgned)
    inline unsigned LookupPacked (int idx, const vector<unsigned>& address) const { 
      return LookupPacked (idx, vec2u ( address, m_Inputs ) ); };

    /// additional lookup function (vector -> vector)
    inline vector<unsigned> Lookup (int idx, const vector<unsigned>& address) const { 
      return Lookup (idx, vec2u ( address, m_Inputs ) ); };

    /// additional lookup function (unsigned -> vector)
    inline vector<unsigned> Lookup (int idx, unsigned address) const {
      return u2vec ( LookupPacked(idx, address), m_Outputs );
    };

    
    /// I/O functions
    
    /// load from LUT file
    void Load (const char* path);
    
    /// save to LUT file
    void Save (const char* path);    

    /// lookup defined as analytic function
    /// to be impemented in derived class
    virtual unsigned LookupFunctionPacked (int idx, unsigned address) const { return 0; };

    /// Add Generate SubClass method
    void MakeSubClass (const char* fname = "", const char* template_file_h = "../interface/L1MuGMTLUT_SubClass.h_template",
		       const char* template_file_cc = "../interface/L1MuGMTLUT_SubClass.cc_template");

    string Name() {return m_name;};

    friend class L1MuGMTLUTConverter;

    /// get the number of Instances
    int numberOfInstances() { return m_NLUTS; };

  protected:

    /// Initialize the LUT
    void Init(const char* name, const vector<string>& instances, 
	      const vector<port>& in_widths, const vector<port>& out_widths, 
	      unsigned vme_addr_width=0, bool distrRAM=false);

    /// generate address or value from composite address or value
    inline unsigned vec2u (const vector <unsigned>& vec, const vector<port>& widths) const;

    /// generate composite address or value from compact unsigned
    inline vector<unsigned> u2vec (unsigned value, const vector<port>& widths) const;

    /// set with single address and value 
    void Set (int idx, unsigned address, unsigned value);  

    class PortDecoder : public vector<port> { 
      typedef vector<port> base;
      public:
      PortDecoder(const vector<port> &pt) : base(pt) {};

      PortDecoder(const string& input){
	// decode string of style "phi(2) eta(4)"
	L1MuGMTLUTHelpers::Tokenizer tok(" ", input);
	for (unsigned int i=0;i<tok.size(); i++) {
	  size_type obrace=tok[i].find("("), cbrace=tok[i].find(")");
	  if (obrace != string::npos && cbrace != string::npos) 
	    push_back( port ( tok[i].substr(0,obrace), (unsigned) atoi (tok[i].substr(obrace+1,cbrace-obrace-1).c_str() ) ) );
	  else 
	    edm::LogWarning("LUTMismatch") << "L1MuGMTLUT::PortDecoder: error decoding port " << tok[i] << endl; 
	}
      };
      string str() {
	string temp;
	for  (unsigned int i=0; i<size();i++) {
	  // ostringstream os; os << (*this)[i].second << ends;
	  //	  temp += (*this)[i].first + "(" + string( os.str() ) + ")";
	  
	  char buf[100]; sprintf(buf,"(%d)",(*this)[i].second);
	  temp += (*this)[i].first + string(buf);

	  if (i!=size()-1) temp += " ";
	}
	return temp;
      };
      
    private:
    };

    bool m_initialized;
    int m_NLUTS; 
    bool m_UseLookupFunction;
    vector <string> m_InstNames;
    vector <vector <unsigned> > m_Contents;
    vector <port> m_Inputs;                  // first port in vector is most significant bits
    vector <port> m_Outputs;
    unsigned m_TotalInWidth;
    unsigned m_TotalOutWidth;
    unsigned m_vme_addr_width;
    bool m_distrRAM;
    string m_name;
    bool m_saveFlag;
};

//--------------------------------------------------------------------------------

unsigned L1MuGMTLUT::vec2u (const vector <unsigned>& vec, const vector<port>& widths) const{
  if (vec.size() != widths.size()) {
    edm::LogWarning("LUTMismatch") << "Error in L1MuGMTLUT::vec2u: number of LUT inputs/outputs does not match definition" << endl;
    return (0);
  }
  
  unsigned value = 0;
  unsigned start_ofs=0;

  for (int i=vec.size()-1; i>=0; i--) {
    if ( vec[i] >= (unsigned) (1 << widths[i].second) ) {
      edm::LogWarning("LUTMismatch") << "Error in L1MuGMTLUT::vec2u: LUT input/output number " << i 
	   << " exceeds range (0 to " << ( (1 << widths[i].second) -1 ) << ")." 
	   << endl;
    }
    else
      value |= vec[i] << start_ofs;
    start_ofs += widths[i].second;
  }
  
  return (value);
}

//--------------------------------------------------------------------------------

vector <unsigned> L1MuGMTLUT::u2vec (unsigned value, const vector<port>& widths) const {
  vector<unsigned> output( widths.size(), 0);
 
  unsigned start_ofs=0;

  for (int i=widths.size()-1; i>=0; i--) {
    int mask = ( (1 << widths[i].second) - 1 ) << start_ofs;
    output[i] = ( value & mask ) >> start_ofs;
    start_ofs += widths[i].second;
  }

  return output;
}

//--------------------------------------------------------------------------------

//
// the main lookup function
// looks up either from the function or the table
// checks the input and result ranges
//
unsigned L1MuGMTLUT::LookupPacked (int idx, unsigned address) const {
  if (! m_initialized) {
    edm::LogWarning("LUTMismatch")  << "Error in L1MuGMTLUT::LookupPacked: LUT not initialized. " << endl;
    return 0;
  }
  if ( idx >= m_NLUTS ) {
    edm::LogWarning("LUTMismatch")  << "Error in L1MuGMTLUT::LookupPacked: LUT index exceeds range (0 to " << ( m_NLUTS -1 ) << ")." 
	 << endl;
    return 0;
  }
  if ( address >= (unsigned) (1 << m_TotalInWidth) ) {
    edm::LogWarning("LUTMismatch")  << "Error in L1MuGMTLUT::LookupPacked: LUT input exceeds range (0 to " << ( (1 << m_TotalInWidth) -1 ) << ")." 
	 << endl;
    return 0;
  }

  unsigned value = 0;
  if (m_UseLookupFunction) {
    value = LookupFunctionPacked (idx, address);
  } else {
    value = m_Contents[idx][address];
  }

  // check range of output
  if ( value >= (unsigned) (1 << m_TotalOutWidth) ) {
    edm::LogWarning("LUTMismatch")  << "Error in L1MuGMTLUT::LookupPacked(): LUT output value " << value
	 << " exceeds range (0 to " << ( (1 << m_TotalOutWidth) -1 ) << ")." 
	 << endl;
    edm::LogWarning("LUTMismatch")  << "  LUT name: " << m_name << endl;
    if (m_UseLookupFunction) 
      edm::LogWarning("LUTMismatch")  << "  Lookup Function has to be corrected!!!" << endl;
    else
      edm::LogWarning("LUTMismatch")  << "  LUT File has to be corrected!!!" << endl;
    return (1 << m_TotalOutWidth) - 1;
  }
  return value;
}
 
#endif












