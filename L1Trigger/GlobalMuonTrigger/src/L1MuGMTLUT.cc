//-------------------------------------------------
//
//   \class L1MuGMTLUT
/**
 *   A general-purpose Look-Up-Table Class
 *   Base class for all LUTs in the GMT
 * 
*/ 
//
//   $Date: 2012/02/10 14:19:28 $
//   $Revision: 1.7 $
//
//   Author :
//   H. Sakulin            HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//---------------
// C++ Headers --
//---------------

#include <L1Trigger/GlobalMuonTrigger/src/L1MuGMTLUT.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"

using namespace std;

L1MuGMTLUT::~L1MuGMTLUT() {
  if (! m_UseLookupFunction ) {
    
    // de-allocate vectors
    // no destuction needed for vector< vector <unsigned> >

    // for (int i=0;i < m_NLUTS; i++) 
    //   m_Contents[i].clear();
    // m_Contents.clear();
  }
}

void L1MuGMTLUT::Init(const char* name, const vector<string>& instances, 
		      const vector<port>& in_widths, const vector<port>& out_widths, 
		      unsigned vme_addr_width, bool distrRAM) 
{
  m_name = name;
  m_InstNames = instances;
  m_NLUTS = instances.size();

  m_Inputs = in_widths;
  m_TotalInWidth = 0; 
  for (unsigned i=0; i< in_widths.size(); i++) m_TotalInWidth += in_widths[i].second;

  m_Outputs = out_widths;
  m_TotalOutWidth = 0; 
  for (unsigned i=0; i< out_widths.size(); i++) m_TotalOutWidth += out_widths[i].second;

  m_vme_addr_width = vme_addr_width;
  m_distrRAM = distrRAM;

  if (m_distrRAM && (m_TotalInWidth != vme_addr_width) ) {
    edm::LogWarning("AddressMismatch")
         << "L1MuGMTLUT::Init(): for distributed RAM the GMT (Input) address width " 
         << "has to match the VME address width. Core Generation will not work."; 
  }

  m_GeneralLUTVersion = L1MuGMTConfig::getVersionLUTs();
  m_initialized = true; 
}


void L1MuGMTLUT::Save(const char* path) {
  if (! m_initialized) {
    edm::LogWarning("LUTNotInitialized") << "L1MuGMTLUT::Save: LUT not initialized. ";
    return;
  }

  m_saveFlag = true;

  ofstream of(path);
  of << "// This is a CMS L1 Global Muon Trigger .lut file. " << endl;
  of << "//  " << endl;
  of << "// It defines a set of look-up-tables(LUTs) of the same type (same inputs and outputs) but" << endl;
  of << "// with different default contents. For example a certain type of LUT can have different" << endl;
  of << "// default values for DT for RPC and for CSC muons. " << endl;
  of << "//  " << endl;
  of << "// NAME           gives the name of the LUT. It should match the base name (name without '.lut')" << endl;
  of << "//                of the LUT file." << endl;
  of << "//                When deriving a C++ sub-class the name is used case sensitive." << endl;
  of << "//                In generated VHDL code the name is used in lower case." << endl;
  of << "//  " << endl;
  of << "// INSTANCES      is the list of instances of the LUT with different default values." << endl;
  of << "//                the lists consists of identifiers for each of the instances separated by spaces." << endl;
  of << "//                the identifiers can be made up of characters that are valid in VHDL lables. " << endl;
  of << "//                In the VHDL code they are used to label the different instances." << endl;
  of << "//  " << endl;
  of << "//                In C++ and VHDL the instance of a LUT is selected by an integer index where 0 " << endl;
  of << "//                corresponds to the leftmost identifier. Integer indices are also used in the CONTENTS_XX" << endl;
  of << "//                statements in this file." << endl;
  of << "//  " << endl;
  of << "// LUT_INPUTS     is the (space-separated) list of inputs of the LUT. Each input is specified in the form" << endl;
  of << "//                <input_name>(<number_of_bits>) where <input_name> is the name of the input and" << endl;
  of << "//                <number_of_bits> is the number of bits of the input. <input_name> has to be a valid" << endl;
  of << "//                identifier both in C++ and in VHDL. In C++ it is represented as an unsigned int." << endl;
  of << "//  " << endl;
  of << "//                All LUT inputs together make up the address of the corresponding memory." << endl;
  of << "//                The first input in the list corresponds to the most-significant bits of the address." << endl;
  of << "//  " << endl;
  of << "// LUT_OUTPUTS    is the (space-separated) list of outputs of the LUT. Each output is specified in the form" << endl;
  of << "//                <output_name>(<number_of_bits>) where <output_name> is the name of the output and" << endl;
  of << "//                <number_of_bits> is the number of bits of the output. <output_name> has to be a valid" << endl;
  of << "//                identifier both in C++ and in VHDL. In C++ it is represented as an unsigned int." << endl;
  of << "//  " << endl;
  of << "//                All LUT outputs together make up the data of the corresponding memory." << endl;
  of << "//                The first output in the list corresponds to the most-significant data bits." << endl;
  of << "//  " << endl;
  of << "// VME_ADDR_WIDTH is the address width of the LUT memory when accessed via VME. When the LUT " << endl;
  of << "//                is implemented as dual-port block RAM, the VME address with can be different " << endl;
  of << "//                from the address width given by the sum of the input widths. It can be larger or" << endl;
  of << "//                smaller by a factor of 2^n. When the LUT is implemented as dual-port distributed RAM," << endl;
  of << "//                the VME address width has to match the input width. " << endl;
  of << "//  " << endl;
  of << "// DISTRIBUTED_RAM is 1 if the LUT is to be implemented in distributed RAM and 0 for Block RAM " << endl;
  of << "//                 Note that for distributed RAM the address width on the GMT side (sum of input widths)" << endl;
  of << "//                 has to match the vme_addr_width" << endl;
  of << "//  " << endl;
  of << "// CONTENTS_XX    specifies the default contents of instance with index XX (see INSTANCES)." << endl;
  of << "//                contents are specified as decimal numbers, one number per line." << endl;
  of << "//  " << endl;
  of << "// Hannes Sakulin / HEPHY Vienna, 2003" << endl;
  of << "// "<< endl;
  
  of << "NAME = " << m_name << endl;
  of << "INSTANCES =" ;
  for (unsigned i=0; i<m_InstNames.size(); i++) of << " " << m_InstNames[i];
  of << endl;
  of << "LUT_INPUTS = " << PortDecoder(m_Inputs).str() <<  endl;
  of << "LUT_OUTPUTS = " << PortDecoder(m_Outputs).str() << endl;
  of << "VME_ADDR_WIDTH = " << m_vme_addr_width << endl;
  of << "DISTRIBUTED_RAM = " << (m_distrRAM?"1":"0") << endl;
  for (int i=0; i<m_NLUTS; i++) {
    of << "// " << m_InstNames[i] << endl;
    of << "CONTENTS_" << i << " = ";
    for (unsigned addr = 0; addr < (unsigned) (1<<m_TotalInWidth); addr ++) {
      of << LookupPacked (i, addr) << endl;
    }
    of << endl;
  }

  m_saveFlag = false;

}


// Rules for file
//
// NAME = VALUE1 [VALUE2 [...]]
// All header variables have to be set before first contents variable
// comments start with "//" and go up to end of line
// 
// when writing back, comments will be save first


//--------------------------------------------------------------------------------

void L1MuGMTLUT::Set (int idx, unsigned address, unsigned value) {
  if (! m_initialized) {
     edm::LogWarning("LUTNotInitialized") << "L1MuGMTLUT::Set: LUT not initialized. ";
    return;
  }

  if ( idx >= m_NLUTS ) {
    edm::LogWarning("LUTRangeViolation") << "L1MuGMTLUT::Set: LUT index exceeds range (0 to " << ( m_NLUTS -1 ) << ").";
    return;
  }
  if ( address >= (unsigned)(1 << m_TotalInWidth) ) {
    edm::LogWarning("LUTRangeViolation") << "Error in L1MuGMTLUT::Set: LUT input exceeds range (0 to " << ( (1 << m_TotalInWidth) -1 ) << ").";
    return;
  }
  if ( value >= (unsigned)(1 << m_TotalOutWidth) ) {
    edm::LogWarning("LUTRangeViolation") << "Error in L1MuGMTLUT::Set: LUT output exceeds range (0 to " << ( (1 << m_TotalOutWidth) -1 ) << ")." ;
    return;
  }
  m_Contents[idx][address] = value;
}
 


void L1MuGMTLUT::Load(const char* path) {
  string lf_name("");
  vector <string> lf_InstNames;
  vector <port> lf_Inputs;
  vector <port> lf_Outputs;
  unsigned lf_vme_addr_width=0;
  bool lf_distrRAM=false;
  vector<string> lf_comments;

  ifstream in (path);
  const int sz=1000; char buf[sz];


  // read header
  
  while ( in.getline(buf, sz) ) {
    string line(buf);
    string::size_type i=0;
    if ( (i=line.find("//")) != string::npos) {
      lf_comments.push_back( line.substr(i) ); // save comments
      line.erase(i); // and strip
    }
    L1MuGMTLUTHelpers::Tokenizer tok("=", line);
    if (tok.size() == 2) {
      L1MuGMTLUTHelpers::replace(tok[0], " ","", false); // skip spaces
      L1MuGMTLUTHelpers::replace(tok[0], "\t","", false); // skip tabs

      L1MuGMTLUTHelpers::replace(tok[1], "\t", " ", false); // convert tabs to spaces
      L1MuGMTLUTHelpers::replace(tok[1], "  ", " ", true); // skip multiple spaces
      tok[1].erase(0, tok[1].find_first_not_of(" ")); // skip leading spaces
      tok[1].erase(tok[1].find_last_not_of(" ")+1); // skip trailing spaces
            
      if (tok[0] == "NAME") lf_name = tok[1];
      else if (tok[0] == "INSTANCES") { lf_InstNames = L1MuGMTLUTHelpers::Tokenizer(" ",tok[1]); }
      else if (tok[0] == "LUT_INPUTS") lf_Inputs = PortDecoder(tok[1]);
      else if (tok[0] == "LUT_OUTPUTS") lf_Outputs = PortDecoder(tok[1]); 
      else if (tok[0] == "VME_ADDR_WIDTH") lf_vme_addr_width = atoi(tok[1].c_str()); 
      else if (tok[0] == "DISTRIBUTED_RAM") lf_distrRAM = ( atoi(tok[1].c_str()) == 1 ); 
    }
    if (tok[0].find("CONTENTS") != string::npos) break;
  }

  if (!m_initialized) { // then initialize
    Init(lf_name.c_str(), lf_InstNames, lf_Inputs, lf_Outputs, lf_vme_addr_width, lf_distrRAM);    
  }
  else { // verify compatibility
    if (m_name != lf_name ||
	m_InstNames != lf_InstNames ||
	m_Inputs != lf_Inputs ||
	m_Outputs != lf_Outputs ||
	m_vme_addr_width != lf_vme_addr_width ||
	m_distrRAM != lf_distrRAM) {
      edm::LogWarning("LUTParmasMismatch") 
          << "L1MuGMTLUT::Load: error: parameters in file do not match configuration of LUT. Load failed.";
      return;
    }
  }
  
  if (m_UseLookupFunction) {
    // allocate vectors
    m_Contents.resize( m_NLUTS );
    for (int i=0;i < m_NLUTS; i++) 
      m_Contents[i].resize( 1 << m_TotalInWidth );

      // switch to table mode
    m_UseLookupFunction = false;
  }  
  
  // continue to read contents (first line should be in buf)
  int maxrows = 1 << m_TotalInWidth;
  int row = 0;
  int current_index = -1;
  do {
    string line(buf);
    string::size_type i=0;
    if ( (i=line.find("//")) != string::npos) line.erase(i); // strip comments
    L1MuGMTLUTHelpers::Tokenizer tok("=", line);

    if (tok.size() == 2 && tok[0].find("CONTENTS") != string::npos) {
      L1MuGMTLUTHelpers::Tokenizer tok1("_",tok[0]);
      if (tok1.size() !=2) {
	edm::LogWarning("LUTParsingProblem") << "L1MuGMTLUT::Load: error parsing contents tag " << tok[0] << ".";
	break;
      }	

      istringstream is(tok1[1].c_str());
      int newindex; 
      is >> newindex;
      if (newindex != current_index+1)
	edm::LogWarning("LUTParsingProblem") << "L1MuGMTLUT::Load: warning: LUTS in LUT file are not in order.";

      if (newindex > m_NLUTS-1) {
	edm::LogWarning("LUTParsingProblem") << "L1MuGMTLUT::Load: warning: LUT file contains LUT with too high index (" 
	     << tok[0] 
	     << "). max = " << m_NLUTS << " skipping.";
	newindex = -1;
      }
      current_index = newindex;

      if (row != 0) {
	if ( row < maxrows ) 
	  edm::LogWarning("LUTParsingProblem") << "L1MuGMTLUT::Load: warning: LUT file only contains part of LUT contents.";
	row = 0;
      }
      istringstream is1(tok[1].c_str());
      unsigned value;
      if (is1 >> value) {
	if (current_index!=-1)
	  Set (current_index, row++, value);	
      }
    }
    else {
      istringstream is1(line.c_str());
      unsigned value;
      if (is1 >> value) {
	if (row < maxrows) {
	  if (current_index!=-1)
	    Set (current_index, row++, value);	
	}
	else
	  edm::LogWarning("LUTParsingProblem") 
                   << "L1MuGMTLUT::Load: warning: LUT file only contains LUT with too many entries. skipping.";
      }
    }
  } while ( in.getline(buf, sz) );
}

//--------------------------------------------------------------------------------
// Generate a SubClass .h file
//
// to be used manually during code development

void L1MuGMTLUT::MakeSubClass(const char* fname, const char* template_file_h, 
			      const char* template_file_cc) {

  // prepare parts
  string ins_name (m_name);
  string ins_name_upper = L1MuGMTLUTHelpers::upperCase (ins_name);
  string ins_instance_string;
  string ins_instances_enum;
  for (unsigned i=0; i<m_InstNames.size(); i++) {
    if (i!=0) ins_instance_string += ' ';
    ins_instance_string +=  m_InstNames[i];

    if (i!=0) ins_instances_enum += ", ";
    ins_instances_enum += m_InstNames[i];
  }
  char ins_vme[100]; 
  sprintf (ins_vme, "%d", m_vme_addr_width);

  char ins_distr_RAM[10];
  sprintf (ins_distr_RAM, "%s", m_distrRAM?"true":"false");
  
  
  string ins_input_decl_list, ins_input_list, ins_input_addr_list;
  for (unsigned i=0; i<m_Inputs.size(); i++) {
    ins_input_decl_list += string(", unsigned ") + m_Inputs[i].first;
    ins_input_list += string(", ") + m_Inputs[i].first;
    char tmp[100]; sprintf (tmp, " ,addr[%d]", i);
    ins_input_addr_list += string(tmp); 
  }
  
  //  string ins_lookup_functions;
  ostringstream os;
  for (unsigned i=0; i<m_Outputs.size(); i++) {
    os << "  /// specific lookup function for " <<  m_Outputs[i].first << endl;
    os << "  unsigned SpecificLookup_" << m_Outputs[i].first << " (int idx" << ins_input_decl_list << ") const {" << endl;
    os << "    vector<unsigned> addr(" << m_Inputs.size() << ");" << endl;
    for (unsigned j=0; j< m_Inputs.size(); j++) {
      os << "    addr[" << j << "] = " << m_Inputs[j].first << ";" << endl;
    }
    os << "    return Lookup(idx, addr) [" << i << "];" << endl;
    os << "  };" << endl << endl;
  }
  os << "  /// specific lookup function for entire output field" << endl;
  os << "  unsigned SpecificLookup (int idx" << ins_input_decl_list << ") const {" << endl;
  os << "    vector<unsigned> addr(" << m_Inputs.size() << ");" << endl;
  for (unsigned j=0; j< m_Inputs.size(); j++) {
    os << "    addr[" << j << "] = " << m_Inputs[j].first << ";" << endl;
  }
  os << "    return LookupPacked(idx, addr);" << endl;
  os << "  };" << endl << endl;

  os << ends;
  string ins_lookup_functions = os.str();

  // substitute in .h file
  string outfn (fname);
  if (outfn.size() == 0) outfn = string("../interface/L1MuGMT") +  m_name + string("LUT.h");
  ifstream of_check(outfn.c_str());
  if (! of_check.good() ) {
    ofstream of(outfn.c_str());
  

    ifstream in(template_file_h); 
    const int sz=1000; char buf[sz];

    while ( in.getline(buf, sz) ) {
      string line(buf);

      L1MuGMTLUTHelpers::replace(line, "###insert_name_upper###", ins_name_upper, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_name###", ins_name, false); 
      L1MuGMTLUTHelpers::replace(line, "###insert_instance_string###", ins_instance_string, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_instances_enum###", ins_instances_enum, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_inputs_string###", string(PortDecoder(m_Inputs).str()), false);
      L1MuGMTLUTHelpers::replace(line, "###insert_outputs_string###", string(PortDecoder(m_Outputs).str()), false);
      L1MuGMTLUTHelpers::replace(line, "###insert_vme_input_width###", string(ins_vme), false);
      L1MuGMTLUTHelpers::replace(line, "###insert_distrRAM###", string(ins_distr_RAM), false);
      L1MuGMTLUTHelpers::replace(line, "###insert_input_decl_list###", ins_input_decl_list, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_input_list###", ins_input_list, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_input_addr_list###", ins_input_addr_list, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_lookup_functions###", ins_lookup_functions, false);
      of << line << endl;
    }    
  }

  // substitute in .cc file
  string outfn_cc (fname);
  if (outfn_cc.size() == 0) outfn_cc = string("../interface/L1MuGMT") +  m_name + string("LUT.cc");

  ifstream of_cc_check( outfn_cc.c_str() );
  if (! of_cc_check.good() ) {
    ofstream of_cc(outfn_cc.c_str());
  

    ifstream in_cc(template_file_cc); 
    const int sz=1000; char buf[sz];

    while ( in_cc.getline(buf, sz) ) {
      string line(buf);

      L1MuGMTLUTHelpers::replace(line, "###insert_name_upper###", ins_name_upper, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_name###", ins_name, false); 
      L1MuGMTLUTHelpers::replace(line, "###insert_instance_string###", ins_instance_string, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_instances_enum###", ins_instances_enum, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_inputs_string###", string(PortDecoder(m_Inputs).str()), false);
      L1MuGMTLUTHelpers::replace(line, "###insert_outputs_string###", string(PortDecoder(m_Outputs).str()), false);
      L1MuGMTLUTHelpers::replace(line, "###insert_vme_input_width###", string(ins_vme), false);
      L1MuGMTLUTHelpers::replace(line, "###insert_distrRAM###", string(ins_distr_RAM), false);
      L1MuGMTLUTHelpers::replace(line, "###insert_input_decl_list###", ins_input_decl_list, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_input_list###", ins_input_list, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_input_addr_list###", ins_input_addr_list, false);
      L1MuGMTLUTHelpers::replace(line, "###insert_lookup_functions###", ins_lookup_functions, false);
      of_cc << line << endl;
    }    
  }
}


