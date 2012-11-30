#ifndef _include_LzmaFile_interface_h_
#define _include_LzmaFile_interface_h_

extern "C" { void lzmaopenfile_(char* name); }
extern "C" { void lzmafillarray_(double* data, const int& length); }
extern "C" { void lzmaclosefile_(); } 
extern "C" { int lzmanextnumber_(double& data); } 
  

#endif
