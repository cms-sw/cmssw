#ifndef _ROOTFILEREADER_H_
#define _ROOTFILEREADER_H_

#include "RecoLuminosity/ROOTSchema/interface/ROOTFileBase.h"

// STL Headers
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

class TChain;
class TBranch;



namespace HCAL_HLX{

  struct LUMI_SECTION;
 
  class ROOTFileReader: public ROOTFileBase {
    
  public:
    ROOTFileReader();
   ~ROOTFileReader();
   
    // For manual replacment of files.
    int SetFileName(const std::string &fileName);
    int CreateFileNameList(); // Call after SetDir.

    int GetEntry(int entry);
    int GetLumiSection(HCAL_HLX::LUMI_SECTION& section);

    unsigned int GetEntries();
    
  private:    

    int ReplaceFile(const std::vector< std::string > &fileNames);
    void CreateTree();

    TChain *mChain_;

    // Branches
    // LUMI_SECTION
    TBranch *b_Header;
    TBranch *b_Summary;
    TBranch *b_Detail;
    
    static const unsigned int NUM_HLXS = 36;

    TBranch *b_ETSum[NUM_HLXS];
    TBranch *b_Occupancy[NUM_HLXS];
    TBranch *b_LHC[NUM_HLXS];

  };
}

#endif
