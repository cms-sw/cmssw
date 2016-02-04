#ifndef __ROOTFILEWRITER_H__
#define __ROOTFILEWRITER_H__

/*

Adam Hunt - Princeton University
ahunt@princeton.edu

*/

#include "RecoLuminosity/ROOTSchema/interface/ROOTFileBase.h"

#include <string>

class TFile;
class TTree;

namespace HCAL_HLX{
  
  struct LUMI_SECTION;

  class ROOTFileWriter: public ROOTFileBase{
    
  public:
    
    ROOTFileWriter();
    ~ROOTFileWriter();

    bool OpenFile(const HCAL_HLX::LUMI_SECTION &lumiSection);
    bool OpenFile(const unsigned int runNumber, 
		  const unsigned int sectionNumber);

    void FillTree(const HCAL_HLX::LUMI_SECTION &localSection);
    bool CloseFile();    

    void SetMerge( const bool bMerge ){ bMerge_ = bMerge; }

  protected:

    void CreateTree();
    template< class T >
      void MakeBranch(const T &in, T **out, int HLXNum);

    void InsertInformation();

    TFile* m_file;
    TTree* m_tree;

    bool bMerge_;
    
  };
}
#endif
