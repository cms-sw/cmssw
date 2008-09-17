#ifndef __ROOTFILEWRITER_H__
#define __ROOTFILEWRITER_H__

/*

Adam Hunt - Princeton University
ahunt@princeton.edu

*/

#include "RecoLuminosity/TCPReceiver/interface/ICTypeDefs.hh"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

#include "RecoLuminosity/ROOTSchema/interface/ROOTFileBase.h"

#include <string>

class TFile;
class TTree;

namespace HCAL_HLX{

  
  class ROOTFileWriter: public HCAL_HLX::ROOTFileBase{
    
  public:
    
    ROOTFileWriter();
    ~ROOTFileWriter();

    void FillTree(const HCAL_HLX::LUMI_SECTION &localSection);
    void InsertInformation();
    void CloseTree();    

  protected:

    void CreateTree();
    template< class T >
      void MakeBranch(const T &in, T **out, int HLXNum);

    TFile* m_file;
    TTree* m_tree;
    
  };
}
#endif
