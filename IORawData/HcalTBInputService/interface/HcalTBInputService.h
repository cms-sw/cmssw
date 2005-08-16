/* -*- C++ -*- */
#ifndef HcalTBInputService_h_included
#define HcalTBInputService_h_included 1

#include <map>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/Retriever.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputServiceDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Retriever.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/ProductDescription.h"

class TFile;
class TTree;
class CDFChunk;
class CDFEventInfo;

namespace cms {
  namespace hcal {

/** \class HcalTBInputService
    
   $Date: 2005/08/04 17:11:44 $
   $Revision: 1.1 $
   \author J. Mans - Minnesota
*/
class HcalTBInputService : public edm::InputService {
public:
explicit HcalTBInputService(const edm::ParameterSet & pset, edm::InputServiceDescription const& desc);
private:
  virtual std::auto_ptr<edm::EventPrincipal> read();
  void unpackSetup(const std::vector<std::string>& params);
  void openFile(const std::string& filename);
  std::vector<std::string> files_;
  edm::Retriever* retriever_;
  TTree* m_tree;
  TFile* m_file;
  int fileCounter_;
  int m_i, m_imax, m_itotal;
  //  int m_duplicateChunkAs;
  int n_chunks;
  static const int CHUNK_COUNT=64; // MAX Chunks
  CDFChunk* m_chunks[CHUNK_COUNT];
  int m_chunkIds[CHUNK_COUNT];
  std::map<std::string,int> m_sourceIdRemap;
  CDFEventInfo* m_eventInfo;
  edm::ProductDescription prodDesc_;
};

}
}

#endif // HcalTBInputService_h_included
