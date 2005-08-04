/* -*- C++ -*- */
#ifndef HcalTBInputService_h_included
#define HcalTBInputService_h_included 1

#include <map>
#include <string>

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
namespace cms {
  namespace hcal {

/** \class HcalTBInputService
    
   $Date: 2005/07/19 23:26:19 $
   $Revision: 1.1 $
   \author J. Mans - Minnesota
*/
class HcalTBInputService : public edm::InputService {
public:
explicit HcalTBInputService(const edm::ParameterSet & pset, edm::InputServiceDescription const& desc);
private:
  virtual std::auto_ptr<edm::EventPrincipal> read();
  void initThis();
  std::string file_;
  edm::Retriever* retriever_;
  TTree* m_tree;
  TFile* m_file;
  int m_i, m_imax;
  int m_hcalFedOffset;
  int m_duplicateChunkAs;
  int n_chunks;
  CDFChunk* m_chunks[1024];
  edm::ProductDescription prodDesc_;
};

}
}

#endif // HcalTBInputService_h_included
