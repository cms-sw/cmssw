/* -*- C++ -*- */
#ifndef TBRUInputSource_h_included
#define TBRUInputSource_h_included 

#include <map>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ExternalInputSource.h"

class TFile;
class TTree;
class TBRU;

/** 
    \class TBRUInputSource
    \author L. Mirabito CERN
*/
class TBRUInputSource : public edm::ExternalInputSource {
  
public:
  
  explicit TBRUInputSource( const edm::ParameterSet & pset, 
			    edm::InputSourceDescription const& desc );
  virtual ~TBRUInputSource() {;}
  
protected:
  
  virtual void setRunAndEventInfo();
  virtual bool produce( edm::Event& e );
  bool checkFedStructure(int i, unsigned int* dest,unsigned int &length) ;
  int getFedId(bool slinkswap,unsigned int* output);

private:
  
  void unpackSetup( const std::vector<std::string>& ) {;}
  void openFile( const std::string& filename );
  
  TTree* m_tree;
  TFile* m_file;
  int m_i, m_fileCounter;
  bool m_quiet;
  int n_fed9ubufs,n_run,m_branches,nfeds,triggerFedId;
  static const int MAX_FED9U_BUFFER=144; // MAX Fed9ubufs
  TBRU* m_fed9ubufs[MAX_FED9U_BUFFER];
 
};

#endif // TBRUInputSource_h_included
