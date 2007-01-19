#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "Cintex/Cintex.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ParameterSetBlob.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Common/interface/BranchType.h"

#include <iostream>
#include <sstream>
#include <assert.h>

namespace {
  struct HistoryNode {
    HistoryNode( const edm::ProcessConfiguration& iConfig,
                 unsigned int iSimpleId): config_(iConfig), simpleId_(iSimpleId) {}
    edm::ProcessConfiguration config_;
    std::vector<HistoryNode> children_;
    unsigned int simpleId_;
    
    typedef std::vector<HistoryNode>::const_iterator const_iterator;
    typedef std::vector<HistoryNode>::iterator iterator;
    
    iterator begin() { return children_.begin();}
    iterator end() { return children_.end();}

    const_iterator begin() const { return children_.begin();}
    const_iterator end() const { return children_.end();}
};
}
static void printHistory( const edm::ProcessHistory& iHist)
{
  const std::string indentDelta("  ");
  std::string indent = indentDelta;
   for(edm::ProcessHistory::const_iterator itH = iHist.begin(), itHEnd = iHist.end();
       itH != itHEnd;
       ++itH) {
      std::cout << indent <<itH->processName()<<" '"<<itH->passID()<<"' '"<<itH->releaseVersion()<<"' ("<<itH->parameterSetID()<<")"<<std::endl;
     indent += indentDelta;
   }
}

static void printHistory( const HistoryNode& iNode, const std::string& iIndent = std::string("  "))
{
  const std::string indentDelta("  ");
  std::string indent = iIndent;
  for(HistoryNode::const_iterator itH = iNode.begin(), itHEnd = iNode.end();
      itH != itHEnd;
      ++itH) {
    std::cout << indent <<itH->config_.processName()<<" '"<<itH->config_.passID()<<"' '"<<itH->config_.releaseVersion()
    <<"' ["<<itH->simpleId_<<"] "<<" ("<<itH->config_.parameterSetID()<<")"<<std::endl;
    printHistory(*itH, indent+indentDelta);
  }
}

int main(int argc, char* argv[]) {
   typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> ParameterSetMap;
   std::stringstream errorLog;
   int exitCode(0);

   try {
      ROOT::Cintex::Cintex::Enable();
      TFile f(argv[1]);

      TTree* meta = dynamic_cast<TTree*>(f.Get(edm::poolNames::metaDataTreeName().c_str()));
      assert(0!=meta);

      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg=&reg;
      meta->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(),&pReg);

      ParameterSetMap psm;
      ParameterSetMap* pPsm =&psm;
      meta->SetBranchAddress(edm::poolNames::parameterSetMapBranchName().c_str(),&pPsm);

      edm::ProcessHistoryMap phm;
      edm::ProcessHistoryMap* pPhm=&phm;
      meta->SetBranchAddress(edm::poolNames::processHistoryMapBranchName().c_str(),&pPhm);

      meta->GetEntry(0);
      assert(0!=pReg);
      pReg->setFrozen();

      //std::cout << meta->GetEntries()<<std::endl;

      //std::cout << pReg->size()<<std::endl;
      edm::ProcessConfiguration dummyConfig;
      HistoryNode historyGraph(dummyConfig,0);
      std::map<edm::ProcessConfigurationID, unsigned int> simpleIDs;

      std::cout << "Processing History:"<<std::endl;
      if( 1 == phm.size() ) {
	 printHistory((phm.begin()->second));
      } else {
        bool multipleHistories =false;
        for(edm::ProcessHistoryMap::const_iterator it = phm.begin(), itEnd = phm.end();
            it != itEnd;
            ++it) {
          //loop over the history entries looking for matches
          HistoryNode* parent = &historyGraph;
          for(edm::ProcessHistory::const_iterator itH = it->second.begin(), itHEnd = it->second.end();
              itH != itHEnd;
              ++itH) {
            if(parent->children_.size() == 0) {
              unsigned int id = simpleIDs[itH->id()];
              if (0 == id) {
                id = 1;
                simpleIDs[itH->id()] = id;
              }
              parent->children_.push_back(HistoryNode(*itH,id));
              parent = &(parent->children_.back());
            } else {
              //see if this is unique
              bool unique = true;
              for(HistoryNode::iterator itChild = parent->children_.begin(), itChildEnd = parent->children_.end();
                  itChild != itChildEnd;
                  ++itChild) {
                if( itChild->config_.id() == itH->id() ) {
                  unique = false;
                  parent = &(*itChild);
                  break;
                }
              }
              if(unique) {
                multipleHistories = true;
                simpleIDs[itH->id()]=parent->children_.size()+1;
                parent->children_.push_back(HistoryNode(*itH,simpleIDs[itH->id()]));
                parent = &(parent->children_.back());
              }
            }
          }
        }
        printHistory(historyGraph);
      }
   
      std::cout <<"------------------"<<std::endl;
      /*
      for(std::vector<edm::ProcessHistory>::const_iterator it = uniqueLongHistories.begin(),
	  itEnd = uniqueLongHistories.end();
	  it != itEnd;
	  ++it) {
	 //ParameterSetMap::const_iterator itpsm = psm.find(psid);
	 for(edm::ProcessHistory::const_iterator itH = it->begin(), itHEnd = it->end();
	     itH != itHEnd;
	     ++itH) {
	    std::cout << edm::ParameterSet(psm[ itH->parameterSetID() ].pset_) <<std::endl;
	 }
      }
       */
//using edm::ParameterSetID as the key does not work
//   typedef std::map<edm::ParameterSetID,std::vector<edm::BranchDescription> > IdToBranches
      typedef std::map<std::string,std::vector<edm::BranchDescription> > IdToBranches;
      typedef std::map<std::pair<std::string,std::string>,IdToBranches> ModuleToIdBranches;
      ModuleToIdBranches moduleToIdBranches;
      //IdToBranches idToBranches;
      for( edm::ProductRegistry::ProductList::const_iterator it = 
	      reg.productList().begin(), itEnd = reg.productList().end();
	   it != itEnd;
	   ++it) {
	 //force it to rebuild the branch name
	 it->second.init();

	 /*
	 std::cout << it->second.branchName()
		   << " id " << it->second.productID() << std::endl;
	 */
	 for(std::set<edm::ParameterSetID>::const_iterator itId = it->second.psetIDs().begin(),
	     itIdEnd = it->second.psetIDs().end();
	     itId != itIdEnd;
	     ++itId) {
	 
	    std::stringstream s;
	    s <<*itId;
	    moduleToIdBranches[std::make_pair(it->second.processName(),it->second.moduleLabel())][s.str()].push_back(it->second);
	    //idToBranches[*itId].push_back(it->second);
	 }
      }
      for(ModuleToIdBranches::const_iterator it = moduleToIdBranches.begin(),
	  itEnd = moduleToIdBranches.end();
	  it != itEnd;
	  ++it) {
	 std::cout <<"Module: "<<it->first.second<<" "<<it->first.first<<std::endl;
	 const IdToBranches& idToBranches = it->second;
	 for(IdToBranches::const_iterator itIdBranch = idToBranches.begin(),
             itIdBranchEnd = idToBranches.end();
             itIdBranch != itIdBranchEnd;
	     ++itIdBranch) {
	    std::cout <<" PSet id:"<<itIdBranch->first<<std::endl;
	    std::cout <<" products: {"<<std::endl;
	    for(std::vector<edm::BranchDescription>::const_iterator itBranch = itIdBranch->second.begin(),
		itBranchEnd = itIdBranch->second.end();
		itBranch != itBranchEnd;
		++itBranch) {
	       std::cout << "  "<< itBranch->branchName()<<std::endl;
	    }
	    std::cout <<"}"<<std::endl;
	    edm::ParameterSetID psid(itIdBranch->first);
	    ParameterSetMap::const_iterator itpsm = psm.find(psid);
	    if (psm.end() == itpsm) {
	       errorLog << "No ParameterSetID for " << psid << std::endl;
	       exitCode = 1;
	    } else {
	       std::cout <<" parameters: "<<
		  edm::ParameterSet((*itpsm).second.pset_)<<std::endl;
	    }
	    std::cout << std::endl;
	 }
      }
   }
   catch ( edm::Exception const& x ) {
      std::cerr << "cms::Exception caught\n";
      std::cerr << x.what() << '\n';
      exitCode = 2;
   }
   catch ( std::exception& x ) {
       std::cerr << "std::exception caught\n";
       std::cerr << x.what() << '\n';
       exitCode = 3;
    }
   catch ( ... ) {
      std::cerr << "Unknown exception caught\n";
      exitCode = 4;
   }
   std::string s(errorLog.str());
   if (s.length() > 0) {
      std::cerr << "Errors:" << std::endl << s;
   }
   return exitCode;
}
