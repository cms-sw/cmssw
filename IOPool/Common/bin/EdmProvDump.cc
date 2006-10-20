#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "Cintex/Cintex.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ParameterSetBlob.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <sstream>
#include <assert.h>

int main(int argc, char* argv[]) {
   typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> ParameterSetMap;
   std::stringstream errorLog;
   int exitCode(0);

   try {
      ROOT::Cintex::Cintex::Enable();
      TFile f(argv[1]);

      TTree* meta = dynamic_cast<TTree*>(f.Get("MetaData"));
      assert(0!=meta);

      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg=&reg;
      meta->SetBranchAddress("ProductRegistry",&pReg);

      ParameterSetMap psm;
      ParameterSetMap* pPsm =&psm;
      meta->SetBranchAddress("ParameterSetMap",&pPsm);

      meta->GetEntry(0);
      assert(0!=pReg);
      pReg->setFrozen();

      std::cout << meta->GetEntries()<<std::endl;

      std::cout << pReg->size()<<std::endl;
//using edm::ParameterSetID as the key does not work
//   typedef std::map<edm::ParameterSetID,std::vector<edm::BranchDescription> > IdToBranches
      typedef std::map<std::string,std::vector<edm::BranchDescription> > IdToBranches;
      typedef std::map<std::string,IdToBranches> ModuleToIdBranches;
      ModuleToIdBranches moduleToIdBranches;
      //IdToBranches idToBranches;
      for( edm::ProductRegistry::ProductList::const_iterator it = 
	      reg.productList().begin();
	   it != reg.productList().end();
	   ++it) {
	 //force it to rebuild the branch name
	 it->second.init();

	 /*
	 std::cout << it->second.branchName()
		   << " id " << it->second.productID() << std::endl;
	 */
	 for(std::set<edm::ParameterSetID>::const_iterator itId=it->second.psetIDs().begin();
	     itId != it->second.psetIDs().end();
	     ++itId) {
	 
	    std::stringstream s;
	    s <<*itId;
	    moduleToIdBranches[it->second.moduleLabel()+" "+it->second.processName()][s.str()].push_back(it->second);
	    //idToBranches[*itId].push_back(it->second);
	 }
      }
      for(ModuleToIdBranches::const_iterator it = moduleToIdBranches.begin();
	  it != moduleToIdBranches.end();
	  ++it) {
	 std::cout <<"Module: "<<it->first<<std::endl;
	 const IdToBranches& idToBranches = it->second;
	 for(IdToBranches::const_iterator itIdBranch = idToBranches.begin();
     itIdBranch != idToBranches.end();
	     ++itIdBranch) {
	    std::cout <<" PSet id:"<<itIdBranch->first<<std::endl;
	    std::cout <<" products: {"<<std::endl;
	    for(std::vector<edm::BranchDescription>::const_iterator itBranch=itIdBranch->second.begin();
		itBranch != itIdBranch->second.end();
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
