#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"
#include <sstream>
#include <string>
#include <vector>

namespace cond {
  
	template<>
	std::string PayLoadInspector<CSCBadChambers>::summary() const {
		std::stringstream ss;
		CSCIndexer indexer;
		/// How many bad chambers are there
		int numberOfChambers = object().numberOfChambers();

		/// Return the container of bad chambers
		std::vector<int> vContainerInts = object().container();
		std::vector<CSCDetId> vCSCDetIds;
		std::vector<int>::const_iterator iInt;
		//get vector of CSCDetId:
		for (iInt = vContainerInts.begin(); iInt != vContainerInts.end(); ++iInt){
			vCSCDetIds.push_back(indexer.detIdFromChamberIndex(*iInt));
		}
		/// Is the given chamber flagged as bad?
		//bool isInBadChamber0 = object().isInBadChamber( const CSCDetId& id );

		//print data:
		ss << "---Total of bad Chambers: " << numberOfChambers << std::endl;
		ss << "--Bad chambers:" << std::endl;
		iInt = vContainerInts.begin();
		int index = 0;
		for (std::vector<CSCDetId>::const_iterator iCSCDetId = vCSCDetIds.begin(); iCSCDetId != vCSCDetIds.end(); ++iCSCDetId, ++iInt){
			ss << index++ << ". "<< "Id[" << *iInt
			   << "]-> Chamber index: " << indexer.chamberIndex( *iCSCDetId )
			   << "; Layer index: " << indexer.layerIndex( *iCSCDetId ) 
			   << ";"<< std::endl;
		}
		
		return ss.str();
	}

}
PYTHON_WRAPPER(CSCBadChambers,CSCBadChambers);
