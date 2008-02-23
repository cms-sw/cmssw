#ifndef L1GtConfigProducers_L1GtVhdlWriterMaps_h
#define L1GtConfigProducers_L1GtVhdlWriterMaps_h

/**
 * \class L1GtVhdlWriterMaps
 * 
 * 
 * Description: Contains conversion maps for conversion of trigger objects to strings etc.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Philipp Wagner
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

// forward declarations

// class declaration
class L1GtVhdlWriterMaps
{

public:

    /// constructor
    L1GtVhdlWriterMaps();

    /// destructor
    virtual ~L1GtVhdlWriterMaps();

	/// converts object type to firmware string
    std::string obj2str(const L1GtObject &type);
	
	/// converts a condition type to firmware string
    std::string type2str(const L1GtConditionType &type);
    
    const std::map<L1GtObject,std::string> getObj2StrMap();
    
    const std::map<L1GtConditionType,std::string> getCond2StrMap();
    
    const std::map<L1GtObject,std::string> getCalo2IntMap();

private:
    
    /// converts L1GtConditionType to firmware string
    std::map<L1GtObject,std::string> objType2Str_;
    
    /// converts L1GtObject to calo_nr
	std::map<L1GtConditionType,std::string> condType2Str_;
	
	/// converts L1GtObject to string
    std::map<L1GtObject,std::string> caloType2Int_;

};

#endif /*L1GtConfigProducers_L1GtVhdlWriterMaps_h*/
