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

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterMaps.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

// system include files
#include <string>

// user include files

// constructor
L1GtVhdlWriterMaps::L1GtVhdlWriterMaps() {
	
	objType2Str_[Mu]="muon";
    objType2Str_[NoIsoEG]="eg";
    objType2Str_[IsoEG]="ieg";
    objType2Str_[ForJet]="fwdjet";
    objType2Str_[TauJet]="tau";
    objType2Str_[CenJet]="jet";
    objType2Str_[JetCounts]="jet_cnts";
    objType2Str_[HTT]="htt";
    objType2Str_[ETT]="ett";
    objType2Str_[ETM]="etm";

    condType2Str_[Type1s]="1_s";
    condType2Str_[Type2s]="2_s";
    condType2Str_[Type2wsc]="2_wsc";
    condType2Str_[Type3s]="3";
    condType2Str_[Type4s]="4";
    condType2Str_[Type2cor]="Type2cor";
    condType2Str_[TypeETM]="cond";
    condType2Str_[TypeETT]="cond";
    condType2Str_[TypeHTT]="cond";
    condType2Str_[TypeJetCounts]="TypeJetCounts";
    
    caloType2Int_[IsoEG]="0";
    caloType2Int_[NoIsoEG]="1";
    caloType2Int_[CenJet]="2";
    caloType2Int_[TauJet]="3";
    caloType2Int_[ForJet]="4";
    caloType2Int_[Mu]="5";
    caloType2Int_[ETM]="6";

}


const std::map<L1GtObject,std::string> L1GtVhdlWriterMaps::getObj2StrMap()
{
    return objType2Str_;
}

const std::map<L1GtConditionType,std::string> L1GtVhdlWriterMaps::getCond2StrMap()
{
    return  condType2Str_;
}

const std::map<L1GtObject,std::string> L1GtVhdlWriterMaps::getCalo2IntMap()
{
    return caloType2Int_;
}


std::string L1GtVhdlWriterMaps::obj2str(const L1GtObject &type) {
	
	return objType2Str_[type];
}

std::string L1GtVhdlWriterMaps::type2str(const L1GtConditionType &type) {
	
	return condType2Str_[type];
}

// destructor
L1GtVhdlWriterMaps::~L1GtVhdlWriterMaps() {

    // empty
    
}


