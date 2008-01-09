#ifndef L1GtConfigProducers_L1GtVhdlWriter_h
#define L1GtConfigProducers_L1GtVhdlWriter_h

/**
 * \class L1GtVhdlWriter
 *
 *
 * Description: write VHDL templates for the L1 GT.
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

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterBitManager.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterCore.h"

class Event;
class EventSetup;
class ParameterSet;

// forward declarations

// class declaration
class L1GtVhdlWriter : public edm::EDAnalyzer
{

	public:

		/// constructor
		explicit L1GtVhdlWriter(const edm::ParameterSet&);

		/// destructor
		virtual ~L1GtVhdlWriter();

		/// returns all condition of the same class. Conditions belong to one class if they are matching
		/// in type (Type1s, Type2s..) category (CondMuon, CondCalo,)and are defined for the same object (Mu, fwdJet..)
		/// \param map is the ConditionMap that shall be filtered
		/// \param conditionToIntegerMap is the condition is added to this conversion map
		/// \param outputMap here the conditions matching the requirements are stored
		bool returnConditionsOfOneClass(const L1GtConditionType &type,  const L1GtConditionCategory &category, const L1GtObject &object, const ConditionMap &map,  ConditionMap &outputMap);

		/// builds a parameter map that can be used by the L1GtVhdlWriterCore class
		/// the output is written into the parameter &muonParameters;
		/// \param conditionToIntegerMap is the condition is added to this conversion map
		/// \param muonParameters here the routine stores the content of the subsitution parameters
		/// \param conditionMap containing the input
		void buildMuonParameterMap(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
			std::map<std::string,std::string> &muonParameters, const std::vector<ConditionMap> &conditionMap);

		bool buildCaloParameterMap(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
			std::map<std::string,std::string> &muonParameters,const L1GtObject &caloObject, const std::vector<ConditionMap> &conditionMap);

		bool buildEnergySumParameter(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
			std::string &energySumParameter, const std::vector<ConditionMap> &conditionMap);

		/// builds the substitution parameters for the cond_chip.vhd
		/// \condChip the condition chip that will be processed
		/// \param conditionToIntegerMap this has to be a already FILLED conversion map.
		///  therefore this routine has to be called after all those which are adding information
		///  to the conditionToIntegerMap map (buildMuonParameterMap, buildCaloParameterMap...)
		/// \param templates in this map the final content for the subsitution parameters is stored in
		/// VHDL template file format
		bool buildCondChipParameters(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
			const std::vector<ConditionMap> &conditionMap, std::map<std::string, L1GtVhdlTemplateFile> &templates, std::map<std::string, std::string> &commonParams);

		bool findObjectType(const L1GtObject &object, ConditionMap &map);
		
		/// builds the parameters particle_common for the cond_chip.vhd's
		bool buildCommonParameter(L1GtVhdlTemplateFile &particle,const L1GtObject &object, const L1GtConditionCategory &category, std::string &parameterStr, const ConditionMap &conditionMap);

		/// processes algorithm map delivered by parser, replaces condition names by types and serial numbers,
		/// splits the map in two seperate ones for the two condition chips
		bool processAlgorithmMap(const AlgorithmMap &algorithmMap,  std::vector<ConditionMap> &conditionMap, 
				 std::map<std::string,int> &conditionToIntegerMap, std::map<int, std::string> &algorithmsChip1, std::map<int, std::string> &algorithmsChip2);
		
		/// produces the firmware code
		bool makeFirmware(L1GtVhdlWriterCore &core, std::vector<ConditionMap> &conditionMap,const AlgorithmMap &algorithmMap);
		
		virtual void analyze(const edm::Event&, const edm::EventSetup&);

	private:
        
		/// converts L1GtConditionType to firmware string
		std::map<L1GtConditionType,std::string> condType2Str_;
		
		/// converts L1GtObject to calo_nr
		std::map<L1GtObject,std::string> caloType2Int_;
		
		/// converts L1GtObject to string
		std::map<L1GtObject,std::string> objType2Str_;
		
		/// list of all possible calo objects
		std::vector<L1GtObject> caloObjects_;
		
		/// list of all possible esums objects
		std::vector<L1GtObject> esumObjects_;
		
		/// templates directory
		std::string vhdlDir_;
		
		/// output directory
		std::string outputDir_;
		
		L1GtVhdlWriterBitManager bm_;

};
#endif											  /*L1GtConfigProducers_L1GtVhdlWriter_h*/
