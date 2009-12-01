// $Id: Configuration.h,v 1.11 2009/11/09 15:40:55 mommsen Exp $
/// @file: TriggerSelector.h 

#ifndef EventFilter_StorageManager_TriggerSelector_h
#define EventFilter_StorageManager_TriggerSelector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Framework/interface/EventSelector.h"

#include "boost/shared_ptr.hpp"

#include <vector>
#include <string>

namespace stor
{

  /**
   * Event selector allowing for and/not combination of triggers/paths
   *
   * $Author: mommsen $
   * $Revision: 1.11 $
   * $Date: 2009/11/09 15:40:55 $
   */

	class TriggerSelector
	{
		public:

			typedef std::vector<std::string> Strings;

			//old mode only
			TriggerSelector(Strings const& pathspecs,
					Strings const& names);

			TriggerSelector(edm::ParameterSet const& pset,
					Strings const& triggernames, bool old_ = false);

			TriggerSelector(std::string const& expression, Strings const& triggernames);

			~TriggerSelector() {};

			bool wantAll() const { return accept_all_; }
			bool acceptEvent(edm::TriggerResults const&) const;
			bool acceptEvent(unsigned char const*, int) const;

			//for testing purposes
			bool returnStatus(edm::HLTGlobalStatus const& trStatus) const {
				return masterElement_->returnStatus(trStatus);
			}

			static  std::string makeXMLString(std::string const& input);

			static std::vector<std::string>
				getEventSelectionVString(edm::ParameterSet const& pset);

		private:

			bool accept_all_;

			void init(std::string const& path, Strings const& triggernames);

			static std::string trim(std::string input);

			class TreeElement {

				enum TreeOperator {
					NonInit = 0,
					AND = 1,
					OR  = 2,
					NOT = 3,
					BR = 4
				};

				public:

				TreeElement(std::string const& inputString,Strings const& tr,TreeElement* parentElement = NULL);
				~TreeElement();

				bool returnStatus(edm::HLTGlobalStatus const& trStatus) const;

				TreeOperator op() const {return op_;}
				TreeElement * parent() const {return parent_;}

				private:

				TreeElement * parent_;
				std::vector<TreeElement*> children_;
				TreeOperator op_;
				int trigBit_;
			};

			boost::shared_ptr<TreeElement> masterElement_;

			//keep a copy of initialization string
			std::string expression_;

			boost::shared_ptr<edm::EventSelector> eventSelector_;
			bool useOld_;

			static const bool debug_ = false;

	};
}

#endif // EventFilter_StorageManager_TriggerSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
