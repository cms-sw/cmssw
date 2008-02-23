#ifndef PhysicsTools_MVAComputer_HelperMacros_h
#define PhysicsTools_MVAComputer_HelperMacros_h

#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceImpl.h"

#define MVA_COMPUTER_RECORD_DEFINE(T)				\
	class T : public edm::eventsetup::EventSetupRecordImplementation<T> {}

#define MVA_COMPUTER_CONTAINER_RECORD_DEFINE(T)			\
	MVA_COMPUTER_RECORD_DEFINE(T)

#define MVA_COMPUTER_RECORD_IMPLEMENT(T)			\
	EVENTSETUP_RECORD_REG(T);				\
	namespace { namespace mva1 {				\
		using namespace ::PhysicsTools::Calibration;	\
		REGISTER_PLUGIN(T, MVAComputer);		\
	}} typedef int mvaDummyTypedef1 ## T

#define MVA_COMPUTER_CONTAINER_RECORD_IMPLEMENT(T)		\
	EVENTSETUP_RECORD_REG(T);				\
	namespace { namespace mva2 {				\
		using namespace ::PhysicsTools::Calibration;	\
		REGISTER_PLUGIN(T, MVAComputerContainer);	\
	}} typedef int mvaDummyTypedef2 ## T

#define MVA_COMPUTER_CONTAINER_FILE_SOURCE_IMPLEMENT(T, P)	\
	namespace { namespace mva3 {				\
		typedef ::PhysicsTools::MVAComputerESSourceImpl<T> P; \
		DEFINE_FWK_EVENTSETUP_SOURCE(P);		\
	}} typedef int mvaDummyTypedef3 ## T

#define MVA_COMPUTER_CONTAINER_DEFINE(N)			\
	MVA_COMPUTER_CONTAINER_RECORD_DEFINE(N ## Rcd)

#define MVA_COMPUTER_CONTAINER_IMPLEMENT(N)			\
	MVA_COMPUTER_CONTAINER_RECORD_IMPLEMENT(N ## Rcd);	\
	MVA_COMPUTER_CONTAINER_FILE_SOURCE_IMPLEMENT(N ## Rcd, N ## FileSource)

#endif // PhysicsTools_MVAComputer_HelperMacros_h
