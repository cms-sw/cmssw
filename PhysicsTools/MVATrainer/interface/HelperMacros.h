#ifndef PhysicsTools_MVATrainer_HelperMacros_h
#define PhysicsTools_MVATrainer_HelperMacros_h

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerSaveImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerFileSaveImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooperImpl.h"

#define MVA_COMPUTER_SAVE_IMPLEMENT(T, P)			\
	namespace { namespace mva11 {				\
		typedef ::PhysicsTools::MVATrainerSaveImpl<T> P; \
		DEFINE_FWK_MODULE(P);			\
	}} typedef int mvaDummyTypedef11 ## T

#define MVA_COMPUTER_CONTAINER_SAVE_IMPLEMENT(T, P)		\
	namespace { namespace mva12 {				\
		typedef ::PhysicsTools::MVATrainerContainerSaveImpl<T> P; \
		DEFINE_FWK_MODULE(P);			\
	}} typedef int mvaDummyTypedef12 ## T

#define MVA_COMPUTER_FILE_SAVE_IMPLEMENT(T, P)	\
	namespace { namespace mva13 {				\
		typedef ::PhysicsTools::MVATrainerFileSaveImpl<T> P; \
		DEFINE_FWK_MODULE(P);			\
	}} typedef int mvaDummyTypedef13 ## T

#define MVA_TRAINER_LOOPER_IMPLEMENT(T, P)			\
	namespace { namespace mva14 {				\
		typedef ::PhysicsTools::MVATrainerLooperImpl<T> P; \
		DEFINE_FWK_LOOPER(P);				\
	}} typedef int mvaDummyTypedef14 ## T

#define MVA_TRAINER_CONTAINER_LOOPER_IMPLEMENT(T, P)		\
	namespace { namespace mva15 {				\
		typedef ::PhysicsTools::MVATrainerContainerLooperImpl<T> P; \
		DEFINE_FWK_LOOPER(P);				\
	}} typedef int mvaDummyTypedef15 ## T

#define MVA_TRAINER_IMPLEMENT(N)				\
	MVA_COMPUTER_CONTAINER_SAVE_IMPLEMENT(N ## Rcd, N ## ContainerSaveCondDB); \
	MVA_COMPUTER_FILE_SAVE_IMPLEMENT(N ## Rcd, N ## SaveFile); \
	MVA_TRAINER_CONTAINER_LOOPER_IMPLEMENT(N ## Rcd, N ## TrainerLooper)
// 	MVA_COMPUTER_SAVE_IMPLEMENT(N ## Rcd, N ## SaveCondDB)
//	MVA_TRAINER_LOOPER_IMPLEMENT(N ## Rcd, N ## TrainerLooper)

#endif // PhysicsTools_MVATrainer_HelperMacros_h
