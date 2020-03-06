#ifndef SonicCMS_Core_SonicClientAsync
#define SonicCMS_Core_SonicClientAsync

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include "SonicCMS/Core/interface/SonicClientBase.h"
#include "SonicCMS/Core/interface/SonicClientTypes.h"

template <typename InputT, typename OutputT=InputT>
class SonicClientAsync : public SonicClientBase, public SonicClientTypes<InputT,OutputT> {
	public:
		virtual ~SonicClientAsync() {}

		//main operation
		void predict(edm::WaitingTaskWithArenaHolder holder) override final {
			holder_ = std::move(holder);
			setStartTime();
			predictImpl();
			//impl calls finish() which calls holder_
		}
};

#endif

