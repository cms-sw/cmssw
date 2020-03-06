#ifndef SonicCMS_Core_SonicClientSync
#define SonicCMS_Core_SonicClientSync

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include "SonicCMS/Core/interface/SonicClientBase.h"
#include "SonicCMS/Core/interface/SonicClientTypes.h"

#include <exception>

template <typename InputT, typename OutputT=InputT>
class SonicClientSync : public SonicClientBase, public SonicClientTypes<InputT,OutputT> {
	public:
		virtual ~SonicClientSync() {}

		//main operation
		void predict(edm::WaitingTaskWithArenaHolder holder) override final {
			holder_ = std::move(holder);
			setStartTime();

			std::exception_ptr eptr;
			try {
				predictImpl();
			}
			catch(...) {
				eptr = std::current_exception();
			}

			//sync Client calls holder at the end
			finish(eptr);
		}		
};

#endif

