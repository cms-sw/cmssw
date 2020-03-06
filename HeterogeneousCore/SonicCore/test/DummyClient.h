#ifndef SonicCMS_Core_test_DummyClient
#define SonicCMS_Core_test_DummyClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "SonicCMS/Core/interface/SonicClientSync.h"
#include "SonicCMS/Core/interface/SonicClientPseudoAsync.h"
#include "SonicCMS/Core/interface/SonicClientAsync.h"

#include <vector>
#include <thread>
#include <chrono>

template <typename Client>
class DummyClient : public Client {
	public:
		//constructor
		DummyClient(const edm::ParameterSet& params) :
			factor_(params.getParameter<int>("factor")),
			wait_(params.getParameter<int>("wait"))
		{}

		//for fillDescriptions
		static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
			edm::ParameterSetDescription descClient;
			descClient.add<int>("factor",-1);
			descClient.add<int>("wait",10);
			iDesc.add<edm::ParameterSetDescription>("Client",descClient);
		}

	protected:
		void predictImpl() override {
			//simulate a blocking call
			std::this_thread::sleep_for(std::chrono::seconds(wait_));

			this->output_ = this->input_*factor_;
		}

		//members
		int factor_;
		int wait_;
};

typedef DummyClient<SonicClientSync<int>> DummyClientSync;
typedef DummyClient<SonicClientPseudoAsync<int>> DummyClientPseudoAsync;
typedef DummyClient<SonicClientAsync<int>> DummyClientAsync;

//specialization for true async
template <>
void DummyClientAsync::predictImpl() {
	this->output_ = this->input_*factor_;
	this->finish();
}

#endif
