#include "CommonTools/Utils/interface/cutParser.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TThread.h"
#include "TObject.h"
#include "TVirtualStreamerInfo.h"

#include <thread>
#include <atomic>
#include <iostream>
#include <string>

namespace {
  class Decrementer {
  public:
    Decrementer(std::atomic<int>& iValue) : value_(iValue), dec_(true) {}
    ~Decrementer() {
      if (dec_) {
        --value_;
      }
    }

    void decrement() {
      --value_;
      dec_ = false;
    }

  private:
    std::atomic<int>& value_;
    bool dec_;
  };
}  // namespace

int main() {
  constexpr int kNThreads = 30;
  std::atomic<int> canStart{kNThreads};
  std::atomic<int> canStartEval{kNThreads};
  std::atomic<bool> failed{false};
  std::atomic<int> threadID{-1};
  std::vector<std::thread> threads;

  TThread::Initialize();
  //When threading, also have to keep ROOT from logging all TObjects into a list
  TObject::SetObjectStat(false);

  //Have to avoid having Streamers modify themselves after they have been used
  TVirtualStreamerInfo::Optimize(false);

  std::string const cut(" pt >= 1 & momentum().x() > 0.5 & quality('highPurity') ");
  StringCutObjectSelector<reco::Track> selectShared(cut, false);
  StringCutObjectSelector<reco::Track> selectSharedLazy(cut, true);
  for (int i = 0; i < kNThreads; ++i) {
    threads.emplace_back([&canStart, &canStartEval, &failed, &selectShared, &selectSharedLazy, &threadID]() {
      auto id = ++threadID;
      bool sharedTestShouldSucceed = ((id % 2) == 0);
      try {
        static thread_local TThread guard;
        reco::Track trk(20.,
                        20.,
                        reco::Track::Point(),
                        reco::Track::Vector(sharedTestShouldSucceed ? 1. : 0., 1., 1.),
                        +1,
                        reco::Track::CovarianceMatrix{});
        trk.setQuality(reco::Track::highPurity);
        std::string const cut(" pt >= 1 & quality('highPurity') ");
        //std::cout <<cut<<std::endl;
        bool const lazy = true;

        --canStart;
        while (canStart > 0) {
        }

        //need to make sure canStartEval is decremented even if we have an exception
        Decrementer decCanStartEval(canStartEval);

        StringCutObjectSelector<reco::Track> select(cut, lazy);

        decCanStartEval.decrement();

        while (canStartEval > 0) {
        }
        if (not select(trk)) {
          std::cout << "selection failed" << std::endl;
          failed = true;
        }
        if (sharedTestShouldSucceed != selectShared(trk)) {
          std::cout << "selection shared failed, expected " << sharedTestShouldSucceed << std::endl;
          failed = true;
        }
        if (sharedTestShouldSucceed != selectSharedLazy(trk)) {
          std::cout << "selection shared lazy failed, expected " << sharedTestShouldSucceed << std::endl;
          failed = true;
        }
      } catch (cms::Exception const& exception) {
        std::cout << exception.what() << std::endl;
        failed = true;
      }
    });
  }
  canStart = true;

  for (auto& thread : threads) {
    thread.join();
  }

  if (failed) {
    std::cout << "FAILED" << std::endl;
    return 1;
  }

  std::cout << "OK" << std::endl;
  return 0;
}
