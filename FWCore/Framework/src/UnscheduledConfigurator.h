#ifndef FWCore_Framework_UnscheduledConfigurator_h
#define FWCore_Framework_UnscheduledConfigurator_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     UnscheduledConfigurator
//
/**\class UnscheduledConfigurator UnscheduledConfigurator.h "UnscheduledConfigurator.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 13 Apr 2016 18:57:55 GMT
//

// system include files
#include <unordered_map>

// user include files

// forward declarations

namespace edm {
  class Worker;
  class UnscheduledAuxiliary;

  class UnscheduledConfigurator {
  public:
    template <typename IT>
    UnscheduledConfigurator(IT iBegin, IT iEnd, UnscheduledAuxiliary const* iAux) : m_aux(iAux) {
      for (auto it = iBegin; it != iEnd; ++it) {
        m_labelToWorker.emplace((*it)->description()->moduleLabel(), *it);
      }
    }

    UnscheduledConfigurator(const UnscheduledConfigurator&) = delete;                   // stop default
    const UnscheduledConfigurator& operator=(const UnscheduledConfigurator&) = delete;  // stop default

    // ---------- const member functions ---------------------
    Worker* findWorker(std::string const& iLabel) const {
      auto itFound = m_labelToWorker.find(iLabel);
      if (itFound != m_labelToWorker.end()) {
        return itFound->second;
      }
      return nullptr;
    }

    UnscheduledAuxiliary const* auxiliary() const { return m_aux; }

  private:
    // ---------- member data --------------------------------
    std::unordered_map<std::string, Worker*> m_labelToWorker;
    UnscheduledAuxiliary const* m_aux;
  };
}  // namespace edm

#endif
