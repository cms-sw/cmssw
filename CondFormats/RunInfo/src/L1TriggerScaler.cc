#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"
L1TriggerScaler::L1TriggerScaler() { m_run.reserve(10000); }
void L1TriggerScaler::printAllValues() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\n lumisegment: " << it->m_lumisegment
              << "\n start_time: " << it->m_start_time << std::endl;

    for (size_t i = 0; i < it->m_GTAlgoRates.size(); i++) {
      std::cout << "m_GTAlgoRates[" << i << "] = " << it->m_GTAlgoRates[i] << std::endl;
    }
    for (size_t i = 0; i < it->m_GTAlgoPrescaling.size(); i++) {
      std::cout << "m_GTAlgoPrescaling[" << i << "] = " << it->m_GTAlgoPrescaling[i] << std::endl;
    }
    for (size_t i = 0; i < it->m_GTTechCounts.size(); i++) {
      std::cout << " m_GTTechCounts[" << i << "] = " << it->m_GTTechCounts[i] << std::endl;
    }
    for (size_t i = 0; i < it->m_GTTechRates.size(); i++) {
      std::cout << " m_GTTechRates[" << i << "] = " << it->m_GTTechRates[i] << std::endl;
    }
    for (size_t i = 0; i < it->m_GTTechPrescaling.size(); i++) {
      std::cout << " m_GTTechPrescaling[" << i << "] = " << it->m_GTTechPrescaling[i] << std::endl;
    }
    for (size_t i = 0; i < it->m_GTPartition0TriggerCounts.size(); i++) {
      std::cout << " m_GTPartition0TriggerCounts[" << i << "] = " << it->m_GTPartition0TriggerCounts[i] << std::endl;
    }
    for (size_t i = 0; i < it->m_GTPartition0TriggerRates.size(); i++) {
      std::cout << " m_GTPartition0TriggerRates[" << i << "] = " << it->m_GTPartition0TriggerRates[i] << std::endl;
    }
    for (size_t i = 0; i < it->m_GTPartition0DeadTime.size(); i++) {
      std::cout << " m_GTPartition0DeadTime[" << i << "] = " << it->m_GTPartition0DeadTime[i] << std::endl;
    }
    for (size_t i = 0; i < it->m_GTPartition0DeadTimeRatio.size(); i++) {
      std::cout << " m_GTPartition0DeadTimeRatio[" << i << "] = " << it->m_GTPartition0DeadTimeRatio[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printRunValue() const {
  LumiIterator it = m_run.begin();
  std::cout << it->m_rn << std::endl;
}

void L1TriggerScaler::printLumiSegmentValues() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\n lumisegment: " << it->m_lumisegment
              << "\n start_time: " << it->m_start_time << std::endl;
  }
}

void L1TriggerScaler::printFormat() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\n lumisegment: " << it->m_lumisegment << std::endl;
    std::cout << "format :" << it->m_string_format << std::endl;
  }
}

void L1TriggerScaler::printGTAlgoCounts() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\n lumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTAlgoCounts.size(); i++) {
      std::cout << "m_GTAlgoCounts[" << i << "] = " << it->m_GTAlgoCounts[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printGTAlgoRates() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\n lumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTAlgoRates.size(); i++) {
      std::cout << "m_GTAlgoRates[" << i << "] = " << it->m_GTAlgoRates[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printGTAlgoPrescaling() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\n lumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTAlgoPrescaling.size(); i++) {
      std::cout << "m_GTAlgoPrescaling[" << i << "] = " << it->m_GTAlgoPrescaling[i] << std::endl;
    }
  }
}
void L1TriggerScaler::printGTTechCounts() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\nlumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTTechCounts.size(); i++) {
      std::cout << "m_GTTechCounts[" << i << "] = " << it->m_GTTechCounts[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printGTTechRates() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\nlumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTTechRates.size(); i++) {
      std::cout << "m_GTTechRates[" << i << "] = " << it->m_GTTechRates[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printGTTechPrescaling() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\nlumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTTechPrescaling.size(); i++) {
      std::cout << "m_GTTechPrescaling[" << i << "] = " << it->m_GTTechPrescaling[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printGTPartition0TriggerCounts() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\nlumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTPartition0TriggerCounts.size(); i++) {
      std::cout << "m_GTPartition0TriggerCounts[" << i << "] = " << it->m_GTPartition0TriggerCounts[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printGTPartition0TriggerRates() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\nlumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTPartition0TriggerRates.size(); i++) {
      std::cout << "m_GTPartition0TriggerRates[" << i << "] = " << it->m_GTPartition0TriggerRates[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printGTPartition0DeadTime() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\nlumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTPartition0DeadTime.size(); i++) {
      std::cout << "m_GTPartition0DeadTime[" << i << "] = " << it->m_GTPartition0DeadTime[i] << std::endl;
    }
  }
}

void L1TriggerScaler::printGTPartition0DeadTimeRatio() const {
  for (LumiIterator it = m_run.begin(); it != m_run.end(); ++it) {
    std::cout << "  run:  " << it->m_rn << "\nlumisegment: " << it->m_lumisegment << std::endl;
    for (size_t i = 0; i < it->m_GTPartition0DeadTimeRatio.size(); i++) {
      std::cout << "m_GTPartition0DeadTimeRatio[" << i << "] = " << it->m_GTPartition0DeadTimeRatio[i] << std::endl;
    }
  }
}
