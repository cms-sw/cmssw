#ifndef FFFNAMINGSCHEMA_H_
#define FFFNAMINGSCHEMA_H_

#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace fffnaming {

  inline std::string runLumiPrefix(const unsigned int run, const unsigned int ls) {
    std::stringstream ss;
    ss << "run" << std::setfill('0') << std::setw(6) << run << "_ls" << std::setfill('0') << std::setw(4) << ls;
    return ss.str();
  }

  inline void runLumiPrefixFill(std::stringstream& ss, const unsigned int run, const unsigned int ls) {
    ss << "run" << std::setfill('0') << std::setw(6) << run << "_ls" << std::setfill('0') << std::setw(4) << ls;
  }

  inline std::string eolsFileName(const unsigned int run, const unsigned int ls) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_EoLS.jsn";
    return ss.str();
  }

  inline std::string bolsFileName(const unsigned int run, const unsigned int ls) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_BoLS.jsn";
    return ss.str();
  }

  inline std::string eorFileName(const unsigned int run) {
    std::stringstream ss;
    ss << "run" << std::setfill('0') << std::setw(6) << run << "_ls0000"
       << "_EoR.jsn";
    return ss.str();
  }

  inline std::string inputJsonFileName(const unsigned int run, const unsigned int ls, const unsigned int index) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_index" << std::setfill('0') << std::setw(6) << index << ".jsn";
    return ss.str();
  }

  inline std::string inputRawFileName(const unsigned int run, const unsigned int ls, const unsigned int index) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_index" << std::setfill('0') << std::setw(6) << index << ".raw";
    return ss.str();
  }

  inline std::string initFileNameWithPid(const unsigned int run, const unsigned int ls, std::string const& stream) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".ini";
    return ss.str();
  }

  inline std::string initTempFileNameWithPid(const unsigned int run, const unsigned int ls, std::string const& stream) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".initemp";
    return ss.str();
  }

  inline std::string initFileNameWithInstance(const unsigned int run,
                                              const unsigned int ls,
                                              std::string const& stream,
                                              std::string const& instance) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_" << instance << ".ini";
    return ss.str();
  }

  inline std::string streamerDataFileNameWithPid(const unsigned int run,
                                                 const unsigned int ls,
                                                 std::string const& stream) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".dat";
    return ss.str();
  }

  inline std::string streamerDataFileNameWithInstance(const unsigned int run,
                                                      const unsigned int ls,
                                                      std::string const& stream,
                                                      std::string const& instance) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_" << instance << ".dat";
    return ss.str();
  }

  inline std::string streamerDataChecksumFileNameWithInstance(const unsigned int run,
                                                              const unsigned int ls,
                                                              std::string const& stream,
                                                              std::string const& instance) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_" << instance << ".checksum";
    return ss.str();
  }

  inline std::string streamerJsonFileNameWithPid(const unsigned int run,
                                                 const unsigned int ls,
                                                 std::string const& stream) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";
    return ss.str();
  }

  inline std::string streamerJsonFileNameWithInstance(const unsigned int run,
                                                      const unsigned int ls,
                                                      std::string const& stream,
                                                      std::string const& instance) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_stream" << stream << "_" << instance << ".jsn";
    return ss.str();
  }

  inline std::string protocolBufferHistogramFileNameWithPid(const unsigned int run,
                                                            const unsigned int ls,
                                                            std::string const& stream) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".pb";
    return ss.str();
  }

  inline std::string protocolBufferHistogramFileNameWithInstance(const unsigned int run,
                                                                 const unsigned int ls,
                                                                 std::string const& stream,
                                                                 std::string const& instance) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_" << instance << ".pb";
    return ss.str();
  }

  inline std::string rootHistogramFileNameWithPid(const unsigned int run,
                                                  const unsigned int ls,
                                                  std::string const& stream) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".root";
    return ss.str();
  }

  inline std::string rootHistogramFileNameWithInstance(const unsigned int run,
                                                       const unsigned int ls,
                                                       std::string const& stream,
                                                       std::string const& instance) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_" << stream << "_" << instance << ".root";
    return ss.str();
  }

  inline std::string genericFileName(const unsigned int run,
                                     const unsigned int ls,
                                     std::string const& typePrefix,
                                     std::string const& typeSuffix,
                                     unsigned int typeWidth,
                                     std::string const& instancePrefix,
                                     std::string const& instanceSuffix,
                                     unsigned int instanceWidth,
                                     std::string const& fileExtension) {
    std::stringstream ss;
    runLumiPrefixFill(ss, run, ls);
    ss << "_";
    if (!typePrefix.empty())
      ss << typePrefix;
    if (typeWidth)
      ss << std::setfill('0') << std::setw(typeWidth);
    ss << typeSuffix;
    if (!instanceSuffix.empty()) {
      ss << "_";
      if (!instancePrefix.empty())
        ss << instancePrefix;
      if (instanceWidth)
        ss << std::setfill('0') << std::setw(instanceWidth);
      ss << instanceSuffix;
    }
    if (!fileExtension.empty())
      ss << "." << fileExtension;
    return ss.str();
  }

};  // namespace fffnaming
#endif
