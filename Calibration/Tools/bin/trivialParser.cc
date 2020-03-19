//#include "Calibration/EcalAlCaRecoProducers/interface/trivialParser.h"
#include "Calibration/Tools/bin/trivialParser.h"
#include <iostream>
#include <cstdlib>

trivialParser::trivialParser(std::string configFile) {
  parse(configFile);
  print("[ctor] ");
}

// ------------------------------------------------------------

double trivialParser::getVal(std::string name) {
  if (m_config.count(name))
    return m_config[name];
  std::cerr << "[trivialParser] no value for " << name << " found\n";
  return -999999.;
}

// ------------------------------------------------------------

void trivialParser::parse(std::string configFile) {
  std::ifstream input(configFile.c_str());
  do {
    std::string linea = getNextLine(input);
    if (linea.empty())
      continue;
    std::string name(linea, 0, linea.find('=', 0));
    eraseSpaces(name);
    std::string valuestring(linea, linea.find('=', 0) + 1, linea.size() - linea.find('=', 0) - 1);
    eraseSpaces(valuestring);
    double value = strtod(valuestring.c_str(), nullptr);
    m_config[name] = value;
  } while (!input.eof());
}

// ------------------------------------------------------------

std::string trivialParser::getNextLine(std::ifstream& input) {
  //  std::cerr << "PG prima cerca " << std::endl ;
  std::string singleLine;
  do {
    getline(input, singleLine, '\n');
    //    std::cerr << "PG guardo " << singleLine << std::endl ;
  } while ((singleLine.find('#', 0) != std::string::npos || singleLine.find('=', 0) == std::string::npos ||
            singleLine.size() < 3) &&
           !input.eof());
  //  std::cerr << "PG trovato " << singleLine << std::endl ;
  return singleLine;
}

// ------------------------------------------------------------

void trivialParser::print(std::string prefix) {
  std::cerr << "read parameters: " << std::endl;
  for (std::map<std::string, double>::const_iterator mapIT = m_config.begin(); mapIT != m_config.end(); ++mapIT) {
    std::cerr << prefix << mapIT->first << " = " << mapIT->second << "\n";
  }
}

// ------------------------------------------------------------

void trivialParser::eraseSpaces(std::string& word) {
  while (word.find(' ', 0) != std::string::npos) {
    word.erase(word.find(' ', 0), 1);
  }
  return;
}
