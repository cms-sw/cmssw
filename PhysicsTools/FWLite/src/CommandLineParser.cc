// -*- C++ -*-
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>

#include "TString.h"

#include "PhysicsTools/FWLite/interface/CommandLineParser.h"
#include "PhysicsTools/FWLite/interface/dout.h"
#include "PhysicsTools/FWLite/interface/dumpSTL.icc"

using namespace std;
using namespace optutl;

const std::string CommandLineParser::kSpaces = " \t";

CommandLineParser::CommandLineParser(const string &usage, unsigned int optionsType)
    : m_argv0(""), m_usageString(usage), m_printOptions(true), m_optionsType(optionsType) {
  if (m_optionsType & kEventContOpt) {
    // Integer options
    addOption("totalSections", kInteger, "Total number of sections", 0);
    addOption("section", kInteger, "This section (from 1..totalSections inclusive)", 0);
    addOption("maxEvents", kInteger, "Maximum number of events to run over (0 for whole file)", 0);
    addOption("jobID", kInteger, "jobID given by CRAB,etc. (-1 means append nothing)", -1);
    addOption("outputEvery", kInteger, "Output something once every N events (0 for never)", 0);
    // String options
    addOption("outputFile", kString, "Output filename", "output.root");
    addOption("storePrepend", kString, "Prepend location on files starting with '/store/'");
    addOption("tag", kString, "A 'tag' to append to output file (e.g., 'v2', etc.)");
    // Bool options
    addOption("logName", kBool, "Print log name and exit");
    // String vector options
    addOption("inputFiles", kStringVector, "List of input files");
    addOption("secondaryInputFiles", kStringVector, "List of secondary input files (a.k.a. two-file-solution");
    addOption("orderedSecondaryFiles", kBool, "Are the secondary files ordered?", false);
    return;
  }
  // If we're still here, we have a type I don't understand.
  cerr << "CommandLineParser() Error: type '" << optionsType << "' is not understood.  Aborting." << endl;
  assert(0);
}

void CommandLineParser::parseArguments(int argc, char **argv, bool returnArgs) {
  bool callHelp = false;
  SVec argsVec;
  m_argv0 = argv[0];
  m_fullArgVec.push_back(argv[0]);
  for (int loop = 1; loop < argc; ++loop) {
    string arg = argv[loop];
    m_fullArgVec.push_back(arg);
    string::size_type where = arg.find_first_of("=");
    if (string::npos != where) {
      if (_setVariableFromString(arg)) {
        continue;
      }
      if (_runVariableCommandFromString(arg)) {
        continue;
      }
      cerr << "Don't understand: " << arg << endl;
      exit(0);
    }  // tag=value strings
    else if (arg.at(0) == '-') {
      string::size_type where = arg.find_first_not_of("-");
      if (string::npos == where) {
        // a poorly formed option
        cerr << "Don't understand: " << arg << endl;
        exit(0);
        continue;
      }
      lowercaseString(arg);
      char first = arg.at(where);
      // Print the values
      if ('p' == first) {
        m_printOptions = true;
        continue;
      }
      if ('n' == first) {
        m_printOptions = false;
        continue;
      }
      // Exit after printing values
      if ('h' == first) {
        callHelp = true;
        continue;
      }
      // if we're still here, then we've got a problem.
      cerr << "Don't understand: " << arg << endl;
      exit(0);
    }  // -arg strings
    if (returnArgs) {
      argsVec.push_back(arg);
    } else {
      cerr << "Don't understand: " << arg << endl;
      exit(0);
    }
  }  // for loop
  if (callHelp) {
    help();
  }
}

void CommandLineParser::help() {
  if (m_usageString.length()) {
    cout << m_argv0 << " - " << m_usageString << endl;
  }
  cout << "--help    - This screen" << endl
       << "--noPrint - Do not print out all settings" << endl
       << "--print   - Print out all settings" << endl;
  printOptionValues();
  exit(0);
}

void CommandLineParser::split(SVec &retval, string line, string match, bool ignoreComments) {
  if (ignoreComments) {
    removeComment(line);
  }  // if ignoreComments
  retval.clear();
  // find the first non-space
  string::size_type start1 = line.find_first_not_of(kSpaces);
  // Is the first non-space character a '#'
  char firstCh = line[start1];
  if ('#' == firstCh) {
    // this line is a comment
    return;
  }

  line += match;  // get last word of line
  string::size_type last = string::npos;
  string::size_type current = line.find_first_of(match);
  while (string::npos != current) {
    string::size_type pos;
    if (string::npos != last) {
      pos = last + 1;
    } else {
      pos = 0;
    }
    string part = line.substr(pos, current - last - 1);
    // don't bother adding 0 length strings
    if (part.length()) {
      retval.push_back(part);
    }
    last = current;
    current = line.find_first_of(match, current + 1);
  }  // while we're finding spaces
}

void CommandLineParser::removeComment(string &line) {
  string::size_type location = line.find("#");
  if (string::npos != location) {
    // we've got a comment.  Strip it out
    line = line.substr(0, location - 1);
  }  // if found
}

void CommandLineParser::removeLeadingAndTrailingSpaces(std::string &line) {
  string::size_type pos = line.find_first_not_of(kSpaces);
  if (string::npos == pos) {
    // we don't have anything here at all.  Just quit now
    return;
  }
  if (pos) {
    // We have spaces at the beginning.
    line = line.substr(pos);
  }
  pos = line.find_last_not_of(kSpaces);
  if (pos + 1 != line.length()) {
    // we've got spaces at the end
    line = line.substr(0, pos + 1);
  }
}

string CommandLineParser::removeEnding(const string &input, const string &ending) {
  string::size_type position = input.rfind(ending);
  if (input.length() - ending.length() == position) {
    // we've got it
    return input.substr(0, position);
  }
  // If we're still here, it wasn't there
  return input;
}

void CommandLineParser::findCommand(const string &line, string &command, string &rest) {
  command = rest = "";
  string::size_type nonspace = line.find_first_not_of(kSpaces);
  if (string::npos == nonspace) {
    // we don't have anything here at all.  Just quit now
    return;
  }
  string::size_type space = line.find_first_of(kSpaces, nonspace);
  if (string::npos == space) {
    // we only have a command and nothing else
    command = line.substr(nonspace);
    return;
  }
  command = line.substr(nonspace, space - nonspace);
  rest = line.substr(space + 1);
  removeLeadingAndTrailingSpaces(rest);
}

void CommandLineParser::printOptionValues() {
  cout << "------------------------------------------------------------------" << left << endl;
  // Print the integers next
  if (!m_integerMap.empty()) {
    cout << endl << "Integer options:" << endl;
  }
  for (SIMapConstIter iter = m_integerMap.begin(); m_integerMap.end() != iter; ++iter) {
    const string &description = m_variableDescriptionMap[iter->first];
    cout << "    " << setw(14) << iter->first << " = " << setw(14) << iter->second;
    if (description.length()) {
      cout << " - " << description;
    }
    cout << endl;
  }  // for iter

  // Print the doubles next
  if (!m_doubleMap.empty()) {
    cout << endl << "Double options:" << endl;
  }
  for (SDMapConstIter iter = m_doubleMap.begin(); m_doubleMap.end() != iter; ++iter) {
    const string &description = m_variableDescriptionMap[iter->first];
    cout << "    " << setw(14) << iter->first << " = " << setw(14) << iter->second;
    if (description.length()) {
      cout << " - " << description;
    }
    cout << endl;
  }  // for iter

  // Print the bools first
  if (!m_boolMap.empty()) {
    cout << endl << "Bool options:" << endl;
  }
  for (SBMapConstIter iter = m_boolMap.begin(); m_boolMap.end() != iter; ++iter) {
    const string &description = m_variableDescriptionMap[iter->first];
    cout << "    " << setw(14) << iter->first << " = " << setw(14);
    if (iter->second) {
      cout << "true";
    } else {
      cout << "false";
    }
    if (description.length()) {
      cout << " - " << description;
    }
    cout << endl;
  }  // for iter

  // Print the strings next
  if (!m_stringMap.empty()) {
    cout << endl << "String options:" << endl;
  }
  for (SSMapConstIter iter = m_stringMap.begin(); m_stringMap.end() != iter; ++iter) {
    const string &description = m_variableDescriptionMap[iter->first];
    cout << "    " << setw(14) << iter->first << " = ";
    const string value = "'" + iter->second + "'";
    cout << setw(14) << "";
    if (description.length()) {
      cout << " - " << description;
    }
    cout << endl << "        " << value << endl;
  }  // for iter

  // Integer Vec
  if (!m_integerVecMap.empty()) {
    cout << endl << "Integer Vector options:" << endl;
  }
  for (SIVecMapConstIter iter = m_integerVecMap.begin(); m_integerVecMap.end() != iter; ++iter) {
    const string &description = m_variableDescriptionMap[iter->first];
    cout << "    " << setw(14) << iter->first << " = ";
    dumpSTL(iter->second);
    cout << endl;
    if (description.length()) {
      cout << "      - " << description;
    }
    cout << endl;
  }  // for iter

  // Double Vec
  if (!m_doubleVecMap.empty()) {
    cout << endl << "Double Vector options:" << endl;
  }
  for (SDVecMapConstIter iter = m_doubleVecMap.begin(); m_doubleVecMap.end() != iter; ++iter) {
    const string &description = m_variableDescriptionMap[iter->first];
    cout << "    " << setw(14) << iter->first << " = ";
    dumpSTL(iter->second);
    cout << endl;
    if (description.length()) {
      cout << "      - " << description;
    }
    cout << endl;
  }  // for iter

  // String Vec
  if (!m_stringVecMap.empty()) {
    cout << endl << "String Vector options:" << endl;
  } else {
    cout << endl;
  }
  for (SSVecMapConstIter iter = m_stringVecMap.begin(); m_stringVecMap.end() != iter; ++iter) {
    const string &description = m_variableDescriptionMap[iter->first];
    cout << "    " << setw(14) << iter->first << " = ";
    if (description.length()) {
      cout << setw(14) << ""
           << " - " << description;
    }
    cout << endl;
    dumpSTLeachEndl(iter->second, 8);
  }  // for iter

  cout << "------------------------------------------------------------------" << right << endl;
}

bool CommandLineParser::_setVariableFromString(const string &arg, bool dontOverrideChange, int offset) {
  string::size_type where = arg.find_first_of("=", offset + 1);
  string varname = arg.substr(offset, where - offset);
  string value = arg.substr(where + 1);
  lowercaseString(varname);
  // check to make sure this is a valid option
  SBMapConstIter sbiter = m_variableModifiedMap.find(varname);
  if (m_variableModifiedMap.end() == sbiter) {
    // Not found.  Not a valid option
    return false;
  }
  // if 'dontOverrideChange' is set, then we are being asked to NOT
  // change any variables that have already been changed.
  if (dontOverrideChange && _valueHasBeenModified(varname)) {
    // don't go any further
    return true;
  }
  // integers
  SIMapIter integerIter = m_integerMap.find(varname);
  if (m_integerMap.end() != integerIter) {
    // we found it
    // use 'atof' instead of 'atoi' to get scientific notation
    integerIter->second = (int)atof(value.c_str());
    m_variableModifiedMap[varname] = true;
    return true;
  }
  // double
  SDMapIter doubleIter = m_doubleMap.find(varname);
  if (m_doubleMap.end() != doubleIter) {
    // we found it
    doubleIter->second = atof(value.c_str());
    m_variableModifiedMap[varname] = true;
    return true;
  }
  // string
  SSMapIter stringIter = m_stringMap.find(varname);
  if (m_stringMap.end() != stringIter) {
    // we found it
    stringIter->second = value;
    m_variableModifiedMap[varname] = true;
    return true;
  }
  // bool
  SBMapIter boolIter = m_boolMap.find(varname);
  if (m_boolMap.end() != boolIter) {
    // we found it
    boolIter->second = 0 != atoi(value.c_str());
    m_variableModifiedMap[varname] = true;
    return true;
  }
  // IntegerVec
  SIVecMapIter integerVecIter = m_integerVecMap.find(varname);
  if (m_integerVecMap.end() != integerVecIter) {
    // we found it
    SVec words;
    split(words, value, ",");
    for (SVecConstIter wordIter = words.begin(); words.end() != wordIter; ++wordIter) {
      integerVecIter->second.push_back((int)atof(wordIter->c_str()));
    }
    // we don't want to mark this as modified because we can add
    // many values to this
    // m_variableModifiedMap[varname] = true;
    return true;
  }
  // DoubleVec
  SDVecMapIter doubleVecIter = m_doubleVecMap.find(varname);
  if (m_doubleVecMap.end() != doubleVecIter) {
    // we found it
    SVec words;
    split(words, value, ",");
    for (SVecConstIter wordIter = words.begin(); words.end() != wordIter; ++wordIter) {
      doubleVecIter->second.push_back(atof(wordIter->c_str()));
    }
    // we don't want to mark this as modified because we can add
    // many values to this
    // m_variableModifiedMap[varname] = true;
    return true;
  }
  // StringVec
  SSVecMapIter stringVecIter = m_stringVecMap.find(varname);
  if (m_stringVecMap.end() != stringVecIter) {
    // we found it
    SVec words;
    split(words, value, ",");
    for (SVecConstIter wordIter = words.begin(); words.end() != wordIter; ++wordIter) {
      stringVecIter->second.push_back(*wordIter);
    }
    // we don't want to mark this as modified because we can add
    // many values to this
    // m_variableModifiedMap[varname] = true;
    return true;
  }
  // We didn't find your variable.  And we really shouldn't be here
  // because we should have know that we didn't find your variable.
  cerr << "CommandLineParser::SetVeriableFromString() Error: "
       << "Unknown variable and internal fault.  Aborting." << endl;
  assert(0);
  return false;
}

bool CommandLineParser::_setVariablesFromFile(const string &filename) {
  ifstream source(filename.c_str(), ios::in);
  if (!source) {
    cerr << "file " << filename << "could not be opened" << endl;
    return false;
  }
  string line;
  while (getline(source, line)) {
    // find the first nonspace
    string::size_type where = line.find_first_not_of(kSpaces);
    if (string::npos == where) {
      // no non-spaces
      continue;
    }
    char first = line.at(where);
    if ('-' != first) {
      continue;
    }
    where = line.find_first_not_of(kSpaces, where + 1);
    if (string::npos == where) {
      // no non-spaces
      continue;
    }
    // Get string starting at first nonspace after '-'.  Copy it to
    // another string without copying any spaces and stopping at the
    // first '#'.
    string withspaces = line.substr(where);
    string nospaces;
    for (int position = 0; position < (int)withspaces.length(); ++position) {
      char ch = withspaces[position];
      if ('#' == ch) {
        // start of a comment
        break;
      } else if (' ' == ch || '\t' == ch) {
        continue;
      }
      nospaces += ch;
    }  // for position
    if (!_setVariableFromString(nospaces, true)) {
      cerr << "Don't understand line" << endl
           << line << endl
           << "in options file '" << filename << "'.  Aborting." << endl;
      exit(0);
    }  // if setting variable failed
  }    // while getline
  return true;
}

bool CommandLineParser::_runVariableCommandFromString(const string &arg) {
  SVec equalWords;
  split(equalWords, arg, "=");
  if (2 != equalWords.size()) {
    return false;
  }
  SVec commandWords;
  split(commandWords, equalWords.at(0), "_");
  if (2 != commandWords.size()) {
    return false;
  }
  string &command = commandWords.at(1);
  lowercaseString(command);
  if (command != "load" && command != "clear") {
    return false;
  }
  const string &key = commandWords.at(0);
  OptionType type = hasOption(key);
  if (type < kIntegerVector || type > kStringVector) {
    cerr << "Command '" << command << "' only works on vectors." << endl;
    return false;
  }

  ///////////
  // Clear //
  ///////////
  if ("clear" == command) {
    if (kIntegerVector == type) {
      integerVector(key).clear();
    } else if (kDoubleVector == type) {
      doubleVector(key).clear();
    } else if (kStringVector == type) {
      stringVector(key).clear();
    } else {
      // If we're here, then I made a coding mistake and want to
      // know about it.
      assert(0);
    }
    return true;
  }

  //////////
  // Load //
  //////////
  const string &filename = equalWords.at(1);
  ifstream source(filename.c_str(), ios::in);
  if (!source) {
    cerr << "file " << filename << "could not be opened" << endl;
    return false;
  }
  string line;
  while (getline(source, line)) {
    // find the first nonspace
    string::size_type where = line.find_first_not_of(kSpaces);
    if (string::npos == where) {
      // no non-spaces
      continue;
    }
    // get rid of leading spaces
    line = line.substr(where);
    // get rid of trailing spaces
    where = line.find_last_not_of(kSpaces);
    if (line.length() - 1 != where) {
      line = line.substr(0, where + 1);
    }
    if ('#' == line.at(0)) {
      // this is a comment line, ignore it
      continue;
    }
    if (kIntegerVector == type) {
      integerVector(key).push_back((int)atof(line.c_str()));
    } else if (kDoubleVector == type) {
      doubleVector(key).push_back(atof(line.c_str()));
    } else if (kStringVector == type) {
      stringVector(key).push_back(line);
    } else {
      // If we're here, then I made a coding mistake and want to
      // know about it.
      assert(0);
    }
  }  // while getline
  return true;
}

void CommandLineParser::_getSectionFiles(const SVec &inputList, SVec &outputList, int section, int totalSections) {
  // Make sure the segment numbers make sense
  assert(section > 0 && section <= totalSections);

  // The Perl code:
  // my $entries    = @list;
  // my $perSection = int ($entries / $totalSections);
  // my $extra      = $entries % $totalSections;
  // --$section; # we want 0..n-1 not 1..n
  // my $start = $perSection * $section;
  // my $num   = $perSection - 1;
  // if ($section < $extra) {
  //    $start += $section;
  //    ++$num;
  // } else {
  //    $start += $extra;
  // };
  // my $end = $start + $num;
  int entries = (int)inputList.size();
  int perSection = entries / totalSections;
  int extra = entries % totalSections;
  --section;  // we want 0..n-1, not 1..n.
  int current = perSection * section;
  int num = perSection - 1;
  if (section < extra) {
    current += section;
    ++num;
  } else {
    current += extra;
  }
  outputList.clear();
  // we want to go from 0 to num inclusive, so make sure we have '<='
  // and not '='
  for (int loop = current; loop <= current + num; ++loop) {
    outputList.push_back(inputList.at(loop));
  }  // for loop
}

void CommandLineParser::_finishDefaultOptions(std::string tag) {
  if (!(m_optionsType & kEventContOpt)) {
    // nothing to see here, folks
    return;
  }
  ////////////////////////
  // Deal with sections //
  ////////////////////////
  if (integerValue("totalSections")) {
    // we have a request to break this into sections.  First, note
    // this in the output file
    tag += Form("_sec%03d", integerValue("section"));
    SVec tempVec;
    _getSectionFiles(stringVector("inputFiles"), tempVec, integerValue("section"), integerValue("totalSections"));
    stringVector("inputFiles") = tempVec;
  }  // if section requested

  //////////////////////
  // Store lfn to pfn //
  //////////////////////
  const string &kStorePrepend = stringValue("storePrepend");
  if (kStorePrepend.length()) {
    string match = "/store/";
    int matchLen = match.length();
    SVec tempVec;
    SVec &currentFiles = stringVector("inputFiles");
    for (SVecConstIter iter = currentFiles.begin(); currentFiles.end() != iter; ++iter) {
      const string &filename = *iter;
      if ((int)filename.length() > matchLen && filename.substr(0, matchLen) == match) {
        tempVec.push_back(kStorePrepend + filename);
      } else {
        tempVec.push_back(filename);
      }
    }
    currentFiles = tempVec;
  }  // if storePrepend.

  //////////////////////////////////
  // //////////////////////////// //
  // // Modify output filename // //
  // //////////////////////////// //
  //////////////////////////////////
  string outputFile = stringValue("outputFile");
  bool modifyOutputFile = (outputFile.length());
  outputFile = removeEnding(outputFile, ".root");
  outputFile += tag;
  if (integerValue("maxEvents")) {
    outputFile += Form("_maxevt%03d", integerValue("maxEvents"));
  }
  if (integerValue("jobID") >= 0) {
    outputFile += Form("_jobID%03d", integerValue("jobID"));
  }
  if (stringValue("tag").length()) {
    outputFile += "_" + stringValue("tag");
  }

  /////////////////////////////////
  // Log File Name, if requested //
  /////////////////////////////////
  if (boolValue("logName")) {
    cout << outputFile << ".log" << endl;
    exit(0);
  }
  outputFile += ".root";
  if (modifyOutputFile) {
    stringValue("outputFile") = outputFile;
  }

  // finally, if they asked us to print all options, let's do so
  // after we've modified all variables appropriately
  if (m_printOptions) {
    printOptionValues();
  }  // if printOptions
}

// friends
ostream &operator<<(ostream &o_stream, const CommandLineParser &rhs) { return o_stream; }
