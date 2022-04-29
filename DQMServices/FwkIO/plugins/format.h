#ifndef DQMServices_FwkIO_format_h
#define DQMServices_FwkIO_format_h
// -*- C++ -*-
//
// Package:     FwkIO
// Class  :     format
//
/**\class format format.h DQMServices/FwkIO/interface/format.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Tue May  3 15:33:16 CDT 2011
//

//These are the different types where each type has its own TTree
enum TypeIndex {
  kIntIndex,
  kFloatIndex,
  kStringIndex,
  kTH1FIndex,
  kTH1SIndex,
  kTH1DIndex,
  kTH2FIndex,
  kTH2SIndex,
  kTH2DIndex,
  kTH3FIndex,
  kTProfileIndex,
  kTProfile2DIndex,
  kTH1IIndex,
  kTH2IIndex,
  kNIndicies,
  kNoTypesStored = 1000
};

static const char* const kTypeNames[] = {"Ints",
                                         "Floats",
                                         "Strings",
                                         "TH1Fs",
                                         "TH1Ss",
                                         "TH1Ds",
                                         "TH2Fs",
                                         "TH2Ss",
                                         "TH2Ds",
                                         "TH3Fs",
                                         "TProfiles",
                                         "TProfile2Ds",
                                         "TH1Is",
                                         "TH2Is"};

//Branches for each TTree type
static const char* const kFullNameBranch = "FullName";
static const char* const kFlagBranch = "Flags";
static const char* const kValueBranch = "Value";

//Storage of Run and Lumi information
static const char* const kIndicesTree = "Indices";
static const char* const kRunBranch = "Run";
static const char* const kLumiBranch = "Lumi";
static const char* const kProcessHistoryIndexBranch = "ProcessHistoryIndex";
static const char* const kBeginTimeBranch = "BeginTime";
static const char* const kEndTimeBranch = "EndTime";
static const char* const kTypeBranch = "Type";
static const char* const kFirstIndex = "FirstIndex";
static const char* const kLastIndex = "LastIndex";

//File GUID
static const char* const kCmsGuid = "cms::edm::GUID";

//Meta data info
static const char* const kMetaDataDirectoryAbsolute = "/MetaData";
static const char* const kMetaDataDirectory = kMetaDataDirectoryAbsolute + 1;

static const char* const kProcessHistoryTree = "ProcessHistories";
static const char* const kPHIndexBranch = "Index";
static const char* const kProcessConfigurationProcessNameBranch = "ProcessName";
static const char* const kProcessConfigurationParameterSetIDBranch = "ParameterSetID";
static const char* const kProcessConfigurationReleaseVersion = "ReleaseVersion";
static const char* const kProcessConfigurationPassID = "PassID";

static const char* const kParameterSetTree = "ParameterSets";
static const char* const kParameterSetBranch = "ParameterSetBlob";
#endif
