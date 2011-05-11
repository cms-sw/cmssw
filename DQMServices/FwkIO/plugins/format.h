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
// $Id$
//


//These are the different types where each type has its own TTree
enum TypeIndex {kIntIndex, kFloatIndex, kStringIndex,
                kTH1FIndex, kTH1SIndex, kTH1DIndex,
                kTH2FIndex,kTH2SIndex, kTH2DIndex, kTH3FIndex,
                kTProfileIndex,kTProfile2DIndex,kNIndicies};

static const char* const kTypeNames[]={"Ints","Floats","Strings",
                                       "TH1Fs","TH1Ss","TH1Ds",
                                       "TH2Fs", "TH2Ss", "TH2Ds",
                                       "TH3Fs", "TProfiles","TProfile2Ds"};

//Branches for each TTree type
static const char* const kFullNameBranch = "FullName";
static const char* const kFlagBranch = "Flags";
static const char* const kValueBranch = "Value";


static const char* const kIndicesTree = "Indices";
static const char* const kRunBranch = "Run";
static const char* const kLumiBranch = "Lumi";
static const char* const kTypeBranch = "Type";
static const char* const kFirstIndex = "FirstIndex";
static const char* const kLastIndex = "LastIndex";

#endif
