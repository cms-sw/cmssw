//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprRootReader.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprRootReader :
//         read the list of root files that contains the data from a file
//         Open the root files and return the data from those.
//
//         The format of the file has to be as follows:
//
//         # comment
//         Tree: name_of_tree_in_file
//         ClassVariable:  name of variable in tree that gives the class
//         ClassVariable is otional and doesn't need to be included if unused
//          
//         WeightVariable: list of variables in the tree that determine 
//           the weight of the events.  The product of all of these variables
//           and the Normal Weight will be applied
//           This line is optional         
//
//         Leaves: name_of_leaf1 name_of_leaf2 ... # no spaces in names
//         Weight: 1.0 # This Weight is used until a different Weight is read
//         # If no Weight is given, 1.0 is assumed
//         File: /full/path/to/file1 event_range 0/1 # weight = 1.0
//         File: /full/path/to/file2 event_range 0/1 # weight = 1.0
//         Weight: 2.4
//         File: /full/path/to/file3 event_range 0/1 # weight = 2.4
//
//         where event_range is either: a-b # read events in the range [a, b)
//                                  or: a-  # read events in the range [a, EOF)
//                                  or: -b  # read events in the range [0, b)
//         Note: event_range may not contain whitespace characters
//         0: background
//         1: signal
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Jan Strube                      Original author
//      Josh Boehm                      Modified to include ClassVariable
//                                        and some error checking
// Copyright Information:
//      Copyright (C) 2005              University of Oregon
//      Copyright (C) 2006              Harvard University
//
// 2006-04-25:  Boehm modified the code to accept weights based on variables
//              present in the tree
//------------------------------------------------------------------------



#ifndef SPRROOTREADER_HH
#define SPRROOTREADER_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include <vector>
#include <string>

class SprAbsFilter;

struct FileInfo {
    std::string name;
    int start;
    int end;
    int fileClass;
    double weight;    
};

class SprRootReader : public SprAbsReader 
{
private:
    SprAbsFilter* readRootObjects(bool);

    std::string treeName_;
    std::vector<std::string> leafNames_;
    std::vector<FileInfo> fileObjects_;
    bool hasSpecialClassifier_;
    std::string classifierVarName_;
    std::vector<std::string> weightLeafNames_;
    
public:
    SprRootReader();
    bool chooseVars(const std::set<std::string>& vars);
    bool chooseAllBut(const std::set<std::string>& vars);
    void chooseAll();
    SprAbsFilter* read(const char* filename);
};

#endif
