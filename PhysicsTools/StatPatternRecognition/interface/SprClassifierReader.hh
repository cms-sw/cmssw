// File and Version Information:
//      $Id: SprClassifierReader.hh,v 1.1 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprClassifierReader :
//          Framework for reading stored classifiers from a file.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2006              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprClassifierReader_HH
#define _SprClassifierReader_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <utility>

class SprAbsClassifier;
class SprAbsTrainedClassifier;
class SprStdBackprop;
class SprTrainedTopdownTree;
class SprTrainedDecisionTree;
class SprTrainedFisher;
class SprTrainedLogitR;
class SprTrainedBinarySplit;
class SprAdaBoost;
class SprBagger;

class SprAbsFilter;
class SprAbsTwoClassCriterion;
class SprIntegerBootstrap;
class SprTrainedNode;


class SprClassifierReader
{
public:
  virtual ~SprClassifierReader() {}

  SprClassifierReader() {}

  // Reads trained classifier from file.
  static SprAbsTrainedClassifier* readTrained(const char* filename,
					      int verbose=0);

  // Reads trained classifier of specified type. Returns ownership.
  // Upon failure returns null pointer.
  template<class T> static T* readTrained(const char* filename, 
					  const char* classifier,
					  int verbose=0);

  // Sets the trainable classifier configuration 
  // to the one from the configuration file. Return false upon failure.
  static bool readTrainable(const char* filename, 
			    SprAbsClassifier* classifier,
			    int verbose=0);

  // Read classifier configurations from a file.
  static bool readTrainableConfig(std::ifstream& file,
				  unsigned& lineCounter,
				  SprAbsFilter* data,
				  bool discreteTree, bool mixedNodesTree,
				  std::vector<SprAbsTwoClassCriterion*>& crits,
				  std::vector<SprIntegerBootstrap*>& bstraps,
				  std::vector<SprAbsClassifier*>& classifiers,
				  std::vector<SprCCPair>& ccPairs,
				  bool readOneEntry=false);

private:
  friend class SprMultiClassReader;

  /*
    Reads classifier name from top of file.
  */
  static std::string readClassifierName(std::ifstream& file);

  /*
    Reads trained classifier from random position in a file.
  */
  static SprAbsTrainedClassifier* readTrainedFromFile(std::ifstream& file,
					       const std::string& classifier,
						      unsigned& lineCounter);

  /*
    Reads trained classifier of known type from random position in a file.
  */
  static SprAbsTrainedClassifier* readSelectedTrained(std::ifstream& file,
					       const std::string& classifier,
						      unsigned& lineCounter);

  // specific reader implementations
  static bool readStdBackprop(std::ifstream& file, 
			      SprStdBackprop* trainable, 
			      unsigned& lineCounter);
  static SprTrainedTopdownTree* readTopdownTree(std::ifstream& file, 
						unsigned& lineCounter);
  static SprTrainedDecisionTree* readDecisionTree(std::ifstream& file, 
						  unsigned& lineCounter);
  static SprTrainedFisher* readFisher(std::ifstream& file, 
				      unsigned& lineCounter);
  static SprTrainedLogitR* readLogitR(std::ifstream& file, 
				      unsigned& lineCounter);
  static SprTrainedBinarySplit* readBinarySplit(std::ifstream& file, 
						unsigned& lineCounter);
  static bool readAdaBoost(std::ifstream& file, 
			   SprAdaBoost* trainable, 
			   unsigned& lineCounter);
  static bool readBagger(std::ifstream& file, 
			 SprBagger* trainable, 
			 unsigned& lineCounter);

  // various helper typedefs
  typedef std::map<int,std::pair<SprTrainedNode*,std::pair<int,int> > > 
  SprTopdownNodeMap;

  // various helper methods
  static bool addTopdownTreeNode(SprTrainedNode* node, 
				 int dau1, int dau2,
				 SprTopdownNodeMap& mapped);
  static bool makeTopdownTreeNodeList(const SprTopdownNodeMap& mapped,
				      std::vector<const SprTrainedNode*>& 
				      nodes);
  static void prepareTopdownTreeExit(const SprTopdownNodeMap& mapped);
};

#endif
